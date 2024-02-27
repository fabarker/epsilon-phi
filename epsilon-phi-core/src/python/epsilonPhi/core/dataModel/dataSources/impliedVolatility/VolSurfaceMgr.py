from epsilonPhi.core.dataModel.alchemist.SessionManager import *
from epsilonPhi.core.dataModel.dataSources.curves.interestRateCurve.IRCurve import IRCurve
from epsilonPhi.core.dataModel.dataSources.curves.abstractCurve.AbstractCurve import AbstractCurve
from epsilonPhi.core.utils.OptionUtils import *
from epsilonPhi.core.dataModel.enums.Database import PriceQuote

sessionMgr = SessionMgr()
session = sessionMgr.getSessionFactory()
_DROP_COLS = ['uid','pricing_location','relative_strike','strike_reference','tenor','security']

class VolSurfaceMgr(object):
    _raw_cache = {}
    _ivols = {}

    def __init__(self, 
                 underlier,
                 pricing_location=None,
                 cross_section=None):

        self._underlier = underlier
        self._pricing_location = pricing_location
        self._strike_reference = None
        self._maturities = None
        self._currency = None

        self._spec = None
        self._surf = None
        self._datasource = None

        self._interest_rate_curve = None
        self._funding_rate_curve = None
        self._spot_prices = None

        self._validate_underlier()

        # Load Data
        self._load_vol_surface_spec()


    @property
    def pricing_location(self):
        return self._pricing_location
    @property
    def strike_reference(self):
        return self._strike_reference
    @property
    def security(self):
        return self.underlier
    @property
    def underlier(self):
        return self._underlier
    @property
    def ticker(self):
        return self._spec.get('ticker')
    @property
    def exchange(self):
        return self._pricing_location
    @property
    def currency(self):
        return self._spec.get('currency')
    @property
    def name(self):
        return self._spec.get('name')
    @property
    def region(self):
        return self._spec.get('region')
    @property
    def category(self):
        return self._spec.get('category')
    @property
    def delta_convention(self):
        return deltaConvention.get(self.underlier, DeltaType.SPOT_DELTA)
    @property
    def maturities(self):
        return self._maturities

    @property
    def ds(self):
        if self._datasource is None:
            from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
            self._datasource = GlobalDataSource()
        return self._datasource

    def _load_vol_surface_spec(self):
        _q = (session.query(ImpliedVolatilitySpec).
              filter(ImpliedVolatilitySpec.security == self._underlier))
        self._spec = sessionMgr.query_format_df(_q).T.to_dict().get(0)

    def _load_spot_prices(self):

        if self.category == 'FX':
           spt = self.ds.get_fx_spot_rates(self.ticker, PriceQuote.BID.value)
        elif self.category in ['Equity Index', 'ETF']:
           spt = self.ds.get_time_series_data_from_ticker(self.ticker, cols='PI')
        elif self.category == 'Future':
           spt = self.ds.get_front_futures_continuous_series_settlement_price(self.ticker[0:3])
        else:
            raise ValueError('Error - category {} not recognized'.format(self.category))

        spt.columns = [self.security]
        self._spot_prices = spt.copy()

    def get_spot_prices(self):
        if self._spot_prices is None:
           self._load_spot_prices()
        return self._spot_prices.copy()

    def _load_interest_rate_curve(self):
        self._interest_rate_curve = IRCurve(region=self.region, type=['Interbank', 'Deposit'])

    def get_interest_rate_curve(self, maturities=None, stack=False):

        if self._interest_rate_curve is None:
            self._load_interest_rate_curve()

        if stack:
            return self._interest_rate_curve.get_curve(maturities).stack()
        else:
            return self._interest_rate_curve.get_curve(maturities)

    def _load_funding_rate_curve(self):

        if self.category == 'FX':
            if self.ticker[0:3] == 'EUR':
               fx = 'DEM'
            else:
               fx = self.ticker[0:3]
            rf = IRCurve(currency=fx, type=['Interbank', 'Deposit']).get_curve()
        elif self.category in ['Equity Index', 'ETF']:
            rf = self.ds.get_time_series_data_from_ticker(self.ticker, cols='DY')
            rf = rf.resample('B').ffill() / 100
            rf.columns = [0]
        elif self.category == 'Future':
            rf = self.ds.get_front_futures_continuous_series_settlement_price(self.ticker[0:3])
        else:
            raise ValueError('Error - category {} not recognized'.format(self.category))
        self._funding_rate_curve = AbstractCurve(rf)

    def get_funding_rate_curve(self, maturities=None, stack=False):

        if self._funding_rate_curve is None:
            self._load_funding_rate_curve()

        if stack:
            return self._funding_rate_curve.get_curve(maturities).stack()
        else:
            return self._funding_rate_curve.get_curve(maturities)

    def get_ivols(self):
        if self.underlier not in self._ivols.keys():
            self._load_ivols()
        return self._ivols.get(self.underlier)

    def _load_ivols(self):

        if self.underlier not in self._raw_cache.keys():

            print('Loading Vol Surface Data for Security {}'.format(self.underlier))
            q = session.query(ImpliedVolatility).filter(ImpliedVolatility.security ==
                                                        self._underlier)

            if self._pricing_location:
                q = q.filter(ImpliedVolatility.pricing_location == self._pricing_location)
            df = sessionMgr.query_format_df(q).dropna(how='all', axis=1)

            # remove weekends
            _dates = pd.to_datetime(df.date.values)
            drop_rows = (_dates.dayofweek == 5) | (_dates.dayofweek == 6)
            df = df.iloc[~drop_rows, :]

            df['t'] = DateUtils.Rdate_to_mat(df['tenor'].values)
            df['days'] = np.round(df['t'] * DateUtils.days_per_year, 0).astype(int)
            df['mat'] = df.get('date') + pd.to_timedelta(df['days'], unit='D')

            self._maturities = np.sort(df.t.unique())
            self._pricing_location = df.get('pricing_location').drop_duplicates().values[0]
            self._strike_reference = df.get('strike_reference').drop_duplicates().values[0]

            idx = df.set_index(['date','t']).index
            df['rd'] = self.get_interest_rate_curve(self.maturities, True).loc[idx].values
            df['rf'] = self.get_funding_rate_curve(self.maturities, True).loc[idx].values
            df['s'] = self.get_spot_prices().loc[idx.get_level_values(0)].values
            df['f'] = df['s'] * np.exp(-df['rf'] * df['t']) / np.exp(-df['rd'] * df['t'])

            dc = self.delta_convention
            if self.strike_reference.lower() == 'delta':
                _ref = 'delta'
                df = df.replace('DN', '-99900')
                df['delta'] = df['relative_strike'].astype(int) / 100
                df['k'] = solve_for_strike(df['s'], df['t'], df['rd'], df['rf'], np.sign(df['del']), df['del'], dc, df['mid'])
                df['mn'] = df['k'] / df['s']
            elif self.strike_reference.lower() in ['spot', 'forward']:
                _ref = 'mn'
                df['mn'] = np.round((1/100) * df['relative_strike'].astype(float), 2)
                df['k'] = df[self.strike_reference[0]] *  df['mn']

                _opts = [(-1, -1), (1, 1), (-1, 0), (1, 0)]
                ivols = [pd.DataFrame()]
                for opt in _opts:
                    ivols.extend([self.__solve_for_deltas(df, opt[0], opt[1])])
                df = pd.concat(ivols, axis=0)
            else:
                raise FinError('Error - strike_reference {} not supported'.format(self.strike_reference))

            df['lmn'] = np.log(df['mn'])
            df['zp'] = df['lmn'] + 0.5 * df['mid'] * df['mid'] * df['t']
            df['zm'] = df['zp'] - df['mid'] * df['mid'] * df['t']
            df['x'] = df['zp'] / (df['mid'] * np.sqrt(df['t']))
            ivols = df.drop(columns=_DROP_COLS)
            self._ivols[self._underlier] = ivols.set_index(['date','t', _ref])

    def __solve_for_deltas(self, _df, opt_type, mny_type):

        if np.sign(mny_type) == 0:
            idx = _df['mn'].values == 1
        elif np.sign(mny_type) == -1:
            idx = _df['mn'].values < 1
        elif np.sign(mny_type) == 1:
            idx = _df['mn'].values > 1
        else:
            raise ValueError('Error')

        _tmp = _df.iloc[idx, :].copy()
        _tmp['delta'] = fast_delta(_tmp['s'],
                                   _tmp['t'],
                                   _tmp['k'],
                                   _tmp['rd'],
                                   _tmp['rf'],
                                   _tmp['mid'],
                                   self.delta_convention,
                                   opt_type)
        return _tmp.copy()

    def _validate_underlier(self):
        if not session.query(exists().where(ImpliedVolatility.security ==
                                            self._underlier)).scalar():
            raise ValueError('Error: {} ticker not supported'.format(self._underlier))


if __name__ == "__main__":

    underlier = 'SPX'
    self = VolSurfaceMgr(underlier)
    df = self.get_ivols()
