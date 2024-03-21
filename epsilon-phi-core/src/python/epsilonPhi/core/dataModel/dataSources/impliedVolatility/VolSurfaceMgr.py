from epsilonPhi.core.dataModel.alchemist.SessionManager import *
from epsilonPhi.core.dataModel.dataSources.curves.interestRateCurve.IRCurve import IRCurve
from epsilonPhi.core.dataModel.dataSources.curves.abstractCurve.AbstractCurve import AbstractCurve
from epsilonPhi.core.utils.OptionUtils import *
from epsilonPhi.core.dataModel.enums.Database import PriceQuote

sessionMgr = SessionMgr()
session = sessionMgr.getSessionFactory()

_DROP_COLS = ['uid', 'pricing_location', 'relative_strike', 'strike_reference', 'tenor', 'security']

_RELATIVE_STRIKE = 'relative_strike'
_STRIKE_REFERENCE = 'strike_reference'
_TENOR = 'tenor'
_MATURITY = 't'

# Default interpolation points for converting between strike reference space
_DELTA_POINTS = np.arange(0.05, 1, 0.05)
_MONEYNESS_POINTS = np.arange(50, 155, 5)/100
_ZSCORE_POINTS = np.arange(-3, 3.5, 0.5)
_CONVEXITY_MNY_POINTS = np.arange(-3, 3.5, 0.1)


class VolSurfaceMgr(object):
    _raw_cache = {}
    _ivol_cache = {}
    _VOL_TOL = 0.02

    def __init__(self, 
                 underlier,
                 pricing_location=None,
                 strike_reference=None):

        # Optional Parameters
        self._underlier = underlier
        self._pricing_location = pricing_location
        self._strike_reference = strike_reference

        # Properties of the raw data
        self._currency = None

        # Spec for the ivols
        self._spec = None
        self._datasource = None

        # Asset Prices and Interest/Funding Rates
        self._rate_curve = None
        self._funding_curve = None
        self._spot_prices = None
        self._forward_prices = None
        self._s = None
        self._rd = None
        self._rf = None
        self._put_deltas = None
        self._call_deltas = None

        # Validate the target surface. i.e make sure
        # we have the data
        self._validate_underlier()

        # Load Data
        self._load_vol_surface_spec()
        self._load_ivols()

    @property
    def dates(self):
        return pd.to_datetime(self.ivols.index)
    @property
    def pricing_location(self):
        return self._pricing_location
    @property
    def strike_reference(self):
        return self._strike_reference
    @property
    def ivols(self):
        return self.get_ivols()
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
    def mids(self):
        return self.ivols.get('mid').values.reshape(-1, 1)
    @property
    def sig(self):
        return self.ivols.get('mid').values.reshape(-1, 1)
    @property
    def t(self):
        return self.ivols.get(_MATURITY).values.reshape(-1, 1)
    @property
    def k(self):
        return self.ivols.get('k').values.reshape(-1, 1)
    @property
    def s(self):
        if self._s is None:
            self._s = self.get_spot_prices(self.dates).values.reshape(-1, 1)
        return self._s
    @property
    def f(self):
        if self._forward_prices is None:
            self._forward_prices = self.get_forward_prices(self.dates,
                                   self.t).values.reshape(-1, 1)
        return self._forward_prices
    @property
    def rd(self):
        if self._rd is None:
            self._rd = self._rate_curve.get_stacked_curve(self.dates, self.t).values.reshape(-1, 1)
        return self._rd
    @property
    def rf(self):
        if self._rf is None:
            self._rf = self._funding_curve.get_stacked_curve(self.dates, self.t).values.reshape(-1, 1)
        return self._rf
    @property
    def strike_references(self):
        return self.ivols.get(_STRIKE_REFERENCE)
    @property
    def relative_strikes(self):
        return self.ivols.get(_RELATIVE_STRIKE)

    @property
    def ds(self):
        if self._datasource is None:
            from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
            self._datasource = GlobalDataSource()
        return self._datasource

    ################ LOAD METHODS ############

    def _load_ivols(self):

        if self.underlier not in self._ivol_cache.keys():

            # Load raw ivols data from database
            df = sessionMgr.get_ivols(self.underlier,
                                      self.pricing_location,
                                      index='date')

            df = df[df.mid > self._VOL_TOL]
            df = df.reset_index(drop=False).drop_duplicates(subset=['date', 'relative_strike', 'tenor']).set_index('date', drop=True)

            if 'k' not in df.columns:
                # Estimate strikes from spot moneyness
                _spt_idx = df[_STRIKE_REFERENCE].values == 'spot'
                df.loc[_spt_idx, 'k'] = self.get_strikes_from_moneyness(df[_RELATIVE_STRIKE][_spt_idx])

                # Estimate strikes from forward moneyness
                _fwd_idx = df[_STRIKE_REFERENCE].values == 'forward'
                df.loc[_fwd_idx, 'k'] = self.get_strikes_from_forward_moneyness(df[_RELATIVE_STRIKE][_fwd_idx],
                                                                                df[_MATURITY][_fwd_idx])
                # Estimates strikes from deltas
                _del_idx = df[_STRIKE_REFERENCE].values == 'delta'
                df.loc[_del_idx, 'k'] = self.get_strikes_from_deltas(df['mid'][_del_idx],
                                                                     df[_MATURITY][_del_idx],
                                                                     df[_RELATIVE_STRIKE][_del_idx],
                                                                     self.delta_convention)
                # Estimate the strikes for delta neutral points
                _DN_idx = df.relative_strike == -999
                df.loc[_DN_idx, 'k'] = self.get_strikes_from_atm_delta_neutral(df['mid'][_DN_idx],
                                                                               df[_MATURITY][_DN_idx],
                                                                               self.delta_convention)


            self._ivol_cache[self.underlier] = df[['mid','t','k','relative_strike']].copy()
            self._ivol_cache[self.underlier][self.strike_reference] = self.get_strike_reference(self.strike_reference)

    def _load_vol_surface_spec(self):
        self._spec = sessionMgr.get_ivol_spec(self._underlier)

    def _load_spot_prices(self):

        if self.category == 'FX':
           spt = self.ds.get_fx_spot_rates(self.ticker, PriceQuote.BID.value)
        elif self.category in ['Equity Index', 'ETF']:
           spt = self.ds.get_time_series_data_from_ticker(self.ticker, cols='PI')
        elif self.category == 'Future':
           spt = self.ds.get_front_futures_continuous_series_settlement_price(self.ticker[0:3])
        else:
            raise ValueError('Error - category {} not recognized'.format(self.category))

        self._spot_prices = pd.Series(spt.values.flatten(),
                                      index=spt.index,
                                      name=self.security)

    def _load_interest_rate_curve(self):
        self._rate_curve = IRCurve(region=self.region,
                                   type=['Interbank', 'Deposit'])
        self._rate_curve._curve_df = self._rate_curve._curve_df * 0

    def _load_funding_rate_curve(self):

        if self.category == 'FX':
            if self.ticker[0:3] == 'EUR':
               fx = 'DEM'
            else:
               fx = self.ticker[0:3]
            rf = IRCurve(currency=fx, type=['Interbank', 'Deposit']).get_curve()
        elif self.category in ['Equity Index', 'ETF']:
            rf = self.ds.get_time_series_data_from_ticker(self.ticker, cols='DY')
            rf = pd.concat([rf] * 2, axis=1).resample('B').ffill() / 100
            rf.columns = [0, 10]
        elif self.category == 'Future':
            rf = self.ds.get_front_futures_continuous_series_settlement_price(self.ticker[0:3])
        else:
            raise ValueError('Error - category {} not recognized'.format(self.category))
        self._funding_curve = AbstractCurve(rf * 0)



    ############### Public Getter Methods ###############

    def get_ivols(self):
        if self.underlier not in self._ivol_cache.keys():
            self._load_ivols()
        return self._ivol_cache.get(self.underlier)

    def get_put_deltas(self):
        if self._put_deltas is None:
           self._put_deltas = fast_delta(self.s,
                                          self.t,
                                          self.k,
                                          self.rd,
                                          self.rf,
                                          self.sig,
                                          self.delta_convention, -1)
        return self._put_deltas.copy()

    def get_call_deltas(self):
        if self._call_deltas is None:
           self._call_deltas = fast_delta(self.s,
                                          self.t,
                                          self.k,
                                          self.rd,
                                          self.rf,
                                          self.sig,
                                          self.delta_convention, 1)
        return self._call_deltas.copy()

    def get_spot_prices(self, dates):

        if self._spot_prices is None:
           self._load_spot_prices()
        return self._spot_prices.loc[dates]

    def get_interest_rates(self, pricing_dates, maturities):

        if self._rate_curve is None:
            self._load_interest_rate_curve()

        return self._rate_curve.get_stacked_curve(pricing_dates,
                                                  np.array(maturities))

    def get_funding_rates(self, pricing_dates, maturities):

        if self._funding_curve is None:
            self._load_funding_rate_curve()
        return self._funding_curve.get_stacked_curve(pricing_dates,
                                                     np.array(maturities))


    def get_forward_prices(self, pricing_dates, maturities):

        _s = self.get_spot_prices(pricing_dates)
        _rf = self.get_funding_rates(pricing_dates, maturities)
        _rd = self.get_interest_rates(pricing_dates, maturities)
        return (_s.values * np.exp(-_rf * np.array(maturities).flatten()) /
                np.exp(-_rd * np.array(maturities).flatten()))


    def get_strikes_from_deltas(self, ivols, t, delta, delta_type):
        if ivols.size > 0:
            _s = self.get_spot_prices(ivols.index).values
            _rf = self.get_funding_rates(ivols.index, t).values
            _rd = self.get_interest_rates(ivols.index, t).values
            return solve_for_strike(_s,
                                    t,
                                    _rd,
                                    _rf,
                                    np.sign(delta),
                                    delta,
                                    delta_type,
                                    ivols)

    def get_strikes_from_atm_delta_neutral(self, ivols, t, deltaTypeValue):
        if ivols.size > 0:
            _s = self.get_spot_prices(ivols.index)
            _rf = self.get_funding_rates(ivols.index, t)
            _rd = self.get_interest_rates(ivols.index, t)
            return atm_delta_neutral_strike(_s.values,
                                            t,
                                            _rd.values,
                                            _rf.values,
                                            ivols,
                                            deltaTypeValue)

    def get_strikes_from_forward_moneyness(self, x, t):
        if x.size > 0:
            return x * self.get_forward_prices(x.index, t).droplevel(1)

    def get_strikes_from_moneyness(self, x):
        if x.size > 0:
            return x * self.get_spot_prices(x.index).values.flatten()

    def get_strikes_from_log_moneyness(self, x, t):
        if x.size > 0:
            return self.get_forward_prices(x.index, t).droplevel(1) * np.exp(x)

    def get_atm_delta_neutral_deltas(self, ivols, t, deltaTypeValue, option_type_value):

        # Get Forward Prices
        rf = self.get_forward_prices(ivols.index, t)
        return delta_from_delta_neutral_straddle_quote(t,
                                                       rf,
                                                       ivols,
                                                       deltaTypeValue,
                                                       option_type_value)

    def get_log_moneyness(self):
        return np.log(self.k / self.f)

    def get_deltas(self, option_type):
        return np.round(fast_delta(self.s,
                                   self.t,
                                   self.k,
                                   self.rd,
                                   self.rf,
                                   self.sig,
                                   self.delta_convention,
                                   option_type), 3)

    def get_convexity_moneyness(self):

        # Compute the convexity adjusted moneyness measure
        sigsq = np.power(self.sig, 2)
        lnm = self.get_log_moneyness()
        zp = lnm + 0.5 * sigsq * self.t
        return zp / (self.sig * np.sqrt(self.t))

    def get_zscore(self):
        lmn = self.get_log_moneyness()
        return lmn / (self.sig * np.sqrt(self.t))

    def get_moneyness(self):
        return np.exp(self.get_log_moneyness())

    def get_strike_reference(self, strike_reference):

        if strike_reference in [StrikeReference.DELTA]:
            return self.get_deltas(1)
        if strike_reference in [StrikeReference.MONEYNESS]:
            return self.get_moneyness()
        if strike_reference in [StrikeReference.LOG_MONEYNESS]:
            return self.get_log_moneyness()
        if strike_reference in [StrikeReference.Z_SCORE]:
            return self.get_zscore()
        if strike_reference in [StrikeReference.CONVEXITY_MN]:
            return self.get_convexity_moneyness()
        else:
            return df.copy()

    def _validate_underlier(self):
        if not session.query(exists().where(ImpliedVolatility.security == self.underlier)).scalar():
            raise ValueError('Error: {} ticker not supported'.format(self._underlier))


if __name__ == "__main__":

        from epsilonPhi.core.dataModel.dataSources.impliedVolatility.Models import *

        underlier = 'GBPUSD'
        self = VolSurfaceMgr(underlier,
                             pricing_location='NYC',
                             strike_reference=StrikeReference.DELTA)

        _DELTAS = [0.1, 0.25, 0.5, 0.75, 0.9]
        _MATS = np.round([1/12, 3/12, 6/12, 12/12], 10)

        _ivols = self.get_ivols().reset_index(drop=False).set_index(['date','t',StrikeReference.DELTA]).get('mid')
        vols = _ivols[~_ivols.index.duplicated()]

        idx_d = vols.index.get_level_values(2).isin([0.1, 0.25, 0.5, 0.75, 0.9])
        idx_m = vols.index.get_level_values(1).isin(_MATS)
        idx = np.logical_and(idx_d, idx_m)

        _vol = vols.iloc[idx].unstack(level=[1, 2]).sort_index(axis=1).sort_index().loc['24-Jan-1996':'13-Oct-2022']
        _vol = _vol[~_vol.index.dayofweek.isin([5, 6])]




