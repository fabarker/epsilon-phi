from epsilonPhi.core.dataModel.alchemist.SessionManager import *
from epsilonPhi.core.utils.DateUtils import DateUtils
from epsilonPhi.core.dataModel.enums.ImpliedVolatility import *
from epsilonPhi.core.utils.OptionUtils import *

sessionMgr = SessionMgr()
session = sessionMgr.getSessionFactory()

class VolSurfaceMgr(object):
    _raw_cache = {}

    def __init__(self, underlier, pricing_location=None):

        self._underlier = underlier
        self._pricing_location = pricing_location
        self._strike_reference = None
        self._currency = None
        self._interest_rate_curve = None
        self._spec = None
        self._datasource = None
        self._validate_underlier()
        self._construct_vol_surface_data()

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
    def ds(self):
        if self._datasource is None:
            from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
            self._datasource = GlobalDataSource()
        return self._datasource

    def _load_spot_prices(self):

        if self.category == 'FX':
           spt = self.ds.get_fx_spot_rates(self.ticker)
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
        return self._spot_prices

    def _load_interest_rate_curve(self):
        self._interest_rate_curve = self.ds.get_interest_rate_curve_for_region(self.region)

    def get_interest_rate_curve(self):
        if self._interest_rate_curve is None:
            self._load_interest_rate_curve()
        return self._interest_rate_curve.copy()

    def get_vol_surface_data(self):
        if self.underlier not in self._raw_cache.keys():
            self._load_vol_surface_data()
        return self._raw_cache.get(self.underlier)

    def _load_vol_surface_data(self):

        if self.underlier not in self._raw_cache.keys():
            q = session.query(ImpliedVolatility).filter(ImpliedVolatility.security ==
                                                        self._underlier)

            if self._pricing_location:
                q = q.filter(ImpliedVolatility.pricing_location == self._pricing_location)

            df__ = sessionMgr.query_format_df(q).set_index(['uid', 'date'])
            df__ = df__.dropna(how='all', axis=1)
            df__['maturity'] = DateUtils.Rdate_to_mat(df__['tenor'].values)
            df__['days'] = np.round(df__['maturity'] * DateUtils.days_per_year, 0).astype(int)
            df__['expiration'] = df__.index.get_level_values(1) + pd.to_timedelta(df__['days'], unit='D')

            self._pricing_location = df__.get('pricing_location').drop_duplicates().values[0]
            self._strike_reference = df__.get('strike_reference').drop_duplicates().values[0]

            if self._strike_reference in ['spot', 'forward']:
               df__['relative_strike'] = np.round((1/100) * df__['relative_strike'].astype(float), 2)
            elif self._strike_reference in ['delta']:
               df__ = df__.replace('DN', '-999')
               df__['relative_strike'] = df__['relative_strike'].astype(int)

            self._raw_cache[self._underlier] = df__.drop(columns=['pricing_location', 'strike_reference'])
            _q = session.query(ImpliedVolatilitySpec).filter(ImpliedVolatilitySpec.security == self._underlier)
            self._spec = sessionMgr.query_format_df(_q).T.to_dict().get(0)

    def _construct_data_from_delta_reference(self):
        ivols = self.get_vol_surface_data()

    def _construct_data_from_moneyness_reference(self):
        pass

    def _construct_vol_surface_data(self):
        self._load_vol_surface_data()

        if self.strike_reference.lower() == 'delta':
            self._construct_data_from_delta_reference()
        elif self.strike_reference.lower() in ['spot', 'forward']:
            self._construct_data_from_moneyness_reference()
        else:
            raise ValueError('Error - {} not supported'.format(self.strike_reference))



    def _validate_underlier(self):
        if not session.query(exists().where(ImpliedVolatility.security ==
                                            self._underlier)).scalar():
            raise ValueError('Error: {} ticker not supported'.format(self._underlier))


if __name__ == "__main__":

    underlier = 'EURUSD'
    self = VolSurfaceMgr(underlier, pricing_location='NYC')
