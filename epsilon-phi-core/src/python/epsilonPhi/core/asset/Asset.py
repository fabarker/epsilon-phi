from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.estimator.estimationMgr import EstimationMgr
from epsilonPhi.core.config.appConfig import CAppConfig
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.core.timeSeries.timeSeriesMain import CSlice
from epsilonPhi.core.asset.CAssetInf import CAssetInf
from epsilonPhi.core.asset.AssetMgr import CAssetMgr
from typing import Optional
import numpy as np

ts_type: TimeSeriesType = TimeSeriesType.LEVELS
class CAsset(CSlice, CAssetInf):

    _metadata = ["_denominated_currency",
                 "_exposure_currency",
                 "_schema",
                 "_assetMgr",
                 "_hedge_ratio",
                 "_ts_hedge_ratio",
                 "_added_attributes",
                 "_type",
                 "_name",
                 "_returns_type",
                 "_alpha"]

    @property
    def _constructor(self):
        def _c(*args, **kwargs):
            return CAsset(*args, **kwargs).__finalize__(self)
        return _c

    def __init__(self,
                 dataframe,
                 schema=None,
                 denominated_currency=None,
                 exposure_currency=None,
                 ts_hedge_ratio=None,
                 ts_type=TimeSeriesType.RETURNS,
                 **kwargs
                 ):

        series = CAssetMgr._prepare_dataframe_for_asset(schema, dataframe, ts_type)
        super(CAsset, self).__init__(data=series,
                                     ts_type=ts_type,
                                     **kwargs
                                     )

        self._denominated_currency = denominated_currency
        self._exposure_currency = exposure_currency
        self._schema = schema
        self._assetMgr = CAssetMgr(schema)
        self._hedge_ratio = None
        self._alpha = 0

        if ((ts_hedge_ratio is not None) and
                (self.denominated_currency != self.exposure_currency)):
            self._ts_hedge_ratio = float(ts_hedge_ratio)
        else:
            self._ts_hedge_ratio = 0

    def _cast_derived_class(self, klass):
        self.__init__(dataframe=klass,
                      schema=klass.schema,
                      denominated_currency=klass.denominated_currency,
                      exposure_currency=klass.exposure_currency,
                      ts_hedge_ratio=klass._ts_hedge_ratio,
                      ts_type=klass.type,
                      returns_type=klass.returns_type,
                      attributes=klass.attributes)

    def deepcopy(self):
        return self.create_new_object(dataframe=self,
                                      schema=self.schema,
                                      denominated_currency=self.denominated_currency,
                                      exposure_currency=self.exposure_currency,
                                      ts_hedge_ratio=self._ts_hedge_ratio,
                                      ts_type=self.type,
                                      returns_type=self.returns_type,
                                      attributes=self.attributes)

    def _create_new_object_same_type(self, data=None, attributes=None, returns_type=None):
        return self.create_new_object(data, attributes, self.type, returns_type)

    def create_new_object(self, *args, **kwargs):
        return self.__class__(schema=kwargs.get('schema', self.schema),
                              dataframe=kwargs.get('data', self),
                              denominated_currency=kwargs.get('denominated_currency', self.denominated_currency),
                              exposure_currency=kwargs.get('exposure_currency', self.exposure_currency),
                              ts_hedge_ratio=kwargs.get('ts_hedge_ratio', self._ts_hedge_ratio),
                              ts_type=kwargs.get('ts_type', self.type),
                              returns_type=kwargs.get('returns_type', self.returns_type),
                              attributes=kwargs.get('attributes', self.attributes))

    ###############

    @property
    def denominated_currency(self):
        return self._denominated_currency
    @property
    def exposure_currency(self):
        return self._exposure_currency
    @property
    def obs_per_year(self):
        return self.frequency.obs_per_year()
    @property
    def frequency(self):
        return self.schema.frequency
    @property
    def schema(self):
        return self._schema
    @property
    def assetMgr(self):
        return self._assetMgr
    @property
    def hedging_ratio(self):
        return self._hedge_ratio
    @property
    def is_time_series_in_local_terms(self):
        return self.denominated_currency == self.exposure_currency

    ###############

    def set_currency_hedge_ratio(self, hedge_ratio: Optional[float, int]):
        assert hedge_ratio >= 0 and hedge_ratio <= 1, 'Error - Currency hedge ratio must be in interval [0,1]'
        assert isinstance(hedge_ratio, float) and isinstance(hedge_ratio, int), 'Error - Currency hedge ratio must be of type float or int'
        self._hedge_ratio = hedge_ratio

    def set_alpha(self, alpha):
        self._alpha = alpha

    ############

    def get_risk_free_asset(self):
        return self.assetMgr.get_risk_free_asset(self.denominated_currency)

    def get_historical_total_return(self, from_date, to_date):
        pass

    def get_historical_sharpe_ratio(self, from_date=None, to_date=None):
        return EstimationMgr.get_historical_Sharpe_ratio(self)

    def get_historical_volatility(self, from_date=None, to_date=None):
        return EstimationMgr.get_historical_volatility(self)

    def get_historical_risk_premium(self, from_date=None, to_date=None):
        return EstimationMgr.get_historical_risk_premia(self)

    def get_excess_return_df(self, from_date=None, to_date=None):
        return EstimationMgr.get_excess_return_timeseries(self)

    def get_historical_value_at_risk(self, horiozon, confidence):
        pass
    def get_historical_conditional_value_at_risk(self, horizon, confidence):
        pass

    def get_historical_PoL(self, horizon):
        pass

    def get_historical_worst_peak_to_trough(self):
        pass

    def get_historical_max_drawdown(self):
        pass

    def get_historical_beta(self):
        pass

    def get_historical_skewness(self):
        pass


    ##############

    def get_return_betas(self, normalized=True):
        betas = CAppConfig.get_estimation_mgr().get_return_betas(self, normalized=True)
        return betas

    def get_return_betas_not_normalized(self):
        return self.get_return_betas(False)

    def get_risk_premia(self):
        return CAppConfig.get_estimation_mgr().get_risk_premium(self)

    def get_total_return(self):
        return self.get_risk_premia() + self._schema.get_risk_free_rate() + self.get_alpha()

    def get_Sharpe_ratio(self):
        return (np.sum(self.get_risk_premia()) + self.get_alpha()) / self.get_volatility()

    def get_alpha(self):
        return self._alpha

    def get_risk_betas(self):
        return CAppConfig.get_estimation_mgr().get_risk_betas(self, self.hedging_ratio)

    def get_asset_risk_betas(self):
        pass

    def get_fx_risk_betas(self):
        pass

    def get_fx_risk_decomposition(self):
        pass

    def get_idiosyncratic_variance(self):
        return CAppConfig.get_estimation_mgr().get_idiosyncratic_variance(self, self.hedging_ratio)

    def get_volatility(self):

        betas = self.get_risk_betas()
        idio = self.get_idiosyncratic_variance()

        factor_covariance = self.schema.get_risk_factor_covariance()
        systematic_var = betas @ factor_covariance @ betas
        return np.sqrt(systematic_var + idio)

    def get_data_length(self):
        return EstimationMgr.get_estimation_length(self)

    def get_beta_and_idio_risk(self, hedging_ratio=None):
        if hedging_ratio is None:
            return CAppConfig.get_estimation_mgr().get_beta_and_idio_variance(self, self.hedging_ratio)
        else:
            return CAppConfig.get_estimation_mgr().get_beta_and_idio_variance(self, hedging_ratio)

    def get_standard_error(self):
        pass

    def get_residuals(self):
        pass

    def get_uncertainty(self):
        pass

    ##############

    def convert_asset_to_currency(self, currency, hedging_ratio):
        return self.assetMgr.convert_asset_to_currency(self, currency, hedging_ratio)

    def simulate(self):
        pass

    def brownian_bridge(self):
        pass



if __name__ == "__main__":

    gds = GlobalDataSource()

    df = gds.get_time_series_data_from_ticker('MSWRLD$','RI')
    rtns = df.get_returns()

    from epsilonPhi.core.schema.Schema import ContextCreator
    schema = ContextCreator(currency='GBP',
                            start_date='30-Nov-1983',
                            end_date='31-Dec-2022').create_context()

    self = CAsset(exposure_currency='WLD', denominated_currency='USD', schema=schema, dataframe=rtns, ts_hedge_ratio=0)
    self.set_asset_hedging_ratio(0)
    betas = self.get_volatility()







