from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType
from epsilonPhi.core.timeSeries.timeSeriesMain import CSlice, CTimeSeries
from epsilonPhi.core.asset.CAssetInf import CAssetInf, _assetmeta
from epsilonPhi.core.estimator.estimationMgr import EstimationMgr
from epsilonPhi.core.config.appConfig import CAppConfig
from epsilonPhi.core.asset.AssetMgr import CAssetMgr
from epsilonPhi.core.schema.Schema import CContext
from typing import Optional, Union
import numpy as np

ts_type: TimeSeriesType = TimeSeriesType.LEVELS
class CAsset(CAssetInf, CSlice):

    _metadata = _assetmeta

    @property
    def _constructor(self):
        def _c(*args, **kwargs):
            return CAsset(*args, **kwargs).__finalize__(self)
        return _c

    def __init__(self,
                 data: Optional[Union[CTimeSeries, CSlice]],
                 schema: Optional[CContext] = None,
                 denominated_currency: Optional[str] = None,
                 exposure_currency: Optional[str] = None,
                 ts_hedge_ratio: Optional[Union[float, int]] = None,
                 returns_type: Optional[ReturnsType] = None,
                 ts_type: Optional[TimeSeriesType] = None,
                 **kwargs
                 ) -> None:

        series = CAssetMgr._prepare_dataframe_for_asset(schema, data)
        super(CAsset, self).__init__(data=series,
                                     returns_type=returns_type,
                                     ts_type=ts_type,
                                     **kwargs
                                     )

        self._denominated_currency = denominated_currency
        self._exposure_currency = exposure_currency
        self._schema = schema
        self._assetMgr = CAssetMgr(schema)
        self._hedge_ratio = None
        self._alpha = None
        self._risk_betas = {}
        self._return_betas = {}

        if ((ts_hedge_ratio is not None) and
                (self.denominated_currency != self.exposure_currency)):
            self._ts_hedge_ratio = float(ts_hedge_ratio)
        else:
            self._ts_hedge_ratio = 0

    def deepcopy(self):
        return super().deepcopy()

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

    def set_currency_hedge_ratio(self, hedge_ratio: Optional[Union[float, int]]):
        if hedge_ratio is not None:
            assert 0 <= hedge_ratio <= 1, 'Error - Currency hedge ratio must be in interval [0,1]'
            assert isinstance(hedge_ratio, (float, int)), 'Error - Currency hedge ratio must be of type float or int'
        self._hedge_ratio = hedge_ratio

    def set_alpha(self, alpha):
        self._alpha = alpha


    ##################### Asset risk free rate ###########################

    def get_risk_free_asset(self):
        return self.assetMgr.get_risk_free_asset(self.denominated_currency)


    ########### Historical Asset Class Performance Metrics ###############

    def get_historical_total_return(self, from_date=None, to_date=None):
        return EstimationMgr.get_historical_total_return(self)

    def get_historical_sharpe_ratio(self, from_date=None, to_date=None):
        return EstimationMgr.get_historical_sharpe_ratio(self[from_date:to_date])

    def get_historical_volatility(self, from_date=None, to_date=None):
        return EstimationMgr.get_historical_volatility(self[from_date:to_date])

    def get_historical_risk_premium(self, from_date=None, to_date=None):
        return EstimationMgr.get_historical_risk_premium(self[from_date:to_date])

    def get_excess_return_df(self, from_date=None, to_date=None):
        return EstimationMgr.get_excess_return_timeseries(self[from_date:to_date])

    def get_historical_value_at_risk(self, horizon=1, confidence=0.99, from_date=None, to_date=None):
        return EstimationMgr.get_historical_value_at_risk(self[from_date:to_date], horizon=horizon, confidence=confidence)

    def get_historical_conditional_value_at_risk(self, horizon=1, confidence=0.99, from_date=None, to_date=None):
        return EstimationMgr.get_historical_conditional_value_at_risk(self[from_date:to_date], horizon=horizon, confidence=confidence)

    def get_historical_probability_of_loss(self, horizon=1, from_date=None, to_date=None):
        return EstimationMgr.get_historical_probability_of_loss(self[from_date:to_date], horizon=horizon)

    def get_historical_worst_peak_to_trough(self, from_date=None, to_date=None):
        return EstimationMgr.get_historical_worst_peak_to_trough(self[from_date:to_date])

    def get_historical_equity_beta(self, from_date=None, to_date=None):
        return EstimationMgr.get_historical_equity_beta(self[from_date:to_date])

    def get_historical_alpha_over_equity(self, from_date=None, to_date=None):
        return EstimationMgr.get_historical_alpha_over_equity(self[from_date:to_date])

    def get_historical_skewness(self, from_date=None, to_date=None):
        return EstimationMgr.get_historical_skewness(self[from_date:to_date])

    def get_historical_worst_period_return(self, period=1, from_date=None, to_date=None):
        return EstimationMgr.get_historical_worst_period_return(self[from_date:to_date], period=period)

    def get_historical_best_period_return(self, period=1, from_date=None, to_date=None):
        return EstimationMgr.get_historical_best_period_return(self[from_date:to_date], period=period)


    ################### Factor Model Asset Metrics ###################

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
        return CAppConfig.get_estimation_mgr().get_risk_betas(self, 1)

    def get_fx_risk_betas(self):
        return self.get_return_betas() - self.get_asset_risk_betas()

    def get_fx_risk_decomposition(self):
        pass

    def get_idiosyncratic_variance(self):
        return CAppConfig.get_estimation_mgr().get_idiosyncratic_variance(self, self.hedging_ratio)

    def get_volatility(self, hedging_ratio=None):

        if hedging_ratio is None:
           hedging_ratio = self.hedging_ratio
        return EstimationMgr.get_risk_factor_stdev(self, hedging_ratio)

    def get_data_length(self):
        return EstimationMgr.get_estimation_length(self)

    def get_beta_and_idio_risk(self, hedging_ratio=None):

        if hedging_ratio is None:
           hedging_ratio = self.hedging_ratio
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

    df = gds.get_time_series_data_from_ticker('MSWRLDL','RI')
    rtns = df.get_returns()

    from epsilonPhi.core.schema.Schema import ContextCreator
    schema = ContextCreator(currency='GBP',
                            start_date='30-Nov-1983',
                            end_date='31-Dec-2022').create_context()

    self = CAsset(exposure_currency='WLD',
                  denominated_currency='USD',
                  schema=schema,
                  data=rtns,
                  ts_hedge_ratio=0,
                  returns_type=rtns.returns_type,
                  ts_type=rtns.type)

    self.set_currency_hedge_ratio(0)

    self.get_risk_premia()
    vol = self.get_volatility()
    beta, idio = self.get_beta_and_idio_risk(0)
    rb = self.get_asset_risk_betas()
    fx = self.get_fx_risk_betas()

    rtn = self.get_historical_total_return()
    sr = self.get_historical_sharpe_ratio()
    vol = self.get_historical_volatility()
    er = self.get_historical_risk_premium()
    mdd = self.get_historical_worst_peak_to_trough()

    beta = self.get_historical_equity_beta()
    alpha = self.get_historical_alpha_over_equity()
    skew = self.get_historical_skewness()

    max_rtn = self.get_historical_best_period_return()
    min_rtn = self.get_historical_worst_period_return()

    var = self.get_historical_value_at_risk()
    cvar = self.get_historical_conditional_value_at_risk()
    pol = self.get_historical_probability_of_loss()








