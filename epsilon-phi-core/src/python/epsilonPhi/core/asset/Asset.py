from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.estimator.estimationMgr import EstimationMgr
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries
from epsilonPhi.core.asset.CAssetInf import CAssetInf
from epsilonPhi.core.asset.AssetMgr import CAssetMgr
from typing import Optional
import pandas as pd
import numpy as np

ts_type: TimeSeriesType = TimeSeriesType.LEVELS
class CAsset(CTimeSeries, CAssetInf):
    def __init__(self,
                 dataframe=None,
                 schema=None,
                 denominated_currency: Optional[str] = None,
                 exposure_currency: Optional[str] = None,
                 ts_type=TimeSeriesType.RETURNS
                 ):

        df_ = CAssetMgr._prepare_dataframe_for_asset(schema, dataframe, ts_type)
        super(CAsset, self).__init__(data=df_,
                                     ts_type=ts_type,
                                     )

        self.__denominated_currency = denominated_currency
        self.__exposure_currency = exposure_currency
        self.__schema = schema
        self.__assetMgr = CAssetMgr(schema)

    @property
    def denominated_currency(self):
        return self.__denominated_currency
    @property
    def exposure_currency(self):
        return self.__exposure_currency
    @property
    def obs_per_year(self):
        return self.frequency.obs_per_year()
    @property
    def name(self):
        if isinstance(self.columns, pd.MultiIndex):
            return self.columns.get_level_values(0)[0]
        else:
            return self.columns[0]

    @property
    def frequency(self):
        return self.schema.frequency
    @property
    def schema(self):
        return self.__schema
    @property
    def assetMgr(self):
        return self.__assetMgr
    def create_new_object(self, *args, **kwargs):
        return self.__class__(schema=self.schema,
                              dataframe=kwargs.get('data', None),
                              denominated_currency=self.denominated_currency,
                              exposure_currency=self.exposure_currency,
                              ts_type=kwargs.get('ts_type', None))

    def get_risk_free_asset(self):
        return self.assetMgr.get_risk_free_asset(self.denominated_currency)

    def get_alpha(self):
        pass

    def set_alpha(self):
        pass

    def get_historical_sharpe_ratio(self):
        return EstimationMgr.get_historical_Sharpe_ratio(self)

    def get_historical_volatility(self):
        return EstimationMgr.get_historical_volatility(self)

    def get_historical_risk_premium(self):
        return EstimationMgr.get_historical_risk_premia(self)

    def get_excess_return_df(self):
        return EstimationMgr.get_excess_return_timeseries(self)

    def get_Sharpe_ratio(self):
        pass

    def get_data_length(self):
        pass

    def get_return_betas(self, normalized=True):
        betas = self._schema.getEstimationMgr().getReturnBetas(self, normalized=True)
        return betas

    def get_return_betas_not_normalized(self):
        return self.get_return_betas(False)

    def get_total_return(self):
        return self.get_risk_premia() + self._schema.get_risk_free_rate() + self.get_alpha()

    def get_risk_betas(self):
        return self._schema.getEstimationMgr().get_risk_betas(self, self.hedging_ratio)

    def get_idiosyncratic_variance(self):
        return self._schema.getEstimationMgr().get_idiosyncratic_variance(self, self.hedging_ratio)

    def get_volatility(self):

        betas = self.get_risk_betas()
        idio = self.get_idiosyncratic_variance()

        factor_covariance = self._schema.get_factor_panels().get_risk_factor_covariance_matrix()
        systematic_var = betas @ factor_covariance @ betas
        return np.sqrt(systematic_var + idio)

    def get_risk_premia(self):
        return self._schema.getEstimationMgr().get_risk_premium(self)

    def set_hedging_ratio(self, hedging_ratio):
        self._hedging_ratio = hedging_ratio


if __name__ == "__main__":

    gds = GlobalDataSource()

    df = gds.get_time_series_data_from_ticker('MSGWLDL','RI')
    rtns = df.get_returns()


    from epsilonPhi.core.schema.Schema import ContextCreator
    schema = ContextCreator(currency='GBP',
                            start_date='31-Dec-1999',
                            end_date='31-Dec-2022').create_context()

    self = CAsset(ticker='MSGWLDL', currency='USD', schema=schema, dataframe=rtns)







