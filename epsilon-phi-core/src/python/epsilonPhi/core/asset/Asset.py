import pandas as pd
from epsilonPhi.core.asset.CAssetInf import CAssetInf
from epsilonPhi.core.asset.AssetMgr import CAssetMgr
from epsilonPhi.core.timeSeries.timeSeriesMain import CSlice, CTimeSeries
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType
from epsilonPhi.core.estimator.estimationMgr import EstimationMgr
from epsilonPhi.core.schema.Schema import CContext
from typing import Optional
import numpy as np

ts_type: TimeSeriesType = TimeSeriesType.LEVELS
class CAsset(CTimeSeries, CAssetInf):
    def __init__(self,
                 dataframe=None,
                 ticker: Optional[str] = None,
                 currency: Optional[str] = None,
                 schema: Optional[CContext] = None,
                 ):

        if isinstance(dataframe, pd.DataFrame):
           assert len(dataframe.columns) == 1, 'Error - dataframe must be single return time series'

        super(CAsset, self).__init__(data=dataframe,
                                     ts_type=TimeSeriesType.RETURNS,
                                     )

        self._ticker = ticker
        self._currency = currency
        self._schema = schema
        self._assetMgr = CAssetMgr(schema)

    @property
    def asset_ticker(self):
        return self._ticker
    @property
    def currency(self):
        return self._currency
    @property
    def name(self):
        return self._ticker

    def get_frequency(self):
        return self._schema.frequency

    def get_currency(self):
        return self._currency

    def get_risk_free_asset(self):
        return self._assetMgr.get_risk_free_asset(self.currency,
                                                  self.frequency)

    def get_alpha(self):
        pass

    def set_alpha(self):
        pass

    def get_historical_sharpe_ratio(self):
        pass

    def get_historical_volatility(self):
        pass

    def get_historical_risk_premium(self):
        pass

    def get_excess_return_df(self):
        pass

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

    def get_hedging_option(self, hedging_option):
        self._hedging_option = hedging_option




if __name__ == "__main__":

    gds = GlobalDataSource()

    df = gds.get_time_series_data_from_ticker('MSGWLDL','RI')
    rtns = df.get_returns()


    from epsilonPhi.core.schema.Schema import ContextCreator
    schema = ContextCreator(currency='GBP',
                            start_date='31-Dec-1999',
                            end_date='31-Dec-2022').create_context()

    asset = CAsset(ticker='MSGWLDL', currency='USD', schema=schema, dataframe=rtns)







