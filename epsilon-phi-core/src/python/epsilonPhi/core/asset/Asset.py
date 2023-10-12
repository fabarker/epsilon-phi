import pandas as pd
from epsilonPhi.core.asset.CAssetInf import CAssetInf
from epsilonPhi.core.asset.AssetMgr import CAssetMgr
from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.core.estimator.estimationMgr import EstimationMgr
from epsilonPhi.core.schema.Schema import CContext
from typing import Optional
import numpy as np

ts_type: TimeSeriesType = TimeSeriesType.LEVELS
class CAsset(CTimeSeries, CAssetInf):
    def __init__(self,
                 schema,
                 ticker,
                 dataframe=None,
                 currency: Optional[str] = None,
                 ts_type=TimeSeriesType.RETURNS
                 ):

        if isinstance(dataframe, pd.DataFrame):
           assert len(dataframe.columns) == 1, 'Error - dataframe must be single return time series'

        if schema is not None and dataframe is not None:
            dataframe = dataframe.reindex(schema.dates)
        super(CAsset, self).__init__(data=dataframe,
                                     ts_type=ts_type,
                                     )

        self._ticker = ticker
        self._currency = currency
        self._schema = schema
        self._assetMgr = CAssetMgr(schema)
        self.frequency = schema.frequency

    @property
    def asset_ticker(self):
        return self._ticker
    @property
    def currency(self):
        return self._currency
    @property
    def name(self):
        return self._ticker
    @property
    def frequency(self):
        return self._frequency
    @frequency.setter
    def frequency(self, frequency):
        self._frequency = frequency

    def create_new_object(self, *args, **kwargs):
        return self.__class__(self._schema,
                              self._ticker,
                              dataframe=kwargs.get('data', None),
                              currency=self.currency,
                              ts_type=kwargs.get('ts_type', None))

    def get_frequency(self):
        return self._frequency

    def get_currency(self):
        return self._currency

    def get_risk_free_asset(self):
        return self._assetMgr.get_risk_free_asset(self.currency)

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

    self = CAsset(ticker='MSGWLDL', currency='USD', schema=schema, dataframe=rtns)







