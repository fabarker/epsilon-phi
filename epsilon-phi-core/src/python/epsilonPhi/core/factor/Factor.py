from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries
import numpy as np
import math

ts_type: TimeSeriesType = TimeSeriesType.LEVELS


class CFactor(CTimeSeries):
    def __init__(self,
                 dataframe=None,
                 ts_type=TimeSeriesType.RETURNS
                 ):

        super(CFactor, self).__init__(data=dataframe,
                                      ts_type=ts_type,
                                      )

    def create_new_object(self, data=None, attributes=None, ts_type=None, returns_type=None):
        newObj = CFactor(dataframe=data, ts_type=ts_type)
        return newObj

    def deepcopy(self):
        return CFactor(dataframe=self, ts_type=self.type)

    def get_historical_risk_premia(self):
        return np.mean(self.values) * math.sqrt(self.obs_per_year)

    def get_historical_Sharpe(self):
        return np.mean(self.values) / np.std(self.values, ddof=1) * math.sqrt(self.obs_per_year)

    def get_historical_volatility(self):
        return np.std(self.values, ddof=1) * math.sqrt(self.obs_per_year)







