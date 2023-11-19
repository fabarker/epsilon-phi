from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries
from epsilonPhi.core.factor.factorMgr import CFactorMgr
import numpy as np

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

    def get_historical_risk_premia(self):
        return np.mean(self.values) * self.frequency.yearfrac()

    def get_historical_Sharpe(self):
        return (np.mean(self.values) * self.frequency.yearfrac()) /\
                    np.std(self.values, ddof=1) * np.sqrt(self.frequency.yearfrac())

    def get_historical_volatility(self):
        return np.std(self.values, ddof=1) * np.sqrt(self.frequency.yearfrac())

    def get_Sharpe_ratio(self):
        pass

    def get_risk_premium(self):
        pass






