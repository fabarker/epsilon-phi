from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType
from epsilonPhi.core.timeSeries.timeSeriesMain import CSlice
import numpy as np
import math

ts_type: TimeSeriesType = TimeSeriesType.LEVELS


class CFactor(CSlice):
    def __init__(self,
                 data=None,
                 name=None,
                 ts_type=TimeSeriesType.RETURNS,
                 returns_type=ReturnsType.SIMPLE,
                 **kwargs,
                 ):

        super(CFactor, self).__init__(data=data,
                                      ts_type=ts_type,
                                      name=name,
                                      returns_type=returns_type,
                                      **kwargs,
                                      )

    @property
    def _constructor(self):
        def _c(*args, **kwargs):
            return CFactor(*args, **kwargs).__finalize__(self)
        return _c

    def _cast_derived_class(self, klass):
        self.__init__(data=klass,
                      ts_type=klass.type,
                      returns_type=klass.returns_type)

    def deepcopy(self):
        return self.create_new_object(data=self,
                                      name=self.name,
                                      ts_type=self.type,
                                      returns_type=self.returns_type)

    def create_new_object(self, *args, **kwargs):
        return self.__class__(*args, **kwargs)

    def get_historical_risk_premia(self):
        return np.mean(self.values) * math.sqrt(self.obs_per_year)

    def get_historical_Sharpe(self):
        return np.mean(self.values) / np.std(self.values, ddof=1) * math.sqrt(self.obs_per_year)

    def get_historical_volatility(self):
        return np.std(self.values, ddof=1) * math.sqrt(self.obs_per_year)







