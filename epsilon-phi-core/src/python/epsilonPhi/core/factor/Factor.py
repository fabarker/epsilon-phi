from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.core.timeSeries.timeSeriesMain import CSlice
import numpy as np
import math

ts_type: TimeSeriesType = TimeSeriesType.LEVELS


class CFactor(CSlice):
    def __init__(self,
                 series=None,
                 name=None,
                 ts_type=TimeSeriesType.RETURNS,
                 **kwargs,
                 ):

        super(CFactor, self).__init__(data=series,
                                      ts_type=ts_type,
                                      name=name,
                                      **kwargs,
                                      )

    @property
    def _constructor(self):
        def _c(*args, **kwargs):
            return CFactor(*args, **kwargs).__finalize__(self)
        return _c

    def _cast_derived_class(self, klass):
        self.__init__(series=klass,
                      ts_type=klass.type,
                      returns_type=klass.returns_type)

    def deepcopy(self):
        return self.create_new_object(series=self,
                                      name=self.name,
                                      ts_type=self.type,
                                      returns_type=self.returns_type,
                                      attributes=self.attributes)

    def create_new_object(self, *args, **kwargs):
        return CFactor(series=kwargs.get('data', None),
                       name=kwargs.get('name', None),
                       ts_type=kwargs.get('ts_type', None),
                       returns_type=kwargs.get('returns_type', None),
                       attributes=kwargs.get('attributes', None))

    def get_historical_risk_premia(self):
        return np.mean(self.values) * math.sqrt(self.obs_per_year)

    def get_historical_Sharpe(self):
        return np.mean(self.values) / np.std(self.values, ddof=1) * math.sqrt(self.obs_per_year)

    def get_historical_volatility(self):
        return np.std(self.values, ddof=1) * math.sqrt(self.obs_per_year)







