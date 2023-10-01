from epsilonPhi.core.timeSeries.timeSeriesMain import CSlice
from abc import ABC, abstractmethod

class AbstractFactor(CSlice, ABC):
    def __init__(self, df=None, factorName=None):
        super(AbstractFactor, self).__init__(df)
        self._factor_name = factorName

    @abstractmethod
    def get_time_series_from_database(self,
                                      ticker,
                                      endDate,
                                      frequency):
        pass

    @abstractmethod
    def construct_factor_time_series(self):
        pass

    @abstractmethod
    def to_frequency(self):
        pass


