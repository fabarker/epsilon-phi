from abc import ABC, abstractmethod
from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries, CSlice
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType

_assetmeta = ["_denominated_currency", "_exposure_currency", "_schema", "_assetMgr", "_hedge_ratio", "_reporting_name",
              "_category",
              "_ts_hedge_ratio", "_added_attributes", "_type", "_name", "_returns_type", "_alpha", "_return_betas",
              "_risk_betas"]

__author__ = 'Francis Barker'
__date__ = '01/07/2023'


class CAssetInf(ABC):

    def _create_new_object_same_type(self, data=None, returns_type=None):
        return self.create_new_object(data=data,
                                      schema=self.schema,
                                      denominated_currency=self.denominated_currency,
                                      exposure_currency=self.exposure_currency,
                                      ts_hedge_ratio=self._ts_hedge_ratio,
                                      returns_type=returns_type,
                                      ts_type=self.type)

    def _create_new_levels_object(self, data=None, returns_type=None):
        return self.create_new_object(data=data,
                                      schema=self.schema,
                                      denominated_currency=self.denominated_currency,
                                      exposure_currency=self.exposure_currency,
                                      ts_hedge_ratio=self._ts_hedge_ratio,
                                      returns_type=returns_type,
                                      ts_type=TimeSeriesType.LEVELS)

    def _create_new_returns_object(self, returns_type, data=None):
        return self.create_new_object(data=data,
                                      schema=self.schema,
                                      denominated_currency=self.denominated_currency,
                                      exposure_currency=self.exposure_currency,
                                      ts_hedge_ratio=self._ts_hedge_ratio,
                                      returns_type=returns_type,
                                      ts_type=TimeSeriesType.RETURNS)

    def _cast_derived_class(self, klass):
        klass._constructor(klass).__finalize__(self)

    def create_new_object(self, *args, **kwargs):
        return self.__class__(*args, **kwargs)

    @abstractmethod
    def get_historical_risk_premium(self):
        pass

    @abstractmethod
    def get_excess_return_df(self):
        pass

    @abstractmethod
    def get_historical_sharpe_ratio(self):
        pass

    @property
    @abstractmethod
    def denominated_currency(self):
        pass

    @property
    @abstractmethod
    def exposure_currency(self):
        pass

    @abstractmethod
    def get_data_length(self):
        pass

    @abstractmethod
    def get_return_betas(self):
        pass

    @abstractmethod
    def get_risk_premia(self):
        pass

    @abstractmethod
    def get_total_return(self):
        pass

    @abstractmethod
    def get_risk_betas(self):
        pass

    @abstractmethod
    def get_idiosyncratic_variance(self):
        pass

    @abstractmethod
    def get_volatility(self):
        pass
