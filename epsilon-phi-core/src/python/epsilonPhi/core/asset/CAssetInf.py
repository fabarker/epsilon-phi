from abc import ABC, abstractmethod

__author__ = 'Francis Barker'
__date__ = '01/07/2023'

class CAssetInf(ABC):

    @property
    @abstractmethod
    def name(self):
        pass

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
