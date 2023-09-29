from abc import ABC, abstractmethod

class CFactorPanelInf(ABC):

    @abstractmethod
    def get_risk_factor_list(self):
        pass

    @abstractmethod
    def get_return_factor_list(self):
        pass

    @abstractmethod
    def get_risk_factor_panel(self):
        pass

    @abstractmethod
    def get_return_factor_panel(self):
        pass

    @abstractmethod
    def get_risk_factors_data(self):
        pass

    @abstractmethod
    def get_return_factors_data(self):
        pass

    @abstractmethod
    def get_risk_factor_covariance(self):
        pass

    @abstractmethod
    def get_risk_factor(self, risk_factor_name):
        pass

    @abstractmethod
    def get_return_factor(self, return_factor_name):
        pass

    @staticmethod
    def get_return_factor_Sharpe_ratios(self):
        pass



