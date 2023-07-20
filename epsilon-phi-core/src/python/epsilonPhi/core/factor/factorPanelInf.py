from abc import ABC, abstractmethod

class CFactorPanelInf(ABC):

    @abstractmethod
    def get_risk_factor_list(self):
        pass

    def get_return_factor_list(self):
        pass

    def get_risk_factor_panel(self):
        pass

    def get_return_factor_panel(self):
        pass

    def get_risk_factor_df(self):
        pass

    def get_return_factor_df(self):
        pass

    def get_risk_factor_covariance(self):
        pass

    def get_risk_factor(self, risk_factor_name):
        pass

    def get_return_factor(self, return_factor_name):
        pass

