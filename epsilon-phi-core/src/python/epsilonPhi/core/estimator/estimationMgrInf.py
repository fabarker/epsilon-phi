from abc import ABC, abstractmethod

class CEStimationMgrInf(ABC):

    @staticmethod
    @abstractmethod
    def get_risk_premium(asset):
        pass

    @staticmethod
    @abstractmethod
    def get_excess_return_timeseries(asset):
        pass

    @staticmethod
    @abstractmethod
    def get_return_betas(asset, hedging_ratio, normalized=True):
        pass

    @staticmethod
    @abstractmethod
    def get_idiosyncratic_variance(asset, hedging_ratio):
        pass

    @staticmethod
    @abstractmethod
    def get_risk_betas(asset, hedging_ratio):
        pass