from abc import ABC, abstractmethod

class CPortfolioInf(ABC):

    @abstractmethod
    def get_historical_risk_premia(self):
        pass

    @abstractmethod
    def get_excess_return_time_series(self):
        pass

    @abstractmethod
    def get_historical_sharpe_ratio(self):
        pass

    @abstractmethod
    def get_data_length(self):
        pass

    @abstractmethod
    def get_return_betas(self, normalized=True):
        pass

    @abstractmethod
    def get_risk_premia(self):
        pass

    @abstractmethod
    def get_total_return(self):
        pass

    @abstractmethod
    def get_hedging_ratio(self):
        pass

    @abstractmethod
    def get_beta_and_idio_risk(self, hedging_ratio):
        pass
    
    @abstractmethod
    def get_risk_std(self):
        pass
