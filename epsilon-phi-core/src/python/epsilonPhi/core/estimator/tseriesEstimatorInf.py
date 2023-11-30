from abc import ABC, abstractmethod
from typing import Optional, Union
import pandas as pd


class CTimeSeriesEstimatorInf(ABC):
    pass

    @abstractmethod
    def get_historical_total_return(self,
                                    from_date=None,
                                    to_date=None) -> float:
        pass

    @abstractmethod
    def get_historical_excess_return_df(self,
                                       currency,
                                       from_date=None,
                                       to_date=None):
        pass

    @abstractmethod
    def get_historical_risk_premium(self,
                                    currency,
                                    from_date=None,
                                    to_date=None) -> float:
        pass

    @abstractmethod
    def get_historical_volatility(self,
                                  from_date=None,
                                  to_date=None) -> float:
        pass

    @abstractmethod
    def get_historical_sharpe_ratio(self,
                                    currency,
                                    from_date=None,
                                    to_date=None) -> float:
        pass

    @abstractmethod
    def get_historical_value_at_risk(self,
                                     from_date=None,
                                     to_date=None,
                                     confidence_level=0.99,
                                     horizon=1) -> float:
        pass

    @abstractmethod
    def get_historical_conditional_value_at_risk(self,
                                                from_date=None,
                                                to_date=None,
                                                confidence_level=0.99,
                                                horizon=1) -> float:
        pass

    @abstractmethod
    def get_historical_probability_of_loss(self,
                                          from_date=None,
                                          to_date=None,
                                          horizon=1) -> float:
        pass

    @abstractmethod
    def get_historical_worst_peak_to_trough(self,
                                            from_date=None,
                                            to_date=None) -> float:
        pass

    @abstractmethod
    def get_historical_equity_beta(self,
                                   currency,
                                   from_date=None,
                                   to_date=None) -> float:
        pass

    @abstractmethod
    def get_historical_alpha_over_equity(self,
                                         currency,
                                         from_date=None,
                                         to_date=None):
        pass


    @abstractmethod
    def get_historical_skewness(self,
                                from_date=None,
                                to_date=None):
        pass

    @abstractmethod
    def get_historical_worst_period_return(self,
                                            from_date=None,
                                            to_date=None,
                                            period=1) -> float:
        pass

    @abstractmethod
    def get_historical_best_period_return(self,
                                           from_date=None,
                                           to_date=None,
                                           period=1) -> float:
        pass

    @abstractmethod
    def get_historical_real_return(self,
                                    currency,
                                    from_date=None,
                                    to_date=None) -> float:
        pass

    @abstractmethod
    def get_historical_inflation_out_performance_frequency(self,
                                                         currency,
                                                         horizon=1,
                                                         from_date=None,
                                                         to_date=None,
                                                         annual_alpha_target=0) -> float:
        pass