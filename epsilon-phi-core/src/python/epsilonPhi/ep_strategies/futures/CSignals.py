from epsilonPhi.ep_strategies.estimators.risk.Volatility import Volatility, Estimator
from typing import Optional, Union
import pandas as pd
from enum import Enum

# Signals consist of the following
# 1. Raw Signal
# 2. Normalize/Standardization/Risk Scailing
# 3. Winzorization/Capping

class Signals(Enum):

    MOVING_AVERAGE = 1
    MOVING_AVERAGE_CROSSOVER = 3
    EXPONENTIAL_MOVING_AVERAGE = 4
    EXPONENTIAL_MOVING_AVERAGE_CROSSOVER = 5
    TIME_SERIES_MOMENTUM = 6
    REALIZED_SKEW = 7
    REALIZED_VOLATILITY = 8
    REALIZED_KURTOSIS = 9
    CARRY = 10
    MOMENTUM = 11
    PRICE_BREAKOUT = 12
    MEAN_REVERSION = 13
    REVERSALS = 14
    ACCELERATION = 15


class SignalFactory():

    @staticmethod
    def get_raw_MOVING_AVERAGE(prices, period):
        pass

    @staticmethod
    def get_raw_MOVING_AVERAGE_CROSSOVER(prices, slow, fast):
        pass

    @staticmethod
    def get_raw_EXPONENTIAL_MOVING_AVERAGE(prices, span):
        pass

    @staticmethod
    def get_raw_EXPONENTIAL_MOVING_AVERAGE_CROSSOVER(prices, slow, fast):
        pass

    @staticmethod
    def get_TIME_SERIES_MOMENTUM(prices, period):
        pass

    @staticmethod
    def get_realized_skewness(prices, period):
        pass

    @staticmethod
    def get_realized_kurtsis(prices, period):
        pass

    @staticmethod
    def get_realized_volatility(prices, period):
        pass

    @staticmethod
    def get_carry(prices, period):
        pass

    @staticmethod
    def get_carry_momentum(prices, period):
        pass


    @staticmethod
    def get_value(prices, period):
        pass

    @staticmethod
    def get_breakout(prices, period):
        pass




class CSignal(object):


    def __init__(self, prices: Optional[Union[pd.Series, pd.DataFrame]] = None):

        if isinstance(prices, pd.DataFrame):
            self.set_prices(prices)
        else:
            self._prices = prices


    def set_prices(self, prices: pd.DataFrame):
        assert isinstance(prices, (pd.DataFrame, pd.Series)), \
            'Error - prices must be a dataframe of price observations indexed by date'
        if isinstance(prices, pd.Series):
            self._prices = prices.to_frame(prices.name)
        else:
            self._prices = prices.sort_index()

    def set_vol_function(self, vol_function):
        self._risk_estimator = vol_function


    def get_signal(self):
        pass



if __name__ == "__main__":



    vol_function = Volatility(Estimator.rolling_close_exponential_weighted, span=252)

    self = CSignal()
    self.set_vol_function(vol_function)

