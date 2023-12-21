from epsilonPhi.ep_strategies.estimators.risk.Volatility import Volatility, Estimator
from typing import Optional, Union
import pandas as pd
import numpy as np
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


class SignalFactory(object):

    @staticmethod
    def get_raw_moving_average(prices, window):
        prices.rolling(window=window, min_periods=1).mean()

    @staticmethod
    def get_raw_moving_average_crossover(prices, slow, fast):
        return prices.rolling(window=fast, min_periods=1).mean() - prices.rolling(window=slow, min_periods=1).mean()

    @staticmethod
    def get_raw_exponential_moving_average(prices, span):
        return prices.ewm(span=span, min_periods=1).mean()

    @staticmethod
    def get_raw_exponentially_weighted_moving_average_crossover(prices, slow, fast):
        return prices.ewm(span=fast, min_periods=1).mean() - prices.ewm(span=slow, min_periods=1).mean()

    @staticmethod
    def get_time_series_momentum(prices, period):
        return prices.pct_change(period)

    @staticmethod
    def get_realized_skewness(prices, period):
        return prices.rolling(span=period, min_periods=1).skewness()

    @staticmethod
    def get_realized_kurtosis(prices, period):
        return prices.rolling(span=period, min_periods=1).kurtosis()

    @staticmethod
    def get_realized_volatility(prices, span):
        return prices.ewm(span=span, min_periods=1).std()

    @staticmethod
    def get_realized_volatility_crossover(prices, fast, slow):
        return prices.ewm(span=fast, min_periods=1).std() - prices.ewm(span=slow, min_periods=1).std()

    @staticmethod
    def get_carry(near_contract, far_contract):
        return np.log(far_contract / near_contract)

    @staticmethod
    def get_carry_momentum(near_contract, far_contract, period):
        carry = SignalFactory.get_carry(near_contract, far_contract)
        return carry.ewm(span=period, min_periods=1).mean()

    @staticmethod
    def get_acceleration(price, fast, slow):
        ewmac = SignalFactory.get_raw_exponentially_weighted_moving_average_crossover(price, fast, slow)
        return ewmac - ewmac.shidt(fast)

    @staticmethod
    def get_breakout(prices, lookback, smooth=None):

        rolling_min = prices.rolling(lookback, min_periods=int(min(len(prices), np.ceil(lookback / 2.0)))).max()
        rolling_max = prices.rolling(lookback, min_periods=int(min(len(prices), np.ceil(lookback / 2.0)))).min()
        rolling_mean = 0.5 * (rolling_max + rolling_min)

        output = 40.0 * ((prices - rolling_mean) / (rolling_max - rolling_min))
        return output.ewm(span=smooth, min_periods=np.ceil(smooth / 2.0)).mean()

    @staticmethod
    def cross_sectional_mean_reversion():
        pass

    @staticmethod
    def get_value(prices, period):
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

    def set_risk_estimator(self, risk_estimator):
        self._risk_estimator = risk_estimator


    def get_signal(self):
        pass



if __name__ == "__main__":



    vol_function = Volatility(Estimator.rolling_close_exponential_weighted, span=252)

    self = CSignal()
    self.set_vol_function(vol_function)

