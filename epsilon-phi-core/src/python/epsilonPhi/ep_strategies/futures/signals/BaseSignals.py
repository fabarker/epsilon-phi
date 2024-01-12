from scipy.stats import zscore
from abc import abstractmethod
import numpy as np
from epsilonPhi.core.lib.Decorators import builtin_types
from epsilonPhi.ep_strategies.estimators.risk.Volatility import Volatility, Estimator

class SignalCatalogue(object):

    @staticmethod
    def get_raw_moving_average(prices, window, *args, **kwargs):
        return prices.rolling(window=window, min_periods=1, *args, **kwargs).mean()

    @staticmethod
    def get_raw_moving_average_crossover(prices, slow, fast):
        return (prices.rolling(window=slow, min_periods=1).mean() -
                prices.rolling(window=fast, min_periods=1).mean())

    @staticmethod
    def get_raw_exponential_moving_average(prices, span):
        return prices.ewm(span=span, min_periods=1).mean()

    @staticmethod
    def get_raw_exponentially_weighted_moving_average_crossover(prices, slow, fast):
        return (prices.ewm(span=fast, min_periods=1,  adjust=False).mean() -
                prices.ewm(span=slow, min_periods=1,  adjust=False).mean())

    @staticmethod
    def get_time_series_momentum(prices, period):
        return np.sign(prices.pct_change(period))

    @staticmethod
    def get_realized_skewness(prices, period):
        return prices.rolling(window=period, min_periods=1).skew()

    @staticmethod
    def get_realized_kurtosis(prices, period):
        return prices.rolling(window=period, min_periods=1).kurt()

    @staticmethod
    def get_realized_volatility(prices, span):
        return prices.ewm(span=span, min_periods=1).std()

    @staticmethod
    def get_realized_volatility_crossover(prices, fast, slow):
        return (prices.ewm(span=slow, min_periods=1).std() -
                prices.ewm(span=fast, min_periods=1).std())

    @staticmethod
    def get_carry(near_contract, far_contract):

        # Carry in normalized based on instrument vol.
        # The idea is that we observe asset expected Sharpe Ratios, and then allocate accordingly.
        # If the correlation between all assets are 0, then the optimal portfolio is weighted in proportion
        # to an assets Sharpe ratio.
        return np.log(far_contract / near_contract)

    @staticmethod
    def get_carry_momentum(near_contract, far_contract, period):
        carry = SignalCatalogue.get_carry(near_contract, far_contract)
        return carry.ewm(span=period, min_periods=1).mean()

    @staticmethod
    def get_acceleration(price, fast, slow):
        ewmac = SignalCatalogue.get_raw_exponentially_weighted_moving_average_crossover(price, fast, slow)
        return ewmac - ewmac.shift(fast)

    @staticmethod
    def get_breakout(prices, lookback, smooth=None):

        rolling_min = prices.rolling(lookback, min_periods=int(min(len(prices), np.ceil(lookback / 2.0)))).max()
        rolling_max = prices.rolling(lookback, min_periods=int(min(len(prices), np.ceil(lookback / 2.0)))).min()
        rolling_mean = 0.5 * (rolling_max + rolling_min)

        output = 40.0 * ((prices - rolling_mean) / (rolling_max - rolling_min))
        return output.ewm(span=smooth, min_periods=np.ceil(smooth / 2.0)).mean()

    @staticmethod
    def get_relative_strength_indicator(prices, rsi_period):
        pass

    @staticmethod
    def cross_sectional_mean_reversion():
        pass

    @staticmethod
    def get_value(prices, period):
        pass

class BaseSignal(object):
    def __init__(self):
        pass

    def get_signal_id(self):
        return '%s(%s)' % (
            type(self).__name__,
            ', '.join('%s=%s' % (item.replace('_',''), str(getattr(self, item))) for item in dir(self) if
                      not callable(item) and not item.startswith("__") and
                      type(getattr(self, item)) in builtin_types)
        )

    @abstractmethod
    def get_raw_signal(self, prices):
        pass

    def load_normalized_signal(self, prices):
        pass


    def get_normalized_signal(self, prices):
        raw_signal = self.get_raw_signal(prices)
        return raw_signal / raw_signal.rolling(window=256, min_periods=10).std()

    def position_integer(self, prices):
        return np.sign(self.get_normalized_signal(prices))


class SignalFactory(object):

        class moving_average(BaseSignal):
            def __init__(self, prices, window):
                BaseSignal.__init__(self, prices)
                self._window = window

            def get_raw_signal(self):
                return SignalCatalogue.get_raw_moving_average(self._prices, self._window)

        class exponential_moving_average_crossover_Baz_et_al(BaseSignal):
            def __init__(self, fast, slow):
                BaseSignal.__init__(self)
                self._fast = fast
                self._slow = slow

            def get_normalized_signal(self, prices):
                x_k = self.get_raw_signal(prices)
                y_k = x_k / self.get_normalizing_factor(prices)
                run_std_y_k = Volatility.estimate_volatility(y_k,
                                                             Estimator.rolling_close_equal_weighted,
                                                             window=252)
                z_k = y_k / run_std_y_k
                return (1/0.89) * (z_k * np.exp( (-1/4) * np.power(z_k, 2) ))

            def get_normalizing_factor(self, prices):
                return Volatility.estimate_volatility(prices.pct_change(),
                                                      Estimator.rolling_close_equal_weighted,
                                                      window=63)


            def get_raw_signal(self, prices):
                return SignalCatalogue.get_raw_exponentially_weighted_moving_average_crossover(prices,
                                                                                               self._slow,
                                                                                               self._fast)


        class moving_average_crossover(BaseSignal):
            def __init__(self, fast, slow):
                BaseSignal.__init__(self, prices, fast, slow)
                self._fast = fast
                self._slow = slow

            def get_raw_signal(self):
                return SignalCatalogue.get_raw_moving_average_crossover(self._prices, self._slow, self._fast)
        class exponential_moving_average(BaseSignal):
            def __init__(self, span):
                BaseSignal.__init__(self, prices, span)
                self._span = span

            def get_raw_signal(self):
                return SignalCatalogue.get_raw_exponential_moving_average(self._prices, self._span)

        class exponential_moving_average_crossover(BaseSignal):
            def __init__(self, fast, slow):
                BaseSignal.__init__(self)
                self._fast = fast
                self._slow = slow

            def get_raw_signal(self, prices):
                return SignalCatalogue.get_raw_exponentially_weighted_moving_average_crossover(prices,
                                                                                               self._slow,
                                                                                               self._fast)

        class time_series_momentum(BaseSignal):
            def __init__(self, lookback):
                BaseSignal.__init__(self, prices, lookback)
                self._lookback = lookback

            def get_raw_signal(self):
                return SignalCatalogue.get_time_series_momentum(self._prices, self._lookback)



if __name__ == "__main__":

    from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource

    gds = GlobalDataSource()
    prices = gds.get_time_series_data_from_ticker('@AAPL', cols='PL')
    prices = prices.loc[prices.index <= '7-Jul-2023']

    s1 = SignalFactory.exponential_moving_average_crossover_Baz_et_al(fast=8, slow=24)
    s2 = SignalFactory.exponential_moving_average_crossover_Baz_et_al(fast=16, slow=48)
    s3 = SignalFactory.exponential_moving_average_crossover_Baz_et_al(fast=32, slow=96)

    s1.get_normalized_signal(prices)

