from epsilonPhi.core.utils.DateUtils import DateUtils
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.ep_strategies.futures.Models import Predictor, PREDICTOR_TYPE
import pandas as pd
import numpy as np


# Signal Construction....
# 1. Estimate one or more signals, such as exponential weighted moving average
# 2. Standardize all the signals and cap them to + or - 3 standard deviations
# 2. For mixed signals, we combine them into a single signal that takes the form of long (1) or (-1)
# 3. We solve for a position size on the portfolio, using the correlation matrix, as an optimization problem


# Signals consist of the following....
# 1. Raw Signal
# 2. Normalize/Standardization/Risk Scailing
# 3. Winzorization/Capping


class SignalExtractor(object):

    _signals = ()
    _scores = {}

    def __init__(self, *args):

        self._price_panel = None
        self.__add_univariate_signals(args)


    @property
    def univariate_signals(self):
        return [x.get_signal_id() for x in self._signals] if self._signals is not None else []

    def set_instrument_price_panel(self, prices: pd.DataFrame):
        assert isinstance(prices, pd.DataFrame), 'Error - class expects dataframe of instrument prices'
        self._price_panel = prices

    def _is_univariate_signal(self, cls_):
        return 'BaseSignals.SignalFactory' in str(cls_.__class__)

    def _isin(self, signal_cls):
        return signal_cls.get_signal_id() in self.univariate_signals

    def __add_univariate_signal_single(self, signal_cls):
        if not self._isin(signal_cls):
            self._signals = self._signals + (signal_cls,)

    def __add_univariate_signals(self, signal_cls):

        if not DateUtils.is_iterable(signal_cls):
            signal_cls = tuple(signal_cls)
        assert np.all(
            [self._is_univariate_signal(x) for x in signal_cls]), 'Error - all inputs must be univariate signals'

        for _sig_cls in signal_cls:
            self.__add_univariate_signal_single(_sig_cls)

    def get_raw_signals(self, prices):
        df = pd.concat([x.get_raw_signal(prices) for x in self._signals], axis=1)
        df.set_attribute_single('signal_id', self.univariate_signals)
        return df.dropna(how='all', axis=0)

    def get_normalized_signals(self, prices):
        df = pd.concat([x.get_normalized_signal(prices) for x in self._signals], axis=1)
        df.set_attribute_single('signal_id', self.univariate_signals)
        return df.dropna(how='all', axis=0)

    def get_rebalancing_dates(self, prices, rebalancing_frequency='BM'):
        return pd.to_datetime(prices.resample(rebalancing_frequency).asfreq().index)


    def get_forecast_variable(self, prices, rebalancing_frequency='BM'):
        periods = int(Frequency(prices.frequency).obs_per_year() / Frequency(rebalancing_frequency).obs_per_year())
        return prices.diff(periods).dropna(how='all', axis=0) / prices.diff(periods).std(ddof=1)

    def get_lagged_predictors(self, prices):
        return self.get_normalized_signals(prices).shift(1).dropna()

    def extract_position_score_rolling(self, prices, estimator_type, sampling_window):
        pass


    def extract_position_score_expanding(self, prices, estimator_type, sampling_window):
        pass



    # Fit a model to
    def get_position_score(self,
                           prices,
                           rebalancing_frequency='BM',
                           estimator_type=PREDICTOR_TYPE.EQUAL_WEIGHTED,
                           sampling_type='expanding',
                           sampling_window=252):

        X = self.get_normalized_signals(prices)
        y = self.get_forecast_variable(prices, rebalancing_frequency)

        # Fit the model out of sample

        predictor = Predictor(estimator_type)
        X_prime, y_prime = predictor.prepare_variables(X, y)

        predicted_score = list()
        if sampling_type == 'expanding':

            for dt in y_prime[sampling_window+1:].index:
                predicted_score.extend([predictor.fit_and_predict(X_prime.loc[X_prime.index < dt],
                                                                 y_prime.loc[y_prime.index < dt],
                                                                 X_prime.loc[dt].values)])

        elif sampling_type == 'rolling':
            self.extract_position_score_expanding(prices, estimator_type, sampling_window)
        else:
            raise ValueError('Error - sampling_type {} not supported'.format())
        return pd.DataFrame(predicted_score, index=y_prime[sampling_window+1:].index)


if __name__ == "__main__":

    from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
    from epsilonPhi.ep_strategies.futures.signals.BaseSignals import SignalFactory
    from epsilonPhi.ep_strategies.estimators.risk.Volatility import Volatility, Estimator
    import pandas as pd

    s1 = SignalFactory.exponential_moving_average_crossover_Baz_et_al(fast=8, slow=24)
    s2 = SignalFactory.exponential_moving_average_crossover_Baz_et_al(fast=16, slow=48)
    s3 = SignalFactory.exponential_moving_average_crossover_Baz_et_al(fast=32, slow=96)

    cls_ = SignalExtractor(s1, s2, s3)

    gds = GlobalDataSource()
    prices = gds.get_time_series_data_from_ticker('@AAPL', cols='PL')
    prices = prices.loc[prices.index <= '7-Jul-2023']

    norm_sig = cls_.get_position_score(prices)

    capital = 100
    target_vol_ann = 0.2
    target_vol_ann_dollar = capital * target_vol_ann
    target_vol_daily_dollar = target_vol_ann_dollar / np.sqrt(252)
    vol_scalar = target_vol_daily_dollar / Volatility.estimate_volatility(prices.diff(),
                                                                          Estimator.rolling_close_exponential_weighted,
                                                                          span=21)

    position = np.sign(norm_sig) * vol_scalar.loc[norm_sig.index].values
    shifted_positions = position.shift(1)
    pnl = prices.diff().loc[shifted_positions.index] * shifted_positions.values
