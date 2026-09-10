import pandas as pd
import numpy as np

# This script implements trend following

path = r'/Users/francisbarker/Desktop/Trend Following/TrendFollowing_CandidateProject_Data.xlsx'
df_ = pd.read_excel(path, sheet_name='AAPL', index_col=0)
df_.index.name = 'dates'
df_ = df_.sort_index()

prc = df_.copy()
rtn = np.log(prc).diff()

num_days_per_year = 256
capital = 100000
target_annual_vol_percentage = 0.2
vol_multiplier = 1

## 1. Vol Calculations

ST_lookback = 44
LT_lookback = num_days_per_year * 10
LT_sig_wt = 0.3

sig_ST = rtn.rolling(window=ST_lookback, min_periods=2).std(ddof=0) * np.sqrt(num_days_per_year)
sig_LT = rtn.rolling(window=LT_lookback, min_periods=2).std(ddof=0) * np.sqrt(num_days_per_year)
sig_smooth = sig_LT * LT_sig_wt + (1 - LT_sig_wt) * sig_ST
sig_smooth_daily = sig_smooth / np.sqrt(num_days_per_year)
daily_price_vol = prc * sig_smooth_daily

## 2. Moving Averages of Price With Set of Spans (4, 16, 64, 256)
ewma_df = pd.concat([prc.ewm(span=span, adjust=False, min_periods=2).mean() for span in [4, 16, 64, 256]], axis=1)
ewma_df.columns = [4, 16, 64, 256]

## 3. Difference in Moving Averages
crossovers = ewma_df.T.diff(-1).T.dropna(axis=1, how='all')

## 4. Volatility Adjust the Moving Averages
VolAdjD = crossovers / daily_price_vol.values

# 5. Absolute Values
AbsVAD = np.abs(VolAdjD)

# 6. Moving Average of Signals
smoothed_signal_lookback = num_days_per_year
AvgVAD = AbsVAD.rolling(window=smoothed_signal_lookback, min_periods=2).mean()

# 7. Average Absolute Signal
average_absolute_signal_mult = 10

# 8. Raw Signals
Signal = (VolAdjD * average_absolute_signal_mult) / np.abs(AvgVAD)

# 9. Cap Signals
CapSignal = Signal.clip(lower=-20, upper=20)

# 10. Moving Average Signal is the average.
MovAvgSignal = CapSignal.mean(axis=1).to_frame('Agg')

# 11. Single Signal Positions
Signals = pd.concat((CapSignal, MovAvgSignal), axis=1)
Posi = (Signals * capital * target_annual_vol_percentage) / (10 * vol_multiplier * prc * sig_smooth).values
AbsPosiWtdSignal = Posi.get('Agg').abs()

PnL = Posi.shift(1) * rtn.values
CumPnL = PnL.cumsum()

# 12. Compute Vols of PnL
rolling_PnL_vol = PnL.rolling(window=ST_lookback, min_periods=2).std(ddof=0) * np.sqrt(num_days_per_year)

# 13. ST Realized PnL Vol / Target Vol % Annualized - Scale Up the PnL to Hit the Target Vol
# This is because we may be missing the target vol
MulitCapSignal = target_annual_vol_percentage / rolling_PnL_vol
strategy_return_daily = MulitCapSignal.shift(1) * PnL
strategy_cum_return = strategy_return_daily.add(1).cumprod()

# 14. Breakout: Mid Point of Backward Looking Range

range_mean_df = pd.concat(
    [prc.rolling(window=span).apply(lambda x: (x.max() + x.min()) / 2, raw=False) for span in [16, 64, 256]], axis=1)
range_len = pd.concat(
    [prc.rolling(window=span).apply(lambda x: (x.max() - x.min()), raw=False) for span in [16, 64, 256]], axis=1)


####################################################################################################################################################################################################################################

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union

_DAYS_PER_YEAR = 256

_VALUE_OF_CONTRACT_PRICE_MOVE_ = {}
_VALUE_OF_CONTRACT_PRICE_MOVE_['AAPL'] = 1
_VALUE_OF_CONTRACT_PRICE_MOVE_['EDOLLAR'] = 2500
_VALUE_OF_CONTRACT_PRICE_MOVE_['SP500'] = 50
_VALUE_OF_CONTRACT_PRICE_MOVE_['US10'] = 1000
_VALUE_OF_CONTRACT_PRICE_MOVE_['LEANHOG'] = 400
_VALUE_OF_CONTRACT_PRICE_MOVE_['JPY'] = 12500000
_VALUE_OF_CONTRACT_PRICE_MOVE_['GOLD'] = 100
_VALUE_OF_CONTRACT_PRICE_MOVE_['WHEAT'] = 50
_VALUE_OF_CONTRACT_PRICE_MOVE_['NIKKEI'] = 100
_VALUE_OF_CONTRACT_PRICE_MOVE_['COPPER'] = 25000
_VALUE_OF_CONTRACT_PRICE_MOVE_['BITCOIN'] = 0.1



class CSignal(object):

    _multiplier = 1
    _average_abs = 10
    _signal_smoothing = _DAYS_PER_YEAR

    def __init__(self, prices: Optional[Union[pd.Series, pd.DataFrame]] = None):

        if isinstance(prices, pd.DataFrame):
            self.set_prices(prices)
        else:
            self._prices = prices

        self._short_vol_period = None
        self._long_vol_period = None
        self._short_vol_weight = None
        self._signals = list()

        self.set_default_parameters()

    @abstractmethod
    def get_capped_signal(self, signal_params):
        pass

    @property
    def instrument_name(self):
        return self.prices.columns[0]

    @property
    def dates(self):
        return pd.to_datetime(self.prices.index)

    @property
    def prices(self):
        if self._prices is not None:
           return self._prices.copy()
        else:
           raise ValueError('Error - must set prices in object')

    @property
    def price_change(self):
        return self.prices.diff()

    @property
    def returns(self):
        return np.log(self.prices).diff()

    def set_prices(self, prices: pd.DataFrame):
        assert isinstance(prices, (pd.DataFrame, pd.Series)), \
            'Error - prices must be a dataframe of price observations indexed by date'
        if isinstance(prices, pd.Series):
            self._prices = prices.to_frame(prices.name)
        else:
            self._prices = prices.sort_index()

    def generate_signals(self):

        signal_dfs = []
        for sig_params in self._signals:
            tmp = self.get_capped_signal(sig_params)
            tmp.columns = [(self.instrument_name,) + tuple(sig_params.values())]
            signal_dfs.append(tmp)

        # Concatenate all signal dataframes at once
        signal_df = pd.concat(signal_dfs, axis=1)

        # Compute weighted signal
        weights = [x.get('weight') for x in self._signals]
        weighted_signal = signal_df.dot(weights).to_frame('Agg')

        # Concatenate the weighted signal with the individual signals
        return pd.concat((signal_df, weighted_signal), axis=1)

    def set_default_parameters(self):
        self._long_vol_period = _DAYS_PER_YEAR * 10
        self._short_vol_period = 44
        self._short_vol_weight = 0.7

    def set_short_vol_periods(self, period: int):
        assert isinstance(period, int), \
            'Error - parameter shortvol_period must be of type int'
        self._short_vol_period = period

    def set_long_vol_periods(self, period: int):
        assert isinstance(period, int), \
            'Error - parameter long_vol_period must be of type int'
        self._long_vol_period = period

    def set_short_vol_weight(self, weight):
        self._short_vol_weight = weight

    def get_short_term_sigma(self):
        return self.get_rolling_sigma(self._short_vol_period)

    def get_long_term_sigma(self):
        return self.get_rolling_sigma(self._long_vol_period)

    def get_smoothed_sigma(self):
        return self._short_vol_weight * self.get_short_term_sigma() +\
                    (1-self._short_vol_weight) * self.get_long_term_sigma()

    def get_rolling_sigma(self, period, annualzied=True):
        sig = self.returns.rolling(window=period,
                                    min_periods=2).std(ddof=0)
        if annualzied:
            return sig * np.sqrt(_DAYS_PER_YEAR)
        return sig

    def get_price_volatility(self):
        return self.prices * self.get_smoothed_sigma() / np.sqrt(_DAYS_PER_YEAR)

class MovingAverageSignal(CSignal):
    def __init__(self, prices: Optional[Union[pd.Series, pd.DataFrame]] = None):
        super().__init__(prices)

        self._multiplier = 1
        self._average_abs = 10
        self._signal_smoothing = _DAYS_PER_YEAR

    def set_signal_smoothing_parameter(self, smoothing_paramter):
        self._signal_smoothing = smoothing_paramter

    def set_signal_multiplier(self, multiplier):
        self._multiplier = multiplier

    def set_single_moving_average_signal_params(self,
                                                fast: int,
                                                slow: int,
                                                cap: tuple,
                                                weight):

        assert fast < slow, 'Error - lookback for fast moving average must be less than slow moving average'
        signal_params = dict(zip(['fast', 'slow', 'cap', 'weight'], (fast, slow, cap, weight)))
        self._signals.extend([signal_params])

    def get_moving_average(self, lookback):
        return self.prices.ewm(span=lookback, adjust=False, min_periods=2).mean()

    def get_moving_average_crossover(self, fast_period, slow_period):
        return (self.get_moving_average(fast_period) -
                self.get_moving_average(slow_period))

    def get_vol_adjusted_moving_average_crossover(self, fast_period, slow_period):
        return self.get_moving_average_crossover(fast_period, slow_period) / self.get_price_volatility()

    def get_smoothed_abs_vol_adjusted_moving_average_crossover(self, fast_period, slow_period, smoothing_parameter):
        sig_adj_signal = self.get_vol_adjusted_moving_average_crossover(fast_period, slow_period)
        return sig_adj_signal.abs().rolling(window=self._signal_smoothing, min_periods=2).mean()

    def get_moving_average_crossover_signal(self, fast_period, slow_period, smoothing_parameter):
        return (self._average_abs * self.get_vol_adjusted_moving_average_crossover(fast_period, slow_period)
                / self.get_smoothed_abs_vol_adjusted_moving_average_crossover(fast_period, slow_period, smoothing_parameter))

    def get_capped_signal(self, signal_params):

        unclipped_signal = self.get_moving_average_crossover_signal(signal_params.get('fast'),
                                                                    signal_params.get('slow'),
                                                                    self._signal_smoothing)
        return unclipped_signal.clip(np.min(signal_params.get('cap')),
                                     np.max(signal_params.get('cap')))

class BreakoutSignal(CSignal):
    def __init__(self, prices: Optional[Union[pd.Series, pd.DataFrame]] = None):
        super().__init__(prices)

    def set_single_breakout_signal_params(self,
                                          lookback: int,
                                          cap: tuple,
                                          weight):

        signal_params = dict(zip(['lookback', 'cap', 'weight'], (lookback, cap, weight)))
        self._signals.extend([signal_params])

    def get_price_range_average(self, lookback):
        return self.prices.rolling(window=lookback).apply(lambda x: (x.max() + x.min()) / 2, raw=False)

    def get_price_range_distance(self, lookback):
        return self.prices.rolling(window=lookback).apply(lambda x: (x.max() - x.min()), raw=False)

    def get_smoothed_breakout_signal(self, lookback):

        # Calculate price range average and distance
        price_range_average = self.get_price_range_average(lookback)
        price_range_distance = self.get_price_range_distance(lookback)

        # Calculate weighting coefficient
        wt = 2 * (1/(np.round(lookback/4, decimals=0) + 1))

        normalized_signal = (40 * (self.prices - price_range_average)/ price_range_distance)
        normalized_signal = normalized_signal.dropna()

        # Initialize the signal list
        signal = np.zeros_like(normalized_signal)
        signal[0] = wt * normalized_signal.iloc[0]

        # Loop through the normalized signal for recursive calculation
        for t in range(1, len(normalized_signal)):
            signal[t] = (1 - wt) * signal[t - 1] + wt * normalized_signal.iloc[t]

        # Convert signal array to Pandas DataFrame
        signal_df = pd.DataFrame(signal, index=normalized_signal.index)

        # Reindex to match price_range_average index
        return signal_df.reindex(price_range_average.index)

    def get_capped_signal(self, signal_params):

        unclipped_signal = self.get_smoothed_breakout_signal(signal_params.get('lookback'))
        return unclipped_signal.clip(np.min(signal_params.get('cap')),
                                     np.max(signal_params.get('cap')))


class CStrategy(object):
    def __init__(self,
                 prices: pd.DataFrame,
                 capital=100000,
                 target_volatility=0.2,
                 multiplier=1):

        self._prices = prices
        self._capital = capital
        self._target_volatility = target_volatility
        self._multiplier = multiplier
        self._signals = list()

    @property
    def instrument_list(self):
        return list(self._prices.columns)

    def get_instrument_prices(self, instrument_name):
        return self._prices.get(instrument_name,
                                pd.DataFrame).dropna()

    def get_instrument_returns(self, instrument_name):
        return np.log(self.get_instrument_prices(instrument_name)).diff()

    def get_instrument_vol(self, instrument_name):
        return self._signals[0].get_smoothed_sigma().get(instrument_name)

    def run_strategy(self):
        pass

    def _append_signals_to_strategy(self, signal):
        self._signals.extend([signal])

    def get_signals_single_instrument(self, instrument_name):

        prices = self.get_instrument_prices(instrument_name)

        signal_dfs = []
        for _signal in self._signals:
            _signal.set_prices(prices)
            tmp = _signal.generate_signals()

            signal_dfs.append(tmp)

        # Concatenate all signal dataframes at once
        return pd.concat(signal_dfs, axis=1)


    def get_positions_single_instrument(self, instrument_name):

        signals = self.get_signals_single_instrument(instrument_name)
        _block_value = 1
        _avg_abs_signal = 10

        num = signals * self._capital * _block_value * self._target_volatility
        den = _avg_abs_signal * self._multiplier * self.get_instrument_prices(instrument_name) * self.get_instrument_vol(instrument_name)
        return num / den.values.reshape(-1, 1)

    def get_pnl_single_instrument(self, instrument):
        positions = self.get_positions_single_instrument(instrument).shift(1)
        returns = self.get_instrument_returns(instrument)
        return positions * returns.values.reshape(-1, 1)

    def get_pnls(self):
        pnl_df = []
        for instrument in self.instrument_list:
            tmp = self.get_pnl_single_instrument(instrument)
            pnl_df.append(tmp)
        return pd.concat(pnl_df, axis=1)

    def get_returns(self):
        rtn_df = []
        for instrument in self.instrument_list:
            tmp = self.get_signal_investment_returns_single_instrument(instrument)
            rtn_df.append(tmp)
        return pd.concat(rtn_df, axis=1)

    def get_signal_investment_returns_single_instrument(self, instrument):
        pnl = self.get_pnl_single_instrument(instrument)
        pnl_vols = self._target_volatility / (pnl.rolling(window=44, min_periods=10).std(ddof=0) * np.sqrt(_DAYS_PER_YEAR))
        return pnl * pnl_vols.shift(1)



if __name__ == "__main__":

    path = r'/Users/francisbarker/Desktop/Trend Following/TrendFollowing_CandidateProject_Data.xlsx'
    df_ = pd.read_excel(path, sheet_name='AAPL', index_col=0)
    df_.index.name = 'dates'
    df_ = df_.sort_index()

    # Moving Average Signal

    MA_signal = MovingAverageSignal()
    MA_signal.set_single_moving_average_signal_params(fast=4, slow=16, cap=(-20, 20), weight=1/3)
    MA_signal.set_single_moving_average_signal_params(fast=16, slow=64, cap=(-20, 20), weight=1/3)
    MA_signal.set_single_moving_average_signal_params(fast=64, slow=256, cap=(-20, 20), weight=1/3)

    MA_Strategy = CStrategy(df_)
    MA_Strategy._append_signals_to_strategy(MA_signal)

    pnls = MA_Strategy.get_pnls()
    rtns = MA_Strategy.get_returns()

    # Breakout Signal
    BO_Signal = BreakoutSignal()
    BO_Signal.set_single_breakout_signal_params(lookback=16, cap=(-20, 20), weight=1/3)
    BO_Signal.set_single_breakout_signal_params(lookback=64, cap=(-20, 20), weight=1 / 3)
    BO_Signal.set_single_breakout_signal_params(lookback=256, cap=(-20, 20), weight=1 / 3)

    BO_Strategy = CStrategy(df_)
    BO_Strategy._append_signals_to_strategy(BO_Signal)
    pnls = BO_Strategy.get_pnls()
    rtns = BO_Strategy.get_returns()
