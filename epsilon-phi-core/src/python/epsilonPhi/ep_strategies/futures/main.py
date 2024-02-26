import pandas as pd
import numpy as np
import os
from epsilonPhi.core.optimizer.riskBudgeting.allocation import LongShortRiskParity, RiskBudgetWithERandVolTarget, LongShortRiskBudgeting
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from abc import abstractmethod
from tqdm import tqdm

import statsmodels.api as sm
from numba import jit
import scipy.stats as sps
from scipy.optimize import minimize
import math

__author__ = 'francis barker'
__date__ = '13/12/2023'

_scalar = StandardScaler()
_svr = SVR()
# %%

# Use Days Per Year From Excel
_DAYS_PER_YEAR = 256

# Contract Information - We Use Contract Units To Price The Strategy
_VALUE_OF_CONTRACT_PRICE_MOVE_ = {'AAPL': 1, 'EDOLLAR': 2500, 'SP500': 50, 'US10': 1000, 'LEANHOG': 400,
                                  'JPY': 12500000,
                                  'GOLD': 100, 'WHEAT': 50, 'NIKKEI': 100, 'COPPER': 25000, 'BITCOIN': 0.1}

# Contract Information - We Need to know what currency the contract prices are quoted in
_PRICE_CURRENCY_DENOMINATION = {'AAPL': 'USD', 'EDOLLAR': 'USD', 'SP500': 'USD', 'US10': 'USD', 'LEANHOG': 'USD',
                                'JPY': 'USD',
                                'GOLD': 'USD', 'WHEAT': 'USD', 'NIKKEI': 'JPY', 'COPPER': 'USD', 'BITCOIN': 'USD'}


class DataHandler(object):
    """ Class that loads, pre-processes and handles raw time series data.
    Also includes some data transformations such as volatility estimation
    and as such the parameters controlling the volatility estimator are managed in
    this class (_fast_vol, _slow_vol, _fast_vol_wt)

    The class assumes that data is stored in a xlsx workbook with dates in the first column (index_col=0)
     and data labels in row 1.

    Usage:
        handler = DataHandler(...data_workbook.xlsx, sheet_name='data_sheet')
        prices = handler.get_price_series('JPY')

    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DataHandler, cls).__new__(cls)
        return cls._instance

    def __init__(self, filepath, sheet_name):

        """
        Args:
            filepath: fullfile path to workbook with data
            sheet_name: sheet containing the time series data
        """

        # Store information of where the file  to data is located
        self._filepath = filepath
        self._sheet_name = sheet_name

        # Set some default parameters for the volatility estimator (See Function)
        self._fast_vol = 44
        self._slow_vol = 10 * _DAYS_PER_YEAR
        self._fast_vol_wt = 0.7

        # Load the Data from the excel spreadsheet
        self.__load_data()

    @staticmethod
    def get_instance():
        return DataHandler._instance

    def set_volatility_parameters(self, fast_vol_days=44, fast_vol_weight=0.7, slow_vol_years=10):

        """ Function for setting the parameters for volatility estimation,
            Default set to equal Keridion excel sheet
        """
        self._fast_vol = fast_vol_days
        self._slow_vol = slow_vol_years * _DAYS_PER_YEAR
        self._fast_vol_wt = fast_vol_weight

    @property
    def labels(self):
        return self._df.columns

    @property
    def dates(self):
        return pd.to_datetime(self._df.index)

    @property
    def df(self):
        return self._df.copy()

    @property
    def price_change(self):
        return self.df.diff()

    def __load_data(self):
        """ Private method for loading raw data from excel """
        df_ = pd.read_excel(self._filepath, sheet_name=self._sheet_name, index_col=0)
        df_.index.name = 'dates'
        self._df = df_.sort_index()

    def get_price_series_denominated_currency(self, series_name):
        """ Returns currency that price series is denominated in """
        return _PRICE_CURRENCY_DENOMINATION.get(series_name)

    def get_price_series(self, series_names):
        """ Returns single price series """
        return self.df.get([series_names]).dropna(how='all')

    def get_price_series_in_given_currency(self, series_name, pricing_currency):
        """ Returns single price series in a given currency converting via spot fx"""
        from_currency = self.get_price_series_denominated_currency(series_name)
        series = self.get_price_series(series_name)
        return self.convert_price_to_currency(series, from_currency, pricing_currency)

    def get_fx_rate(self, bbid):
        """ Returns fx rate for a given currency """
        assert bbid in self._df.columns, 'Error - currency {} not supported'.format(bbid)
        return self._df.get([bbid])

    def get_fx_rate_for_instrument(self, instrument_name, base_currency):

        """ Returns fx rate for a given instrument """

        assert base_currency in ['USD'], 'Error - currently only USD base currency is supported'
        instrument_currency = _PRICE_CURRENCY_DENOMINATION.get(instrument_name)

        if instrument_currency == base_currency:
            return self.get_price_series(instrument_name) / self.get_price_series(instrument_name)
        else:
            return self.get_fx_rate(instrument_currency)

    def convert_price_to_currency(self, price_series, from_currency, to_currency):
        """ Returns price series converted from_currency to to_currency at spot rate """

        assert to_currency == 'USD', 'Error - conversion to currency {} not supported'.format(to_currency)
        assert from_currency in self._df.columns, 'Error - fx rate {}{} not supported'.format(from_currency,
                                                                                              to_currency)

        # Get the FX Rate
        fx_rate = self._df.get(from_currency)
        common_dates = np.intersect1d(fx_rate.index, price_series.index)
        # Do the conversion
        return price_series.loc[common_dates] * fx_rate.loc[common_dates].values.reshape(-1, 1)

    def get_series_slow_volatility(self, series, annualized=True):
        """ Returns long-term slow moving volatility for a price series """
        return self.get_rolling_price_volatility(series, self._slow_vol, annualized).get(series)

    def get_series_fast_volatility(self, series, annualized=True):
        """ Returns short-term fast moving volatility for a price series """
        return self.get_rolling_price_volatility(series, self._fast_vol, annualized).get(series)

    def get_series_price_volatility(self, series, annualized=True):
        """ Returns price volatiltiy for a series weighted between slow and fast
            volatility estimates """
        return self._fast_vol_wt * self.get_series_fast_volatility(series, annualized) + \
            (1 - self._fast_vol_wt) * self.get_series_slow_volatility(series, annualized)

    def get_rolling_price_volatility(self, series, period, annualized=True):
        """ Returns rolling volatiltiy for a price series given lookback period """
        sig = self.get_price_series(series).diff().rolling(window=period, min_periods=10).std(ddof=0)

        if annualized:
            return sig * np.sqrt(_DAYS_PER_YEAR)
        return sig

    def get_contract_unit(self, instrument_name):
        return _VALUE_OF_CONTRACT_PRICE_MOVE_.get(instrument_name)

    def get_contract_units(self, instruments):
        return np.array([self.get_contract_unit(x) for x in instruments])

    def get_instrument_price_pnl(self, instrument_name):
        return self.get_price_series(instrument_name).diff()

    def get_instrument_contract_pnl(self, instrument_name):
        return (self.get_instrument_price_pnl(instrument_name) *
                self.get_contract_unit(instrument_name))

    def get_instruments_contract_pnls(self):
        return pd.concat([self.get_instrument_contract_pnl(x) for x in self.labels], axis=1)

    @staticmethod
    def fast_linear_model(X_, y, x_star):

        # Re-scalled Xs
        X = (X_ - np.mean(X_)) / np.std(X_)

        X_prime = np.vstack((np.ones(X.shape[0]), X)).T
        XtX = np.dot(X_prime.T, X_prime)
        XtY = np.dot(X_prime.T, y)
        beta = np.linalg.solve(XtX, XtY)
        return beta * x_star

    @staticmethod
    @jit('float64(float64[:])', nopython=True)
    def hurst_exponent(log_returns):

        log_returns = log_returns.flatten()
        n = log_returns.shape[0]
        N = np.floor(np.log(n) / np.log(2))

        X = np.arange(2, N+1)
        Y = np.empty(X.shape[0])
        for p_idx, p in enumerate(X):

            m = int(2**p)
            s = int(2**(N-p))
            rs_array = np.empty(s)
            for i in range(s):
                tmp = log_returns[i*m:(i+1)*m]
                mu = np.mean(tmp)
                deviate = np.cumsum(tmp-mu)
                diff_d = np.max(deviate) - np.min(deviate)
                sig = np.std(tmp)
                scaled_rng = diff_d / sig if sig != 0 else 0
                rs_array[i] = scaled_rng
            Y[p_idx] = np.log2(np.mean(rs_array))

        X_prime = np.vstack((np.ones(X.shape[0]), X)).T
        XtX = np.dot(X_prime.T, X_prime)
        XtY = np.dot(X_prime.T, Y)
        beta = np.linalg.solve(XtX, XtY)

        #residuals = Y - np.dot(X_prime, beta)
        #degrees_of_freedom = X_prime.shape[0] - X_prime.shape[1]
        #sigma_squared = np.dot(residuals.T, residuals) / degrees_of_freedom

        # Calculate covariance matrix of the coefficients
        #cov_matrix = sigma_squared * np.linalg.inv(XtX)

        # Standard error of the coefficients is the square root of the diagonal of the covariance matrix
        #tstat = (beta - 0.5) / np.sqrt(np.diag(cov_matrix))
        return beta[1]

    def get_contract_hurst_exponent(self, instrument, start_date=None, end_date=None):
        prices = self.get_price_series(instrument).loc[start_date:end_date]
        logRtns = np.log(prices).diff().replace(0, np.nan).dropna()
        return DataHandler.hurst_exponent(logRtns.values.flatten())

    def get_contract_rolling_hurst_exponent(self, instrument, window=256):
        logRtns = np.log(self.get_price_series(instrument)).diff().dropna()
        return logRtns.rolling(window=window).apply(lambda x: DataHandler.hurst_exponent(x.values))



class CSignal(object):

    """ Base class representing a trading signal.
        Encapsulates and handles more abstract methods
        common across all types of signals (MAC, Breakout ect...)

        For signal design see subclasses: MovingAverageSignal, BreokoutSignal

        Attributes:
            _multiplier: scales the positions proportionally (Default is 1)
            _signal_smoothing: lookback window for estimating the volatility multiplier (Default is 1 Year)
            _average_abs_target: target value for volatility of signal (default is 10)
            _prices: the price series of the instrument
            _signals: set of distinct signals within this signal class type
            _datasource: data handler class

    """

    _multiplier = 1
    _signal_smoothing = _DAYS_PER_YEAR
    _average_abs_target = 10

    def __init__(self):

        self._prices = None
        self._signals = list()
        self._datasource = DataHandler.get_instance()

    @abstractmethod
    def get_capped_signal(self, signal_params):
        """ Abstract Method implemented in subclasses """
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
            self._prices = prices.sort_index().dropna()

    def _signal_name(self, sig_params):
        """ Construct a signal name from paramters. Useful for graphing ect..  """

        if isinstance(self, MovingAverageSignal):
           return 'Moving Average ({},{})'.format(sig_params.get('fast'),
                                                      sig_params.get('slow'))
        elif isinstance(self, BreakoutSignal):
           return 'Breakout ({})'.format(sig_params.get('lookback'))
        else:
           return self.__class__.__name__

    def generate_signals(self):
        """ Method for generating all signals for a given instrument  """

        signal_dfs = []
        for sig_params in self._signals:

            signal_name = self._signal_name(sig_params)
            print('Constructing {} signal for instrument {}'.format(signal_name, self.instrument_name))

            tmp = self.get_capped_signal(sig_params)
            tmp.columns = [self.instrument_name + ' ' + signal_name]
            signal_dfs.append(tmp)

        # Concatenate all signal dataframes at once
        return pd.concat(signal_dfs, axis=1)

    def generate_signals_for_instrument(self, instrument_name):
        """ Method for generating all signals for a given instrument  """

        ds = DataHandler.get_instance()
        self.set_prices(ds.get_price_series(instrument_name))
        return self.generate_signals()


class MovingAverageSignal(CSignal):

    """ Subclass inheritning class definition CSignal.
        Represents Moving Average Crossover Trading Rule
        mirroring the logic in Keridion Capital excel sheet

        Attributes:
            _multiplier: scales the positions proportionally (Default is 1)
            _signal_smoothing: lookback window for estimating the volatility multiplier
            _average_abs_target: target value for volatility of signal
            _prices: the price series of the instrument
            _signals: set of distinct signals within this signal class type
            _datasource: data handler class

        Usage:
        signal = MovingAverageSignal()
        signal.set_single_moving_average_signal_params(fast=4, slow=16, cap_lower=-20, cap_upper=20)
        capped_signal = signal.get_capped_signal()

    """

    def __init__(self):
        super().__init__()

    def set_single_moving_average_signal_params(self,
                                                fast: int,
                                                slow: int,
                                                cap_lower: float,
                                                cap_upper: float,
                                                ):
        assert fast < slow, 'Error - lookback for fast moving average must be less than slow moving average'
        signal_params = dict(zip(['fast', 'slow', 'lower', 'upper'], (fast, slow, cap_lower, cap_upper)))
        self._signals.extend([signal_params])

    def get_moving_average(self, lookback):
        """ Method for getting the ewm average given a lookback window  """
        return self.prices.ewm(span=lookback, adjust=False, min_periods=10).mean()

    def get_moving_average_crossover(self, fast_period, slow_period):
        """ Returns the moving average crossover for fast and slow legs of signal  """
        return (self.get_moving_average(fast_period) -
                self.get_moving_average(slow_period))

    def get_vol_adjusted_moving_average_crossover(self, fast_period, slow_period):
        """ Returns the volatility adjusted moving average crossover """
        return (self.get_moving_average_crossover(fast_period, slow_period) /
                self._datasource.get_series_price_volatility(self.instrument_name, False).values.reshape(-1, 1))

    def get_smoothed_abs_vol_adjusted_moving_average_crossover(self, fast_period, slow_period, smoothing_parameter):
        """ Returns the rolling average absolute volatilitites """
        sig_adj_signal = self.get_vol_adjusted_moving_average_crossover(fast_period, slow_period)
        return sig_adj_signal.abs().rolling(window=smoothing_parameter, min_periods=10).mean()

    def get_normalized_moving_average_crossover_signal(self, fast_period, slow_period, smoothing_parameter):
        """ Returns the normalized moving average crossover for fast and slow legs of signal, normalized over a
            period defined by the smoothing parameter """
        sig_adjusted_ma_crossover = self.get_vol_adjusted_moving_average_crossover(fast_period, slow_period)
        trailing_avg_abs_sig_adj_mac = self.get_smoothed_abs_vol_adjusted_moving_average_crossover(fast_period,
                                                                                                   slow_period,
                                                                                                   smoothing_parameter)
        return self._average_abs_target * sig_adjusted_ma_crossover / trailing_avg_abs_sig_adj_mac


    def get_capped_signal(self, signal_params):
        " Returns the capped signal, after all normalizations "
        unclipped_signal = self.get_normalized_moving_average_crossover_signal(signal_params.get('fast'),
                                                                               signal_params.get('slow'),
                                                                               self._signal_smoothing)
        return unclipped_signal.clip(np.min(signal_params.get('cap_lower')),
                                     np.max(signal_params.get('cap_upper')))


class BreakoutSignal(CSignal):

    """ Subclass inheritning class definition CSignal.
        Represents Breakout Trading Rule
        mirroring the logic in Keridion Capital excel sheet

        Attributes:
            _multiplier: scales the positions proportionally (Default is 1)
            _signal_smoothing: lookback window for estimating the volatility multiplier
            _average_abs_target: target value for volatility of signal
            _prices: the price series of the instrument
            _signals: set of distinct signals within this signal class type
            _datasource: data handler class

        Usage:
            signal = BreakoutSignal()
            signal.set_single_breakout_signal_params(lookback=16, cap_lower=-20, cap_upper=20)
            capped_signal = signal.get_capped_signal()

        """

    def __init__(self):
        super().__init__()

    def set_single_breakout_signal_params(self,
                                          lookback: int,
                                          cap_lower: float,
                                          cap_upper: float):
        signal_params = dict(zip(['lookback', 'cap_lower', 'cap_upper'], (lookback, cap_lower, cap_upper)))
        self._signals.extend([signal_params])

    def get_price_range_average(self, lookback):
        " Returns the price range average "
        return self.prices.rolling(window=lookback).apply(lambda x: (x.max() + x.min()) / 2, raw=False)

    def get_price_range_distance(self, lookback):
        """ Returns the distance between previous high and lows  """
        return self.prices.rolling(window=lookback).apply(lambda x: (x.max() - x.min()), raw=False)

    def get_smoothed_breakout_signal(self, lookback):
        """ Returns the breakout signal for a given lookback window  """

        # Calculate price range average and distance
        price_range_average = self.get_price_range_average(lookback)
        price_range_distance = self.get_price_range_distance(lookback)

        # Calculate weighting coefficient
        wt = 2 * (1 / (np.round(lookback / 4, decimals=0) + 1))

        normalized_signal = (40 * (self.prices - price_range_average) / price_range_distance)
        normalized_signal = normalized_signal.dropna()

        # Initialize the signal list
        #signal = np.zeros_like(normalized_signal)
        #signal[0] = wt * normalized_signal.iloc[0]

        # Loop through the normalized signal for recursive calculation
        #for t in range(1, len(normalized_signal)):
        #    signal[t] = (1 - wt) * signal[t - 1] + wt * normalized_signal.iloc[t]
        signal = normalized_signal.ewm(alpha=wt, adjust=False, min_periods=2).mean().to_numpy()

        # Convert signal array to Pandas DataFrame
        signal_df = pd.DataFrame(signal, index=normalized_signal.index)

        # Reindex to match price_range_average index
        return signal_df.reindex(price_range_average.index)

    def get_capped_signal(self, signal_params):
        " Returns the capped signal, after all normalizations "
        unclipped_signal = self.get_smoothed_breakout_signal(signal_params.get('lookback'))
        return unclipped_signal.clip(np.min(signal_params.get('cap_lower')),
                                     np.max(signal_params.get('cap_upper')))


class CStrategy(object):

    """ Class representing a trading strategy. The class accepts a set of
        signal classes and converts signals into positions and positions into PnLs

        The class also has some simple methods for portfolio construction, including
        equal weighted and covariance weighted. For the covariance weighted,
        it runs monthly optimizations to solve for the equal risk weighted (risk parity)
        weights of the portfolio on an out of sample basis, with a target volatility constraint. The covariance is
        based on 256 day trailing exponentially weighted covariance matrix, and the optimal weights held
        fixed for 1 month before rebalancing.

        Usage:
            strategy = CStrategy(instrument_list=['US10'],
                                 base_currency='USD',
                                 capital=100000,
                                 target_annual_vol=0.2)

            strategy.add_signal_to_strategy(signal)
            cum_pnl_equal_weighted_portfolio = strategy.get_equal_weighted_portfolio_cumulative_pnl()

            """

    def __init__(self, instrument_list, base_currency='USD', capital=100000, target_annual_vol=0.2, multiplier=1):

        self._ds = DataHandler.get_instance()
        self._capital = capital
        self._base_currency = base_currency
        self._target_volatility = target_annual_vol
        self._multiplier = multiplier
        self._signals = list()
        self._optimal_weights = None
        self.__set_prices(instrument_list)
        self._signal_cache = {}

    def __set_prices(self, instrument_list):

        # Get a list of instruments we have data for
        cols = np.intersect1d(instrument_list, self._ds.df.columns)

        # If we dont have any of the data, throw and error
        assert len(cols) > 0, 'Error - No data for instruments {}'.format(instrument_list)

        # Set the instrument prices in the strategy object
        self._prices = self._ds.df.get(cols)

        # If there were instruments in the list with no data, display a warning
        missing_data = np.setdiff1d(instrument_list, cols)
        if len(missing_data) > 0:
            print('Warning: No data for instruments {}. These will be excluded form the model'.format(missing_data))

    @property
    def instrument_list(self):
        """ Returns list of instruments included in the strategy model """
        return list(self._prices.columns)

    @property
    def target_cash_volatility(self, annualize=True):
        """ Returns strategy target annualized volatility """

        if annualize:
            return self._capital * self._target_volatility
        else:
            return self._capital * self._target_volatility / np.sqrt(_DAYS_PER_YEAR)

    def get_instrument_value_per_tick(self, instrument_name):
        """ Returns the contract unit for a given instrument """
        return _VALUE_OF_CONTRACT_PRICE_MOVE_.get(instrument_name)

    def get_instrument_prices(self, instrument_name):
        """ Returns the price series for a given instrument """
        return self._prices.get(instrument_name, pd.DataFrame).dropna()

    def get_instrument_returns(self, instrument_name):
        """ Returns the price returns (in currency units) for a given instrument """
        return self.get_instrument_prices(instrument_name).diff()

    def get_fx_rate_for_instrument(self, instrument_name):
        """ Returns the fx rate for a given instrument """
        return (self._ds.get_fx_rate_for_instrument(instrument_name, self._base_currency)
                .reindex(self.get_instrument_prices(instrument_name).index))

    def _reset_signals(self):
        """Clear signals from strategy"""
        self._signals = []

    def add_signal_to_strategy(self, signal):
        """ Method to add a signal object to the strategy"""
        self._signals.extend([signal])

    def get_signals_for_optimization(self):
        signals = [self.get_signals_single_instrument(x).mean(axis=1).to_frame(x) for x in self.instrument_list]
        return pd.concat(signals, axis=1)

    def get_signals_single_instrument(self, instrument_name) -> pd.DataFrame:
        """ Returns time series of capped signals for an instrument  """

        if instrument_name in self._signal_cache.keys():
            return self._signal_cache.get(instrument_name)

        # 1. Get the instrument prices (we need to set these in the signal object)
        prices = self.get_instrument_prices(instrument_name)

        signal_dfs = []
        for _signal in self._signals:
            # 2. Set the prices in the signal object and generate signals
            _signal.set_prices(prices)
            tmp = _signal.generate_signals()
            signal_dfs.append(tmp)

        # Concatenate all signal dataframes at once
        signals_df = pd.concat(signal_dfs, axis=1)
        self._signal_cache[instrument_name] = signals_df.copy()
        return signals_df.copy()

    def get_positions_single_instrument(self, instrument_name):
        """ Returns time series of instrument positions for a given strategy model
            This includes scaling positions to the target strategy volatility """

        # 1. Get the normalized signals
        signals = self.get_signals_single_instrument(instrument_name)

        # 2. Get the contract unit for the instrument we are trading
        _contract_unit = self.get_instrument_value_per_tick(instrument_name)

        # 3. Compute the vol multiplier - This ensures we are getting close to our annual vol target
        # adjusted for the contract size

        # CAREFUL of the currency mismatch. Our target cash volatility is in base currency units,
        # so we have to convert any foreign currency vols back to base at the prevailing spot rate.
        price_volatility = self._ds.get_series_price_volatility(instrument_name)
        fx_rate = self.get_fx_rate_for_instrument(instrument_name)

        assert np.all(fx_rate.index == price_volatility.index), 'Error - dates not aligned'
        vol_multiplier = self.target_cash_volatility / (_contract_unit * price_volatility * fx_rate.values.flatten())

        # Below represents the number of contracts, long/short to hold - Note: It is not rounded to nearest contract
        positions = signals * vol_multiplier.values.reshape(-1, 1) / 10

        # DUE TO POTENITAL DATA ERRORS - POSITIONS IN NIKKEI ARE FORCED TO ZERO FOR THIS PERIOD
        if instrument_name == 'NIKKEI':
            positions['10-Jun-2020':'1-Jan-2022'] = np.nan
        return positions.copy()

    def get_instrument_expected_return(self, instrument, window=252, horizon=1):

        signals = self.get_signals_single_instrument(instrument).dropna(how='all')
        prices = self.get_instrument_prices(instrument).loc[signals.index]
        returns = prices.pct_change(round(horizon))

        start = 21
        end = signals.shape[0] - horizon + 1
        rtns = np.ones((end, 1)) * np.nan
        for i in tqdm(range(start + horizon, end), desc='Estimating returns for ' + instrument):

            K = i - horizon
            T = max(0, K - window)

            # Get the necessary ranges
            rng_X = np.arange(T, i, 1)
            rng_X_reg = np.arange(T, K, 1)
            rng_y = rng_X_reg + horizon

            # target variance
            y_prime = returns.values[rng_y]

            # predictors
            x_prior = signals.values[rng_X]
            keepcols = ~np.any(np.isnan(x_prior), axis=0)

            x_prime = _scalar.fit_transform(x_prior[:,keepcols])
            x__p = x_prime[rng_X_reg - rng_X_reg[0]]
            x_star = x_prime[-1,:]

            _fit = _svr.fit(x__p, y_prime)
            rtns[i] = _fit.predict(x_star.reshape(1, -1))

        return pd.DataFrame(rtns, columns=[instrument], index=signals.index[:end])


    def get_instrument_signals(self):
        pnl_df = []
        for instrument in self.instrument_list:
            tmp = self.get_positions_single_instrument(instrument)
            pnl_df.append(tmp)
        return pd.concat(pnl_df, axis=1)

    def get_pnl_single_instrument(self, instrument_name):
        """ Returns time series of PnLs for a given instrument, after translating back to base currency """
        price_returns = self.get_instrument_returns(instrument_name)
        contract_unit = self.get_instrument_value_per_tick(instrument_name)
        fx_rate = self.get_fx_rate_for_instrument(instrument_name)
        positions = self.get_positions_single_instrument(instrument_name)

        assert np.all(fx_rate.index == price_returns.index), 'Error - dates not aligned'
        assert np.all(positions.index == price_returns.index), 'Error - dates not aligned'
        return positions.shift(1) * fx_rate.values * price_returns.values.reshape(-1, 1) * contract_unit

    def __get_raw_signal_pnls(self):
        """ Returns raw time series of PnLs for all signal/ instruments in the strategy before vol scailing"""

        pnl_df = []
        for instrument in self.instrument_list:
            tmp = self.get_pnl_single_instrument(instrument)
            pnl_df.append(tmp)
        return pd.concat(pnl_df, axis=1)

    def get_signal_pnl_volatility(self, lookback=44):
        """ Returns time series of annualized vols (in currency units) for strategy PnLS using simple weights """
        return self.__get_raw_signal_pnls().rolling(window=lookback, min_periods=10).std(ddof=0) * np.sqrt(
            _DAYS_PER_YEAR)

    def get_signal_vol_multiplier(self, lookback=44):
        """ Returns time series of vol multipliers,
        which estimates how much we are under/overshooting the target vol """
        return (self._target_volatility * self._capital) / self.get_signal_pnl_volatility(lookback)

    def get_signal_pnls(self):
        """ Returns time series of return streams, after adjusting for the target volatility """
        return self.__get_raw_signal_pnls() * self.get_signal_vol_multiplier().shift(1)

    def get_signal_returns(self):

        """ Returns time series of return streams, after adjusting for the target volatility """
        returns = self.__get_raw_signal_pnls() / self._capital
        return returns * self.get_signal_vol_multiplier().shift(1)

    def get_signal_cumulative_pnl(self):
        """ Returns time series of return streams, after adjusting for the target volatility """
        return self.get_signal_pnls().cumsum(axis=0).ffill()

    #####  Equal Weighted Portfolio Related

    def get_equal_weighted_portfolio_pnl(self):
        """ Returns time series of periodic PnLs for the equal weighted portfolio """
        return self.get_signal_pnls().mean(axis=1).to_frame('Equal Weighted')

    def get_equal_weighted_portfolio_returns(self):
        """ Returns time series of periodic returns for the equal weighted portfolio """
        return self.get_signal_returns().mean(axis=1).to_frame('Equal Weighted')

    def get_equal_weighted_portfolio_cumulative_pnl(self):
        """ Returns time series of cumulative PnLs for the equal weighted portfolio """
        return self.get_equal_weighted_portfolio_pnl().cumsum(axis=0)

    #####  Covariance Weighted Portfolio Related

    def get_covariance_weighted_portfolio_pnl(self):
        """ Returns time series of periodic pnls for the covariance weighted portfolio """
        return (self.get_signal_pnls() *
                self.get_optimized_risk_parity_weights().shift(1).values).sum(axis=1).to_frame('Covariance Weighted')

    def get_covariance_weighted_portfolio_returns(self):
        """ Returns time series of periodoc returns for the covariance weighted portfolio """
        return (self.get_signal_returns() *
                self.get_optimized_risk_parity_weights().shift(1).values).sum(axis=1).to_frame('Covariance Weighted')

    def get_covariance_weighted_portfolio_cumulative_pnl(self):
        """ Returns time series of cumulative PnLs for the covariance weighted portfolio """
        return self.get_covariance_weighted_portfolio_pnl().cumsum(axis=0)

    def get_optimized_risk_parity_weights(self):
        """ Returns dataframe for point in time optimal risk parity weights """
        if self._optimal_weights is None:
            self._compute_optimal_weights()
        return self._optimal_weights

    @staticmethod
    def print_progress_bar(iteration, total, length=50):
        percent = f"{100 * (iteration / total):.1f}"
        filled_length = int(length * iteration // total)
        bar = '█' * filled_length + '-' * (length - filled_length)
        print(f'\rProgress: |{bar}| {percent}% Complete', end="\r")
        if iteration == total:
            print()

    def get_asset_class_pnls(self):

        pnls = self.get_signal_pnls()
        cols = [(x.split(' ')[0], x) for x in pnls.columns]
        pnls.columns = pd.MultiIndex.from_tuples(cols)
        return pnls.T.groupby(level=0).mean().T

    def get_asset_class_number_of_contracts(self):
        pnls = self.get_asset_class_pnls()
        pnl_c = self._ds.get_instruments_contract_pnls()
        return (pnls / pnl_c.get(pnls.columns)).shift(-1)

    def get_signal_number_of_contracts(self):

        pnls = self.get_signal_pnls()

        df_N = [pd.DataFrame()]
        for col in pnls.columns:

            sig_pnl = pnls.get(col)

            instr_name = sig_pnl.name.split(' ')[0]
            price_returns = self.get_instrument_returns(instr_name)
            contract_unit = self.get_instrument_value_per_tick(instr_name)
            df_N.extend([sig_pnl / (price_returns * contract_unit)])

        N = pd.concat(df_N, axis=1)
        N.columns = pnls.columns
        return N.shift(-1)

    def get_asset_class_expected_returns(self, window=252, horizon=21):
        rt_df = []
        for x in self.instrument_list:
            tmp = self.get_instrument_expected_return(x, horizon=horizon, window=window)
            rt_df.append(tmp)
        return pd.concat(rt_df, axis=1)

    def _get_target_risk_for_optimiztion(self):
        return self.get_equal_weighted_portfolio_returns().std().values.item() * np.sqrt(256)

    def _get_rebalancing_dates(self, dates, rebalance_frequency):

        unique_dates = np.unique(dates)
        if rebalance_frequency == 'D':
            return pd.to_datetime(unique_dates)
        elif rebalance_frequency == 'W':
            return pd.to_datetime(unique_dates[unique_dates.dayofweek == 4])
        elif rebalance_frequency == 'M':
            yearMonths = unique_dates.month + unique_dates.year * 100
            return pd.to_datetime([unique_dates[yearMonths == x].max() for x in np.unique(yearMonths)])
        else:
            return pd.to_datetime(unique_dates)

    def get_covariance_matrix(self):

        returns = self.get_instrument_returns_for_optimization()

        _sig_f = returns.rolling(window=44, min_periods=7).std(ddof=0)
        _sig_s = returns.rolling(window=10 * _DAYS_PER_YEAR, min_periods=7).std(ddof=0)
        sig = _sig_f * 0.7 + _sig_s * 0.3

        corrs = returns.ewm(span=252 * 5, min_periods=7).corr()
        _unique_dates = np.intersect1d(sig.index, corrs.index.unique(0))

        corrs = corrs.loc[_unique_dates]
        sig = sig.loc[_unique_dates]

        _sig = sig.loc[_unique_dates].values
        _corrs = corrs.values

        corr_dates = corrs.index.get_level_values(0)
        corrmats = list()
        for t in tqdm(range(len(_unique_dates))):
            tmp = _corrs[corr_dates == _unique_dates[t], :] * np.outer(_sig[t], _sig[t])
            corrmats.extend([tmp])
        return pd.DataFrame(np.vstack(corrmats), columns=corrs.columns, index=corrs.index)

    def optimize_strategy_hyperparameters(self):

        alpha = [0.1]
        beta = [0]
        rho = [0.18, 0.2, 0.22]

        bmk = self.get_benchmark_pnl_contributions()

        hyperams = {}
        for a in alpha:
            for b in beta:
                for r in rho:
                    pnl = self.optimize_strategy_risk_budget(covar_lookback=r,
                                                             cov_shrink=b,
                                                             signal_shrink=a)
                    ma = Evaluator(pnl, bmk)
                    spread_ = ma.get_strategy_performance_attribution().values[-1]
                    hyperams[(a, b, r)] = ma.get_strategy_performance_attribution().values[-1]
                    print(spread_)




    def optimize_strategy_risk_budget(self, rebalance_frequency='D',
                                      covar_lookback=1,
                                      cov_shrink=0,
                                      signal_shrink=0.1):

        # Get the target risk for the optimization
        #risk = self._get_target_risk_for_optimiztion()
        risk = 0.09461731341723605

        # Get the returns for the instruments
        returns = self.get_instrument_returns_for_optimization()

        # Construct rolling covariance estimates
        T = _DAYS_PER_YEAR * covar_lookback
        covs_ = returns.ewm(min_periods=10, span=round(T), ignore_na=True).cov() * _DAYS_PER_YEAR

        # Get Instrument Signals
        sigs = self.get_signals_for_optimization()
        sigs = sigs.clip(-20, 20).dropna(how='all')
        mu_ = sigs.mean(axis=1).values.reshape(-1,1)
        shrunk_signals_ = mu_ + (sigs - mu_) * signal_shrink

        # Intersect of Unique Dates
        unique_dates = np.intersect1d(covs_.index.unique(0),
                                      shrunk_signals_.index)

        # Get the rebalancing/re-optimization dates
        signals = shrunk_signals_.loc[unique_dates].values
        covs = covs_.loc[unique_dates].values
        cov_dates = covs_.loc[unique_dates].index.get_level_values(0)

        rebal_dates = self._get_rebalancing_dates(unique_dates, rebalance_frequency)

        optWts = np.full((len(rebal_dates), covs.shape[1]), np.nan)
        for t in tqdm(range(len(rebal_dates)), desc="Optimizing Strategy"):

            try:
                tmp_cov = covs[rebal_dates[t] == cov_dates, :]
                tmp_sig = signals[t]

                keep_sig = (~np.isnan(tmp_sig) & ~np.all(np.isnan(tmp_cov), axis=0))

                cov_ = tmp_cov[keep_sig,:][:,keep_sig]
                score_ = tmp_sig[keep_sig]
                if cov_.size > 0:
                    scovar = cov_shrink * np.diag(np.diag(cov_)) + (1-cov_shrink) * cov_
                    opt = LongShortRiskBudgeting(scovar, risk, budgets=score_)
                    opt.solve()
                    optWts[t, keep_sig] = opt.x
            except:
                pass

        _wts = pd.DataFrame(optWts, columns=covs_.columns, index=unique_dates)
        _wts = _wts.reindex(returns.index).ffill()
        _rtns = returns[_wts.columns].values * _wts.shift(1)

        sig_mult = risk / (_rtns.sum(axis=1).std() * np.sqrt(256))
        _pnls = self._capital * sig_mult * returns[_wts.columns].values * _wts.shift(1)

        cols = [(x, 'risk_parity') for x in _pnls.columns]
        _pnls.columns = pd.MultiIndex.from_tuples(cols)
        _pnls.columns.names = ['instrument', 'strategy']
        return _pnls.copy()

    def optimize_strategy_risk_parity(self, rebalance_frequency='D', covar_lookback=1):

        # Get the target risk for the optimization
        risk = self._get_target_risk_for_optimiztion()

        # Get the returns for the instruments
        returns = self.get_instrument_returns_for_optimization()

        # Construct rolling covariance estimates
        T = _DAYS_PER_YEAR * covar_lookback
        covs = returns.ewm(min_periods=10, span=round(T), ignore_na=True).cov() * _DAYS_PER_YEAR

        # Get the rebalancing/re-optimization dates
        rebal_dates = self._get_rebalancing_dates(covs.index.get_level_values(0), rebalance_frequency)

        # Get Instrument Signals
        signals = self.get_asset_class_number_of_contracts()
        _signed_signals = np.sign(signals)

        optWts = []
        for t in tqdm(range(len(rebal_dates)), desc="Optimizing Strategy"):
            date = rebal_dates[t]

            try:
                covmat = covs.loc[date].dropna(how='all', axis=0).dropna(how='all', axis=1)
                score = _signed_signals.loc[date].replace(0, np.nan).dropna()

                common = np.intersect1d(covmat.columns, score.index)
                cov_ = covmat.loc[common][common]
                score_ = score.loc[common]

                if covmat.size > 0:
                    # Get the number of assets we have data for
                    target_risk = risk.loc[date].values.item()
                    opt = LongShortRiskParity(cov_.values, risk=target_risk, score=score_.values)
                    opt.solve()
                    optWts.append(pd.DataFrame(opt.x, index=cov_.index, columns=[date]).T)
            except:
                pass

        _wts = pd.concat(optWts, axis=0).reindex(returns.index).ffill()
        _pnls = self._capital * returns[_wts.columns].values * _wts.shift(1)

        cols = [(x, 'risk_parity') for x in _pnls.columns]
        _pnls.columns = pd.MultiIndex.from_tuples(cols)
        _pnls.columns.names = ['instrument', 'strategy']
        return _pnls.copy()


    def optimize_strategy_risk_parity_with_return(self, rebalance_frequency='D', covar_lookback=1, horizon=1, window=252):


        # Get the target risk for the optimization
        target_risk = self._get_target_risk_for_optimiztion()

        # Get the returns for the instruments
        returns = self.get_instrument_returns_for_optimization()

        # Construct rolling covariance estimates
        T = _DAYS_PER_YEAR * covar_lookback
        covs = returns.ewm(min_periods=10, span=round(T), ignore_na=True).cov() * _DAYS_PER_YEAR

        # Get the rebalancing/re-optimization dates
        rebal_dates = self._get_rebalancing_dates(covs.index.get_level_values(0), rebalance_frequency)

        # Get Instrument Signals/Expected Returns
        signals = self.get_asset_class_expected_returns(horizon=horizon, window=window)

        optWts = []
        for t in tqdm(range(len(rebal_dates)), desc="Optimizing Strategy"):
            date = rebal_dates[t]

            try:
                covmat = covs.loc[date].dropna(how='all', axis=0).dropna(how='all', axis=1)
                score = signals.loc[date].replace(0, np.nan).dropna()

                common = np.intersect1d(covmat.columns, score.index)
                cov_ = covmat.loc[common][common]
                score_ = score.loc[common]

                if covmat.size > 0:
                    # Get the number of assets we have data for
                    risk = target_risk.loc[date].values.item()
                    opt = RiskBudgetWithERandVolTarget(cov_.values,
                                                       risk=risk,
                                                       pi=score_.values)
                    opt.solve()
                    optWts.append(pd.DataFrame(opt.x, index=cov_.index, columns=[date]).T)
            except:
                pass

        _wts = pd.concat(optWts, axis=0).reindex(returns.index).ffill()
        _pnls = self._capital * returns[_wts.columns].values * _wts.shift(1)

        cols = [(x, 'rp_with_returns') for x in _pnls.columns]
        _pnls.columns = pd.MultiIndex.from_tuples(cols)
        _pnls.columns.names = ['instrument', 'strategy']
        return _pnls.copy()

    def get_instrument_returns_for_optimization(self):
        returns = self._prices.pct_change()
        returns.loc['10-Jun-2020':'1-Jan-2022', 'NIKKEI'] = np.nan
        return returns.copy()

    def get_strategy_pnl_contributions(self,
                                       method='rp',
                                       rebalance_freq='D',
                                       covar_lookback=256,
                                       horizon=None,
                                       window=None):

        if method.lower() in ['rp','erc']:
           return self.optimize_strategy_risk_parity(rebalance_freq, covar_lookback)
        elif method.lower() in ['rb']:
           return self.optimize_strategy_risk_budget(rebalance_freq, covar_lookback)
        elif method.lower() in ['rpr']:
           return self.optimize_strategy_risk_parity_with_return(rebalance_freq, covar_lookback, horizon, window)
        else:
            raise ValueError('Error - method {} not recognized'.format(method))

    def get_benchmark_pnl_contributions(self):
        benchmark_pnls = self.get_signal_pnls()
        cols = [(x.split(' ')[0], x, 'benchmark') for x in benchmark_pnls.columns]
        benchmark_pnls.columns = pd.MultiIndex.from_tuples(cols)
        benchmark_pnls.columns.names = ['instrument', 'signal', 'strategy']

        N = benchmark_pnls.shape[1] - benchmark_pnls.isna().sum(axis=1)
        return (benchmark_pnls / N.values.reshape(-1, 1))

    @staticmethod
    def get_moving_average_strategy():

        instrument_list = ['US10', 'EDOLLAR', 'SP500', 'WHEAT', 'JPY', 'LEANHOG', 'NIKKEI', 'BITCOIN', 'GOLD', 'COPPER']
        ma_signal = MovingAverageSignal()

        ma_signal.set_single_moving_average_signal_params(fast=4, slow=16, cap_lower=-20, cap_upper=20)
        ma_signal.set_single_moving_average_signal_params(fast=16, slow=64, cap_lower=-20, cap_upper=20)
        ma_signal.set_single_moving_average_signal_params(fast=64, slow=256, cap_lower=-20, cap_upper=20)
        str = CStrategy(instrument_list=instrument_list,
                        base_currency='USD',
                        capital=100000,
                        target_annual_vol=0.2)
        str.add_signal_to_strategy(ma_signal)
        return str

    @staticmethod
    def get_breakout_strategy():

        instrument_list = ['US10', 'EDOLLAR', 'SP500', 'WHEAT', 'JPY', 'LEANHOG', 'NIKKEI', 'BITCOIN', 'GOLD', 'COPPER']
        breakout_signal = BreakoutSignal()
        breakout_signal.set_single_breakout_signal_params(lookback=16, cap_lower=-20, cap_upper=20)
        breakout_signal.set_single_breakout_signal_params(lookback=64, cap_lower=-20, cap_upper=20)
        breakout_signal.set_single_breakout_signal_params(lookback=256, cap_lower=-20, cap_upper=20)

        # 2c. Instantiate Strategy Class
        bo_str = CStrategy(instrument_list=instrument_list,
                           base_currency='USD',
                           capital=100000,
                           target_annual_vol=0.2)
        bo_str.add_signal_to_strategy(breakout_signal)
        return bo_str

class Evaluator(object):

    def __init__(self, strategy_pnls, benchmark_pnls, capital=100000):

        self._capital = capital
        self._strategy_pnls = strategy_pnls[strategy_pnls.first_valid_index():].droplevel('strategy', axis=1)
        self._benchmark_pnls = benchmark_pnls[benchmark_pnls.first_valid_index():]
        self._ds = DataHandler.get_instance()

        self._start_date = np.maximum(np.min(self._strategy_pnls.index), np.min(self._benchmark_pnls.index))
        self._end_date = np.maximum(np.max(self._strategy_pnls.index), np.max(self._benchmark_pnls.index))

    def get_cumulative_pnls(self):
        return pd.concat((self.get_strategy_cumulative_pnl(),
                          self.get_benchmark_cumulative_pnl()), axis=1)
    def get_strategy_instrument_pnl_contribution(self):
        return self._strategy_pnls[self._start_date:self._end_date].fillna(0)
    def get_benchmark_instrument_pnl_contribution(self):
        return self._benchmark_pnls[self._start_date:self._end_date].T.groupby(level=0).sum().T
    def get_strategy_pnl(self):
        return self.get_strategy_instrument_pnl_contribution().sum(axis=1).to_frame('strategy')
    def get_benchmark_pnl(self):
        return self.get_benchmark_instrument_pnl_contribution().sum(axis=1).to_frame('benchmark')
    def get_strategy_instrument_cumulative_pnl_contribution(self):
        return self.get_strategy_instrument_pnl_contribution().cumsum(axis=0)
    def get_benchmark_instrument_cumulative_pnl_contribution(self):
        return self.get_benchmark_instrument_pnl_contribution().cumsum(axis=0)
    def get_strategy_cumulative_pnl(self):
        return self.get_strategy_pnl().cumsum(axis=0)

    def get_benchmark_cumulative_pnl(self):
        return self.get_benchmark_pnl().cumsum(axis=0)
    def get_strategy_instrument_return_contribution(self):
        return self.get_strategy_instrument_pnl_contribution() / self._capital
    def get_benchmark_instrument_return_contribution(self):
        return self.get_benchmark_instrument_pnl_contribution() / self._capital

    def get_strategy_instrument_pnl_attribution(self):

        bmk = self.get_benchmark_instrument_pnl_contribution()
        str = self.get_strategy_instrument_pnl_contribution()

        unique_dates = bmk.index.append(str.index).unique()

        bmk_ = bmk.reindex(unique_dates).ffill()
        str_ = str.reindex(unique_dates).ffill()
        return str_ - bmk_.get(str_.columns)

    def get_strategy_instrument_attribution(self):

        bmk = self.get_benchmark_instrument_cumulative_pnl_contribution()
        str = self.get_strategy_instrument_cumulative_pnl_contribution()

        unique_dates = bmk.index.append(str.index).unique()

        bmk_ = bmk.reindex(unique_dates).ffill()
        str_ = str.reindex(unique_dates).ffill()
        return str_ - bmk_.get(str_.columns)

    def get_strategy_performance_attribution(self):
        return self.get_strategy_instrument_attribution().sum(axis=1)

    def get_strategy_returns(self):
        return self.get_strategy_pnl() / self._capital

    def get_benchmark_returns(self):
        return self.get_benchmark_pnl() / self._capital

    def get_instrument_contract_pnl(self, instr_name):
        return self._ds.get_instrument_contract_pnl(instr_name)

    def get_strategy_instrument_number_of_contracts(self):
        pos_pnl = self.get_strategy_instrument_pnl_contribution()
        con_pnls = self._ds.get_instruments_contract_pnls().get(pos_pnl.columns).reindex(pos_pnl.index)
        return (pos_pnl / con_pnls.values).shift(-1).dropna(how='all')

    def get_benchmark_instrument_number_of_contracts(self):
        pos_pnl = self.get_benchmark_instrument_pnl_contribution()
        con_pnls = self._ds.get_instruments_contract_pnls().get(pos_pnl.columns).reindex(pos_pnl.index)
        return (pos_pnl / con_pnls.values).shift(-1).dropna(how='all')

    def get_strategy_instrument_position_value(self):
        prices = self._ds._df.copy()
        return (prices *
                self.get_strategy_instrument_number_of_contracts().get(prices.columns) *
                self._ds.get_contract_units(prices.columns))

    def get_benchmark_instrument_position_value(self):
        prices = self._ds._df.copy()
        return (prices *
                self.get_benchmark_instrument_number_of_contracts().get(prices.columns) *
                self._ds.get_contract_units(prices.columns))

    def get_strategy_effective_weights(self):
        return self.get_strategy_instrument_position_value() / self._capital

    def get_benchmark_effective_weights(self):
        return self.get_benchmark_instrument_position_value() / self._capital

    @staticmethod
    def run_risk_decomposition(_df, period=256):
        covs = _df.ewm(min_periods=period, span=period, ignore_na=True).cov()

        frames = [pd.DataFrame()]
        for d in covs.index.unique(level=0):

            tmp = covs.loc[d].dropna(how='all', axis=0).dropna(how='all', axis=1)
            N, _ = tmp.shape

            sig_sq = np.ones(N) @ tmp @ np.ones(N)
            sig = np.sqrt(np.ones(N) @ tmp @ np.ones(N)) * np.sqrt(256)
            cont = (sig * ((tmp.dot(np.ones(tmp.shape[0])) * np.ones(tmp.shape[0])) / sig_sq)).to_frame(d)
            frames.extend([cont])
        return pd.concat(frames, axis=1).T

    def get_benchmark_volatility_decomposition(self, period=256):
        rtns = self.get_benchmark_instrument_return_contribution()
        return Evaluator.run_risk_decomposition(rtns, period)

    def get_strategy_volatility_decomposition(self, period=256):
        rtns = self.get_strategy_instrument_return_contribution()
        return Evaluator.run_risk_decomposition(rtns, period)

    def get_benchmark_risk_decomposition(self, period=256):
        rtns = self.get_benchmark_instrument_return_contribution()
        vol = Evaluator.run_risk_decomposition(rtns, period)
        return vol / vol.sum(axis=1).values.reshape(-1, 1)

    def get_strategy_risk_decomposition(self, period=256):
        rtns = self.get_strategy_instrument_return_contribution()
        vol = Evaluator.run_risk_decomposition(rtns, period)
        return vol / vol.sum(axis=1).values.reshape(-1, 1)

    def get_single_instrument_number_of_contrcts(self, instr_name):
        bmk = self.get_benchmark_instrument_number_of_contracts()
        str = self.get_strategy_instrument_number_of_contracts()
        cols = (str.get(instr_name).to_frame((instr_name, 'str')),
                bmk.get(instr_name).to_frame((instr_name, 'bmk')))
        return pd.concat(cols, axis=1)

    def get_signal_distribution(self, strategy):

        dist = [strategy.get_signals_single_instrument(x).dropna().mean(axis=1).to_frame(x).describe()
                for x in strategy.instrument_list]
        return pd.concat(dist, axis=1)

    def get_average_instrument_exposure_in_signal_buckets(self, strategy, instrument):

        # Benchmark Number of Contracts
        _pnls = self.get_instrument_contract_pnl(instrument).replace(0, np.nan).dropna()
        _rtns = np.log(self._ds.get_price_series(instrument)).diff().dropna()
        bmk = self.get_benchmark_instrument_number_of_contracts().get(instrument).dropna()
        str = self.get_strategy_instrument_number_of_contracts().get(instrument).dropna()

        common_dates = np.intersect1d(np.intersect1d(bmk.index, str.index), _pnls.index)

        _sigs = strategy.get_signals_single_instrument(instrument).mean(axis=1).dropna().to_frame('sig').loc[common_dates]

        _sigs['q'] = pd.qcut(_sigs.values.flatten(), 5, labels=False)
        _sigs['bmk'] = bmk.loc[common_dates].values
        _sigs['str'] = str.loc[common_dates].values
        _sigs['pnl'] = _pnls.loc[common_dates].values
        _sigs['return_contribution'] = _sigs['pnl'] / strategy._capital
        _sigs['ds'] = _rtns.loc[common_dates].values * 252
        _sigs = _sigs.reset_index(drop=False).set_index(['dates','q'])

        t = _sigs.groupby(level='q').mean()
        t['vol'] = _sigs.get('ds').divide(252).groupby(level='q').std(ddof=1) * np.sqrt(252)
        t['return_contribution_vol'] = _sigs.get('return_contribution').groupby(level='q').std(ddof=1) * np.sqrt(252)
        return t.copy()


if __name__ == "__main__":

    # Define the path to the spreadsheet
    path = r'/Users/francisbarker/Desktop/Trend Following'
    workbook_name = 'Keridion Candidate Project Data.xlsx'
    fullfile_path = os.path.join(path, workbook_name)

    # Instantiate the data handler, inputs are the path to the sheet containing instrument data
    handler = DataHandler(fullfile_path, sheet_name='Data')

    ##

    ma_str = CStrategy.get_moving_average_strategy()
    ma_str.optimize_strategy_hyperparameters()

    ma_str_pnls_rb_1 = ma_str.get_strategy_pnl_contributions(method='rb', rebalance_freq='D', covar_lookback=0.2)

    ma_bmk_pnls = ma_str.get_benchmark_pnl_contributions()
    ma = Evaluator(ma_str_pnls_rb_1, ma_bmk_pnls)

    ma_str_pnls_er = ma_str.get_strategy_pnl_contributions(method='rpr', rebalance_freq='D', covar_lookback=1, horizon=21, window=252)
    ma_str_pnls_rp = ma_str.get_strategy_pnl_contributions(method='rp', rebalance_freq='D', covar_lookback=1)
    ma_bmk_pnls = ma_str.get_benchmark_pnl_contributions()


    bo_str = CStrategy.get_breakout_strategy()
    bo_str_pnls = bo_str.get_strategy_pnl_contributions(method='rb', rebalance_freq='D', covar_lookback=1)
    bo_bmk_pnls = bo_str.get_benchmark_pnl_contributions()



    #################

    ma = Evaluator(ma_str_pnls_rb_1, ma_bmk_pnls)
    bo = Evaluator(bo_str_pnls, bo_bmk_pnls)

    # 0. Exposure Analysis
    E_MA = ma.get_average_instrument_exposure_in_signal_buckets(ma_str, 'EDOLLAR')
    L_MA = ma.get_average_instrument_exposure_in_signal_buckets(ma_str, 'LEANHOG')
    J_MA = ma.get_average_instrument_exposure_in_signal_buckets(ma_str, 'JPY')

    E_BO = ma.get_average_instrument_exposure_in_signal_buckets(bo_str, 'EDOLLAR')
    L_BO = ma.get_average_instrument_exposure_in_signal_buckets(bo_str, 'LEANHOG')
    J_BO = ma.get_average_instrument_exposure_in_signal_buckets(bo_str, 'JPY')

    # 1. Get Cumulative PnLs
    ma_perf_att = ma.get_strategy_performance_attribution().to_frame('Moving Average')
    bo_perf_att = bo.get_strategy_performance_attribution().to_frame('Breakout')

    # 2. Get Relative PnLs
    risk_decomp_ma_str = ma.get_strategy_volatility_decomposition(period=252*1)
    risk_decomp_ma_bmk = ma.get_benchmark_volatility_decomposition(period=252 * 1)

    # 3. Asset Class Attribution
    rel_pnl_inst = ma.get_strategy_instrument_attribution()

    # 4. Get Number of Contracts
    bmk_contracts = ma.get_benchmark_instrument_number_of_contracts()
    str_contracts = ma.get_strategy_instrument_number_of_contracts()

    # 5. Vol of Vol and Kurtosis Analysis for JPY
    prices = handler.get_price_series('JPY')

    rtns = np.log(prices).diff().dropna()
    _dates = rtns.index

    sig = rtns.rolling(window=21).std(ddof=1).dropna() * np.sqrt(252)
    rtns['sig'] = sig.reindex(_dates)

    sigsig = np.power(np.log(sig).diff(), 2).dropna()
    rtns['sigsig'] = sigsig.reindex(_dates)

    volofvol = np.log(sig).diff().rolling(window=21).std().dropna() * np.sqrt(252)
    rtns['volofvol'] = volofvol.reindex(_dates)

    _rtns = rtns.dropna()

    qs = pd.DataFrame()
    for col in _rtns.columns:

        bmk_tmp = ma.get_benchmark_instrument_pnl_contribution().get('JPY').to_frame('bmk')
        str_tmp = ma.get_strategy_instrument_pnl_contribution().get('JPY').to_frame('str')
        tmp_pnls = pd.concat((str_tmp, bmk_tmp), axis=1)

        _tmp = _rtns.get(col).reindex(bmk_tmp.index)
        q = pd.DataFrame(pd.qcut(_tmp.values.flatten(), 5, labels=False), index=_tmp.index, columns=[col])

        tmp_pnls[col] = _tmp.values.flatten()
        tmp_pnls.index = q.values.flatten()
        tmp_pnls.index.names = ['q']

        tmp_t = tmp_pnls.groupby('q').mean()
        tmp_t.columns = pd.MultiIndex.from_tuples([ (x, col) for x in tmp_t.columns ])
        qs = pd.concat((qs, tmp_t), axis=1)












