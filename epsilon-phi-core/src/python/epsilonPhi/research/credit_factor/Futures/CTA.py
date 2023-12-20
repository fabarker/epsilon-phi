import pandas as pd
import numpy as np
import os
from abc import abstractmethod
from scipy.optimize import minimize

__author__ = 'francis barker'
__date__ = '13/12/2023'

#%%

# Use Days Per Year From Excel
_DAYS_PER_YEAR = 256

# Contract Information - We Use Contract Units To Price The Strategy
_VALUE_OF_CONTRACT_PRICE_MOVE_ = {'AAPL': 1, 'EDOLLAR': 2500, 'SP500': 50, 'US10': 1000, 'LEANHOG': 400, 'JPY': 12500000,
                                  'GOLD': 100, 'WHEAT': 50, 'NIKKEI': 100, 'COPPER': 25000, 'BITCOIN': 0.1}

# Contract Information - We Need to know what currency the contract prices are quoted in
_PRICE_CURRENCY_DENOMINATION = {'AAPL': 'USD', 'EDOLLAR': 'USD', 'SP500': 'USD', 'US10': 'USD', 'LEANHOG': 'USD', 'JPY': 'USD',
                                'GOLD': 'USD', 'WHEAT': 'USD', 'NIKKEI': 'JPY', 'COPPER': 'USD', 'BITCOIN': 'USD'}

class DataHandler(object):

    """ Class that loads, pre-processes and handles raw time series data.
    Also includes some data transformations such as volatility estimation

    The class assumes that data is stored in a xlsx workbook with dates in column 1 and data labels in row 1.

    Usage:
        handler = DataHandler(...data_workbook.xlsx, sheetname='data_sheet')
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
        self._load_data()

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

    def _load_data(self):
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

        assert to_currency == 'USD', 'Error - coversion to curency {} not supported'.format(to_currency)
        assert from_currency in self._df.columns, 'Error - fx rate {}{} not supported'.format(from_currency, to_currency)

        fx_rate = self._df.get(from_currency)
        common_dates = np.intersect1d(fx_rate.index, price_series.index)

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


class CSignal(object):

    """ Base class representing a trading signal.
        Encapsulates and handles more abstract methods common across all types of signals (MAC, Breakout ect...)


        Usage:
            handler = DataHandler(...data_workbook.xlsx, sheetname='data_sheet')
            prices = handler.get_price_series('JPY')

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

    def generate_signals_for_instrument(self, instrument_name):

        ds = DataHandler.get_instance()
        self.set_prices(ds.get_price_series(instrument_name))
        return self.generate_signals()

    def generate_signals(self):

        signal_dfs = []
        for sig_params in self._signals:
            tmp = self.get_capped_signal(sig_params)
            tmp.columns = [(self.instrument_name,) + tuple(sig_params.values())]
            signal_dfs.append(tmp)

        # Concatenate all signal dataframes at once
        return pd.concat(signal_dfs, axis=1)

class MovingAverageSignal(CSignal):
    """ Base class representing a trading signal.
        Encapsulates and handles more abstract methods common across all types of signals (MAC, Breakout ect...)


            Usage:
                handler = DataHandler(...data_workbook.xlsx, sheetname='data_sheet')
                prices = handler.get_price_series('JPY')

        """
    def __init__(self):
        super().__init__()

    def set_signal_smoothing_parameter(self, smoothing_paramter):
        self._signal_smoothing = smoothing_paramter

    def set_signal_multiplier(self, multiplier):
        self._multiplier = multiplier

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
        return self.prices.ewm(span=lookback, adjust=False, min_periods=10).mean()

    def get_moving_average_crossover(self, fast_period, slow_period):
        return (self.get_moving_average(fast_period) -
                self.get_moving_average(slow_period))

    def get_normalized_moving_average_crossover_signal(self, fast_period, slow_period, smoothing_parameter):

        sig_adjusted_ma_crossover = self.get_vol_adjusted_moving_average_crossover(fast_period, slow_period)
        trailing_avg_abs_sig_adj_mac = self.get_smoothed_abs_vol_adjusted_moving_average_crossover(fast_period, slow_period, smoothing_parameter)
        return self._average_abs_target * sig_adjusted_ma_crossover / trailing_avg_abs_sig_adj_mac

    def get_vol_adjusted_moving_average_crossover(self, fast_period, slow_period):
        return (self.get_moving_average_crossover(fast_period, slow_period) /
                self._datasource.get_series_price_volatility(self.instrument_name, False).values.reshape(-1, 1))

    def get_smoothed_abs_vol_adjusted_moving_average_crossover(self, fast_period, slow_period, smoothing_parameter):
        sig_adj_signal = self.get_vol_adjusted_moving_average_crossover(fast_period, slow_period)
        return sig_adj_signal.abs().rolling(window=smoothing_parameter, min_periods=10).mean()

    def get_capped_signal(self, signal_params):

        unclipped_signal = self.get_normalized_moving_average_crossover_signal(signal_params.get('fast'),
                                                                               signal_params.get('slow'),
                                                                               self._signal_smoothing)
        return unclipped_signal.clip(np.min(signal_params.get('cap_lower')),
                                     np.max(signal_params.get('cap_upper')))

class BreakoutSignal(CSignal):
    def __init__(self):
        super().__init__()

    def set_single_breakout_signal_params(self,
                                          lookback: int,
                                          cap_lower: float,
                                          cap_upper: float):

        signal_params = dict(zip(['lookback', 'cap_lower', 'cap_lower'], (lookback, cap_lower, cap_upper)))
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
        return unclipped_signal.clip(np.min(signal_params.get('cap_lower')),
                                     np.max(signal_params.get('cap_upper')))


class CStrategy(object):

    def __init__(self, instrument_list, base_currency='USD', capital=100000, target_annual_vol=0.2, multiplier=1):

        self._ds = DataHandler.get_instance()
        self._capital = capital
        self._base_currency = base_currency
        self._target_volatility = target_annual_vol
        self._multiplier = multiplier
        self._signals = list()
        self._optimal_weights = None
        self.__set_prices(instrument_list)

    def __set_prices(self, instrument_list):

        # Get a list of instruments we have data for
        cols = np.intersect1d(instrument_list, self._ds.df.columns)

        # If we dont have nay of the data, throw and error
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
        """ Returns the contract unit """
        return _VALUE_OF_CONTRACT_PRICE_MOVE_.get(instrument_name)

    def get_instrument_prices(self, instrument_name):
        """ Returns the price series for a given instrument """
        return self._prices.get(instrument_name, pd.DataFrame).dropna()

    def get_instrument_returns(self, instrument_name):
        """ Returns the price returns (in currency units) for a given instrument """
        return self.get_instrument_prices(instrument_name).diff()

    def get_fx_rate_for_instrument(self, instrument_name):
        return (self._ds.get_fx_rate_for_instrument(instrument_name, self._base_currency)
                .reindex(self.get_instrument_prices(instrument_name).index))

    def _reset_signals(self):
        """Clear signals from strategy"""
        self._signals = []

    def add_signal_to_strategy(self, signal):
        """ Method to add a signal object to the strategy"""
        self._signals.extend([signal])

    def get_signals_single_instrument(self, instrument_name) -> pd.DataFrame:
        """ Returns time series of capped signals for an instrument  """

        # 1. Get the instrument prices (we need to set these in the signal object)
        prices = self.get_instrument_prices(instrument_name)

        signal_dfs = []
        for _signal in self._signals:

            # 2. Set the prices in the signal object and generate signals
            _signal.set_prices(prices)
            tmp = _signal.generate_signals()
            signal_dfs.append(tmp)

        # Concatenate all signal dataframes at once
        return pd.concat(signal_dfs, axis=1)


    #TODO: THERE IS A MULTIPLIER MISSING ON THE SIGNAL CALCULATION
    def get_positions_single_instrument(self, instrument_name):
        """ Returns time series of instrument positions for a given strategy model
            This includes scaling positions to the target strategy volatility """

        # 1. Get the normalized signals
        signals = self.get_signals_single_instrument(instrument_name)

        # 2. Get the contract unit for the instrument we are tradung
        _contract_unit = self.get_instrument_value_per_tick(instrument_name)

        # 3. Compute the vol multiplier - This ensures we are getting close to our annual vol target
        # Adjust fot the contract size, as a single contract may give us exposure to multiple "units" of the instrument

        # CAREFUL of the currency mismatch. Our target cash volatility is in base currency units, so we have to convert any
        # Foreign currency vols back to base at the prevailing spot rate.
        price_volatility = self._ds.get_series_price_volatility(instrument_name)
        fx_rate = self.get_fx_rate_for_instrument(instrument_name)

        assert np.all(fx_rate.index == price_volatility.index), 'Error - dates not aligned'
        vol_multiplier = self.target_cash_volatility / (_contract_unit * price_volatility * fx_rate.values.flatten())

        # Below represents the number of contracts, long/short to hold
        positions = signals * vol_multiplier.values.reshape(-1, 1) / 10

        # DUE TO POTENITAL DATA ERRORS - POSITIONS IN NIKKEI ARE FORCED TO ZERO FOR THIS PERIOD
        if instrument_name == 'NIKKEI':
            positions['10-Jun-2020':'1-Jan-2022'] = np.nan
        return positions.copy()


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
        """ Returns time series of PnLs for all signal/ instruments in the strategy """

        pnl_df = []
        for instrument in self.instrument_list:
            tmp = self.get_pnl_single_instrument(instrument)
            pnl_df.append(tmp)
        return pd.concat(pnl_df, axis=1)

    def get_signal_pnl_volatility(self, lookback=44):
        """ Returns time series of annualized vols (in currency units) for strategy PnLS using simple weights """
        return self.__get_raw_signal_pnls().rolling(window=lookback, min_periods=10).std(ddof=0) * np.sqrt(_DAYS_PER_YEAR)

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
        return self.get_signal_pnls().mean(axis=1).to_frame('ew_pnl')

    def get_equal_weighted_portfolio_returns(self):
        return self.get_signal_returns().mean(axis=1).to_frame('ew_returns')

    def get_equal_weighted_portfolio_cumulative_pnl(self):
        return self.get_equal_weighted_portfolio_pnl().cumsum(axis=0)

    #####  Covariance Weighted Portfolio Related

    def get_covariance_weighted_portfolio_pnl(self):
        return (self.get_signal_pnls() * self.get_optimized_risk_parity_weights()).sum(axis=1).to_frame('cw_pnl')

    def get_covariance_weighted_portfolio_returns(self):
        return (self.get_signal_returns() * self.get_optimized_risk_parity_weights()).sum(axis=1).to_frame('cw_returns')

    def get_covariance_weighted_portfolio_cumulative_pnl(self):
        return self.get_covariance_weighted_portfolio_pnl().cumsum(axis=0)


    def get_optimized_risk_parity_weights(self):
        if self._optimal_weights is None:
            self._compute_optimal_weights()
        return self._optimal_weights

    def _compute_optimal_weights(self):

        # Get the returns streams for each instrument/strategy
        returns = self.get_signal_returns()

        # Get the exponentially weighted covariance matrix of the returns streams
        covs = returns.ewm(min_periods=_DAYS_PER_YEAR, span=_DAYS_PER_YEAR, ignore_na=True).cov() * _DAYS_PER_YEAR

        # Define some inline functions that we will use for optimization
        # Define the portfolio variance function
        portfolio_variance = lambda weights, covmat: weights @ covmat @ weights
        # Define the marginal contribution function
        marginal_contribution = lambda weights, covmat: weights * (covmat @ weights)
        # Define the risk contributions function
        risk_contributions = lambda weights, covmat: marginal_contribution(weights, covmat) / portfolio_variance(weights, covmat)
        # Define the objective function for ERC
        minfunc = lambda weights, covmat: np.sum(np.square(risk_contributions(weights, covmat) - np.mean(risk_contributions(weights, covmat))))
        # Define a constraint that we hit the target volatility
        constraints = {'type': 'eq', 'fun': lambda w: np.sqrt(portfolio_variance(w, covmat)) - self._target_volatility}

        # Extract Month Ends
        unique_dates = covs.index.get_level_values(0).unique()
        yearMonths = unique_dates.month + unique_dates.year * 100
        MEnds = [unique_dates[yearMonths == x].max() for x in np.unique(yearMonths)]

        optWts = []
        for date in MEnds:
            print(date)

            try:
                covmat = covs.loc[date].dropna(how='all', axis=0).dropna(how='all', axis=1)
                if covmat.size > 0:

                    # Get the number of assets we have data for
                    n = covmat.shape[0]

                    # Initalize a set of equal weights
                    weights = np.full(n, 1 / n)

                    # Minimize the objective function by solving for the weights
                    res = minimize(minfunc, weights, args=(covmat,), method='SLSQP', constraints=constraints)
                    optWts.append(pd.DataFrame(res.x, index=covmat.index, columns=[date]).T)
            except:
                pass

        self._optimal_weights = pd.concat(optWts, axis=0).reindex(returns.index).ffill()




if __name__ == "__main__":

    # Define the path to the spreadsheet
    path = r'/Users/francisbarker/Desktop/Trend Following'
    workbook_name = 'Keridion Candidate Project Data.xlsx'
    fullfile_path = os.path.join(path, workbook_name)

    # Instantiate the data handler, inputs are the path to the sheet containing instrument data
    handler = DataHandler(fullfile_path, sheet_name='Data')


    # Define the instrument list - which represents tradable instruments
    instrument_list = ['US10', 'EDOLLAR', 'SP500', 'WHEAT', 'JPY', 'LEANHOG', 'NIKKEI', 'BITCOIN', 'GOLD', 'COPPER']


    # Define some other parameters for the strategy
    base_currency = 'USD'                               # USD currently the only supported currency
    capital = 100000                                    # In base currency units
    target_annualized_percentage_volatility = 0.2       # Volatility we will target





    ############## Model 1 - Moving Average Model ##############

    # 1a. Instantiate a Moving Average Signal Class
    ma_signal = MovingAverageSignal()

    # 1b. Populate the class paramters using the signal set method, which supports any number of EWMC signals
    ma_signal.set_single_moving_average_signal_params(fast=4, slow=16, cap_lower=-20, cap_upper=20)
    ma_signal.set_single_moving_average_signal_params(fast=16, slow=64, cap_lower=-20, cap_upper=20)
    ma_signal.set_single_moving_average_signal_params(fast=64, slow=256, cap_lower=-20, cap_upper=20)

    # 1c. Instantiate Strategy Class
    ma_str = CStrategy(instrument_list=instrument_list,
                       base_currency=base_currency,
                       capital=capital,
                       target_annual_vol=target_annualized_percentage_volatility)

    # 1d. Add this signal class to the strategy object
    ma_str.add_signal_to_strategy(ma_signal)

    # 1e. Get strategy and single instrument performance measures
    returns_single_signals_instruments_crossover = ma_str.get_signal_returns()

    cum_pnl_single_instruments_crossover = ma_str.get_signal_cumulative_pnl()
    cum_pnl_equal_weighted_portfolio_crossover = ma_str.get_equal_weighted_portfolio_cumulative_pnl()
    cum_pnl_covar_weighted_portfolio_crossover = ma_str.get_covariance_weighted_portfolio_cumulative_pnl()


    ############## Model 2 - Breakout Model ##############

    # 2a. Instantiate a Moving Average Signal Class
    breakout_signal = BreakoutSignal()

    # 2b. Populate the class paramters using the signal set method, which supports any number of EWMC signals
    breakout_signal.set_single_breakout_signal_params(lookback=16, cap_lower=-20, cap_upper=20)
    breakout_signal.set_single_breakout_signal_params(lookback=64, cap_lower=-20, cap_upper=20)
    breakout_signal.set_single_breakout_signal_params(lookback=256, cap_lower=-20, cap_upper=20)

    # 2c. Instantiate Strategy Class
    bo_str = CStrategy(instrument_list=instrument_list,
                        base_currency=base_currency,
                        capital=capital,
                        target_annual_vol=target_annualized_percentage_volatility)

    # 2d. Add this signal class to the strategy object
    bo_str.add_signal_to_strategy(ma_signal)

    # 2e. Get strategy and single instrument performance measures
    returns_single_signals_instruments_breakout = bo_str.get_signal_returns()

    cum_pnl_single_instruments_breakout = bo_str.get_signal_cumulative_pnl()
    cum_pnl_equal_weighted_portfolio_breakout = bo_str.get_equal_weighted_portfolio_cumulative_pnl()
    cum_pnl_covar_weighted_portfolio_breakout = bo_str.get_covariance_weighted_portfolio_cumulative_pnl()



    ############## Compare Strategy Performance ##############







