import numpy as np
import pandas as pd
from arch import arch_model
import math

class vol_functions:

    @staticmethod
    def close_close_exponential_moving_std(series, span):

        """
        Estimate volatility using Exponentially Weighted Moving Standard Deviation.

        :param series: DataFrame with columns containing stationary series
        :param span: Span parameter for EWM
        :return: Series of exponentially weighted moving standard deviations
        """

        return series.ewm(span=span, adjust=False).std()

    @staticmethod
    def close_close_equal_weighted_moving_std(series, window):

        """
        Calculate the Equal Weighted Moving Average of a returns series, which can be used as a measure of volatility.

        :param series: DataFrame containing stationary time series
        :param window: The window size for the moving average calculation
        :return: Series of equal weighted moving averages
        """

        return series.rolling(window=window, axis=0).std()

    @staticmethod
    def garch(data, p, q):

        """
        Estimate volatility using ARCH (Autoregressive Conditional Heteroskedasticity) model.
        By default, this function uses the GARCH(1,1) model, but it can be adjusted for different ARCH type models.

        :param price_data: DataFrame with at least one column containing returns
        :param returns_column: The name of the column containing returns
        :param vol_model: Type of volatility model to use ('ARCH', 'GARCH', etc.)
        :param p: Order of the autoregressive component
        :param q: Order of the moving average component
        :return: Fitted ARCH model
        """

        # Define and fit the model
        model = arch_model(data, mean='Zero', vol='GARCH', p=p, q=q)
        model_fit = model.fit(disp='off')

        return model_fit.conditional_volatility
    @staticmethod
    def arch(data, p, q):

        """
        Estimate volatility using ARCH (Autoregressive Conditional Heteroskedasticity) model.
        By default, this function uses the GARCH(1,1) model, but it can be adjusted for different ARCH type models.

        :param price_data: DataFrame with at least one column containing returns
        :param returns_column: The name of the column containing returns
        :param vol_model: Type of volatility model to use ('ARCH', 'GARCH', etc.)
        :param p: Order of the autoregressive component
        :param q: Order of the moving average component
        :return: Fitted ARCH model
        """

        # Define and fit the model
        model = arch_model(data, mean='Zero', vol='ARCH', p=p, q=q)
        model_fit = model.fit(disp='off')

        return model_fit.conditional_volatility
    @staticmethod
    def garman_klass(price_data):

        """
        Estimate the Garman-Klass volatility of a price series.
        The formula for Garman-Klass volatility is:
        sigma_GK = sqrt((1/n) * sum(0.5 * log(High_i/Low_i)^2 - (2 * log(2) - 1) * log(Close_i/Open_i)^2))

        :param price_data: DataFrame with columns ['Open', 'High', 'Low', 'Close']
        :return: Garman-Klass volatility
        """

        # Calculate the required log returns
        log_hl = np.log(price_data['High'] / price_data['Low']) ** 2
        log_co = np.log(price_data['Close'] / price_data['Open']) ** 2

        # Calculate Garman-Klass volatility
        n = len(price_data)
        sigma_gk = np.sqrt((1 / n) * np.sum(0.5 * log_hl - (2 * np.log(2) - 1) * log_co))

        return sigma_gk

    @staticmethod
    def yang_zhang(price_data):

        """
        Estimate the Yang-Zhang volatility of a price series.
        The formula for Yang-Zhang volatility is:
        sigma_YZ^2 = k * (sigma_O^2 + sigma_C^2 + sigma_RS^2)
        where k = 0.34 / (1.34 + (n+1)/(n-1)),
        sigma_O is the overnight volatility,
        sigma_C is the close-to-close volatility,
        and sigma_RS is the Rogers-Satchell volatility.

        :param price_data: DataFrame with columns ['Open', 'High', 'Low', 'Close']
        :return: Yang-Zhang volatility
        """

        n = len(price_data)


        log_hl = np.log(price_data['PH'] / price_data['PL'])
        log_co = np.log(price_data['PS'] / price_data['PO'])
        log_oc = np.log(price_data['PO'] / price_data['PS'].shift(1))
        log_oc = log_oc[1:]  # Remove NaN

        # Calculate components of the volatility
        sigma_o = np.std(log_oc)
        sigma_c = np.std(log_co)
        sigma_rs = np.sqrt((1 / n) * np.sum((log_hl * (log_co - 0.5 * log_hl)) ** 2))

        # Calculate k
        k = 0.34 / (1.34 + (n + 1) / (n - 1))

        # Calculate Yang-Zhang volatility
        return np.sqrt(k * (sigma_o ** 2 + sigma_c ** 2 + sigma_rs ** 2))

    @staticmethod
    def roger_satchell(price_data):

        """
        Estimate the Rogers-Satchell volatility of a price series.
        The formula for Rogers-Satchell volatility is:
        sigma_RS = sqrt((1/n) * sum(High_i/Low_i * (log(High_i/Close_i) * log(High_i/Open_i) + log(Low_i/Close_i) * log(Low_i/Open_i))))

        :param price_data: DataFrame with columns ['Open', 'High', 'Low', 'Close']
        :return: Rogers-Satchell volatility
        """

        # Calculate the required log returns
        log_hc = np.log(price_data['High'] / price_data['Close'])
        log_ho = np.log(price_data['High'] / price_data['Open'])
        log_lc = np.log(price_data['Low'] / price_data['Close'])
        log_lo = np.log(price_data['Low'] / price_data['Open'])

        # Calculate Rogers-Satchell volatility
        n = len(price_data)
        sigma_rs = np.sqrt((1 / n) * np.sum((log_hc * log_ho) + (log_lc * log_lo)))

        return sigma_rs




