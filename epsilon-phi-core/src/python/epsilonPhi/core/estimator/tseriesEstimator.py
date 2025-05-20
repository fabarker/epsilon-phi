from epsilonPhi.core.estimator.tseriesEstimatorInf import CTimeSeriesEstimatorInf
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries
from epsilonPhi.core.utils.TimeSeriesUtils import TimeSeriesUtils
from epsilonPhi.core.timeSeries.regression import LinearRegression
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType
from typing import Union, Optional
import scipy.stats as stats
import pandas as pd
import numpy as np
import pandas as _pd
import quantstats as qs

gds = GlobalDataSource()


class CTimeSeriesEstimator(CTimeSeriesEstimatorInf):
    def __init__(self,
                 ts: Optional[Union[pd.Series, pd.DataFrame]],
                 returns_type: ReturnsType,
                 time_series_type: TimeSeriesType):

        super(CTimeSeriesEstimator, self).__init__()
        self._ts = CTimeSeries(data=ts, returns_type=returns_type, ts_type=time_series_type)
        self._ts = self._ts.get_levels()
        self.__return_type = returns_type
        self._bmk = None

    @property
    def obs_per_year(self):
        return self._ts.obs_per_year

    @property
    def levels(self):
        return self._ts.get_levels().deepcopy()

    @property
    def returns(self):
        return self._ts.get_returns().deepcopy()

    @property
    def return_type(self):
        return self.__return_type

    def set_benchmark_returns(self, df, returns_type, time_series_type):
        self._bmk = CTimeSeries(data=df, returns_type=returns_type, ts_type=time_series_type)

    def get_returns(self, from_date, to_date):
        return self.levels[from_date:to_date].get_returns()

    def return_total(self, from_date=None, to_date=None) -> float:
        """ Returns the total return over a given period, annualized """

        returns = self.get_returns(from_date, to_date)
        if self.return_type == ReturnsType.LOG:
            return float(-1 + np.exp(np.sum(returns.values) * self.obs_per_year/returns.data_length))
        elif self.return_type == ReturnsType.SIMPLE:
            return -1 + np.power(returns.add(1).product(axis=0), self.obs_per_year/returns.data_length)
        else:
            raise ValueError('Returns type {} not supported'.format(self.return_type))

    def cagr(self, from_date=None, to_date=None):
        """ Returns the compound annual growth rate """
        return self.return_total(from_date, to_date)

    def comp(self, from_date=None, to_date=None):
        """ Returns the compound cumulative growth/return """
        returns = self.get_returns(from_date, to_date)
        return returns.add(1).prod() - 1

    def avg_rtn(self, from_date=None, to_date=None):
        """ Returns the simple mean of returns """
        return np.mean(self.get_returns(from_date, to_date))

    def excess_return_series(self, currency, from_date=None, to_date=None):
        """ Returns the time series of strategy excess returns over cash """

        levels = self.levels[from_date:to_date]
        rfr = gds.get_risk_free_rate_for_currency_region(currency)
        rfr = rfr.get_levels()
        x, y = levels.intersect_over_dates(rfr)

        rtn = x.get_returns()
        return rtn.subtract_over_common_dates(rfr.get_returns())

    def risk_premium(self, currency, from_date=None, to_date=None) -> np.array:
        """ Returns the mean annualized excess return over cash """

        rx = self.excess_return_series(currency, from_date, to_date).values
        return np.nanmean(rx, axis=0) * self.obs_per_year

    def volatility(self, from_date=None, to_date=None) -> np.array:
        """ Returns the strategy annualized volatility """

        rtn = self.get_returns(from_date, to_date).values
        return np.nanstd(rtn, ddof=1, axis=0) * np.sqrt(self.obs_per_year)

    def downside_volatility(self, from_date=None, to_date=None):
        """ Returns the strategy annualized downside volatility """

        returns = self.get_returns(from_date, to_date).values
        return np.sqrt((returns[returns < 0] ** 2).sum() / len(returns))

    def sharpe_ratio(self, currency, from_date=None,  to_date=None) -> float:
        """ Returns the strategy annualized Sharpe ratio """

        return self.risk_premium(currency, from_date, to_date) / self.volatility(from_date, to_date)

    def var(self, from_date=None, to_date=None, confidence_level=0.99, horizon=1) -> float:
        """ Returns the strategy empirical value at risk for a given horizon and confidence level """

        rr = self.levels[from_date:to_date].pct_change(round(horizon * self.obs_per_year))
        return np.quantile(rr.dropna(), 1-confidence_level, axis=0)

    def cvar(self, from_date=None, to_date=None, confidence_level=0.99, horizon=1) -> float:
        """ Returns the strategy empirical conditional value at risk for a given horizon and confidence level """

        rr = self.levels[from_date:to_date].pct_change(horizon * self.obs_per_year)
        ub = self.var(from_date, to_date, confidence_level, horizon)
        return rr[rr.values < ub].mean(skipna=True, axis=0).values

    def expected_shortfall(self, from_date=None, to_date=None, confidence_level=0.99, horizon=1) -> float:
        """ Returns the strategy expected shortfall for a given horizon and confidence level """
        return self.cvar(from_date, to_date, confidence_level, horizon)

    def frequency_of_loss(self, from_date=None, to_date=None, horizon=1) -> float:
        """ Returns the strategy frequency of return observations for a given horizon that are < 0 """

        rr = self.levels[from_date:to_date].pct_change(round(horizon * self.obs_per_year))
        return rr.apply(lambda x: np.mean(x.dropna() < 0), axis=0).values

    def drawdown_series(self, from_date=None, to_date=None):
        """ Returns the strategy rolling drawdown series """

        lvl = self.levels[from_date:to_date]
        return -1 + (lvl / lvl.expanding().max())

    def max_drawdown(self, from_date=None, to_date=None):
        """ Returns the strategy maximum peak to trough loss/drawdown/downdraft """
        return self.worst_peak_to_trough(from_date, to_date)

    def worst_peak_to_trough(self, from_date=None, to_date=None) -> float:
        """ Returns the strategy maximum peak to trough loss/drawdown/downdraft """
        return self.drawdown_series(from_date, to_date).min().values

    def outliers(self, quantile=0.95, from_date=None, to_date=None):
        """Returns series of outliers"""
        returns = self.get_returns(from_date, to_date)
        return returns[returns > returns.quantile(quantile)].dropna(how="all")

    def remove_outliers(self, quantile=0.95, from_date=None, to_date=None):
        """Returns series of returns without the outliers"""
        returns = self.get_returns(from_date, to_date)
        return returns[returns < returns.quantile(quantile)]

    def best(self, period=1, from_date=None, to_date=None):
        """Returns the best period return"""
        returns = self.levels[from_date:to_date].pct_change(round(period * self.obs_per_year))
        return returns.max()

    def worst(self, period=1, from_date=None, to_date=None):
        """Returns the worst period return"""
        returns = self.levels[from_date:to_date].pct_change(round(period * self.obs_per_year))
        return returns.min()

    def consecutive_wins(self, period=1, from_date=None, to_date=None):
        """Returns the maximum consecutive wins by period"""
        returns = self.levels[from_date:to_date].pct_change(round(period * self.obs_per_year)) > 0
        return TimeSeriesUtils.count_consecutive(returns).max()

    def consecutive_losses(self, period=1, from_date=None, to_date=None):
        """ Returns the maximum consecutive losses by period """
        returns = self.levels[from_date:to_date].pct_change(round(period * self.obs_per_year)) < 0
        return TimeSeriesUtils.count_consecutive(returns).max()

    def win_rate(self, period=None, from_date=None, to_date=None):
        """Calculates the win ratio for a period"""

        if period is None:
            period = 1/self.obs_per_year

        returns = self.levels[from_date:to_date].pct_change(round(period * self.obs_per_year)).dropna()
        return np.mean(returns[returns != 0] > 0).item()

    def avg_win(self, from_date=None, to_date=None):
        """
        Calculates the average winning
        return/trade return for a period
        """
        returns = self.get_returns(from_date, to_date)
        return returns[returns > 0].dropna().mean()

    def avg_loss(self, from_date=None, to_date=None):
        """
        Calculates the average low if
        return/trade return for a period
        """
        returns = self.get_returns(from_date, to_date)
        return returns[returns < 0].dropna().mean()

    def autocorr_penalty(self, from_date=None, to_date=None):
        """Metric to account for auto correlation"""
        returns = self.get_returns(from_date, to_date).values.flatten()

        # returns.to_csv('/Users/ran/Desktop/test.csv')
        num = len(returns)
        coef = np.abs(np.corrcoef(returns[:-1], returns[1:])[0, 1])
        corr = [((num - x) / num) * coef ** x for x in range(1, num)]
        return np.sqrt(1 + 2 * np.sum(corr))

    # ======= METRICS =======

    def smart_sharpe(self, currency, from_date=None, to_date=None):
        """
        Calculates the smart sharpe ratio
        which adjust the volatility to account
        for auto correlation in strategy time series
        """

        rx = self.excess_return_series(currency, from_date, to_date)
        return np.sqrt(self.obs_per_year) * (rx.mean() / (rx.std(ddof=1) * self.autocorr_penalty(from_date, to_date)))

    def sortino(self, currency, from_date=None, to_date=None):
        """Returns the sortino ratio"""
        rx = self.excess_return_series(currency, from_date, to_date)
        downside = np.sqrt((rx[rx < 0] ** 2).sum() / len(rx))
        return np.sqrt(self.obs_per_year) * (rx.mean() / downside)

    def smart_sortino(self, currency, from_date=None, to_date=None):
        """
        Calculates the smart sortino ratio
        which adjust the volatility to account
        for auto correlation in strategy time series
        """
        rx = self.excess_return_series(currency, from_date, to_date)
        downside = np.sqrt((rx[rx < 0] ** 2).sum() / len(rx)) * self.autocorr_penalty(from_date, to_date)
        return np.sqrt(self.obs_per_year) * (rx.mean() / downside)

    def adjusted_sortino(self, currency, from_date=None, to_date=None):
        """
        Jack Schwager's version of the Sortino ratio allows for
        direct comparisons to the Sharpe. See here for more info:
        https://archive.is/wip/2rwFW
        """
        return self.sortino(currency, from_date, to_date) / np.sqrt(2)

    def skew(self, from_date=None, to_date=None):
        """Returns the strategy skewness"""
        returns = self.get_returns(from_date, to_date)
        return stats.skew(returns)

    def kurtosis(self, from_date=None, to_date=None):
        """Returns the strategy kurtosis"""
        returns = self.get_returns(from_date, to_date)
        return stats.kurtosis(returns)

    def probabilistic_ratio(self, currency, base='sharpe', from_date=None, to_date=None):
        """Returns the strategy probabalistic ratio"""

        if base.lower() == "sharpe":
            base = self.sharpe_ratio(currency, from_date, to_date) / np.sqrt(self.obs_per_year)
        elif base.lower() == "sortino":
            base = self.sortino(currency, from_date, to_date) / np.sqrt(self.obs_per_year)
        elif base.lower() == "adjusted_sortino":
            base = self.adjusted_sortino(currency, from_date, to_date) / np.sqrt(self.obs_per_year)
        else:
            raise Exception(
                "`metric` must be either `sharpe`, `sortino`, or `adjusted_sortino`"
            )

        series = self.excess_return_series(currency, from_date, to_date)
        skew_no = self.skew(from_date, to_date)
        kurtosis_no = self.kurtosis(from_date, to_date)
        n = len(series)

        sigma_sr = np.sqrt(
            (
                    1
                    + (0.5 * base ** 2)
                    - (skew_no * base)
                    + (((kurtosis_no - 3) / 4) * base ** 2)
            )
            / (n - 1)
        )

        ratio = base / sigma_sr
        return stats.norm.cdf(ratio) * np.sqrt(self.obs_per_year)

    def probabilistic_sharpe_ratio(self, currency, from_date=None, to_date=None):
        """Returns the strategy probabalistic sharpe ratio"""
        return self.probabilistic_ratio(currency, 'sharpe', from_date, to_date)

    def probabilistic_sortino_ratio(self, currency, from_date=None, to_date=None):
        """Returns the strategy probabalistic sortino ratio"""
        return self.probabilistic_ratio(currency, 'sortino', from_date, to_date)

    def probabilistic_adjusted_sortino_ratio(self, currency, from_date=None, to_date=None):
        """Returns the strategy probabalistic adjusted sortino ratio"""
        return self.probabilistic_ratio(currency, 'adjusted_sortino', from_date, to_date)

    def omega(self, required_return=0.0, from_date=None, to_date=None):
        """
        Determines the Omega ratio of a strategy.
        See https://en.wikipedia.org/wiki/Omega_ratio for more details.
        """
        returns = self.get_returns(from_date, to_date)
        if len(returns) < 2:
            return np.nan

        if required_return <= -1:
            return np.nan

        periods = self.obs_per_year
        if periods == 1:
            return_threshold = required_return
        else:
            return_threshold = (1 + required_return) ** (1.0 / periods) - 1

        returns_less_thresh = returns - return_threshold
        numer = returns_less_thresh[returns_less_thresh > 0.0].sum().values[0]
        denom = -1.0 * returns_less_thresh[returns_less_thresh < 0.0].sum().values[0]

        if denom > 0.0:
            return numer / denom
        return np.nan

    def gain_to_pain_ratio(self, currency, from_date=None, to_date=None, resolution="D"):
        """
        Jack Schwager's GPR. See here for more info:
        https://archive.is/wip/2rwFW
        """
        rx = self.excess_return_series(currency, from_date, to_date)
        returns = rx.resample(resolution).sum()
        downside = abs(returns[returns < 0].sum())
        return returns.sum() / downside

    def calmar(self, from_date=None, to_date=None):
        """Calculates the calmar ratio (CAGR% / MaxDD%)"""

        _cagr = self.cagr(from_date, to_date)
        max_dd = self.max_drawdown(from_date, to_date)
        return _cagr / abs(max_dd)

    def ulcer_index(self, from_date=None, to_date=None):
        """Calculates the ulcer index score (downside risk measurment)"""
        returns = self.get_returns(from_date, to_date)
        dd = self.drawdown_series(from_date, to_date)
        return np.sqrt(np.divide((dd ** 2).sum(), returns.shape[0] - 1))

    def ulcer_performance_index(self, currency, from_date=None, to_date=None):
        """
        Calculates the ulcer index score
        (downside risk measurment)
        """
        rx = self.excess_return_series(currency, from_date, to_date)
        comp_rx = rx.add(1).product() - 1
        return comp_rx / self.ulcer_index(from_date, to_date)

    def serenity_index(self, currency, from_date=None, to_date=None):
        """
        Calculates the serenity index score
        (https://www.keyquant.com/Download/GetFile?Filename=%5CPublications%5CKeyQuant_WhitePaper_APT_Part1.pdf)
        """

        rx = self.excess_return_series(currency, from_date, to_date).get_returns(ReturnsType.LOG)
        dd = self.drawdown_series(from_date, to_date)

        dd_var = np.quantile(dd, 1-0.95)
        dd_cvar = dd[dd < dd_var].mean()
        pitfall = -dd_cvar / rx.std()
        return rx.sum() / (self.ulcer_index(from_date, to_date) * pitfall)

    def risk_of_ruin(self, from_date=None, to_date=None):
        """
        Calculates the risk of ruin
        (the likelihood of losing all one's investment capital)
        """
        returns = self.get_returns(from_date, to_date)
        wins = self.win_rate(from_date, to_date)
        return ((1 - wins) / (1 + wins)) ** len(returns)

    def outlier_loss_ratio(self, quantile=0.01, from_date=None, to_date=None):
        """
        Calculates the outlier losers ratio
        1st percentile of returns / mean negative return
        """
        returns = self.get_returns(from_date, to_date)
        return returns.quantile(quantile).mean() / returns[returns < 0].mean()

    def tail_ratio(self, cutoff=0.95, from_date=None, to_date=None):
        """
        Measures the ratio between the right
        (95%) and left tail (5%).
        """
        returns = self.get_returns(from_date, to_date)
        return np.abs(returns.quantile(cutoff) / returns.quantile(1 - cutoff))

    def payoff_ratio(self, from_date=None, to_date=None):
        """Measures the payoff ratio (average win/average loss)"""
        return self.avg_win(from_date, to_date) / abs(self.avg_loss(from_date, to_date))

    def win_loss_ratio(self, from_date=None, to_date=None):
        """Shorthand for payoff_ratio()"""
        return self.payoff_ratio(from_date, to_date)

    def profit_ratio(self, from_date=None, to_date=None):
        """Measures the profit ratio (win ratio / loss ratio)"""

        returns = self.get_returns(from_date, to_date)
        wins = returns[returns >= 0]
        loss = returns[returns < 0]

        win_ratio = abs(wins.mean() / wins.count())
        loss_ratio = abs(loss.mean() / loss.count())
        try:
            return win_ratio / loss_ratio
        except ZeroDivisionError:
            return 0.0

    def profit_factor(self, from_date=None, to_date=None):
        """Measures the profit ratio (wins/loss)"""

        returns = self.get_returns(from_date, to_date)
        return abs(returns[returns >= 0].sum() / returns[returns < 0].sum())

    def cpc_index(self, from_date=None, to_date=None):
        """
        Measures the cpc ratio
        (profit factor * win % * win loss ratio)
        """
        return (self.profit_factor(from_date, to_date) *
                self.win_rate(from_date) *
                self.win_loss_ratio(from_date, to_date))

    def common_sense_ratio(self, from_date=None, to_date=None):
        """Measures the common sense ratio (profit factor * tail ratio)"""

        return self.profit_factor(from_date, to_date) * self.tail_ratio(from_date=from_date, to_date=to_date)

    def outlier_win_ratio(self, quantile=0.99, from_date=None, to_date=None):
        """
        Calculates the outlier winners ratio
        99th percentile of returns / mean positive return
        """
        returns = self.get_returns(from_date, to_date)
        return returns.quantile(quantile).mean() / returns[returns >= 0].mean()

    def recovery_factor(self, from_date=None, to_date=None):
        """Measures how fast the strategy recovers from drawdowns"""
        total_returns = self.comp(from_date, to_date)
        max_dd = self.max_drawdown(from_date, to_date)
        return abs(total_returns) / abs(max_dd)

    def beta(self, currency, from_date=None, to_date=None) -> list:
        """Returns the strategy OLS beta to global equity market"""

        def _beta(y_p, x_p):
            _reg = LinearRegression().fit(x_p, y_p.flatten())
            if hasattr(_reg, 'coef_'):
                return _reg.coef_

        rx = self.excess_return_series(currency, from_date, to_date).get_levels()
        mkt = gds.get_excess_return_series_from_ticker('MSWRLD$', TimeSeriesType.LEVELS)

        _betas = list()
        for col in rx.columns:
            x, y = rx.get([col]).intersect_over_dates(mkt)
            x = x.get_returns()
            y = y.get_returns()
            _betas.extend([_beta(y.get_returns().values,
                                 x.get_returns().values)])
        return _betas

    def alpha(self, currency, from_date=None, to_date=None):
        """Returns the strategy OLS intercept to global equity market"""

        def _alpha(y_p, x_p):
            _reg = LinearRegression().fit(x_p, y_p.flatten())
            if hasattr(_reg, 'intercept_'):
                return _reg.intercept_

        rx = self.excess_return_series(currency, from_date, to_date).get_levels()
        mkt = gds.get_excess_return_series_from_ticker('MSWRLD$', TimeSeriesType.LEVELS)

        alphas = list()
        for col in rx.columns:
            x, y = rx.get([col]).intersect_over_dates(mkt)
            alphas.extend([_alpha(y.get_returns().values,
                                  x.get_returns().values)])
        return alphas

    def residual_volatility(self, currency, from_date=None, to_date=None):
        """Returns the strategy volatility of residual of OLS regression on equity market"""

        def _residual(y_p, x_p):
            _reg = LinearRegression().fit(x_p, y_p.flatten())
            if hasattr(_reg, 'coef_'):
                return y_p.flatten() - x_p @ _reg.coef_

        rx = self.excess_return_series(currency, from_date, to_date).get_levels()
        mkt = gds.get_excess_return_series_from_ticker('MSWRLD$', TimeSeriesType.LEVELS)

        rr = list()
        for col in rx.columns:
            x, y = rx.get([col]).intersect_over_dates(mkt)
            rr.extend([np.std(_residual(y.get_returns().values,
                                        x.get_returns().values))])
        return np.sqrt(self.obs_per_year) * np.array(rr)

    def kelly_criterion(self, from_date=None, to_date=None):
        """
        Calculates the recommended maximum amount of capital that
        should be allocated to the given strategy, based on the
        Kelly Criterion (http://en.wikipedia.org/wiki/Kelly_criterion)
        """

        win_loss_ratio = self.payoff_ratio(from_date, to_date)
        win_prob = self.win_rate(from_date, to_date)
        lose_prob = 1 - win_prob
        return ((win_loss_ratio * win_prob) - lose_prob) / win_loss_ratio

    def drawdown_details(self, from_date=None, to_date=None):
        """
        Calculates drawdown details, including start/end/valley dates,
        duration, max drawdown and max dd for 99% of the dd period
        for every drawdown period
        """

        def _drawdown_details(drawdown):
            # mark no drawdown
            no_dd = drawdown == 0

            if not isinstance(no_dd, pd.DataFrame):
                no_dd = pd.DataFrame(no_dd)

            # extract dd start dates, first date of the drawdown
            starts = ~no_dd & no_dd.shift(1)
            starts = list(starts[starts.values].index)

            # extract end dates, last date of the drawdown
            ends = no_dd & (~no_dd).shift(1)
            ends = ends.shift(-1, fill_value=False)
            ends = list(ends[ends.values].index)

            # no drawdown :)
            if not starts:
                return _pd.DataFrame(
                    index=[],
                    columns=[
                        "start",
                        "valley",
                        "end",
                        "days",
                        "max drawdown",
                        "99% max drawdown",
                    ],
                )

            # drawdown series begins in a drawdown
            if ends and starts[0] > ends[0]:
                starts.insert(0, drawdown.index[0])

            # series ends in a drawdown fill with last date
            if not ends or starts[-1] > ends[-1]:
                ends.append(drawdown.index[-1])

            # build dataframe from results
            data = []
            for i, _ in enumerate(starts):
                dd = drawdown[starts[i]: ends[i]]
                clean_dd = -TimeSeriesUtils.remove_outliers(-dd, 0.99)
                data.append(
                    (
                        starts[i],
                        dd.idxmin(),
                        ends[i],
                        (ends[i] - starts[i]).days + 1,
                        dd.min() * 100,
                        clean_dd.min() * 100,
                    )
                )

            df = _pd.DataFrame(
                data=data,
                columns=[
                    "start",
                    "valley",
                    "end",
                    "days",
                    "max drawdown",
                    "99% max drawdown",
                ],
            )
            df["days"] = df["days"].astype(int)
            df["max drawdown"] = df["max drawdown"].astype(float)
            df["99% max drawdown"] = df["99% max drawdown"].astype(float)

            df["start"] = df["start"].dt.strftime("%Y-%m-%d")
            df["end"] = df["end"].dt.strftime("%Y-%m-%d")
            df["valley"] = df["valley"].dt.strftime("%Y-%m-%d")

            return df

        drawdown_series = self.drawdown_series(from_date, to_date)
        if isinstance(drawdown_series, _pd.DataFrame):
            _dfs = {}
            for col in drawdown_series.columns:
                _dfs[col] = _drawdown_details(drawdown_series[col])
            return _pd.concat(_dfs, axis=1)

        return _drawdown_details(drawdown_series)

    def returns_matrix(self):
        """Returns the matrix of monthly returns for strategy"""

        mnthly_rtns = self.returns.get_bmonthly_returns()

        if mnthly_rtns is None:
            return None

        for col in mnthly_rtns.columns:
            col_mnthly = mnthly_rtns.get([col])
            col_mnthly['Year'] = mnthly_rtns.index.year
            col_mnthly['Month'] = mnthly_rtns.index.month
            matrix = col_mnthly.groupby(['Year', 'Month']).first().unstack(level=-1)
            matrix[(col, 'YTD')] = matrix.add(1).product(axis=1) - 1
        return matrix

    def rolling(self, period_length=252, func_apply=None, **kwargs):

        T = len(self._ts.index) - period_length + 1
        res = list()
        for i in range(T):
            print(i)

            start_date = self._ts.index[i]
            end_date = self._ts.index[i + period_length - 1]
            res.extend([func_apply(from_date=start_date, to_date=end_date).values[0]])
        return pd.DataFrame(res, index=self._ts.index[period_length - 1:], columns=[func_apply.__name__])

