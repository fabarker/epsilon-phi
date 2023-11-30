from epsilonPhi.core.estimator.tseriesEstimatorInf import CTimeSeriesEstimatorInf
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.config.configUtil import CAppConfig
from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.timeSeries.regression import LinearRegression
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType
from typing import Union, Optional
import pandas as pd
import numpy as np
import math
import scipy

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

    def get_historical_total_return(self,
                                    from_date=None,
                                    to_date=None) -> float:

        rtns = self.levels[from_date:to_date].get_returns()
        if self.return_type == ReturnsType.LOG:
           return float(-1 + np.exp(np.sum(rtns.values) * self.obs_per_year/rtns.data_length))
        elif self.return_type == ReturnsType.SIMPLE:
           return float(-1 + np.power(np.prod((1+rtns.values), axis=0), self.obs_per_year/rtns.data_length))
        else:
            raise ValueError('Returns type {} not supported'.format(self.return_type))

    def get_historical_excess_return_df(self,
                                        currency,
                                        from_date=None,
                                        to_date=None):

        levels = self.levels[from_date:to_date]
        rfr = gds.get_risk_free_rate_for_currency_region(currency)
        rfr = rfr.get_levels()
        x, y = levels.intersect_over_dates(rfr)

        rtn = x.get_returns()
        return rtn.subtract_over_common_dates(rfr.get_returns())

    def get_historical_risk_premium(self,
                                    currency,
                                    from_date=None,
                                    to_date=None) -> np.array:
        rx = self.get_historical_excess_return_df(currency, from_date, to_date).values
        return np.nanmean(rx, axis=0) * self.obs_per_year

    def get_historical_volatility(self,
                                  from_date=None,
                                  to_date=None) -> np.array:

        rtn = self.levels[from_date:to_date].get_returns().values
        return np.nanstd(rtn, ddof=1, axis=0) * np.sqrt(self.obs_per_year)

    def get_historical_sharpe_ratio(self,
                                    currency,
                                    from_date=None,
                                    to_date=None) -> float:

        mu = self.get_historical_risk_premium(currency, from_date, to_date)
        sig = self.get_historical_volatility(from_date, to_date)
        return mu / sig


    def get_historical_value_at_risk(self,
                                     from_date=None,
                                     to_date=None,
                                     confidence_level=0.99,
                                     horizon=1) -> float:

        rr = self.levels[from_date:to_date].pct_change(round(horizon * self.obs_per_year))
        return np.quantile(rr.dropna(), 1-confidence_level, axis=0)

    def get_historical_conditional_value_at_risk(self,
                                                 from_date=None,
                                                 to_date=None,
                                                 confidence_level=0.99,
                                                 horizon=1) -> float:

        rr = self.levels[from_date:to_date].pct_change(horizon * self.obs_per_year)
        UB = self.get_historical_value_at_risk(from_date, to_date, confidence_level, horizon)
        return rr[rr.values < UB].mean(skipna=True, axis=0).values

    def get_historical_probability_of_loss(self,
                                          from_date=None,
                                          to_date=None,
                                          horizon=1) -> float:

        rr = self.levels[from_date:to_date].pct_change(round(horizon * self.obs_per_year))
        return rr.apply(lambda x: np.mean(x.dropna() < 0), axis=0).values

    def get_historical_worst_peak_to_trough(self,
                                            from_date=None,
                                            to_date=None) -> float:
        return (-1 + (self.levels / self.levels.expanding().max())).min().values

    def get_historical_equity_beta(self,
                                   currency,
                                   from_date=None,
                                   to_date=None) -> float:

        rx = self.get_historical_excess_return_df(currency, from_date, to_date).get_levels()
        mkt = gds.get_excess_return_series_from_ticker('MSWRLD$', TimeSeriesType.LEVELS)

        reg = LinearRegression()
        betas = list()
        for col in rx.columns:
            X, y = rx.get([col]).intersect_over_dates(mkt)
            betas.extend(reg.fit(X.get_returns(), y.get_returns().values.flatten()).coef_)
        return np.asarray(betas)

    def get_historical_alpha_over_equity(self,
                                         currency,
                                         from_date=None,
                                         to_date=None):

        rx = self.get_historical_excess_return_df(currency, from_date, to_date).get_levels()
        mkt = gds.get_excess_return_series_from_ticker('MSWRLD$', TimeSeriesType.LEVELS)

        reg = LinearRegression()
        betas = list()
        for col in rx.columns:
            X, y = rx.get([col]).intersect_over_dates(mkt)
            betas.extend([reg.fit(X.get_returns(), y.get_returns().values.flatten()).intercept_])
        return np.asarray(betas)


    def get_historical_skewness(self,
                                from_date=None,
                                to_date=None):
        return scipy.stats.skew(self.returns[from_date:to_date], axis=0)

    def get_historical_worst_period_return(self,
                                           from_date=None,
                                           to_date=None,
                                           period=1) -> float:

        res = self.levels[from_date:to_date].pct_change(round(period * self.obs_per_year)).max(axis=0)
        return res.values

    def get_historical_best_period_return(self,
                                         from_date=None,
                                         to_date=None,
                                         period=1) -> float:

        res = self.levels[from_date:to_date].pct_change(round(period * self.obs_per_year)).min(axis=0)
        return res.values

    def get_historical_real_return(self,
                                   currency,
                                   from_date=None,
                                   to_date=None) -> float:

        cpi = gds.get_consumer_price_index_for_region(currency)
        lvl = self.levels[from_date:to_date]
        rr = lvl.divide_over_common(cpi)
        return CTimeSeriesEstimator(rr, return_type=rr.return_type, ts_type=rr.type).get_historical_total_return()

    def get_historical_inflation_out_performance_frequency(self,
                                                          currency,
                                                          horizon=1,
                                                          from_date=None,
                                                          to_date=None,
                                                          annual_alpha_target=0) -> float:

        cpi = gds.get_consumer_price_index(currency)
        lvl = self.levels[from_date:to_date]
        ptf, bmk = lvl.intersect_over_common_dates(cpi)

        x = bmk.pct_change(horizon * self.obs_per_year) + (annual_alpha_target / self.obs_per_year)
        y = ptf.pct_change(horizon * self.obs_per_year)
        return np.nanmean(y > x)


if __name__ == "__main__":

    from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
    gds = GlobalDataSource()
    ts = gds.get_dataframe_from_tickers('MSWRLD$', 'RI')
    self = CTimeSeriesEstimator(ts, returns_type=ReturnsType.SIMPLE, time_series_type=TimeSeriesType.LEVELS)
    self.get_historical_real_return('United States')
