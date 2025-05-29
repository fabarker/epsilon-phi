import pandas as pd
from typing import Union
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.dataModel.enums.Database import Datatype
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.dataModel.dataSources.curves.yieldCurve.YieldCurveMgr import YieldCurveMgr
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.timeSeries.timeSeriesMain import TimeSeriesType, CTimeSeries
from epsilonPhi.core.utils.DateUtils import DateUtils
from epsilonPhi.core.lib.bonds.BondUtils import BondUtils
import numpy as np
from datetime import timedelta

_QUANTLIB_CUTOFF = '31-Dec-1901'
gds = GlobalDataSource()

class YieldCurve(object):
    def __init__(self, region):
        self._sessionMgr = SessionMgr()
        self._yieldCurveMgr = YieldCurveMgr()
        self._region = region
        self.__set_yield_curve_df()

    @property
    def df(self):
        return self.__df.copy()

    def __set_yield_curve_df(self):
        self.__df = self._yieldCurveMgr.get_yield_curve_dataframe(self._region)

    def get_yield(self, date, maturity):
        return self.get_yields(maturity).loc[date].values[0]

    def get_yields(self, maturities):
        if not DateUtils.is_iterable(maturities):
            maturities = [maturities]

        mats = np.setdiff1d(maturities, self.df.columns)
        if len(mats) > 0:
            self.interpolate_curve_linear(mats)

        return self.df[maturities]

    def interpolate_curve_linear(self, target_maturities):
        df = self.df
        df[target_maturities] = np.nan
        self.__df = -100 + np.exp(np.log(100 + df[np.sort(df.columns)].interpolate(method='linear',
                                                                                   fill_value="extrapolate",
                                                                                   limit_direction="both", axis=1)))

    @staticmethod
    def get_days_to_maturity(pricing_date, issue_date, maturity_at_issue):
        return (YieldCurve.get_maturity_date(issue_date, maturity_at_issue) - pricing_date).days

    @staticmethod
    def get_time_to_maturity(pricing_date, issue_date, maturity_at_issue):
        return YieldCurve.get_days_to_maturity(pricing_date, issue_date, maturity_at_issue) / DateUtils.days_per_year

    @staticmethod
    def get_maturity_date(issue_date, years_to_maturity):
        return issue_date + timedelta(days=DateUtils.days_per_year * years_to_maturity)

    def get_holding_period_return(self, from_date, to_date, maturity_date):
        price_T = self.price_fixed_rate_bond(to_date, from_date, maturity_date, price_type='dirty')
        return -1 + (price_T / 100)

    def get_constant_maturity_total_return_time_series(self, maturity, frequency):
        if (self._region, maturity, frequency) not in self._yieldCurveMgr._cache.keys():
            self._yieldCurveMgr._cache[(self._region, maturity, frequency)] = \
                self.construct_total_return_constant_maturity_vector(maturity, frequency)
        return self._yieldCurveMgr._cache[(self._region, maturity, frequency)]

    def get_govt_bond_total_return_index(self, maturity):
        tickers = self._sessionMgr.get_govt_bond_tickers_for_region(self._region, maturity)
        return gds.get_time_series_data_from_ticker(tickers, Datatype.RETURN_INDEX.value)

    def construct_total_return_constant_maturity_vector(self, maturity, frequency):

        if isinstance(frequency, Frequency):
           frequency = frequency.value

        df = -1 + (self.price_fixed_rate_bonds(bond_term_to_maturity=maturity, frequency=frequency) / 100)
        df_db = self.get_govt_bond_total_return_index(maturity).get_returns()
        df.columns = df_db.columns

        df_ = pd.concat((df_db, df[df.index < df_db.index.min()]), axis=0).sort_index()
        return df_.add(1).cumprod().resample(frequency).asfreq().dropna()

    @staticmethod
    def get_total_return_time_series(region, maturity, frequency='B', ts_type=TimeSeriesType.RETURNS):
        ts = CTimeSeries(YieldCurve(region).get_constant_maturity_total_return_time_series(maturity, frequency),
                                ts_type=TimeSeriesType.LEVELS)
        if ts_type == TimeSeriesType.RETURNS:
            return ts.get_returns()
        else:
            return ts

    def price_fixed_rate_bonds(self,
                               bond_term_to_maturity,
                               coupon_frequency=2,
                               frequency='B'):

        if isinstance(frequency, Frequency):
           frequency = frequency.value

        # Constants and Parameters
        DAYS_PER_YEAR = DateUtils.days_per_year
        REDEMPTION_VALUE = 100
        TOTAL_COUPON_PERIODS = bond_term_to_maturity * coupon_frequency

        # Get market data
        df = self.get_yields(bond_term_to_maturity)
        df_ = df.resample(frequency).asfreq().dropna()

        # Extract Dates
        issue_dates = df_.index[:-1]
        settlement_dates = df_.index[1:]

        # Calculate Days Elapsed and Days to Next Coupon
        days_elapsed_since_issue = np.array((settlement_dates - issue_dates).days).reshape(-1, 1)
        days_per_coupon = round(DAYS_PER_YEAR / coupon_frequency)
        days_to_next_coupon = days_per_coupon - days_elapsed_since_issue

        # Extract Coupons and Yields
        coupon_rates = df_.values[:-1].reshape(-1, 1)
        current_yields = df_.values[1:].reshape(-1, 1)

        # Calculate Coupon Periods (N values) and Exponent for Discounting
        N_values_matrix = np.arange(1, TOTAL_COUPON_PERIODS + 1).reshape(1, -1)
        N_repeated = np.repeat(N_values_matrix, current_yields.shape[0], axis=0) - 1
        exponent_matrix = N_repeated + (days_to_next_coupon / days_per_coupon).repeat(TOTAL_COUPON_PERIODS, axis=1)

        # Calculate Discount Factors, Cash Flows, and Discounted Cash Flows
        discount_factors = np.power(1 + (current_yields.repeat(TOTAL_COUPON_PERIODS, axis=1) / coupon_frequency),
                                    exponent_matrix)
        cash_flows = np.tile(coupon_rates, TOTAL_COUPON_PERIODS) * REDEMPTION_VALUE / coupon_frequency
        cash_flows[:, -1] += REDEMPTION_VALUE  # Adding the redemption value to the last cash flow
        discounted_cash_flows = cash_flows / discount_factors

        # Calculate Dirty Price
        dirty_price = np.sum(discounted_cash_flows, axis=1).reshape(-1, 1)

        # Calculate Accrued Interest and Clean Price
        # accrued_interest = REDEMPTION_VALUE * (coupon_rates / coupon_frequency) * (
        #        days_elapsed_since_issue / days_per_coupon)

        return CTimeSeries(pd.DataFrame(dirty_price,
                            index=settlement_dates,
                            columns=['price']), ts_type=TimeSeriesType.RETURNS)

    def price_fixed_rate_bond(self,
                              pricing_date,
                              issue_date,
                              maturity_at_issue: Union[float, pd.Timestamp],
                              price_type='clean'):

        pricing_date = pd.to_datetime(pricing_date)
        issue_date = pd.to_datetime(issue_date)
        print(pricing_date)

        if isinstance(maturity_at_issue, pd.Timestamp):
           maturity_at_issue = self.get_time_to_maturity(issue_date, issue_date, maturity_at_issue)

        coupon = self.get_yield(issue_date, maturity_at_issue)
        yld = self.get_yield(pricing_date, maturity_at_issue)
        return BondUtils.price(issue_date=issue_date,
                               pricing_date=pricing_date,
                               maturity_at_issue=maturity_at_issue,
                               coupon_rate=coupon,
                               yield_tm=yld,
                               coupon_frequency=2,
                               price_type=price_type,
                               day_count_convention="Actual/Actual")




if __name__ == "__main__":

    fast_returns = YieldCurve.get_total_return_time_series('United Kingdom',
                                                           maturity=10,
                                                           frequency='BM',
                                                           ts_type=TimeSeriesType.RETURNS)

