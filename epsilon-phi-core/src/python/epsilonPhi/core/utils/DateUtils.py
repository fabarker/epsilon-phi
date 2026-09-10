import collections, re, six
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Union, Optional
from epsilonPhi.core.cpp.dates import cDates
from epsilonPhi.core.cpp.fastfind.find_1st import *
from numba import njit
from numba import types, int64
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency

class Offsets(object):
    @staticmethod
    def getOffset(frequency, periods):

        if isinstance(frequency, Frequency):
            frequency = frequency.value

        if isinstance(frequency, pd.DateOffset):
            frequency = frequency.name

        if frequency.lower() in ['d']:
            return pd.tseries.offsets.Day(periods)
        if frequency.lower() in ['bd','b']:
            return pd.tseries.offsets.BDay(periods)
        if frequency.lower() in ['w']:
            return pd.tseries.offsets.Week(periods)
        if frequency.lower() in ['m']:
            return pd.tseries.offsets.MonthEnd(periods)
        if frequency.lower() in ['bm']:
            return pd.tseries.offsets.BMonthEnd(periods)
        if frequency.lower() in ['q']:
            return pd.tseries.offsets.QuarterEnd(periods)
        if frequency.lower() in ['bq']:
            return pd.tseries.offsets.BQuarterEnd(periods)
        if frequency.lower() in ['y','a']:
            return pd.tseries.offsets.YearEnd(periods)
        if frequency.lower() in  ['bm','ba']:
            return pd.tseries.offsets.BYearEnd(periods)

class DateUtils(object):

    #days_per_year = (365*3+366) * (1/4)
    days_per_year = 365
    ERROR_TOLERANCE = 1e-4

    @staticmethod
    def is_iterable(arg):
        return (isinstance(arg, collections.Iterable)
                and not isinstance(arg, six.string_types))

    @staticmethod
    def to_ordinals(datetimes):
        dt = pd.to_datetime(np.array(datetimes))
        return cDates.to_ordinals(dt.to_pydatetime())

    @staticmethod
    def from_ordinals(ords):
        if not DateUtils.is_iterable(ords):
          ords = [ords]
        return pd.to_datetime(cDates.from_ordinals(np.array(ords, dtype=np.int32)))

    @staticmethod
    def find_first_date_loc(a, b):
        ord_A = DateUtils.to_ordinals(np.array(a)).reshape(-1, 1)
        ord_B = DateUtils.to_ordinals(np.array(b)).reshape(-1, 1)
        return find_1st(ord_A, ord_B)

    @staticmethod
    def find_date_locs(arr: np.array) -> np.array:
        mask = np.array([[isinstance(x, datetime) for x in row] for row in arr])
        return np.transpose(np.where(mask))

    @staticmethod
    def is_date(arr: np.array) -> np.array:
        return np.array([isinstance(x, datetime) for x in arr])

    @staticmethod
    def merge(dates: list):

        if not DateUtils.is_iterable(dates):
           dates = [dates]

        new_series = pd.concat([pd.Series(pd.to_datetime(x)) for x in dates], axis=0).unique()
        return pd.to_datetime(np.sort(new_series))



    @staticmethod
    def shift_date(date, offset, periods):
        return date + Offsets.getOffset(offset, periods)

    @staticmethod
    def get_date_range(start_date, end_date, periodicity):
        if isinstance(periodicity, Frequency):
            return pd.date_range(start_date, end_date, freq=periodicity.value)
        else:
            return pd.date_range(start_date, end_date, freq=periodicity)

    @staticmethod
    def get_daily_dates(start_date,
                        end_date):
        return pd.date_range(start_date, end_date)

    @staticmethod
    def Rdate_to_mat(Rdate):

        if not DateUtils.is_iterable(Rdate):
            matStrs = [Rdate]
        else:
            matStrs = Rdate

        T = len(matStrs)
        nYears = np.full((T,), np.nan)
        for rdate in np.unique(matStrs):

            idx = rdate == matStrs
            if re.search("st", rdate.lower()):
                rdate = '1d'
            if re.search("on", rdate.lower()):
                rdate = '1d'
            if re.search("tn", rdate.lower()):
                rdate = '2d'
            if re.search("sw", rdate.lower()):
                rdate = '1w'

            if re.search("d", rdate.lower()):
                nPeriods = DateUtils.days_per_year
                strg = "d"
            elif re.search("bm", rdate.lower()):
                nPeriods = 12
                strg = "BM"
            elif re.search("w", rdate.lower()):
                nPeriods = DateUtils.days_per_year / 7
                strg = "w"
            elif re.search("m", rdate.lower()):
                nPeriods = 12
                strg = "m"
            elif re.search("q", rdate.lower()):
                nPeriods = 4
                strg = "q"
            elif re.search("y", rdate.lower()):
                nPeriods = 1
                strg = "y"
            else:
                raise Exception("Error - relative date {} not recongnized".format(rdate))

            loc = rdate.find(strg)

            if T == 1:
                return float((rdate[0:loc]) )/ nPeriods
            else:
                nYears[idx] = float(rdate[0:loc]) / nPeriods

        if len(nYears) > 1:
            return nYears
        else:
            return float(nYears)

    @staticmethod
    def mat_to_Rdate(nYears: Optional[Union[float, np.array, list]]) -> list:

        if not DateUtils.is_iterable(nYears):
            nYears = [nYears]

        tol = DateUtils.ERROR_TOLERANCE
        tau = DateUtils.days_per_year
        T = len(nYears)

        relativeDates = np.array(["Nan"] * T, dtype=object)
        for i in range(T):
            maturity = nYears[i]

            if abs(maturity) < tol:
                relativeDates[i] = '0m'
            elif abs(maturity % 1) < tol:
                relativeDates[i] = '%.f'% maturity + 'y'
            elif abs(maturity % (1/12)) < tol:
                relativeDates[i] = '%.f'% (maturity*12) + 'm'
            elif np.logical_and(maturity > (25/DateUtils.days_per_year), maturity < (40/DateUtils.days_per_year)):
                relativeDates[i] = '%.f'% (maturity*12) + 'm'
            elif abs(maturity % (7/DateUtils.days_per_year)) < tol:
                relativeDates[i] = '%.f'% (maturity*52) + 'w'
            elif abs(maturity % round((1/tau),10)) < tol:
                relativeDates[i] = '%.f'% (maturity*tau) + 'd'
            else:
                relativeDates[i] = ''

        return list(relativeDates)

    @staticmethod
    def get_date_delta(from_date, to_date, year_frac=False):
        days = np.array((pd.to_datetime(to_date) -
                         pd.to_datetime(from_date))/np.timedelta64(1, 'D'), dtype=int)
        if year_frac:
            return days / DateUtils.days_per_year
        else:
            return days

    @staticmethod
    @njit(fastmath=True, cache=True)
    def shift_off_holidays(dates, holidays, term):

        res = np.empty(dates.shape, dtype=np.int32)
        for i in range(len(dates)):

            t = term[i]
            date = dates[i]

            if 'd' in t or 'w' in t:

                while date in holidays:
                   date += 1
                res.flat[i] = date
            else:

                while date in holidays:
                   date -= 1
                res.flat[i] = date
        return res

    @staticmethod
    def expiry_from_settlement_single_date(settlement_date, term, holidays=None):

        def to_ord(dates):
            return DateUtils.to_ordinals(dates)
        def f_ords(ordinals):
            return DateUtils.from_ordinals(ordinals)

        expiry = cDates.get_expiry_date(settlement_date.to_pydatetime(),
                                        np.asarray(term, str).item())

        if holidays is not None:
           expiry = f_ords(DateUtils.shift_off_holidays(to_ord(expiry),
                                                        term,
                                                        to_ord(holidays)))

        return pd.to_datetime(expiry)



    @staticmethod
    def expiry_from_settlement(settlement_dates, term, holidays=None):

        SD = pd.to_datetime(np.array(settlement_dates))
        if isinstance(SD, pd.Timestamp):
           return DateUtils.expiry_from_settlement_single_date(SD, term, holidays)

        if not DateUtils.is_iterable(term):
            term = [term] * len(settlement_dates)

        expiry = pd.to_datetime(cDates.shiftdates(SD.to_pydatetime(), list(term)))
        if holidays is not None:
           ex_ord = DateUtils.to_ordinals(expiry)
           hd_ord = DateUtils.to_ordinals(holidays)
           is_holiday = np.isin(ex_ord, hd_ord)
           ex_ord[is_holiday] = DateUtils.shift_off_holidays(ex_ord[is_holiday],
                                                             hd_ord,
                                                             np.array(term)[is_holiday])
           return DateUtils.from_ordinals(ex_ord)
        else:
           return expiry


    @staticmethod
    def get_daterange_frequency(date_range):

        N = len(date_range)
        if N < 2:
           return None

        minDateDiff = np.min(np.diff(date_range))
        if isinstance(minDateDiff, np.timedelta64):
            minDateDiff = int(minDateDiff / np.timedelta64(1,'D'))

        if minDateDiff == 1:
            if np.any(date_range.dayofweek.isin([5, 6])):
                return 'D'
            else:
                return 'B'
        elif 5 <= minDateDiff and minDateDiff <= 7:
            return 'W'
        elif 25 <= minDateDiff and minDateDiff <= 35:
            return 'M'
        elif 80 <= minDateDiff and minDateDiff <= 100:
            return 'Q'
        elif 350 <= minDateDiff and minDateDiff <= 370:
            return 'A'
        else:
            return ''

    @staticmethod
    def shift_dates_in_range(dates, reference, periods):

        @njit(cache=True)
        def shift_forward(ordinals, reference, periods):

            unique_ref = np.sort(np.unique(reference))

            T = len(unique_ref)
            N = len(ordinals)

            res = np.full(ordinals.shape, np.nan)
            for i in range(N):
                for t in range(T):
                    if (unique_ref[t] == ordinals[i]) and (t == T-1):
                        res[i] = np.max(unique_ref) + periods
                        break
                    elif (unique_ref[t] == ordinals[i]):
                        res[i] = unique_ref[t+periods]
                        break
            return res

        _shifted = shift_forward(DateUtils.to_ordinals(dates),
                                 DateUtils.to_ordinals(reference),
                                 periods)

        return DateUtils.from_ordinals(_shifted)



if __name__ == "__main__":

    dr1 = pd.date_range('31-Dec-2001', '31-Dec-2003', freq='A')
    dr2 = pd.date_range('31-Dec-2001', '31-Dec-2002', freq='W')
    dr3 = '31-Dec-2013'
    dr4 = datetime(year=2015, month=12, day=31)

    rng = DateUtils.merge([dr1, dr2, dr3, dr4])

    ords = DateUtils.to_ordinals(rng)
    dt = DateUtils.from_ordinals(ords)

    dr2 = pd.date_range('31-Dec-2019', '31-Dec-2021', freq='D')


    res = DateUtils.shift_dates_in_range(dr2[:-1], dr2, 1)






