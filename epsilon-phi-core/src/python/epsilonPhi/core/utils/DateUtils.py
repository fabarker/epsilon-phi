import collections, re, six
import pandas as pd
import numpy as np
from datetime import datetime
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

class DateUtils(object):

    days_per_year = (365*3+366) * (1/4)
    ERROR_TOLERANCE = 1e-4

    @staticmethod
    def is_iterable(arg):
        return (isinstance(arg, collections.Iterable)
                and not isinstance(arg, six.string_types))

    @staticmethod
    def find_date_locs(arr: np.array) -> np.array:
        mask = np.array([[isinstance(x, datetime) for x in row] for row in arr])
        return np.transpose(np.where(mask))
    @staticmethod
    def is_date(arr: np.array) -> np.array:
        return np.array([isinstance(x, datetime) for x in arr])

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
            matStrs = [Rdate.lower()]
        else:
            matStrs = [x.lower() for x in Rdate]

        T = len(matStrs)
        nYears = np.array(["Nan"] * T, dtype=float)
        for i in range(T):
            matStr = matStrs[i]

            if re.search("on", matStr):
                matStr = '1d'
            if re.search("tn", matStr):
                matStr = '2d'
            if re.search("sw", matStr):
                matStr = '1w'

            if re.search("d", matStr):
                nPeriods = DateUtils.days_per_year
                strg = "d"
            elif re.search("w", matStr):
                nPeriods = DateUtils.days_per_year / 7
                strg = "w"
            elif re.search("m", matStr):
                nPeriods = 12
                strg = "m"
            elif re.search("q", matStr):
                nPeriods = 4
                strg = "q"
            elif re.search("y", matStr):
                nPeriods = 1
                strg = "y"
            else:
                raise Exception("Error - relative date {} not recongnized".format(matStr))

            loc = matStr.find(strg)
            nYears[i] = round(float(matStr[0:loc]) / nPeriods, 10)

        if len(nYears) > 1:
            return nYears
        else:
            return float(nYears)

    @staticmethod
    def mat_to_Rdate(nYears: float) -> list:

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
        if year_frac:
            return np.round((pd.to_datetime(to_date) - pd.to_datetime(from_date))/np.timedelta64(1, 'D') / DateUtils.days_per_year, 10)
        else:
            return (pd.to_datetime(to_date) - pd.to_datetime(from_date))/np.timedelta64(1, 'D')

    @staticmethod
    def get_daterange_frequency(date_range):

        N = len(date_range)
        if N < 2:
           return None

        minDateDiff = np.min(np.diff(date_range))
        if minDateDiff == 1:
            return 'D'
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




