from enum import Enum
import numpy as np
import pandas as pd
from pandas import offsets

class Frequency(Enum):

    SECONDLY = 'S'
    HOURLY = 'H'
    BUSINESS_DAILY = 'B'
    DAILY = 'D'
    WEEKLY = 'W'
    MONTHLY = 'M'
    BUSINESS_MONTHLY = 'BM'
    QUARTERLY = 'Q'
    BUSINESS_QUARTERLY = 'BQ'
    YEARLY = 'Y'
    BUSINESS_YEARLY = 'BY'
    NOT_DEFINED = ''

    def yearfrac(self):
        from epsilonPhi.core.utils.DateUtils import DateUtils
        return DateUtils.Rdate_to_mat('1' + self.value)

    def rdate(self):
        from epsilonPhi.core.utils.DateUtils import DateUtils
        return DateUtils.mat_to_Rdate(DateUtils.Rdate_to_mat('1' + self.value))

    def obs_per_year(self):
        from epsilonPhi.core.utils.DateUtils import DateUtils

        if self.value in ['D']:
           return DateUtils.days_per_year
        elif self.value in ['B']:
           return 252
        elif self.value in ['W']:
           return 52
        elif self.value in ['M','BM']:
            return 12
        elif self.value in ['Q','BQ',]:
            return 4
        elif self.value in ['Y','BY']:
            return 1
        else:
            return np.nan

    def get_period_ends(self, dates):

        if self in [Frequency.DAILY, Frequency.BUSINESS_DAILY]:
           return dates
        if self in [Frequency.WEEKLY, Frequency.WEEKLY.value]:
           return dates[dates.weekday == dates.weekday[0]]
        else:
            if self in [Frequency.BUSINESS_YEARLY,
                        Frequency.BUSINESS_MONTHLY,
                        Frequency.BUSINESS_QUARTERLY]:

                sat_sun = (dates.weekday == 5) | (dates.weekday == 6)
                dates = dates[~sat_sun]

            yearMonths = dates.year * 100 + dates.month
            EOM_locs = np.append(np.diff(yearMonths) != 0, True)
            monthEnds = dates[EOM_locs]
            if (monthEnds[-1] + pd.tseries.offsets.MonthEnd(0)) - monthEnds[-1] > pd.to_timedelta(1, 'D'):
                if monthEnds[-1].weekday() in [5, 6]:
                    monthEnds = monthEnds[:-1]

            if self in [Frequency.MONTHLY, Frequency.BUSINESS_MONTHLY]:
               return monthEnds
            elif self in [Frequency.BUSINESS_QUARTERLY, Frequency.QUARTERLY]:
               return monthEnds[np.mod(monthEnds.month, 3) == 0]
            elif self in [Frequency.YEARLY, Frequency.BUSINESS_YEARLY]:
               return monthEnds[np.mod(monthEnds.month, 12) == 0]





    @staticmethod
    def get_frequency(frequency_str: str):
        assert isinstance(frequency_str, str), 'Error - must be a string'
        phi = frequency_str.lower()
        if phi == 's':
            return Frequency.SECONDLY
        if phi == 'h':
            return Frequency.HOURLY
        if phi == 'd':
            return Frequency.DAILY
        if phi == 'w':
            return Frequency.WEEKLY
        if phi == 'm':
            return Frequency.MONTHLY
        if phi == 'bm':
            return Frequency.BUSINESS_MONTHLY
        if phi == 'q':
            return Frequency.QUARTERLY
        if phi == 'bq':
            return Frequency.BUSINESS_QUARTERLY
        if phi in ['y','a']:
            return Frequency.YEARLY
        if phi in ['by','ba']:
            return Frequency.BUSINESS_YEARLY
        else:
            return Frequency.NOT_DEFINED

if __name__ == "__main__":

    import pandas as pd

    freq = Frequency.WEEKLY
    freq.rdate()


