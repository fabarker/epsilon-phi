from enum import Enum
import numpy as np
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
    freq = Frequency.MONTHLY