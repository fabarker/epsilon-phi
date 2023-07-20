from enum import Enum

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