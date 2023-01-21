from enum import Enum

class Frequency(Enum):
    REAL_TIME = 1
    HOURLY = 2
    FEW_HOURS = 3
    DAILY = 4
    WEEKLY = 5
    MONTHLY = 6
    QUARTERLY = 7
    SEMI_ANNUALLY = 8
    YEARLY = 9
    NOT_DEFINED = 99

    @staticmethod
    def get_frequency(frequency_str: str):
        assert isinstance(frequency_str, str), 'Error - must be a string'
        phi = frequency_str.lower()
        if phi == 'r':
            return Frequency.REAL_TIME
        if phi == 'h':
            return Frequency.HOURLY
        if phi == 'f':
            return Frequency.FEW_HOURS
        if phi == 'd':
            return Frequency.DAILY
        if phi == 'w':
            return Frequency.WEEKLY
        if phi == 'm':
            return Frequency.MONTHLY
        if phi == 'q':
            return Frequency.QUARTERLY
        if phi in ['y','a']:
            return Frequency.YEARLY
        else:
            return Frequency.NOT_DEFINED

if __name__ == "__main__":
    freq = Frequency.MONTHLY