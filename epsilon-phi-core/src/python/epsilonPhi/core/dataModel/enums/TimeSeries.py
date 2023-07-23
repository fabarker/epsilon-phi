from enum import Enum

class TimeSeriesType(Enum):

    LEVELS = 0
    GROWTH = 1
    RETURNS = 2

class ReturnsType(Enum):

    SIMPLE = 'simple'
    LOG = 'log'
    DIFFERENCE = 'difference'
