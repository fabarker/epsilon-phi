import pandas as pd
from operator import add
import numpy as np
import collections, re, six

class TimeSeriesUtils(object):
    pass

    @staticmethod
    def convert_timeseries_to_return_index(df):
        ts_type = df.columns

        if ts_type == 'TR':
            return TimeSeriesUtils.RI_from_TR(df)
        elif ts_type == 'RY':
            return TimeSeriesUtils.RI_from_RY(df)
        elif ts_type == 'IN':
            return TimeSeriesUtils.RI_from_IN(df)
        elif ts_type == 'YTW':
            return TimeSeriesUtils.RI_from_YTW(df)
        elif ts_type == 'PI':
            return TimeSeriesUtils.RI_from_PI(df)
        elif ts_type == 'RI':
            return TimeSeriesUtils.RI_from_RI(df)
        elif ts_type == 'X':
            return TimeSeriesUtils.RI_from_X(df)
        elif ts_type == 'IB':
            return TimeSeriesUtils.RI_from_IB(df)
        elif ts_type == 'IR':
            return TimeSeriesUtils.RI_from_IR(df)
        elif ts_type == 'RY':
            return TimeSeriesUtils.RI_from_IO(df)
        elif ts_type == 'IO':
            return TimeSeriesUtils.RI_from_RY(df)
        else:
            return df.copy()

    @staticmethod
    def RI_from_TR(df):
        pass

    @staticmethod
    def RI_from_RY(df):
        pass

    @staticmethod
    def RI_from_IN(df):
        pass

    @staticmethod
    def RI_from_YTW(df):
        pass

    @staticmethod
    def RI_from_PI(df):
        pass

    @staticmethod
    def RI_from_RI(df):
        pass

    @staticmethod
    def RI_from_X(df):
        pass

    @staticmethod
    def RI_from_IB(df):
        pass

    @staticmethod
    def RI_from_IR(df):
        pass

    @staticmethod
    def RI_from_IO(df):
        pass