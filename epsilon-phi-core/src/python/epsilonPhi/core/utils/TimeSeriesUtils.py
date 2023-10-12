import numpy as np
import pandas as pd
from epsilonPhi.core.utils.DateUtils import DateUtils
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency

class TimeSeriesUtils(object):
    pass

    @staticmethod
    def convert_timeseries_to_return_index(df):

        ts_rtns = df._create_new_levels_object()

        columns = df.columns
        for col in columns:

            if 'TR' in col:
                ts_rtns = ts_rtns.concat(TimeSeriesUtils.RI_from_TR(df))
            elif 'RY' in col:
                return TimeSeriesUtils.RI_from_RY(df)
            elif 'YTW' in col:
                return TimeSeriesUtils.RI_from_YTW(df)
            elif 'PI' in col:
                ts_rtns = ts_rtns.concat(TimeSeriesUtils.RI_from_PI(df))
            elif 'RI' in col:
                ts_rtns = ts_rtns.concat(TimeSeriesUtils.RI_from_RI(df))
            elif 'IN' in col:
                ts_rtns = ts_rtns.concat(TimeSeriesUtils.RI_from_IN(df))
            elif 'IB' in col:
                ts_rtns = ts_rtns.concat(TimeSeriesUtils.RI_from_rate(df))
            elif 'IR' in col:
                ts_rtns = ts_rtns.concat(TimeSeriesUtils.RI_from_rate(df))
            elif 'IO' in col:
                ts_rtns = ts_rtns.concat(TimeSeriesUtils.RI_from_rate(df))
            else:
                raise ValueError('Error - col {} not mapped to transfromation'.format(col))

        ts_rtns.columns = pd.MultiIndex.from_tuples([(x, 'RI') for x in ts_rtns.columns.get_level_values(0)])
        return ts_rtns.get_period_ends(Frequency.DAILY)

    @staticmethod
    def RI_from_TR(df):
        if df.is_levels:
            df = df._create_new_returns_object(returns_type=df.type, data=df/100)
        return df.get_levels()

    @staticmethod
    def RI_from_RY(df):
        pass

    @staticmethod
    def RI_from_YTW(df):
        pass

    @staticmethod
    def RI_from_IN(df):
        return df._create_new_levels_object(data=df+100)

    @staticmethod
    def RI_from_PI(df):
        return df._create_new_levels_object(data=df)

    @staticmethod
    def RI_from_RI(df):
        return df._create_new_levels_object(data=df)

    @staticmethod
    def RI_from_rate(df):

        if df.is_levels:
           df = df._create_new_returns_object(returns_type=df.type, data=df)
        df_ = -1+np.power((1+df.get_period_ends(frequency=Frequency.DAILY)/ 100), 1/DateUtils.days_per_year)
        return df_.get_levels().get_period_ends(Frequency.DAILY)


