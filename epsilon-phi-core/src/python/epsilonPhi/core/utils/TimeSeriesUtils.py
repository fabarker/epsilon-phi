import numpy as np
import pandas as pd
from epsilonPhi.core.utils.DateUtils import DateUtils
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.lib.bonds.BondUtils import BondUtils
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType

class TimeSeriesUtils(object):
    pass

    @staticmethod
    def convert_timeseries_to_return_index(df):

        ts_rtns = df._create_new_levels_object()

        columns = df.columns
        for col in np.unique(columns):
            df_col = df.get([col])

            if 'TR' in col:
                ts_rtns = ts_rtns.concat(TimeSeriesUtils.RI_from_TR(df_col))
            elif 'RY' in col:
                ts_rtns = ts_rtns.concat(BondUtils.convertYield(df_col, 10))
            elif 'YTW' in col:
                pass
            elif 'RI' in col:
                ts_rtns = ts_rtns.concat(TimeSeriesUtils.RI_from_RI(df_col))
            elif 'IN' in col:
                ts_rtns = ts_rtns.concat(TimeSeriesUtils.RI_from_IN(df_col))
            elif 'IB' in col:
                ts_rtns = ts_rtns.concat(TimeSeriesUtils.RI_from_rate(df_col))
            elif 'IR' in col:
                ts_rtns = ts_rtns.concat(TimeSeriesUtils.RI_from_rate(df_col))
            elif 'IO' in col:
                ts_rtns = ts_rtns.concat(TimeSeriesUtils.RI_from_rate(df_col))
            else:
                pass

        #cols = pd.MultiIndex.from_tuples([(x[0],'RI') for x in ts_rtns.columns])
        #atts = ts_rtns.attributes.copy()

        #atts.columns = cols
        #ts_rtns.columns = cols
        #ts_rtns.set_attributes(atts)
        return ts_rtns.get_period_ends(Frequency.DAILY)

    @staticmethod
    def RI_from_TR(df):
        if df.is_levels:
            df = df._create_new_returns_object(returns_type=df.type,
                                               data=df/100)
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
           df = df._create_new_returns_object(returns_type=df.type, data=df, attributes=df.attributes)
        df_ = -1+np.power((1+df.get_period_ends(frequency=Frequency.DAILY) / 100), 1/DateUtils.days_per_year)
        return df_.get_levels().get_period_ends(Frequency.DAILY)


