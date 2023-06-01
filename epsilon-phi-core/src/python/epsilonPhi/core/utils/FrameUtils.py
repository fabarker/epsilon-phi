import pandas as pd
from operator import add
import numpy as np
import collections, re, six

class FrameUtils(object):
    pass

    @staticmethod
    def is_iterable(arg):
        return (isinstance(arg, collections.Iterable)
                and not isinstance(arg, six.string_types))

    @staticmethod
    def add_index_to_multi_index(multiIndex, new_index, index_name):
        return pd.MultiIndex.from_tuples(list(map(add, multiIndex, zip(new_index))), names=multiIndex.names + [index_name])

    @staticmethod
    def select_subset_level(df, level_name, values):
        if isinstance(df.columns, pd.MultiIndex)\
                and level_name in df.columns.names \
                and np.isin(df.columns.get_level_values(level_name), values).any():
           new_df = df.iloc[:, np.isin(df.columns.get_level_values(level_name), values)].dropna(how='all')
           new_df.columns = pd.MultiIndex.from_tuples(new_df.columns.values)
           new_df.columns.names = df.columns.names
           return new_df.copy()
        else:
            return df.copy()

    @staticmethod
    def drop_subset_level(df, level_name, values):

        if FrameUtils.is_iterable(values) is False:
            values = [values]

        if isinstance(df.columns, pd.MultiIndex)\
                and level_name in df.columns.names \
                and np.isin(df.columns.get_level_values(level_name), values).any():
           new_df = df.loc[:, ~df.columns.get_level_values('maturity').isin(values)]
           new_df.columns = pd.MultiIndex.from_tuples(new_df.columns.values)
           new_df.columns.names = df.columns.names
           return new_df.copy()
        else:
            return df.copy()


    @staticmethod
    def select_subset_levels(df, level_names, levels_values):
        pass