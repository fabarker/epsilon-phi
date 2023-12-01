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
            return pd.DataFrame()

    @staticmethod
    def vectorize(df, column_name=None):
        df_ = pd.melt(df.reset_index(), id_vars=df.index.name if df.index.name else 'index')
        cols = np.setdiff1d(df_.columns, 'value')
        df_.index = list(map(tuple, df_[cols].to_numpy()))
        if column_name is None:
            return df_.get('value').to_frame()
        else:
            return df_.get('value').to_frame(column_name)

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
    def set_levels(df, level_values, level_name):

        level_names = df.columns.names
        if level_name in level_names:
           df.columns = df.columns.droplevel(level_name)
           df.columns = FrameUtils.add_index_to_multi_index(df.columns, level_values, level_name)
           df.columns = df.columns.reorder_levels(level_names)
        else:
           df.columns = FrameUtils.add_index_to_multi_index(df.columns, level_values, level_name)
        return df.copy()

    @staticmethod
    def sort_by_level(df, level_name):
        idx = np.argsort(df.columns.get_level_values(level_name))
        return df.get(df.columns[idx])