import pandas as pd
from operator import add
from epsilonPhi.core.utils.DateUtils import DateUtils
import numpy as np
import collections, re, six
from epsilonPhi.core.utils.MathUtils import linear_interpolate, flat_forward_interpolation, flat_forward_interp, forward_flat_interpolation_N
import warnings

# Suppress FutureWarning messages
warnings.filterwarnings("ignore", category=FutureWarning)

class FrameUtils(object):
    pass

    @staticmethod
    def rowise_linear_interpolate_on_groups(df, x_var, x_lev, group):
        tmp = (df.groupby(axis=1, level=group).
                apply(lambda x: FrameUtils.linear_interpolate_frame_rows(x, x_var, x_lev)))
        tmp.columns = tmp.columns.rename({None: x_lev}).reorder_levels(df.columns.names)
        return tmp.copy()

    @staticmethod
    def rowise_flat_forward_interpolation_on_groups(df, x_var, x_lev, group):
        tmp = (df.groupby(axis=1, level=group).
                apply(lambda x: FrameUtils.flat_forward_interpolate_frame_rows(x, x_var, x_lev)))
        tmp.columns = tmp.columns.rename({None:x_lev}).reorder_levels(df.columns.names)
        return tmp.copy()

    @staticmethod
    def flat_forward_interpolate_frame_rows(x, x_var, x_lev=False):

        if np.isscalar(x_var):
           x_var = np.array([x_var])

        if x_lev is False:
            x = x.sort_index(axis=1)
            x_lev = np.array(x.columns)
        else:
            x = x.sort_index(axis=1, level=x_lev)
            x_lev = np.array(x.columns.get_level_values(x_lev))

        _type = np.result_type(np.float64, np.float64)
        res = forward_flat_interpolation_N(x_var,
                                          x_lev,
                                          x.values,
                                          _type)
        return pd.DataFrame(res, index=x.index, columns=x_var)

    @staticmethod
    def linear_interpolate_frame_rows(x, x_var, x_lev=False):

        if np.isscalar(x_var):
           x_var = np.array([x_var])

        if x_lev is False:
            x = x.sort_index(axis=1)
            x_lev = np.array(x.columns)
        else:
            x = x.sort_index(axis=1, level=x_lev)
            x_lev = np.array(x.columns.get_level_values(x_lev))


        return linear_interpolate(x_lev,
                                  x,
                                  x_var).set_axis(x_var, axis=1)

    @staticmethod
    def is_iterable(arg):
        return (isinstance(arg, collections.Iterable)
                and not isinstance(arg, six.string_types))

    @staticmethod
    def add_level(multiIndex, new_index, index_name):

        if not DateUtils.is_iterable(new_index):
           new_index = [new_index]

        if len(new_index) == 1:
           new_index = new_index * len(multiIndex)

        assert len(new_index) == len(multiIndex), 'Error in multi-index'
        return pd.MultiIndex.from_tuples([(*mi, ni) for mi, ni in zip(multiIndex, new_index)],
                                         names=multiIndex.names + [index_name])

    @staticmethod
    def select_subset_level(df, level_name, values, drop_nans=True):
        if isinstance(df.columns, pd.MultiIndex)\
                and level_name in df.columns.names \
                and np.isin(df.columns.get_level_values(level_name), values).any():

           if drop_nans:
              new_df = df.iloc[:, np.isin(df.columns.get_level_values(level_name), values)].dropna(how='all')
           else:
               new_df = df.iloc[:, np.isin(df.columns.get_level_values(level_name), values)]

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
           new_df = df.loc[:, ~df.columns.get_level_values(level_name).isin(values)]
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
           df.columns = FrameUtils.add_level(df.columns, level_values, level_name)
           df.columns = df.columns.reorder_levels(level_names)
        else:
           df.columns = FrameUtils.add_level(df.columns, level_values, level_name)
        return df.copy()

    @staticmethod
    def sort_by_level(df, level_name):
        idx = np.argsort(df.columns.get_level_values(level_name))
        return df.get(df.columns[idx])

    @staticmethod
    def multiindex(*args):
        return pd.MultiIndex.from_tuples(list(zip(*args)))

    @staticmethod
    def flatten_columns_or_name(df):
        if isinstance(df, pd.DataFrame):
            if isinstance(df.columns, pd.MultiIndex):
                # Take first level of multi-level columns
                df.columns = df.columns.get_level_values(0)
            elif isinstance(df.columns[0], tuple):
                # Flatten tuple columns by taking the first element
                df.columns = [col[0] for col in df.columns]

        elif isinstance(df, pd.Series):
            if isinstance(df.name, tuple):
                df.name = df.name[0]

        return df

