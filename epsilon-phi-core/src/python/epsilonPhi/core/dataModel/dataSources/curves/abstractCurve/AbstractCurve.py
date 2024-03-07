import numpy as np

from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.utils.FrameUtils import FrameUtils
import pandas as pd
import warnings

warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')

class AbstractCurve(object):
    _cache = dict()

    def __init__(self, df):
        assert isinstance(df, pd.DataFrame)
        self._curve_df = None
        self.set_curve_dataframe(df)

    @property
    def tenors(self):
        return np.array(self._curve_df.columns)

    def set_curve_dataframe(self, df):
        _df = df.sort_index(axis=1)
        self._curve_df = self.interpolate_df(_df)

    def interpolate_df(self, df):
        return df.interpolate(method='linear',
                              fill_value="extrapolate",
                              limit_direction="both",
                              axis=1)

    def get_curve(self, dates=None, maturities=None):

        if maturities is None:
           _curve = self._curve_df.dropna(how='all', axis=0)
        else:
           _unique_mats = np.unique(maturities)
           _curve = FrameUtils.linear_interpolate_frame_rows(self._curve_df, _unique_mats)

        if dates is None:
           return _curve.copy()
        else:
           _unique_dates = np.unique(dates)
           return _curve.loc[_unique_dates]

    def get_stacked_curve(self, dates=None, maturity=None):
        unique_dates = np.unique(dates)
        unique_mats = np.unique(maturity)

        idx = FrameUtils.multiindex(dates, maturity.flatten())
        return self.get_curve(unique_dates, unique_mats).stack().loc[idx]