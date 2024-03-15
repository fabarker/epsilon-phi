import numpy as np
from epsilonPhi.core.utils.MathUtils import interp_N
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
        return df.copy()

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

        # Get the curve on the dates that we need observations
        _ivols = self._curve_df.loc[np.unique(dates)]
        # Get the unique maturities
        _mats = np.unique(maturity)

        # Build arrays for interpolator
        fp = _ivols.values
        xp = np.array(_ivols.columns)

        # Interpolate
        _type = np.result_type(np.float64, np.float64)
        res = interp_N(_mats, xp, fp, _type)

        # Extract
        dates_ = _ivols.index.repeat(len(_mats))
        mats_ = _mats.reshape(1, -1).repeat(len(dates), 0).flatten()
        vals_ = res.flatten()

        rf = pd.DataFrame(vals_, index=zip(dates_, mats_))
        return rf.loc[zip(dates, maturity)]