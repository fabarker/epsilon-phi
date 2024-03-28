import numpy as np
from epsilonPhi.core.utils.MathUtils import interp_N, interp_N_vect
from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.utils.FrameUtils import FrameUtils
from epsilonPhi.core.cpp.fastfind.find_1st import *
import pandas as pd
import warnings

warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')

class AbstractCurve(object):
    _cache = dict()

    def __init__(self, df):
        assert isinstance(df, pd.DataFrame)
        self._curve_df = None
        self._ordinals = None
        self.set_curve_dataframe(df)

    @property
    def tenors(self):
        return np.array(self._curve_df.columns)
    @property
    def ordinals(self):
        return self._ordinals

    def set_curve_dataframe(self, df):
        _df = df.sort_index(axis=1)
        self._curve_df = self.interpolate_df(_df, np.array(_df.columns)).sort_index(axis=0)
        self._ordinals = DateUtils.to_ordinals(self._curve_df.index)

    def interpolate_df(self, df, xdense):
        _type = np.result_type(np.float64, np.float64)
        _cols = np.array(df.columns)
        return pd.DataFrame(interp_N(xdense, _cols, df.values, _type),
                            index=df.index, columns=xdense).dropna(how='all', axis=0)

    def get_curve(self, dates=None, maturities=None, is_stacked=False):

        if is_stacked:
            return self.get_stacked_curve(dates, maturities)

        if maturities is None:
           _curve = self._curve_df.dropna(how='all', axis=0)
        else:
           _unique_mats = np.unique(maturities)
           _curve = self.interpolate_df(self._curve_df, _unique_mats)
           _curve = _curve[maturities]

        if dates is None:
           return _curve.copy()
        else:
           _unique_dates = np.unique(dates)
           return _curve.loc[_unique_dates]


    def get_stacked_curve(self, dates=None, maturity=None):

        dates = np.array(dates)
        maturity = np.array(maturity)

        D_M = DateUtils.to_ordinals(dates)
        _idx = find_1st(self.ordinals.reshape(-1, 1),
                        D_M.reshape(-1, 1))


        # Get the curve on the dates that we need observations
        _ivols = self._curve_df.iloc[_idx, :]

        # Build arrays for interpolator
        fp = _ivols.values
        xp = np.array(_ivols.columns)

        mat = maturity.reshape(-1, 1)
        # Interpolate
        _type = np.result_type(np.float64, np.float64)
        res = interp_N_vect(mat, xp, fp, _type)
        return pd.Series(res.flatten(), index=[dates, maturity.flatten()])