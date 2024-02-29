from epsilonPhi.core.dataModel.alchemist.DataModel import *
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

    def get_curve(self, maturities=None):

        if maturities is None:
           return self._curve_df.dropna(how='all', axis=0)

        _mats = np.setdiff1d(maturities, self.tenors)
        if len(_mats) > 0:
            self._curve_df[_mats] = np.nan
            self._curve_df = self._curve_df.interpolate(method='linear',
                                                        fill_value="extrapolate",
                                                        limit_direction="both",
                                                        axis=1)
        return self._curve_df[maturities].dropna(how='all', axis=0)