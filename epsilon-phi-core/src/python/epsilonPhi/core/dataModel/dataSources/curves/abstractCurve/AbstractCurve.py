from epsilonPhi.core.dataModel.alchemist.DataModel import *
import pandas as pd
import warnings

warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')

class AbstractCurve(object):
    _cache = dict()

    def __init__(self, df):

        assert isinstance(df, pd.DataFrame)
        self._curve_df = df.copy().sort_index(axis=1)

    def get_curve(self, maturities=None):

        rates = self._curve_df.copy()
        if maturities is None:
           maturities = rates.columns

        _mats = np.setdiff1d(maturities, rates.columns)
        rates[_mats] = np.nan
        rates = rates.sort_index(axis=1, level=0)
        _rates = rates.interpolate(method='linear', fill_value="extrapolate", limit_direction="both", axis=1)
        return _rates[maturities].dropna(how='all', axis=0)