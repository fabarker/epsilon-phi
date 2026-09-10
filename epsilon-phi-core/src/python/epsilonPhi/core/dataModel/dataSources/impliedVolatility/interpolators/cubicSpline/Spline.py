import numpy as np
import pandas as pd
from epsilonPhi.core.utils.SolverUtils import brentsmethod, brent
from epsilonPhi.core.utils.FrameUtils import FrameUtils
from epsilonPhi.core.lib.curve_fitting.cubic_spline.cubic_spline import cubic_spline as cubicspline
from epsilonPhi.core.lib.curve_fitting.cubic_spline.cubic_spline import calc_spline_params, piece_wise_spline, fit_cubic_spline, eval_cubic_spline, fit_spline
from epsilonPhi.core.dataModel.dataSources.curves.abstractCurve.AbstractCurve import AbstractCurve
from epsilonPhi.core.utils.NumpyUtils import NumpyUtils as npu
from scipy.optimize import brentq
from epsilonPhi.core.utils.FrameUtils import FrameUtils
from numba import jit
from tqdm import tqdm
from epsilonPhi.core.utils.OptionUtils import *
from abc import abstractmethod
from numba.typed import Dict
from epsilonPhi.core.dataModel.enums.ImpliedVolatility import *
from epsilonPhi.core.utils.MathUtils import cubic_spline, vanna_volga_1d
from epsilonPhi.core.dataModel.enums.ImpliedVolatility import MaturityType
from typing import Optional, Union
from epsilonPhi.core.dataModel.dataSources.impliedVolatility.interpolators.abstractInterpolator.AbstractInterpolator import AbstractInterpolator


class Spline(AbstractInterpolator):
    def __init__(self,
                 market_df: pd.DataFrame,
                 **kwargs):

        super(Spline, self).__init__(market_df=market_df,
                                     **kwargs)
        self.fit_surface()

    def find_date_idx(self, dates, opt_type):
        return DateUtils.find_first_date_loc(self._fit_df[opt_type].index, dates)

    def fit_surface(self):

        self._fit = {}
        self._nan_fits = {}
        self._fit_df = {}
        self._fit_maturities = {}
        for i in [-1, 1]:

            if i == -1:
               col = 'put_delta'
            else:
               col = 'call_delta'

            mkt_data = self._market_df.set_index(['date', 't', 'relative_strike']).get(['mid', col])
            idx = ~np.isnan(mkt_data.index.get_level_values(2))

            XY = mkt_data.iloc[idx].unstack(level=2)

            Y = XY.get('mid').sort_index(level=1)
            Y = Y[(~Y.isna()).sum(axis=1) > 3]

            X = XY.get(col).sort_index(level=1).replace(0, np.nan)
            X = X[(~X.isna()).sum(axis=1) > 3]

            common_idx = X.index.intersection(Y.index)
            Y_prime = Y.loc[common_idx]
            X_prime = X.loc[common_idx].get(Y_prime.columns)

            fy = np.asarray(Y_prime.values, dtype=np.float64)
            fx = np.asarray(X_prime.values, dtype=np.float64)

            tmp = list()
            for t in range(fy.shape[0]):
                tmp.extend([fit_cubic_spline(fx[t], fy[t])])

            _fit_df = pd.DataFrame(tmp, index=Y_prime.index).unstack(level=1).sort_index()

            self._fit_maturities[i] = np.unique(_fit_df.columns.get_level_values(1)).astype(np.float64)
            K = len(self._fit_maturities[i])
            N = len(np.unique(_fit_df.columns.get_level_values(0)))

            self._fit[i] = _fit_df.values.reshape((_fit_df.shape[0], N, K)).swapaxes(2, 1)
            self._nan_fits[i] = _fit_df.isna().values.reshape((_fit_df.shape[0], N, K)).swapaxes(2, 1)
            self._fit_df[i] = _fit_df.swaplevel(1, 0, axis=1).sort_index(level=0, axis=1)

    def solve_for_strike(self, dates, expiries, strikes, option_type=-1):

        s =  np.asarray(self.get_s(dates)).flat
        rf = np.asarray(self.get_rf(dates, expiries, True))
        rd = np.asarray(self.get_rd(dates, expiries, True))

        def fast_fit(s, r, q, fy, ft, m, x, id, _nans):

            sigs = np.full(len(id), np.nan)
            for k in range(len(id)):
                i = id[k]
                if i > 0:
                    sigs.flat[k] = solve_for_sig_strike_spline(s[i],
                                                               r[i],
                                                               q[i],
                                                               fy[i],
                                                               ft,
                                                               m[k],
                                                               x[k],
                                                               _nans[i])
            return sigs

        ids = self.find_date_idx(dates, option_type)
        t = np.unique(self._fit_df[option_type].columns.get_level_values(0))

        rd_ = self.get_rd(self._fit_df[option_type].index, t, is_stacked=False).values
        rf_ = self.get_rf(self._fit_df[option_type].index, t, is_stacked=False).values
        s_ = self.get_s(self._fit_df[option_type].index).values.flatten()

        sigs = fast_fit(s_, rd_, rf_,  self._fit[option_type], t, expiries, strikes, ids, self._nan_fits[option_type])
        sigs[sigs <= 0] = np.nan

        df = pd.DataFrame((sigs, expiries, strikes), index=['mid', 't', 'k']).T
        df['x'] = nb_delta(s, expiries, strikes, rd, rf, sigs, 2, option_type)
        df['date'] = dates
        return df

    def solve_for_delta(self, dates, expiries, deltas):

        assert np.size(np.unique(np.sign(deltas))) == 1, 'Error - cannot mix put and call deltas'
        opt_type = int(np.unique(np.sign(deltas)))

        s = np.asarray(self.get_s(dates)).flat
        rf = np.asarray(self.get_rf(dates, expiries, True))
        rd = np.asarray(self.get_rd(dates, expiries, True))

        def fast_fit(fy, ft, m, x, id, _nans):

            sigs = np.full(len(id), np.nan)
            for k in range(len(id)):
                i = id[k]
                if i > 0:
                    sigs.flat[k] = solve_for_sig_delta_spline(fy[i],
                                                              ft,
                                                              m[k],
                                                              x[k],
                                                              _nans[i])
            return sigs

        ids = self.find_date_idx(dates, opt_type)
        t = np.unique(self._fit_df[opt_type].columns.get_level_values(0))
        sigs = fast_fit(self._fit[opt_type], t, expiries, deltas, ids, self._nan_fits[opt_type])
        sigs[sigs <= 0] = np.nan

        df = pd.DataFrame((sigs, expiries, deltas), index=['mid', 't', 'x']).T
        df['k'] = nb_strike(s, expiries, rd, rf, np.sign(deltas).astype(np.int32), deltas, 2, sigs)
        df['date'] = dates
        return df


if __name__ == "__main__":

        from epsilonPhi.core.dataModel.dataSources.impliedVolatility.Models import *
        from epsilonPhi.core.dataModel.dataSources.impliedVolatility.VolSurfaceMgr import VolSurfaceMgr

        underlier = 'GBPUSD'
        vsm = VolSurfaceMgr(underlier,
                             pricing_location='NYC',
                             strike_reference=StrikeReference.DELTA)


        ### Get ivols
        ivols = vsm.get_ivols()

        s = vsm.get_spot_prices()
        rd = vsm.get_rate_curve()
        rf = vsm.get_funding_curve()

        self = Spline(ivols)
        self.set_deposit_rate_curve(rd)
        self.set_funding_rate_curve(rf)

        sigs = ivols[ivols.t == ivols.t.unique()[1]]
        D = np.unique(sigs.index)[-400:]

        k = sigs.loc[D].get('k') * 0.9999
        d = pd.to_datetime(k.index)
        t = ivols.t.unique()[2]

        fxivols = self.get_ivols(d[-1],
                                 strike_reference=StrikeReference.DELTA,
                                 relative_strike=np.arange(-0.9, -0.1, 0.01),
                                 maturity_type=MaturityType.YEARFRAC,
                                 maturities=[7 / 365, 1 / 12, 1])

        m = fxivols.set_index(['t','x']).get('mid').unstack(level=0)

