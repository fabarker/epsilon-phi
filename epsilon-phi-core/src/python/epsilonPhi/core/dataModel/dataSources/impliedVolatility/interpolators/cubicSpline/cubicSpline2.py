import numpy as np
import pandas as pd
from epsilonPhi.core.utils.FrameUtils import FrameUtils
from epsilonPhi.core.lib.curve_fitting.cubic_spline.cubic_spline import cubic_spline as cubicspline
from epsilonPhi.core.dataModel.dataSources.curves.abstractCurve.AbstractCurve import AbstractCurve
from epsilonPhi.core.utils.NumpyUtils import NumpyUtils as npu
from numba import jit
from tqdm import tqdm
from epsilonPhi.core.utils.OptionUtils import *
from epsilonPhi.core.dataModel.enums.ImpliedVolatility import *
from epsilonPhi.core.utils.MathUtils import cubic_spline


class CubicSpline(object):
    def __init__(self, ivols, s, rd, rf):

        # Set ivols panel..
        self.set_ivols(ivols)

        # Sset spot rates
        self.set_spot_prices(s)

        # Set interest and funding rate curves
        self.set_risk_free_rate_curve(rd)
        self.set_funding_rate_curve(rf)
        self.set_forward_prices()
        self._x = None


    def set_ivols(self, ivols):

        _ivols = ivols.reset_index(drop=False).drop(labels=['relative_strike'], axis=1)
        _loc = [x not in ['date', 'mid', 't', 'k', 'relative_strike'] for x in _ivols.columns]

        # Extract the strike reference and observations
        self._strike_reference = _ivols.columns[_loc][0]
        _ivols = _ivols.rename(columns={self._strike_reference: 'x'})

        vols = _ivols.set_index(['date', 't', 'x'], drop=True)
        self._ivols = vols[~vols.index.duplicated(keep='first')].sort_index(level='date')
        self._ivols = self._ivols.unstack(level=['t', 'x']).sort_index(axis=1)

    def set_spot_prices(self, df):
        self._spot = df.loc[self.dates].to_frame()

    def set_risk_free_rate_curve(self, rd):
        self._rate_curve = rd
        self.set_risk_free_rate()

    def set_funding_rate_curve(self, rf):
        self._funding_curve = rf
        self.set_funding_rate()

    def set_forward_prices(self):
        self._f = self._spot.values * np.exp(-1 * self._rf * self.t) / np.exp(-1 * self._rd * self.t)
        self._forward_curve = AbstractCurve(self._f)

    def set_risk_free_rate(self):
        self._rd = self._rate_curve.get_curve(self.dates, self.t[0,:])

    def set_funding_rate(self):
        self._rf = self._funding_curve.get_curve(self.dates, self.t[0,:])

    @property
    def k_ref(self):
        return self._strike_reference
    @property
    def T(self):
        return len(self.dates)
    @property
    def dates(self):
        return self.ivols.index
    @property
    def ivols(self):
        return self._ivols.copy()
    @property
    def sig(self):
        return self.ivols.get('mid')
    @property
    def x(self):
        return self.strikes.repeat(self.T, axis=0)
    @property
    def strikes(self):
        return self.ivols.columns.get_level_values(self._strike_reference).values[None, :]
    @property
    def mat(self):
        return self.ivols.columns.get_level_values('t').values[None, :]
    @property
    def t(self):
        return self.mat.repeat(self.T, axis=0)
    @property
    def k(self):
        return self.ivols.get('k')
    @property
    def rf(self):
        return self._rf.values
    @property
    def rd(self):
        return self._rd.values
    @property
    def s(self):
        return self._spot.values
    @property
    def f(self):
        return self._f.values

    def get_f(self, pricing_dates, maturities):
        return self._forward_curve.get_curve(pricing_dates, maturities)
    def get_rd(self, pricing_dates, maturities, is_stacked=False):
        return self._rate_curve.get_curve(pricing_dates, maturities, is_stacked)
    def get_rf(self, pricing_dates, maturities, is_stacked=False):
        return self._funding_curve.get_curve(pricing_dates, maturities, is_stacked)
    def get_s(self, pricing_dates):
        return self._spot.loc[pricing_dates]

    def get_deltas(self, sig):

        _t = sig.get('t').values.flatten().astype(np.float64)
        _s = self.get_s(sig.get('date')).values.flatten().astype(np.float64)
        _v = sig.get('mid').values.astype(np.float64)
        _k = sig.get('k').values.astype(np.float64)

        rd = self.get_rd(sig.get('date'), _t, True).values.astype(np.float64)
        rf = self.get_rd(sig.get('date'), _t, True).values.astype(np.float64)

        res = nb_delta(_s, _t, _k, rd, rf, _v, 2, -1)
        return pd.DataFrame(res, columns=['x']).replace(0, np.nan)

    def get_ivols(self, pricing_dates, strike_reference, relative_strike, maturities):

        mats = np.round(maturities, 10)
        if pricing_dates is None:
            _p, _t, _k = npu.flat_meshgrid(self.dates, mats, relative_strike)
        elif pricing_dates.shape != mats.shape:
            _p, _t, _k = npu.flat_meshgrid(pricing_dates, mats, relative_strike)
        else:
            assert pricing_dates.shape == mats.shape, 'Error - dimension mis-match'
            assert pricing_dates.shape == relative_strike.shape, 'Error - dimension mis-match'
            _p, _t, _k = pricing_dates, maturities, relative_strike

        return self.load_ivols(_p, _t, _k, strike_reference)

    def load_ivols(self, pricing_dates, maturities, relative_strike, strike_reference):

        # Interpolate along the time direction - Flat Forward Interpolation.
        _sigs = self.interpolate_maturities(maturities)
        _sigs['k'] = self.get_strikes(_sigs, self._strike_reference)

        # Compute the surface values for the strike reference we care about
        if strike_reference in [StrikeReference.DELTA, StrikeReference.DELTA.value]:
            _sigs['xk'] = self.get_deltas(_sigs)
        elif strike_reference in [StrikeReference.STRIKE_PRICE, StrikeReference.STRIKE_PRICE.value]:
            _sigs['xk'] = _sigs.get('k')
        else:
            raise ValueError('Error - strike reference {} not supported'.format(strike_reference))

        # Stack vols
        vk = _sigs.set_index(['date','t']).pivot(columns='x').reindex(zip(pricing_dates, maturities))
        v = vk.get('mid').sort_index(axis=1)
        k = vk.get('xk').sort_index(axis=1)

        _ivols = self.interpolate(v, k, relative_strike)
        _ivols['k'] = self.get_strikes(_ivols, strike_reference)
        return _ivols

    def get_strikes(self, ivols, strike_reference):

        # Compute the surface values for the strike reference we care about
        if strike_reference in [StrikeReference.DELTA, StrikeReference.DELTA.value]:
            return self.get_strikes_from_delta(ivols)
        elif strike_reference in [StrikeReference.STRIKE_PRICE, StrikeReference.STRIKE_PRICE.value]:
            return ivols.get('x')
        else:
            raise ValueError('Error - strike reference {} not supported'.format(strike_reference))

    def get_strikes_from_delta(self, df_):

        _v = df_.get('mid').values.astype(np.float64)
        _x = df_.get('x').values.astype(np.float64)
        _s = self.get_s(df_.get('date')).values.ravel().astype(np.float64)
        _t = df_.get('t').values.astype(np.float64)

        rd = self.get_rd(df_.get('date'), _t, True).values.astype(np.float64)
        rf = self.get_rf(df_.get('date'), _t, True).values.astype(np.float64)
        ot = np.sign(_x).astype(np.int64)
        k = nb_strike(_s, _t, rd, rf, ot, _x, 1, _v)
        return pd.DataFrame(k, index=df_.index, columns=['k'])


    @staticmethod
    #@njit([float64[:](float64[:,:], float64[:,:], float64[:])], cache=True)
    def spline_interp(x, y, xp):

        xp_ = np.asarray(xp, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        z = np.full(len(xp), np.nan)
        for t in range(len(xp)):
            x_ = x[t, ~np.isnan(x[t, :])]
            y_ = y[t, ~np.isnan(y[t, :])]

            if len(x_) > 1:
                if len(x_) == len(y_):
                    res = cubicspline(x_,
                                      y_,
                                      xp_[t]).item()
                    z.flat[t] = res
            elif np.any(np.abs(xp_[t] - x_) < 1e-9):
                z.flat[t] = y_[np.abs(xp_[t] - x_) < 1e-9].item()
        return z


    def interpolate(self, v, k, x):

        # Target Observations...
        fy = np.asarray(v, dtype=np.float64)
        fx = np.asarray(k, dtype=np.float64)
        xp = np.asarray(x, dtype=np.float64)

        res = np.column_stack((CubicSpline.spline_interp(fx, fy, xp), x))
        return pd.DataFrame(res, index=v.index, columns=['mid', 'x']).reset_index()


    def interpolate_maturities(self, maturities):

        if maturities is None:
           return

        res = FrameUtils.rowise_flat_forward_interpolation_on_groups(self.sig,
                                                                      np.unique(maturities),
                                                                      x_lev='t',
                                                                      group='x').dropna(how='all', axis=1)
        return res.stack(level=[0, 1]).to_frame('mid').reset_index(drop=False)



if __name__ == "__main__":

        from epsilonPhi.core.dataModel.dataSources.impliedVolatility.Models import *
        from epsilonPhi.core.dataModel.dataSources.impliedVolatility.VolSurfaceMgr import VolSurfaceMgr


        @njit([int64[:](int64[:], int64[:])], nopython=True)
        def index(a, b):

            res = np.empty(a.shape, dtype=np.int64)
            for t in range(len(a)):
                tmp = a[t]
                loc = np.where(b == tmp)[0]
                if len(loc) > 0:
                   res.flat[t] = loc.item()
                else:
                   res.flat[t] = -1
            return res


        # Binary Search in python

        def binary_search(arr, xs):


            for t in range(len(xs)):
                low = 0
                high = len(arr) - 1

                while low <= high:
                    mid = (low + high) // 2
                    mid_val = arr[mid]

                    if mid_val < x:
                        low = mid + 1
                    elif mid_val > x:
                        high = mid - 1
                    else:
                        return mid
                return -1

        b = np.array(range(100000)).astype(np.int64)
        a = np.flip(b).astype(np.int64)

        loc = index(a, b)


        underlier = 'GBPUSD'
        vsm = VolSurfaceMgr(underlier,
                             pricing_location='NYC',
                             strike_reference=StrikeReference.DELTA)


        ### Get ivols
        ivols = vsm._ivol_cache[underlier].copy()

        s = vsm._spot_prices
        rd = vsm._rate_curve
        rf = vsm._funding_curve

        self = CubicSpline(ivols,
                           s,
                           rf,
                           rd)

        strike_reference = StrikeReference.DELTA
        strikes = [-0.1, -0.25, -0.5, -0.75, -0.9]
        maturities = [1 / 12, 1/12 + 0.5/12, 3 / 12, 4/12, 6 / 12, 9/12, 12/12]
        fxivols = self.get_ivols(None, strike_reference, strikes, maturities)


        mats = fxivols.get('t') - (1/365)
        dates = fxivols.get('date') + np.timedelta64(1, 'D')
        ks = fxivols.get('k')

        b = mats.values
        a = np.flip(mats).values


        @njit(int64(float64, float64[:]))
        def fast_lookup(a, b):

            ctr = 0
            for t in range(len(b)):
                if a == b[0]:
                   return ctr
                else:
                   ctr += 1

            ctr = np.float64(np.nan)
            return ctr


