import numpy as np
import pandas as pd
from epsilonPhi.core.utils.FrameUtils import FrameUtils
from epsilonPhi.core.lib.curve_fitting.cubic_spline.cubic_spline import cubic_spline as cubicspline
from numba import jit
from tqdm import tqdm
from epsilonPhi.core.utils.OptionUtils import *
from epsilonPhi.core.dataModel.enums.ImpliedVolatility import *
from epsilonPhi.core.utils.MathUtils import cubic_spline


class CubicSpline(object):
    def __init__(self, ivols, s, rd, rf):

        # Set ivols panel
        self.set_ivols(ivols)

        # Sset spot rates
        self.set_spot_prices(s)

        # Set interest and funding rate curves
        self.set_risk_free_rate_curve(rd)
        self.set_funding_rate_curve(rf)
        self._x = None


    def set_ivols(self, ivols):

        _ivols = ivols.reset_index(drop=False).drop(labels=['relative_strike'], axis=1)
        _loc = [x not in ['date', 'mid', 't', 'k', 'relative_strike'] for x in _ivols.columns]

        # Extract the strike reference and observations
        self._strike_reference = _ivols.columns[_loc][0]
        vols = _ivols.drop_duplicates(['date', 't', self._strike_reference])
        self._ivols = vols.set_index(['date', 't', self._strike_reference], drop=True)
        self._ivols = self._ivols[~self._ivols.index.duplicated(keep='first')].sort_index(level='date')

    def set_spot_prices(self, df):
        self._spot = df.copy()

    def set_risk_free_rate_curve(self, rd):
        self._rate_curve = rd

    def set_funding_rate_curve(self, rf):
        self._funding_curve = rf

    @property
    def keys(self):
        return self.ivols.index
    @property
    def unique_dates(self):
        return pd.to_datetime(np.unique(self.dates))
    @property
    def ivols(self):
        return self._ivols.copy()
    @property
    def dates(self):
        return pd.to_datetime(self.ivols.index.get_level_values('date'))
    @property
    def maturities(self):
        return np.sort(np.unique(self.t))
    @property
    def sig(self):
        return self.ivols.get('mid').values.reshape(-1, 1)
    @property
    def x(self):
        return np.array(self.ivols.index.get_level_values(self._strike_reference)).reshape(-1, 1)
    @property
    def t(self):
        return np.array(self.ivols.index.get_level_values('t')).reshape(-1, 1)
    @property
    def k(self):
        return self.ivols.get('k').values.reshape(-1, 1)
    def rf(self):
        return self._rate_curve.get_stacked_curve(self.dates, self.t).values.reshape(-1, 1)
    def rd(self):
        return self._funding_curve.get_stacked_curve(self.dates, self.t).values.reshape(-1, 1)
    def s(self):
        return self._spot.loc[self.dates].values.reshape(-1, 1)
    def f(self):
        return self.get_forward_prices(self.dates, self.t)
    def ivol_frame(self):
        return self._ivols.get('mid').unstack(level=['t', self._strike_reference])
    def ik_frame(self):
        return self._ivols.get('k').unstack(level=['t', self._strike_reference])

    def delta(self):
        return fast_delta(self.s(),
                          self.t,
                          self.k,
                          self.rd(),
                          self.rf(),
                          self.sig,
                          1,
                          -1)
    def convexity_adj_moneyness(self):
        lnm = self.log_moneyness()
        zp = lnm + 0.5 * np.power(self.sig, 2) * self.t
        return zp / (self.sig * np.sqrt(self.t))

    def log_moneyness(self):
        return np.log(self.k / self.f)
    def moneyness(self):
        return self.k / self.s
    def z_score(self):
        lmn = self.log_moneyness()
        return lmn / (self.sig * np.sqrt(self.t))

    def get_ivols(self, pricing_dates, strike_reference, relative_strike, maturities):

        mats = np.round(maturities, 10)
        if pricing_dates is None:
            _p, _t, _k = np.meshgrid(self.unique_dates, mats, relative_strike)
        elif pricing_dates.shape != mats.shape:
            _p, _t, _k = np.meshgrid(pricing_dates, mats, relative_strike)
        else:
            assert pricing_dates.shape == mats.shape, 'Error - dimension mis-match'
            assert pricing_dates.shape == relative_strike.shape, 'Error - dimension mis-match'
            _p, _t, _k = pricing_dates, maturities, strike_reference

        self.load_ivols(_p, _t, _k, strike_reference)
        _vols = FrameUtils.multiindex(_p.flatten(), _t.flatten(), _k.flatten())
        return self.ivols.loc[_vols]

    def load_ivols(self, pricing_dates, maturities, relative_strike, strike_reference):

        # Interpolate along the time direction - Flat Forward Interpolation.
        self.interpolate_term_structure(maturities)

        # Compute the surface values for the strike reference we care about
        if strike_reference in [StrikeReference.DELTA, StrikeReference.DELTA.value]:
            _x = self.delta()
        elif strike_reference in [StrikeReference.MONEYNESS, StrikeReference.MONEYNESS.value]:
            _x = self.moneyness()
        elif strike_reference in [StrikeReference.LOG_MONEYNESS, StrikeReference.LOG_MONEYNESS.value]:
            _x = self.log_moneyness()
        elif strike_reference in [StrikeReference.Z_SCORE, StrikeReference.Z_SCORE.value]:
            _x = self.z_score()
        elif strike_reference in [StrikeReference.CONVEXITY_MN, StrikeReference.CONVEXITY_MN.value]:
            _x = self.convexity_adj_moneyness()
        elif strike_reference in [StrikeReference.STRIKE_PRICE, StrikeReference.STRIKE_PRICE.value]:
            _x = self.k
        else:
            raise ValueError('Error - strike reference {} not supported'.format(strike_reference))

        __x = self.ivols.copy()
        __x['x'] = _x
        self._x = __x[['mid','x']].unstack(self._strike_reference)

        _ivols = self.interpolate(pricing_dates, maturities, relative_strike)
        _strikes = self.get_strikes(_ivols, strike_reference)
        _sig = pd.concat((_ivols, _strikes), axis=1)

        self._ivols = pd.concat((self._ivols, _sig), axis=0).sort_index(level='date')

    def get_strikes(self, ivols, strike_reference):

        # Compute the surface values for the strike reference we care about
        if strike_reference in [StrikeReference.DELTA, StrikeReference.DELTA.value]:
            return self.get_strikes_from_delta(ivols)
        elif strike_reference in [StrikeReference.MONEYNESS, StrikeReference.MONEYNESS.value]:
            return self.get_strikes_from_moneyness(ivols)
        elif strike_reference in [StrikeReference.LOG_MONEYNESS, StrikeReference.LOG_MONEYNESS.value]:
            return self.get_strikes_from_log_moneyness(ivols)
        elif strike_reference in [StrikeReference.Z_SCORE, StrikeReference.Z_SCORE.value]:
            return self.get_strikes_from_z_score(ivols)
        elif strike_reference in [StrikeReference.CONVEXITY_MN, StrikeReference.CONVEXITY_MN.value]:
            return self.get_strikes_from_convexity_moneyness(ivols)
        else:
            raise ValueError('Error - strike reference {} not supported'.format(strike_reference))

    def get_strikes_from_delta(self, ivols):

        _t = np.array(ivols.index.get_level_values('t'))
        _x = np.array(ivols.index.get_level_values(self._strike_reference)).reshape(-1, 1)
        _T = _t.reshape(-1, 1)

        _rd = self.get_risk_free_rates(ivols.index.get_level_values(0), _t)
        _rf = self.get_funding_rates(ivols.index.get_level_values(0), _t)
        _s = self.get_spot_rates(ivols.index.get_level_values(0))
        k = pd.DataFrame(solve_for_strike(_s,
                                          _T,
                                          _rd,
                                          _rf,
                                          np.sign(_x),
                                          _x,
                                         1,
                                          ivols), index=ivols.index)
        k.columns = ['k']
        return k

    def get_strikes_from_moneyness(self, ivols):
        _x = np.array(ivols.columns.get_level_values(self._strike_reference)).reshape(1, -1).repeat(ivols.shape[0], axis=0)
        _t = np.array(ivols.index.get_level_values(1))
        _f = self.get_forward_prices(ivols.index.get_level_values(0),
                                     _t)
        k =  pd.DataFrame(_x * self.f, index=ivols.index)
        k.columns = [('k', x) for x in ivols.columns.get_level_values(1)]
        return k

    def get_strikes_from_log_moneyness(self, ivols):
        _x = np.array(ivols.index.get_level_values(self._strike_reference)).reshape(-1, 1)
        _t = np.array(ivols.index.get_level_values('t'))
        _f = self.get_forward_prices(ivols.index.get_level_values(0),
                                     _t)
        k = pd.DataFrame(_f * np.exp(_x), index=ivols.index, columns=['k'])
        return k

    def get_strikes_from_z_score(self, ivols):
        _x = np.array(ivols.columns.get_level_values(self._strike_reference)).reshape(1, -1).repeat(ivols.shape[0], axis=0)
        _t = np.array(ivols.index.get_level_values(1))
        _f = self.get_forward_prices(ivols.index.get_level_values(0),
                                     _t)

        k = _f * np.exp(_x * (ivols * np.sqrt(_t)))
        k.columns = [('k', x) for x in k.columns.get_level_values(1)]
        return k

    def get_strikes_from_convexity_moneyness(self, ivols):
        sigsq = np.power(ivols, 2)

        _t = np.array(ivols.index.get_level_values('t'))
        _f = self.get_forward_prices(ivols.index.get_level_values('date'), _t)
        _x = np.array(ivols.index.get_level_values(self._strike_reference)).reshape(-1, 1)
        _T = _t.reshape(-1, 1)
        k = _f * np.exp(_x * (ivols * np.sqrt(_T)) - 0.5 * sigsq * _T)
        k.columns = ['k']
        return k

    def get_forward_prices(self, pricing_dates, maturities):
        return (self.get_spot_rates(pricing_dates) *
               np.exp(-self.get_funding_rates(pricing_dates, maturities) * maturities.reshape(-1, 1)) /
               np.exp(-self.get_risk_free_rates(pricing_dates, maturities) * maturities.reshape(-1, 1)))

    def get_risk_free_rates(self, pricing_dates, maturities):
        return self._rate_curve.get_stacked_curve(pricing_dates, maturities).values.reshape(-1, 1)

    def get_funding_rates(self, pricing_dates, maturities):
        return self._funding_curve.get_stacked_curve(pricing_dates, maturities).values.reshape(-1, 1)

    def get_spot_rates(self, pricing_dates):
        return self._spot.loc[pricing_dates].values.reshape(-1, 1)

    @staticmethod
    @njit
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


    def interpolate(self, dates=None, maturities=None, strikes=None):

        assert dates.shape == maturities.shape, 'Error - dimension mis-match'
        assert dates.shape == strikes.shape, 'Error - dimension mis-match'

        tau = FrameUtils.multiindex(dates.flatten(),
                                    maturities.flatten(),
                                    strikes.flatten())

        # Target Observations...
        _tau = tau.difference(self.keys)
        _tau.names = self.keys.names
        if _tau.size > 0:

            xp = np.array(_tau.get_level_values(2))

            Y_p = self._x.reindex(_tau.droplevel(2))
            _y = Y_p.get('mid').values
            _x = Y_p.get('x').values

            res = CubicSpline.spline_interp(_x, _y, xp)

            idx = (res < np.nanmin(_y)) | (res > np.nanmax(_y))
            res[idx] = np.nan
            return pd.DataFrame(res, index=_tau, columns=['mid'])
        else:
            return pd.DataFrame(index=_tau, columns=['mid'])


    def interpolate_term_structure(self, maturities):

        if maturities is None:
           return

        unique_mats = np.unique(maturities)
        interp = FrameUtils.rowise_flat_forward_interpolation_on_groups(self.ivol_frame(),
                                                                        unique_mats,
                                                                        x_lev='t',
                                                                        group=self._strike_reference)

        interp_ = interp.dropna(how='all', axis=1).stack(level=[0, 1]).to_frame('mid')

        # Get the corresponding strikes for our interpolated tenors
        _strikes = self.get_strikes(interp_, self._strike_reference)
        _vols = pd.concat((interp_, _strikes), axis=1)

        # Concatenate the new interpolated vols onto the ivols
        _ivols = pd.concat((self._ivols, _vols.loc[_vols.index.difference(self.ivols.index)]), axis=0)
        self._ivols = _ivols[~_ivols.index.duplicated(keep='first')].sort_index(level='date')


if __name__ == "__main__":

        from epsilonPhi.core.dataModel.dataSources.impliedVolatility.Models import *
        from epsilonPhi.core.dataModel.dataSources.impliedVolatility.VolSurfaceMgr import VolSurfaceMgr

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

        fxivols_2 = self.get_ivols(None, strike_reference, strikes, maturities)

        _vols = fxivols.get('mid').unstack(level=[strike_reference, 't']).sort_index(level=0)
        _ivols = _vols.loc['01/10/1997':'30/06/2023']

        #
