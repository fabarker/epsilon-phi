import numpy as np
import pandas as pd
from epsilonPhi.core.utils.FrameUtils import FrameUtils
from numba import jit
from tqdm import tqdm
from cubic_spline import *
from scipy.interpolate import CubicSpline as spline

from epsilonPhi.core.utils.OptionUtils import *
from epsilonPhi.core.dataModel.enums.ImpliedVolatility import *

class CubicSpline(object):
    def __init__(self, ivols, s, rd, rf):

        # Set ivols panel
        self.set_ivols(ivols)

        # Sset spot rates
        self.set_spot_prices(s)

        # Set interest and funding rate curves
        self.set_risk_free_rate_curve(rd)
        self.set_funding_rate_curve(rf)
        self._keys = None


    def set_ivols(self, ivols):

        _ivols = ivols.reset_index(drop=False).drop(labels=['relative_strike'], axis=1)
        _loc = [ x not in ['date','mid','t','k','relative_strike'] for x in _ivols.columns ]

        # Extract the strike reference and observations
        self._strike_reference = _ivols.columns[_loc][0]
        self._ivols = _ivols.drop_duplicates(['date', 't', self._strike_reference]).set_index('date', drop=True)

    def set_spot_prices(self, df):
        self._spot = df.copy()

    def set_risk_free_rate_curve(self, rd):
        self._rate_curve = rd

    def set_funding_rate_curve(self, rf):
        self._funding_curve = rf

    @property
    def keys(self):
        return self.ivols.reset_index(drop=False).set_index(['date', 't', self._strike_reference]).index

    @property
    def unique_dates(self):
        return pd.to_datetime(np.unique(self.dates))

    @property
    def ivols(self):
        return self._ivols.copy()
    @property
    def dates(self):
        return pd.to_datetime(self.ivols.index)
    @property
    def maturities(self):
        return np.sort(np.unique(self._ivols.get('t')))
    @property
    def sig(self):
        return self.ivols.get('mid').values.reshape(-1, 1)
    @property
    def t(self):
        return self.ivols.get('t').values.reshape(-1, 1)
    @property
    def k(self):
        return self.ivols.get('k').values.reshape(-1, 1)
    @property
    def rf(self):
        return self._rate_curve.get_stacked_curve(self.dates, self.t).values.reshape(-1, 1)
    @property
    def rd(self):
        return self._funding_curve.get_stacked_curve(self.dates, self.t).values.reshape(-1, 1)
    @property
    def s(self):
        return self._spot.loc[self.dates].values.reshape(-1, 1)
    @property
    def f(self):
        return self.get_forward_prices(self.dates, self.t)
    def delta(self):
        return fast_delta(self.s,
                          self.t,
                          self.k,
                          self.rd,
                          self.rf,
                          self.sig,
                          1,
                          1)
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

        # Compute the surface values for the strike reference we care about
        if strike_reference in [StrikeReference.DELTA, StrikeReference.DELTA.value]:
            self.x = self.delta()
        elif strike_reference in [StrikeReference.MONEYNESS, StrikeReference.MONEYNESS.value]:
            self.x = self.moneyness()
        elif strike_reference in [StrikeReference.LOG_MONEYNESS, StrikeReference.LOG_MONEYNESS.value]:
            self.x = self.log_moneyness()
        elif strike_reference in [StrikeReference.Z_SCORE, StrikeReference.Z_SCORE.value]:
            self.x = self.z_score()
        elif strike_reference in [StrikeReference.CONVEXITY_MN, StrikeReference.CONVEXITY_MN.value]:
            self.x = self.convexity_adj_moneyness()
        elif strike_reference in [StrikeReference.STRIKE_PRICE, StrikeReference.STRIKE_PRICE.value]:
            self.x = self.k
        else:
            raise ValueError('Error - strike reference {} not supported'.format(strike_reference))

        if pricing_dates is None:
            _p, _t, _k = np.meshgrid(self.unique_dates, mats, relative_strike)
            _ivols = self.interpolate(_p, _t, _k)
        elif pricing_dates.shape != mats.shape:
            _p, _t, _k = np.meshgrid(pricing_dates, mats, relative_strike)
            _ivols = self.interpolate(_p, _t, _k)
        else:
            assert pricing_dates.shape == mats.shape, 'Error - dimension mis-match'
            assert pricing_dates.shape == relative_strike.shape, 'Error - dimension mis-match'
            _ivols = self.interpolate(pricing_dates, mats, relative_strike)
        _strikes = self.get_strikes(_ivols, strike_reference)
        return pd.concat((_ivols, _strikes), axis=1)

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
        _x = np.array(ivols.index.get_level_values('x')).reshape(-1, 1)
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
        _x = np.array(ivols.columns.get_level_values(1)).reshape(1, -1).repeat(ivols.shape[0], axis=0)
        _t = np.array(ivols.index.get_level_values(1))
        _f = self.get_forward_prices(ivols.index.get_level_values(0),
                                     _t)
        k =  pd.DataFrame(_x * self.f, index=ivols.index)
        k.columns = [('k', x) for x in ivols.columns.get_level_values(1)]
        return k

    def get_strikes_from_log_moneyness(self, ivols):
        _x = np.array(ivols.index.get_level_values('x')).reshape(-1, 1)
        _t = np.array(ivols.index.get_level_values('t'))
        _f = self.get_forward_prices(ivols.index.get_level_values(0),
                                     _t)
        k = pd.DataFrame(_f * np.exp(_x), index=ivols.index, columns=['k'])
        return k

    def get_strikes_from_z_score(self, ivols):
        _x = np.array(ivols.columns.get_level_values(1)).reshape(1, -1).repeat(ivols.shape[0], axis=0)
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
        _x = np.array(ivols.index.get_level_values('x')).reshape(-1, 1)
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
    def interpolate(self, dates=None, maturities=None, strikes=None):

        assert dates.shape == maturities.shape, 'Error - dimension mis-match'
        assert dates.shape == strikes.shape, 'Error - dimension mis-match'

        # Interpolate along the time direction - Flat Forward Interpolation.
        self.interpolate_term_structure(maturities)

        tau = FrameUtils.multiindex(dates.flatten(), maturities.flatten(), strikes.flatten())
        _tau = tau.difference(self.keys)

        _ds = tau.get_level_values(0)
        _ts = tau.get_level_values(1).astype(float).to_numpy()
        _xs = tau.get_level_values(2).astype(float).to_numpy()

        _dates = np.unique(_ds)
        ND = len(_ds)

        ts_ = self.t.astype(float).flatten()
        xs_ = self.x.astype(float).flatten()
        sig = self.sig.astype(float).flatten()
        ds_ = self.dates

        interps = list()
        for t in tqdm(range(ND), desc="Interpolating Vol Surface"):

            idx_T = _dates[t] == _ds
            idx_S = _dates[t] == ds_

            interp = self.interpolate_single_date(ts_[idx_S],
                                                  np.unique(_ts[idx_T]),
                                                  xs_[idx_S],
                                                  np.unique(_xs[idx_T]),
                                                  sig[idx_S])

            result = np.column_stack((interp, _ds[idx_T].astype(float)))
            interps.extend([result])


        # build ivols
        _ivols = pd.DataFrame(np.vstack((interps)))
        _ivols.columns = ['sig', 't', 'x', 'date']
        _ivols.loc[:, 'date'] = pd.to_datetime(_ivols.get('date'))
        _vols = _ivols.set_index(['date','x','t'], drop=True)
        return _vols.copy()

    @staticmethod
    def interpolate_single_date(z, z_dense, x, x_dense, y):

        vols = np.full((len(z_dense), len(x_dense)), np.nan)
        mats = z_dense.reshape((-1, 1)).repeat(len(x_dense), axis=1)
        stks = x_dense.reshape((1, -11)).repeat(len(z_dense), axis=0)
        for ctr in range(len(z_dense)):
            idx = z_dense[ctr] == z
            if np.sum(idx) > 2:
               locs = np.argsort(x[idx])
               v = cubic_y(x[idx][locs], x_dense, y[idx][locs])
               vols[ctr,:] = v
        return np.column_stack((vols.flatten(), mats.flatten(), stks.flatten()))

    def interpolate_term_structure(self, maturities):

        if maturities is None:
           return

        _ivols = self._ivols[['mid', 't', self._strike_reference]].pivot(columns=['t', self._strike_reference]).get('mid')
        unique_mats = np.setdiff1d(np.round(maturities, 10), self.maturities)
        group = np.setdiff1d(_ivols.columns.names, 't').item()

        if len(unique_mats) > 0:
            interp = FrameUtils.rowise_flat_forward_interpolation_on_groups(_ivols,
                                                                            unique_mats,
                                                                            x_lev='t',
                                                                            group=group)
            cols = np.setdiff1d(interp.columns, self._ivols.columns)
            self._ivols = pd.concat((self._ivols, interp.get(cols)),
                                    axis=1).sort_index(axis=1, level='t')


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
        strikes = [0.1, 0.25, 0.5, 0.75, 0.9]
        maturities = [1 / 12, 3 / 12, 6 / 12, 9/12, 12/12]
        fxivols = self.get_ivols(None, strike_reference, strikes, maturities)

        _vols = fxivols.get('sig').unstack(level=['x','t']).sort_index(level=0)

        _ivols = _vols.loc['01/10/1997':'30/06/2023']
