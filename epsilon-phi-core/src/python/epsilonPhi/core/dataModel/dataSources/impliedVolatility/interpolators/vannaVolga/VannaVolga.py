import numpy as np
import pandas as pd
from epsilonPhi.core.utils.FrameUtils import FrameUtils
from epsilonPhi.core.utils.OptionUtils import *
from epsilonPhi.core.dataModel.dataSources.impliedVolatility.Models import *

class VannaVolga(object):

    _raw_cache = {}
    _DELTA_PILLARS = [-0.25, -999, 0.25]
    _DEFAULT_DELTA_POINTS = [-0.1, -0.15, -0.25, 0.5, 0.25, 0.15, 0.1]

    def __init__(self, ivols, s, rd, rf, delta_type):

        # Extract and set the vv pillar vols (-25, ATM, 25) points
        self.set_pillar_vols(ivols)

        # Set the spot prices
        self.set_spot_prices(s)

        # Set interest and funding rate curves
        self.set_risk_free_rate_curve(rd)
        self.set_funding_rate_curve(rf)

        # Set the type of delta of the instrument
        self._delta_type = delta_type

        # Set the data maturities
        self.set_dates(self._ivols.index.get_level_values(0))
        self.set_maturities(self.maturities)

    def set_spot_prices(self, df):
        self._spot = df.copy()

    def set_risk_free_rate_curve(self, rd):
        self._rate_curve = rd

    def set_funding_rate_curve(self, rf):
        self._funding_curve = rf

    def set_pillar_vols(self, ivols):
        ivols = ivols.reset_index(drop=False).set_index(['date'])
        idx = ivols['relative_strike'].isin(self._DELTA_PILLARS)
        ivols_ = ivols[idx].pivot(columns=['t', 'relative_strike']).get('mid')
        ivols_.columns.names = ['t', 'x']
        self._ivols = ivols_.sort_index(axis=1, level=0)

    def set_dates(self, dates):
        self._dates = dates

    @property
    def maturities(self):
        return np.sort(np.unique(self._ivols.columns.get_level_values('t')))

    @property
    def pillar_references(self):
        return np.unique(self._ivols.columns.get_level_values('relative_strike'))
    @property
    def dates(self):
        return pd.to_datetime(self._dates)

    @property
    def N(self):
        return self.t.shape[0]
    @property
    def t(self):
        return self._t.reshape(-1, 1)
    @property
    def f(self):
        return self._f
    @property
    def s(self):
        return self._s.values.reshape(-1, 1)
    @property
    def rf(self):
        return self._rf.values.reshape(-1, 1)
    @property
    def rd(self):
        return self._rd.values.reshape(-1, 1)
    @property
    def sig_atm(self):
        return self._interp_vols.get(-999).values.reshape(-1, 1)
    @property
    def sig_call(self):
        return self._interp_vols.get(0.25).values.reshape(-1, 1)
    @property
    def sig_put(self):
        return self._interp_vols.get(-0.25).values.reshape(-1, 1)
    @property
    def k_put(self):
        return self._k.get(-0.25).values.reshape(-1, 1)
    @property
    def k_call(self):
        return self._k.get(0.25).values.reshape(-1, 1)
    @property
    def k_atm(self):
        return self._k.get(-999).values.reshape(-1, 1)

    def get_sigma(self, strike):
        return self._interp_vols.get(strike).values.reshape(-1, 1)

    def set_maturities(self, maturities=None):

        # Make sure we have the maturity points we care about
        if maturities is None:
           maturities = self.maturities

        mats = np.round(maturities, 10)
        self.interpolate_term_structure(mats)

        vols = FrameUtils.select_subset_level(self._ivols, 't', mats)
        self._interp_vols = vols.stack(level='t')[self._DELTA_PILLARS]

        # Set the dates in the object
        self.set_dates(self._interp_vols.index.get_level_values('date'))
        # Set the interpolated maturity vector
        self._t = np.array(self._interp_vols.index.get_level_values('t'))

        # Update the spot prices
        self._s = self._spot.loc[self.dates]
        assert np.all(self._s.index == self.dates), 'Error - axis not aligned'

        # Set the domestic rate in the object
        self._rd = self._rate_curve.get_stacked_curve(self.dates, self._t)
        assert np.all(self._rd.index.get_level_values(0) == self.dates), 'Error - axis not aligned'

        # Set funding rates
        self._rf = self._funding_curve.get_stacked_curve(self.dates, self._t)
        assert np.all(self._rf.index.get_level_values(0) == self.dates), 'Error - axis not aligned'

        # Set forward rates
        self._f = self.s * np.exp(-self.rf * self.t) / np.exp(-self.rd * self.t)

        # Set Pillar Strikes
        self.set_strikes()


    def set_strikes(self):

        K = len(self.pillar_references)
        ks = np.full((self.N, K), np.nan)
        for t in range(K):

            strk = self.pillar_references[t]
            if strk == -999:
                ks[:, t] = self.get_atm_strikes(self.get_sigma(strk)).flatten()
            elif strk in self._DELTA_PILLARS:
                ks[:, t] = self.get_wing_strikes(strk,
                                                self.get_sigma(strk)).flatten()
        self._k = pd.DataFrame(ks, index=self._interp_vols.index, columns=self.pillar_references)

    def get_wing_strikes(self, delta, ivols):
        return solve_for_strike(self.s,
                                self.t,
                                self.rd,
                                self.rf,
                                np.sign(delta),
                                delta,
                                self._delta_type,
                                ivols)

    def get_atm_strikes(self, ivols):
        return atm_delta_neutral_strike(self.s,
                                        self.t,
                                        self.rd,
                                        self.rf,
                                        ivols,
                                        self._delta_type)

    def get_ivols(self, pricing_dates=None, strike_reference=None, relative_strike=None, maturities=None):

        if pricing_dates is None:
           pricing_dates = self.dates

        relative_strike = np.array(relative_strike)
        maturities = np.array(maturities)

        # Set the maturities which interpolates in forward space
        self.set_maturities(maturities)

        # Compute the surface values for the strike reference we care about
        if strike_reference in [StrikeReference.DELTA, StrikeReference.DELTA.value]:
            return self.get_ivols_delta_space(relative_strike).loc[pricing_dates]
        elif strike_reference in [StrikeReference.MONEYNESS, StrikeReference.MONEYNESS.value]:
            return self.get_ivols_moneyness_space(relative_strike).loc[pricing_dates]
        elif strike_reference in [StrikeReference.LOG_MONEYNESS, StrikeReference.LOG_MONEYNESS.value]:
            return self.get_ivols_log_moneyness_space(relative_strike).loc[pricing_dates]
        elif strike_reference in [StrikeReference.Z_SCORE, StrikeReference.Z_SCORE.value]:
            return self.get_ivols_zscore_space(relative_strike).loc[pricing_dates]
        elif strike_reference in [StrikeReference.CONVEXITY_MN, StrikeReference.CONVEXITY_MN.value]:
            return self.get_ivols_convexity_moneyness_space(relative_strike).loc[pricing_dates]
        else:
            return None

    def k_rng(self):
        return np.linspace(self.f * np.exp((-5 * self.sig_atm * np.sqrt(self.t))),
                           self.f * np.exp((5 * self.sig_atm * np.sqrt(self.t))),
                        100)[:, :, 0].T


    def get_ivols_delta_space(self, relative_strike=None):

        if relative_strike is None:
           relative_strike = np.array(self._DEFAULT_DELTA_POINTS)
        else:
           relative_strike = np.array(relative_strike)

        # Get the number of relative strikes we want to solve for
        T = len(relative_strike.flatten())

        # Build the space of candidate strikes
        k_rng = self.k_rng()

        # Fit the implied vols across each candidate strike
        ivols = self.fit_strikes(k_rng)

        # For each strike and vol pair, compute the delta
        pd_ = self.deltas(ivols, k_rng, -1, self._delta_type)
        cd_ = self.deltas(ivols, k_rng, 1, self._delta_type)

        # Find the location of the delta closest to our target deltas
        _strikes = np.full((self.N, T), np.nan)
        for k in range(T):
            if relative_strike[k] < 0:
                idx = np.abs(pd_ - relative_strike[k]) == np.min(np.abs(pd_ - relative_strike[k]), axis=1, keepdims=True)
                keep_rows = np.any(idx, axis=1)
                _strikes[keep_rows, k] = k_rng[keep_rows, :][idx[keep_rows, :]]
            else:
                idx = np.abs(cd_ - relative_strike[k]) == np.min(np.abs(cd_ - relative_strike[k]), axis=1, keepdims=True)
                keep_rows = np.any(idx, axis=1)
                _strikes[keep_rows, k] = k_rng[keep_rows, :][idx[keep_rows, :]]

        _vols = pd.DataFrame(self.fit_strikes(_strikes), columns=[('ivol', x) for x in relative_strike], index=self._interp_vols.index)
        _k = pd.DataFrame(_strikes, columns=[('k', x) for x in relative_strike], index=_vols.index)
        res = pd.concat((_vols, _k), axis=1)
        res.columns = pd.MultiIndex.from_tuples(res.columns)
        return res.copy()

    def get_ivols_moneyness_space(self, relative_strike=None):

        if relative_strike is None:
           relative_strike = np.arange(0.4, 1.6, 0.1)

        _strikes = self.f * np.array(relative_strike)
        _vols = self.fit_strikes(_strikes)

        _vols = pd.DataFrame(_vols, columns=[('ivol', x) for x in relative_strike], index=self._interp_vols.index)
        _k = pd.DataFrame(_strikes, columns=[('k', x) for x in relative_strike], index=self._interp_vols.index)
        res = pd.concat((_vols, _k), axis=1)
        res.columns = pd.MultiIndex.from_tuples(res.columns)
        return res.copy()

    def get_ivols_log_moneyness_space(self, relative_strike=None):

        if relative_strike is None:
           relative_strike = np.round(np.arange(-0.4, 0.5, 0.05), 3)

        _strikes = self.f * np.exp(relative_strike)
        _vols = self.fit_strikes(_strikes)

        _vols = pd.DataFrame(_vols, columns=[('ivol', x) for x in relative_strike], index=self._interp_vols.index)
        _k = pd.DataFrame(_strikes, columns=[('k', x) for x in relative_strike], index=self._interp_vols.index)
        res = pd.concat((_vols, _k), axis=1)
        res.columns = pd.MultiIndex.from_tuples(res.columns)
        return res.copy()

    def get_ivols_zscore_space(self, relative_strike=None):

        if relative_strike is None:
           relative_strike = np.round(np.arange(-4, 4.5, 0.5), 3)

        _strikes = self.f / np.exp(relative_strike * self.sig_atm * np.sqrt(self.t))
        _vols = self.fit_strikes(_strikes)

        _vols = pd.DataFrame(_vols, columns=[('ivol', x) for x in relative_strike], index=self._interp_vols.index)
        _k = pd.DataFrame(_strikes, columns=[('k', x) for x in relative_strike], index=self._interp_vols.index)
        res = pd.concat((_vols, _k), axis=1)
        res.columns = pd.MultiIndex.from_tuples(res.columns)
        return res.copy()

    def get_ivols_convexity_moneyness_space(self, relative_strike=None):

        if relative_strike is None:
            relative_strike = np.round(np.arange(-3, 3.5, 0.5), 3)

        # Get the number of relative strikes we want to solve for
        T = len(relative_strike.flatten())

        # Build the space of candidate strikes
        k_rng = self.k_rng()

        # Fit the implied vols across each candidate strike
        ivols = self.fit_strikes(k_rng)

        # For each strike and vol pair, compute the delta
        cnx_mny = (np.log(self.f / k_rng)) / (ivols * np.sqrt(self.t))

        # Find the location of the delta closest to our target deltas
        _strikes = np.full((self.N, T), np.nan)
        for k in range(T):
            idx = np.abs(cnx_mny - relative_strike[k]) == np.min(np.abs(cnx_mny - relative_strike[k]), axis=1, keepdims=True)
            keep_rows = np.any(idx, axis=1)
            _strikes[keep_rows, k] = k_rng[keep_rows, :][idx[keep_rows, :]]


        _vols = pd.DataFrame(self.fit_strikes(_strikes), columns=[('ivol', x) for x in relative_strike], index=self._interp_vols.index)
        _k = pd.DataFrame(_strikes, columns=[('k', x) for x in relative_strike], index=self._interp_vols.index)
        res = pd.concat((_vols, _k), axis=1)
        res.columns = pd.MultiIndex.from_tuples(res.columns)
        return res

    def deltas(self, ivols, strikes, option_type, delta_type):
        return fast_delta(self.s,
                          self.t,
                          strikes,
                          self.rd,
                          self.rf,
                          ivols,
                          delta_type,
                          option_type)

    def fit_strikes(self, strikes):
        return vanna_volga_2d(self.f,
                              strikes,
                              self.t,
                              self.k_put,
                              self.k_atm,
                              self.k_call,
                              self.sig_put,
                              self.sig_atm,
                              self.sig_call)

    def interpolate_term_structure(self, maturities):

        if maturities is None:
           return

        unique_mats = np.setdiff1d(np.round(maturities, 10), self.maturities)
        group = np.setdiff1d(self._ivols.columns.names, 't').item()
        interp = FrameUtils.rowise_flat_forward_interpolation_on_groups(self._ivols,
                                                                        unique_mats,
                                                                        x_lev='t',
                                                                        group=group)
        cols = np.setdiff1d(interp.columns, self._ivols.columns)
        self._ivols = pd.concat((self._ivols, interp.get(cols)),
                                axis=1).sort_index(axis=1, level='t')



if __name__ == "__main__":

        from epsilonPhi.core.dataModel.dataSources.impliedVolatility.Models import *
        from epsilonPhi.core.dataModel.dataSources.impliedVolatility.VolSurfaceMgr import VolSurfaceMgr

        underlier = 'EURUSD'
        vsm = VolSurfaceMgr(underlier,
                             pricing_location='NYC',
                             strike_reference=StrikeReference.DELTA)


        ### Get ivols
        ivols = vsm._ivol_cache[underlier].copy()

        s = vsm._spot_prices
        rd = vsm._rate_curve
        rf = vsm._funding_curve

        self = VannaVolga(ivols,
                          s,
                          rf,
                          rd,
                          1)

        dates = pd.to_datetime('31-Dec-2012')
        strike_reference = StrikeReference.DELTA
        strikes = [-0.1, -0.25, 0.5, 0.25, 0.1]
        maturities = [1/12, 1.5/12, 2/12]
        fxivols = self.get_ivols(None, strike_reference, strikes, maturities)

        fxivols_ = self.get_ivols(None, strike_reference=StrikeReference.Z_SCORE, maturities=maturities)


