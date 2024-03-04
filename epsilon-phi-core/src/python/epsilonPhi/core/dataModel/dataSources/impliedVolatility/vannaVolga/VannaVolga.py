import pandas as pd

from epsilonPhi.core.utils.OptionUtils import *
from epsilonPhi.core.dataModel.dataSources.impliedVolatility.Models import *

class VannaVolga(object):

    _raw_cache = {}
    _DELTA_PILLARS = [-0.25, -999, 0.25]
    _DEFAULT_DELTA_POINTS = [-0.1, -0.15, -0.25, 0.5, 0.25, 0.15, 0.1]

    def __init__(self, ivols, s, rd, rf, delta_type):

        self.set_pillar_vols(ivols)

        self._t = np.array(self._ivols.index.get_level_values('t'))

        self._s = s.loc[self.dates]
        assert np.all(self._s.index == self.dates), 'Error - axis not aligned'

        self._rd = rd[~rd.index.duplicated()].loc[self._ivols.index]
        assert np.all(self._rd.index == self._ivols.index), 'Error - axis not aligned'

        self._rf = rf[~rd.index.duplicated()].loc[self._ivols.index]
        assert np.all(self._rf.index == self._ivols.index), 'Error - axis not aligned'

        self._f = self.s * np.exp(-self.rf * self.t) / np.exp(-self.rd * self.t)
        self._delta_type = delta_type

        # Set Pillar Strikes
        self.set_strikes()


    def set_pillar_vols(self, ivols):
        ivols = ivols.reset_index(drop=False).set_index(['date', 't'])
        idx = ivols['relative_strike'].isin(self._DELTA_PILLARS)
        ivols = ivols[idx].pivot(columns='relative_strike').get('mid')
        self._ivols = ivols[self._DELTA_PILLARS]

    @property
    def pillar_references(self):
        return self._ivols.columns.unique()
    @property
    def dates(self):
        return self._ivols.index.get_level_values(0)
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
        return self._ivols.values[:, 1].reshape(-1, 1)
    @property
    def sig_call(self):
        return self._ivols.values[:, 2].reshape(-1, 1)
    @property
    def sig_put(self):
        return self._ivols.values[:, 0].reshape(-1, 1)
    @property
    def k_put(self):
        return self._k.values[:, 0].reshape(-1, 1)
    @property
    def k_call(self):
        return self._k.values[:, 2].reshape(-1, 1)
    @property
    def k_atm(self):
        return self._k.values[:, 1].reshape(-1, 1)
    @property
    def index(self):
        return self._ivols.index
    def get_sigma(self, strike):
        return self._ivols.get(strike).values.reshape(-1, 1)

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
        self._k = pd.DataFrame(ks, index=self.index, columns=self.pillar_references)

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

    def get_ivols(self, pricing_dates, strike_reference, strikes, maturities):

        if strike_reference in [StrikeReference.DELTA, StrikeReference.DELTA.value]:
            return self.get_ivols_delta_space()
        elif strike_reference in [StrikeReference.MONEYNESS, StrikeReference.MONEYNESS.value]:
            return self.get_ivols_moneyness_space()
        elif strike_reference in [StrikeReference.LOG_MONEYNESS, StrikeReference.LOG_MONEYNESS.value]:
            return self.get_ivols_log_moneyness_space()
        elif strike_reference in [StrikeReference.Z_SCORE, StrikeReference.Z_SCORE.value]:
            return self.get_ivols_zscore_space()
        elif strike_reference in [StrikeReference.CONVEXITY_MN, StrikeReference.CONVEXITY_MN.value]:
            return self.get_ivols_convexity_moneyness_space()
        else:
            raise ValueError('Error - strike reference {} not supported'.format(strike_reference))

    def k_rng(self):
        return np.linspace(self.f * np.exp((-3 * self.sig_atm * np.sqrt(self.t))),
                           self.f * np.exp((3 * self.sig_atm * np.sqrt(self.t))),
                        1000)[:, :, 0].T


    def get_ivols_delta_space(self, relative_strike=None):

        if relative_strike is None:
           relative_strike = np.array(self._DEFAULT_DELTA_POINTS)

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

        _vols = pd.DataFrame(self.fit_strikes(_strikes), columns=[('ivol', x) for x in relative_strike], index=self.index)
        _k = pd.DataFrame(_strikes, columns=[('k', x) for x in relative_strike], index=self.index)
        res = pd.concat((_vols, _k), axis=1)
        res.columns = pd.MultiIndex.from_tuples(res.columns)
        return res

    def get_ivols_moneyness_space(self, relative_strike=None):

        if relative_strike is None:
           relative_strike = np.array(self._DEFAULT_DELTA_POINTS)
        pass
    def get_ivols_log_moneyness_space(self):
        pass
    def get_ivols_zscore_space(self):
        pass
    def get_ivols_convexity_moneyness_space(self):
        pass

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


    def fit_cross_sections(self):
        pass


    def fit_maturities(self):
        pass

if __name__ == "__main__":

        from epsilonPhi.core.dataModel.dataSources.impliedVolatility.Models import *
        from epsilonPhi.core.dataModel.dataSources.impliedVolatility.VolSurfaceMgr import VolSurfaceMgr

        underlier = 'EURUSD'
        vsm = VolSurfaceMgr(underlier,
                             pricing_location='NYC',
                             strike_reference=StrikeReference.DELTA)


        ### Get ivols
        ivol_panel = vsm.get_ivols()


        ivols = ivol_panel.get(['mid','t','relative_strike']).reset_index(drop=False)
        ivols = ivols.drop_duplicates(['date', 't', 'relative_strike'])
        s = vsm.get_spot_prices()

        rd = vsm.get_interest_rates(ivols.get('date'), ivols.get('t'))
        rd = pd.DataFrame(rd)
        rd.index = ivols.set_index(['date', 't']).index

        rf = vsm.get_funding_rates(ivols.get('date'), ivols.get('t'))
        rf = pd.DataFrame(rf)
        rf.index = ivols.set_index(['date', 't']).index

        self = VannaVolga(ivols,
                              s,
                              rf,
                              rd,
                              1)

        fxivols = self.get_ivols_delta_space()

