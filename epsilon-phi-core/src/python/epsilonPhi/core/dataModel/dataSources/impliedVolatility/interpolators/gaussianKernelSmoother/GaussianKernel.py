import numpy as np
import pandas as pd
from epsilonPhi.core.utils.FrameUtils import FrameUtils
from numba import jit
from tqdm import tqdm
from epsilonPhi.core.dataModel.enums.ImpliedVolatility import *

class GaussianKernel(object):
    def __init__(self, ivols, s, rd, rf):

        # Set ivols panel
        self.set_ivols(ivols)

        # Sset spot rates
        self.set_spot_prices(s)

        # Set interest and funding rate curves
        self.set_risk_free_rate_curve(rd)
        self.set_funding_rate_curve(rf)


    def set_ivols(self, ivols):
        _ivols = ivols.reset_index(drop=False)
        self._ivols = _ivols.drop_duplicates(ivols.columns).set_index('date', drop=True)

    def set_spot_prices(self, df):
        self._spot = df.copy()

    def set_risk_free_rate_curve(self, rd):
        self._rate_curve = rd

    def set_funding_rate_curve(self, rf):
        self._funding_curve = rf

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
                          self.ivols,
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
        return lmn / (ivols * np.sqrt(self.t))

    def get_ivols(self, pricing_dates=None, strike_reference=None, relative_strike=None, maturities=None):

        if maturities is None:
           mats = self.maturities
        else:
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
        else:
            raise ValueError('Error - strike reference {} not supported'.format(strike_reference))

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

        _x = np.array(ivols.columns.get_level_values(1)).reshape(1, -1).repeat(ivols.shape[0], axis=0)
        _t = np.array(ivols.index.get_level_values(1))
        _rd = self.get_risk_free_rates(ivols.index.get_level_values(0), _t)
        _rf = self.get_funding_rates(ivols.index.get_level_values(0), _t)
        _s = self.get_spot_rates(ivols.index.get_level_values(0))
        k = pd.DataFrame(solve_for_strike(s,
                                          _t,
                                          _rd,
                                          _rf,
                                          np.sign(_x),
                                          _x,
                                         1,
                                          ivols), index=ivols.index)
        k.columns = [('k', x) for x in ivols.columns.get_level_values(1)]
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
        _x = np.array(ivols.columns.get_level_values(1)).reshape(1, -1).repeat(ivols.shape[0], axis=0)
        _t = np.array(ivols.index.get_level_values(1))
        _f = self.get_forward_prices(ivols.index.get_level_values(0),
                                     _t)
        k = pd.DataFrame(_f * np.exp(_x), index=ivols.index)
        k.columns = [('k', x) for x in ivols.columns.get_level_values(1)]
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
        _t = np.array(ivols.index.get_level_values(1))
        _f = self.get_forward_prices(ivols.index.get_level_values(0),
                                     _t)
        _x = np.array(ivols.columns.get_level_values(1)).reshape(1, -1).repeat(_t.shape[0], axis=0)
        _T = _t.reshape(-1, 1)
        k = _f * np.exp(_x * (ivols * np.sqrt(_T)) - 0.5 * sigsq * _T)
        k.columns = [ ('k', x) for x in k.columns.get_level_values(1) ]
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

        if dates is None:
           dates = self.dates

        unique_d = np.unique(dates)
        unique_m = np.unique(maturities).reshape(1, -1).astype(float)
        unique_x = np.unique(strikes).reshape(1, -1).astype(float)

        NX = len(unique_x.flatten())
        NM = len(unique_m.flatten())
        ND = len(unique_d.flatten())

        t_vec = self.t.astype(float).flatten()
        x_vec = self.x.astype(float).flatten()
        v_vec = self.sig.astype(float).flatten()

        interp = np.full(shape=(NX*NM, ND), fill_value=np.nan)
        for t in tqdm(range(ND), desc="Interpolating Vol Surface"):
            idx = (unique_d[t] == dates)
            interp[:, t] = self.interpolate_single_date(t_vec[idx],
                                                         unique_m,
                                                         x_vec[idx],
                                                         unique_x,
                                                         v_vec[idx])

        idx_m = unique_x.T.repeat(NM, axis=0).flatten()
        idx_x = unique_m.repeat(NX, 0).flatten()

        vols = pd.DataFrame(interp, columns=unique_d)
        vols.index = pd.MultiIndex.from_tuples(list(zip(idx_m, idx_x)), names=['x','t'])
        vols.columns.names = ['date']

        # build ivols
        _ivols = vols.T.stack(level=1)
        _ivols.columns = pd.MultiIndex.from_tuples(list(zip(['vol'] * NX, _ivols.columns)))
        return _ivols.copy()

    @staticmethod
    @jit(nopython=True, fastmath=True, cache=True)
    def interpolate_single_date(mat, unique_m, x, unique_x, mids):
        N = len(mids)

        # get the std of the reference on each date
        sig_x = np.std(x)
        h_x = 1.0 * np.power(4.0 / 3.0, 1.0 / 5.0) * sig_x / np.power(N, 1.0 / 5.0)

        # do the same in the maturity direction
        sig_lm = np.log(mat).std()
        h_m = 2.0 * np.power(4.0 / 3.0, 1.0 / 5.0) * sig_lm / np.power(N, 1.0 / 5.0) * 0.1

        lnmat_ = np.log(mat).reshape((-1, 1))

        m_diff = np.abs(lnmat_ - np.log(unique_m.repeat(N).reshape((unique_m.size, N)).T)) / h_m
        x_diff = np.abs(x.reshape((-1, 1)) - unique_x.repeat(N).reshape((unique_x.size, N)).T) / h_x

        x_wts = np.exp(-1.0 * np.power(x_diff, 2.0) / 2.0)
        x_wts = np.expand_dims(x_wts, -1).repeat(unique_m.size).reshape((x_wts.shape[0], x_wts.shape[1], unique_m.size))

        m_wts = np.exp(-1.0 * np.power(m_diff, 2.0) / 2.0).reshape((m_diff.shape[0], 1, m_diff.shape[1]))
        m_wts_ = np.full(x_wts.shape, np.nan) * np.nan
        for dim in range(x_wts.shape[1]):
            m_wts_[:, dim, :] = m_wts[:, 0, :]

        wts = (x_wts * m_wts) / np.sum(x_wts * m_wts_, axis=0)
        ivol = mids.repeat(wts.shape[1] * unique_m.size).reshape(m_wts_.shape)
        ivols = np.sum(wts * ivol, axis=0).flatten()
        return ivols


if __name__ == "__main__":

        from epsilonPhi.core.dataModel.dataSources.impliedVolatility.Models import *
        from epsilonPhi.core.dataModel.dataSources.impliedVolatility.VolSurfaceMgr import VolSurfaceMgr

        underlier = 'EURUSD'
        vsm = VolSurfaceMgr(underlier,
                             pricing_location='NYC',
                             strike_reference=StrikeReference.DELTA)


        ### Get ivols
        ivol_panel = vsm.get_ivols()
        ivols = ivol_panel.get(['mid','t','k']).reset_index(drop=False)
        ivols = ivols.drop_duplicates(['date', 't', 'k'])

        s = vsm.get_spot_prices()
        rd = vsm._interest_rate_curve
        rf = vsm._funding_rate_curve

        # Gaussian Kernel Smoother
        self = GaussianKernel(ivols, s, rd, rf)
        fxivols_ = self.get_ivols(strike_reference=StrikeReference.CONVEXITY_MN,
                                  relative_strike=[-3, -2, -1, 0, 1, 2, 3])
