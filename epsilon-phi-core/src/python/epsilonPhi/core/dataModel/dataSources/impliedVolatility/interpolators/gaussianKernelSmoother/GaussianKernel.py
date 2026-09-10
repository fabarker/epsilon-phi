import numpy as np
import pandas as pd
from epsilonPhi.core.utils.FrameUtils import FrameUtils
from numba import jit
from tqdm import tqdm
from epsilonPhi.core.utils.OptionUtils import *
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

        _target_t = maturities.flatten().astype(float)
        _target_k = strikes.flatten().astype(float)
        _target_d = dates.flatten()

        _unique_d = np.unique(_target_d)
        ND = len(_unique_d.flatten())

        _t_vec = self.t.astype(float).flatten()
        _x_vec = self.x.astype(float).flatten()
        _v_vec = self.sig.astype(float).flatten()
        _d_vec = self.dates

        interps = list()
        for t in tqdm(range(ND), desc="Interpolating Vol Surface"):

            idx_T = _unique_d[t] == _target_d
            idx_S = _unique_d[t] == _d_vec

            interp = self.interpolate_single_date(_t_vec[idx_S],
                                                  _target_t[idx_T],
                                                  _x_vec[idx_S],
                                                  _target_k[idx_T],
                                                  _v_vec[idx_S])

            result = np.column_stack((interp, _target_d[idx_T].astype(float)))
            interps.extend([result])


        # build ivols
        _ivols = pd.DataFrame(np.vstack((interps)))
        _ivols.columns = ['sig', 'x', 't', 'date']
        _ivols.loc[:, 'date'] = pd.to_datetime(_ivols.get('date'))
        _vols = _ivols.set_index(['date','x','t'], drop=True)
        return _vols.copy()

    @staticmethod
    #@jit(nopython=True, fastmath=True, cache=True)
    def interpolate_single_date(mat, unique_m, x, unique_x, mids):

        # Underscore is what we have already
        _t = np.log(mat).reshape((-1, 1))
        _x = x.reshape((-1, 1))

        t = np.log(unique_m.reshape((1, -1)))
        x = unique_x.reshape((1, -1))

        # get the std of the reference on each date
        sig_x = np.std(_x)
        h_x = 1.0 * np.power(4.0 / 3.0, 1.0 / 5.0) * sig_x / np.power(len(mids), 1.0 / 5.0)

        # do the same in the maturity direction
        sig_lm = _t.std()
        h_m = 2.0 * np.power(4.0 / 3.0, 1.0 / 5.0) * sig_lm / np.power(len(mids), 1.0 / 5.0) * 0.1

        m_diff = (_t - t) / h_m
        x_diff = (_x - x) / h_x

        x_wts = np.exp(-1.0 * np.power(x_diff, 2.0) / 2.0)
        m_wts = np.exp(-1.0 * np.power(m_diff, 2.0) / 2.0)

        wts = (x_wts * m_wts) / np.sum(x_wts * m_wts, axis=0)
        ivols = np.sum(wts * mids.reshape((-1, 1)), axis=0)
        return np.vstack((ivols, unique_x, unique_m)).T


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
