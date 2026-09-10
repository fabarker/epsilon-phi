from epsilonPhi.core.dataModel.dataSources.impliedVolatility.VolSurface import AbstractVolSurface
from epsilonPhi.core.dataModel.enums.ImpliedVolatility import *
from epsilonPhi.core.lib.smoothing.GaussianKernel import smooth_2d
from scipy.optimize import lsq_linear
from epsilonPhi.core.utils.DateUtils import DateUtils
from epsilonPhi.core.utils.OptionUtils import *
from epsilonPhi.core.utils.FrameUtils import FrameUtils
import numpy as np
from numba import njit
import pandas as pd
from scipy.stats import t as tstudent

class SmileStatArb(object):

    _DELTAS = [-0.10, -0.25, -0.5, -0.75, -0.90]
    #_DELTAS = [-0.1]
    _MATURITIES = np.array([1, 3, 6, 9, 12]) / 12


    def __init__(self, underlier, start_date=None, end_date=None):

        # Set the underlier in the object
        self._underlier = underlier

        self._vs = AbstractVolSurface.get_volatility_surface(self._underlier,
                                                             interpolation_method=Interpolator.CUBIC_SPLINE)

        # Set dates
        self.set_dates(start_date, end_date)
        self._ivols = None
        self._load_ivols()

        # Strategy Cache
        self._cache = {}
        self._parameters = {}

    def set_dates(self, start_date, end_date):
        self._start_date = start_date
        self._end_date = end_date
    @property
    def start_date(self):
        return self._start_date
    @property
    def end_date(self):
        return self._end_date
    @property
    def dates(self):
        return self._dates
    @property
    def cols(self):
        return self._I.columns
    @property
    def x_cols(self):
        return np.array(self.cols.get_level_values('x'))
    @property
    def t_cols(self):
        return np.array(self.cols.get_level_values('t'))
    @property
    def t(self):
        return self.t_cols.reshape(1, -1).repeat(self.T, axis=0)
    @property
    def x(self):
        return self.x_cols.reshape(1, -1).repeat(self.T, axis=0)
    @property
    def T(self):
        return len(self.dates)
    @property
    def N(self):
        return self._I.shape[1]
    @property
    def z(self):

        if hasattr(self, '_I') and not hasattr(self, '_z'):
           self.__load_z()
        elif not hasattr(self, '_z'):
           return None
        return self._z.copy()

    def __load_z(self):
        f = self._vs.get_forward_prices(self._I.date, self._I.t.values, True)
        zp_ = np.log(self._I.k / f.values) + 0.5 * np.power(self._I.mid, 2)
        self._z = zp_ / (self._I.mid * np.array(np.sqrt(self._I.t)))

    def _load_ivols(self):

        ivols = self.get_floating_strike_ivols()
        ivols['x'] = np.abs(ivols['x'])
        idxs = ivols.open == ivols.date
        self._I = ivols[idxs].set_index(['date', 'x', 't']).get('mid')

        # Store the market data
        self._market_data = ivols.copy()
        self._market_data['x'] = np.abs(self._market_data['x'])
        self._market_data['f'] = self._vs.get_forward_prices(self._market_data.date, np.array(self._market_data.t), True).values

        # Get dI
        It = ivols[ivols.open != ivols.date].sort_values('open').set_index(['k', 'expiry'])
        Ft = self._vs.get_forward_prices(It.date, np.array(It.t), True).values

        I0 = ivols[ivols.open == ivols.date].set_index(['k', 'expiry']).loc[It.index]
        F0 = self._vs.get_forward_prices(I0.date, np.array(I0.t), True).values

        assert np.all(It.date > I0.date), 'Error - dates not aligned correctly'
        assert np.all(It.index == I0.index), 'Error - index not matched'

        dI = pd.DataFrame(-1 + (It.get('mid').values / I0.get('mid').values),
                          index=[It.date, I0.x.values, I0.t.values], columns=['dI']).reset_index()
        dI.columns = ['date', 'x', 't', 'fy']
        dI['z'] = np.array(
            (np.log(np.array(I0.index.get_level_values('k')) / F0) + 0.5 * np.power(I0.mid, 2) * I0.t) / (
                        I0.mid * np.sqrt(I0.t)))
        dI['y'] = np.array(np.log(np.array(I0.t)))
        df_array = dI.set_index('date').get(['t', 'x', 'fy', 'z', 'y']).pivot(columns=['x', 't'])
        _dI = pd.DataFrame(self.smooth_dataframe(df_array),
                           index=df_array.index,
                           columns=df_array.get('fy').columns).stack(level=[0, 1])

        _dF = pd.Series(-1 + (Ft / F0), index=[It.date, I0.x, I0.t])

        common_dates = self._I.index.intersection(_dI.index)

        self._I = self._I.loc[common_dates].unstack(level=[1, 2]).sort_index()
        self._F = self._market_data[['f', 'date', 't', 'x']].set_index(['date', 'x', 't']).loc[common_dates].unstack(level=[1, 2]).sort_index().get('f')
        self._dI = _dI.loc[common_dates].unstack(level=[1,2]).sort_index()
        self._dF = _dF.loc[common_dates].unstack(level=[1,2]).sort_index()

        # Set dates in object
        self._dates = pd.to_datetime(self._I.index)

    def load_ks(self):
        self._k = pd.DataFrame(black_strike(1,
                              self.t_cols[None, :].repeat(len(self.dates), 0),
                              self.get_rd().values,
                              -1,
                              -1 * self.x_cols[None, :].repeat(len(self.dates), 0),
                              2,
                              self.get_I().values), index=self.dates, columns=self.cols)


    def shift_dates(self, dates, periods=1):
        return DateUtils.shift_dates_in_range(np.array(dates),
                                              np.unique(dates),
                                              periods)

    def get_fixed_strike_and_maturity_ivols(self, dates, strikes, maturities):
        return self._vs.get_ivols(dates, StrikeReference.STRIKE_PRICE, strikes, maturities, MaturityType.YEARFRAC)

    def get_floating_strike_ivols(self, dates=None):

        if dates is None:
           dates = self._vs.common_dates

        locs = ((dates >= pd.to_datetime(self._start_date)) & (dates <= pd.to_datetime(self._end_date)))
        vols = self._vs.get_ivols(dates[locs],
                                  StrikeReference.DELTA,
                                  self._DELTAS,
                                  self._MATURITIES,
                                  MaturityType.YEARFRAC).dropna()

        capped_pricing_dates = self._vs.interpolator.get_expiry_dates_from_settlement_dates_tenor(vols.date, '1d')
        _dates, k, open_date, expiry_date, t = self._vs.get_contract_pricing_dates(vols.date, vols.expiry, vols.k,
                                                                                  capped_pricing_dates)

        idx = (_dates >= self._start_date) & (_dates <= self._end_date) & (_dates != open_date)
        i_df = self._vs.get_ivols(_dates[idx], StrikeReference.STRIKE_PRICE, k[idx], expiry_date[idx], MaturityType.EXPIRY_DATE)

        ivols = pd.concat((vols, i_df), axis=0).reset_index(drop=True)
        sorted_ivols = ivols.sort_values(['expiry', 'k', 'date']).reset_index(drop=True)

        # sorted_ivols['mid'] = sorted_ivols.groupby(['expiry', 'k'], group_keys=False, dropna=False).apply(lambda x: x.mid.ffill())
        sorted_ivols['open'] = sorted_ivols.groupby(['expiry', 'k'])['date'].transform('first')
        return sorted_ivols

    def smooth_dataframe(self, df):

        x = np.asarray(df.get('z'), dtype=np.float64)
        y = np.asarray(df.get('y'), dtype=np.float64)
        z = np.asarray(df.get('fy'), dtype=np.float64)

        @njit
        def smooth(x, y, z):

            res = np.empty(shape=z.shape, dtype=np.float64)
            for t in range(res.shape[0]):
                for k in range(res.shape[1]):
                    res[t, k] =  smooth_2d(x[t, :], y[t, :], x[t, k], y[t, k], z[t, :])
            return res
        return smooth(x, y, z)

    def get_k(self):
        if not hasattr(self, '_k'):
           self.load_ks()
        return self._k.loc[self.dates].get(self.cols)

    def get_S(self):
        return self._vs.get_spot_prices(self.dates).to_frame(self._underlier)

    def get_S_prime(self):
        return pd.concat([self.get_S()] * self.N, axis=1) / self.get_S().values

    def get_dS(self):
        return np.log(self._vs.spot_prices).diff().loc[self.dates].to_frame(self._underlier)

    def get_dSdS(self):
        return np.power(self.get_dS(), 2)

    def get_rf(self):
        return self._vs.get_funding_rate(self.dates, np.unique(self.t_cols)).get(self.t_cols)

    def get_rd(self):
        return self._vs.get_risk_free(self.dates, np.unique(self.t_cols)).get(self.t_cols)

    def get_F(self):
        return self._F.loc[self.dates][self.cols]

    def get_F_prime(self):
        df = self.get_S().values * (np.exp(-1 * self.get_rf() * self.t) /
                                    np.exp(-1 * self.get_rd() * self.t))
        return df / df.values

    def get_dF(self):
        return self._dF.loc[self.dates].get(self.cols)

    def get_dFdF(self):
        return np.power(self.get_dF(), 2)

    def get_I(self):
        if self._I is None:
           self._load_ivols()
        return self._I.loc[self.dates].get(self.cols)

    def get_dI(self):
        if self._dI is None:
            self._load_ivols()
        return self._dI.loc[self.dates].get(self.cols)

    def get_dFdI(self):
        if self._dI is None:
            self._load_ivols()
        return self.get_dI() * self.get_dF().values
    def get_slope(self):
        return self.get_I().get(0.5).get(1) - self.get_I().get(0.5).get(1/12)

    def get_skew(self):
        return self.get_I().get(0.9).get(3/12) - self.get_I().get(0.1).get(3/12)

    def get_dIdI(self):
        return np.power(self.get_dI(), 2).get(self.cols)

    def get_nu(self, period=21):
        return self.get_dFdF().rolling(window=period).mean() * 252

    def get_mu(self, period=21):
        return self.get_dI().rolling(window=period).mean().get(self.cols) * 252

    def get_gamma(self, period=21):
        return self.get_dFdI().mul(252).rolling(window=period).mean().get(self.cols)

    def get_omega(self, period=21):
        return self.get_dIdI().mul(252).rolling(window=period).mean().get(self.cols)

    def get_rho(self, period=252):
        return self.get_gamma(period) / self.get_sqrt_nu_omega(period)

    def get_rho_hat(self, period=252):
        return self.get_rho(period).mean(axis=1).to_frame()

    def get_sqrt_nu_omega(self, period=21):
        return np.sqrt(self.get_nu(period).values * self.get_omega(period))

    def black_delta(self, option_type=-1):
        return black_delta(1,
                           self.t,
                           self.get_k().values,
                           self.get_rf().values,
                           self.get_I(),
                           2,
                           option_type)
    def black_gamma(self):
        return black_gamma(1,
                           self.t,
                           self.get_k().values,
                           self.get_rd().values,
                           self.get_I())

    def cash_gamma(self):
        return self.get_I() * self.black_gamma().get(self.cols)

    def black_theta(self, option_type=-1):
        return black_theta(1,
                           self.t,
                           self.get_k().values,
                           self.get_rd().values,
                           self.get_rf().values,
                           self.get_I(),
                           option_type)

    def black_vega(self):
        return black_vega(1,
                          self.t,
                          self.get_k().values,
                          self.get_rd().values,
                          self.get_I())

    def black_vanna(self):
        return black_vanna(1,
                          self.t,
                          self.get_k().values,
                          self.get_rf().values,
                          self.get_I())

    def black_volga(self):
        return black_volga(1,
                           self.t,
                           self.get_k().values,
                           self.get_rd().values,
                           self.get_I())

    def get_X1(self, period=21):
        return self.black_vega() * self.get_I() * self.get_omega(period)

    def get_X2(self, period=21):
        return self.black_vega() * self.get_I() * self.get_mu(period)

    def get_X3(self, period=21):
        return (0.5 * self.black_gamma() * np.power(self.get_F_prime(), 2).values * self.get_nu(period).values)

    def get_X4(self, period=21):
        return (0.5 * self.black_volga() * np.power(self.get_I(), 2) * self.get_omega(period))

    def get_X5(self, period=21):
        return (self.black_vanna() * self.get_F_prime().values * self.get_I() * self.get_sqrt_nu_omega(period))

    def get_Y(self):
        return -1 * self.black_theta(-1)

    def get_independent_variables(self, period):

        g = self.cash_gamma()
        X1 = FrameUtils.set_levels(self.get_X1(period), 'vega', 'var')
        X2 = FrameUtils.set_levels(self.get_X2(period), 'mu', 'var')
        X3 = FrameUtils.set_levels(self.get_X3(period), 'gamma', 'var')
        X4 = FrameUtils.set_levels(self.get_X4(period), 'volga', 'var')
        X5 = FrameUtils.set_levels(self.get_X5(period), 'vanna', 'var')

        return pd.concat([X1, X2, X3, X4, X5], axis=1).swaplevel(2, 0, axis=1)


    def estimate_market_price_of_risk(self, period=21):

        Y_ = self.get_Y()
        X1 = self.get_X1(period).get(Y_.columns).reindex(Y_.index).values
        X2 = self.get_X2(period).get(Y_.columns).reindex(Y_.index).values
        X3 = self.get_X3(period).get(Y_.columns).reindex(Y_.index).values
        X4 = self.get_X4(period).get(Y_.columns).reindex(Y_.index).values
        X5 = self.get_X5(period).get(Y_.columns).reindex(Y_.index).values

        g = self.cash_gamma().get(Y_.columns).reindex(Y_.index)

        # Thursday 9th Feb 2017
        def estimate_betas(y, x1, x2, x3, x4, x5):

            T, K = y.shape
            b = np.full((T, 5), np.nan)
            r = np.full((T, K), np.nan)
            p = np.full((T, 5), np.nan)
            c = np.full((T, 5), np.nan)
            q = np.full((T, 1), np.nan)

            for t in range(T):

                theta = y[t, :]
                X = np.column_stack((x1[t, :],
                                     x2[t, :],
                                     x3[t, :],
                                     x4[t, :],
                                     x5[t, :]
                                     ))


                XtX = np.dot(X.T, X)
                XtY = np.dot(X.T, theta)
                YtY = np.dot(theta.T, theta)

                betas = np.linalg.solve(XtX, XtY)
                residuals = theta - np.dot(X, betas)

                b[t,:] = betas
                r[t, :] = residuals
                q[t] = 1 - (np.dot(residuals.T, residuals) / YtY)

                dof = X.shape[0] - X.shape[1]
                cov_matrix = np.linalg.inv(XtX) * np.dot(residuals.T, residuals) / dof
                tstat = (betas - 0.5) / np.sqrt(np.diag(cov_matrix))

                p[t, :] = 2 * tstudent.sf(np.abs(tstat), dof)
                c[t, :] = (betas * np.dot(XtX, betas)) / YtY
            return b, r, p, c, q

        b, r, p, c, q = estimate_betas(Y_.values, X1, X2, X3, X4, X5)

        e = r / g
        b_df = pd.DataFrame(b, index=e.index, columns=['vega', 'trend', 'gamma', 'volga', 'vanna'])
        c_df = pd.DataFrame(c, index=e.index, columns=['vega', 'trend', 'gamma', 'volga', 'vanna'])
        p_df = pd.DataFrame(p, index=e.index, columns=['vega', 'trend', 'gamma', 'volga', 'vanna'])
        q_df = pd.DataFrame(q, index=e.index, columns=['rsq'])
        r_df = pd.DataFrame(r, index=e.index, columns=e.columns)

        beta_0 = pd.concat([self.get_rho_hat() * 0,
                  self.get_rho_hat() * 0,
                  self.get_rho_hat() / self.get_rho_hat(),
                  self.get_rho_hat() / self.get_rho_hat(),
                  self.get_rho_hat()], axis=1)

        rp = b_df - beta_0.reindex(b_df.index).values
        return rp, r_df, e, b_df, c_df, p_df, q_df

    def get_sig(self, _HOLDING_DAYS):
        ED = np.max(self.get_open_dates(_HOLDING_DAYS))
        I_stacked = self.get_I().loc[:ED].stack([0, 1])
        I_stacked.index.names = ['date', 'x', 't']
        return I_stacked

    def get_strikes(self, _HOLDING_DAYS=21):
        res = nb_strike(self.get_S().values.repeat(self.N, 1),
                        self.t,
                        self.get_rd().values,
                        self.get_rf().values, -1, -1 * self.x,
                       2,
                        self.get_I().values)

        ED = self.get_last_open_date(_HOLDING_DAYS)
        K_stacked = pd.DataFrame(res, index=self.dates, columns=self.cols).loc[:ED].stack([0, 1])
        K_stacked.index.names = ['date', 'x', 't']
        return K_stacked

    def get_open_dates(self, _HOLDING_DAYS=21):
        return self.dates[:-_HOLDING_DAYS]

    def get_open_date_paths(self, _HOLDING_DAYS=21):
        return np.array(self.get_open_vols(_HOLDING_DAYS).index.get_level_values('date'))[:, None]

    def get_open_maturities(self, _HOLDING_DAYS=21):
        return np.array(self.get_open_vols(_HOLDING_DAYS).index.get_level_values('t'))[:, None]

    def get_open_vols(self, _HOLDING_DAYS=21):
        return self.get_sig(_HOLDING_DAYS)

    def get_open_strikes(self, _HOLDING_DAYS=21):
        ED = np.max(self.get_open_dates(_HOLDING_DAYS))
        k = self.get_k().loc[:ED].stack([0, 1])
        k.index.names = ['date', 'x', 't']
        return k

    def get_last_open_date(self, _HOLDING_DAYS):
        return np.max(self.get_open_dates(_HOLDING_DAYS))

    def get_path_idxs(self, _HOLDING_DAYS=21):

        _sig = self.get_sig(_HOLDING_DAYS)
        _dates = np.array(_sig.index.get_level_values(0))

        ODs = self.get_open_dates(_HOLDING_DAYS)
        idx = DateUtils.find_first_date_loc(ODs, _dates).reshape(-1, 1)
        return np.arange(0, _HOLDING_DAYS + 1, 1).reshape(1, -1).repeat(len(idx), 0) + idx

    def get_pricing_date_paths(self,  _HOLDING_DAYS):
        idx = self.get_path_idxs(_HOLDING_DAYS)
        return pd.DataFrame(self.dates.values[idx],
                            index=self.get_sig(_HOLDING_DAYS).index)


    def get_s_paths(self, _HOLDING_DAYS):
        idx = self.get_path_idxs(_HOLDING_DAYS)
        return pd.DataFrame(self.get_S().values.flatten()[idx],
                            index=self.get_sig(_HOLDING_DAYS).index)

    def get_maturity_paths(self, _HOLDING_DAYS):
        paths = self.get_pricing_date_paths(_HOLDING_DAYS)
        ED = self._market_data.set_index(['date', 'x', 't']).loc[paths.index].get('expiry')
        mat_paths = pd.concat(([ED] * (_HOLDING_DAYS + 1)), axis=1)
        mat_paths.columns = range(_HOLDING_DAYS+1)
        return mat_paths

    def get_ttm_paths(self, _HOLDING_DAYS):

        if ('t', _HOLDING_DAYS) not in self._cache.keys():
            paths = self.get_pricing_date_paths(_HOLDING_DAYS)
            ED = self._market_data.set_index(['date', 'x', 't']).loc[paths.index].get('expiry')
            self._cache[('t', _HOLDING_DAYS)] = pd.DataFrame(np.asarray((ED.values.reshape(-1, 1) - paths) / pd.to_timedelta(1, 'd')).astype(np.int32) / 365,
                                                             index=self.get_sig(_HOLDING_DAYS).index)
        return self._cache[('t', _HOLDING_DAYS)]

    def get_rf_paths(self, _HOLDING_DAYS):

        if ('rf', _HOLDING_DAYS) not in self._cache.keys():

            p = self.get_pricing_date_paths(_HOLDING_DAYS)
            m = self.get_ttm_paths(_HOLDING_DAYS).values
            rf = self._vs.get_risk_free(p.values.flatten(), m.flatten(), True)
            self._cache[('rf', _HOLDING_DAYS)] = pd.DataFrame(rf.values.reshape(p.shape), index=p.index)
        return self._cache[('rf', _HOLDING_DAYS)]

    def get_rd_paths(self, _HOLDING_DAYS):

        if ('rd', _HOLDING_DAYS) not in self._cache.keys():

            p = self.get_pricing_date_paths(_HOLDING_DAYS)
            m = self.get_ttm_paths(_HOLDING_DAYS).values
            rd = self._vs.get_funding_rate(p.values.flatten(), m.flatten(), True)
            self._cache[('rd', _HOLDING_DAYS)] = pd.DataFrame(rd.values.reshape(p.shape), index=p.index)
        return self._cache[('rd', _HOLDING_DAYS)]

    def get_f_paths(self, _HOLDING_DAYS):

        if ('f', _HOLDING_DAYS) not in self._cache.keys():
            _rf = self.get_rf_paths(_HOLDING_DAYS)
            _rd = self.get_rd_paths(_HOLDING_DAYS)
            _t = self.get_ttm_paths(_HOLDING_DAYS)
            _s = self.get_s_paths(_HOLDING_DAYS)
            _s_re = _s / _s.values[:,0].reshape(-1, 1)

            t1 = _s_re * np.exp(-_rf * _t) / np.exp(-_rd * _t)
            self._cache[('f', _HOLDING_DAYS)] = t1 / t1.values[:,0].reshape(-1, 1)
            #self._cache[('f', _HOLDING_DAYS)] = t1
        return self._cache[('f', _HOLDING_DAYS)]

    def get_k_paths(self, _HOLDING_DAYS):
        k = self.get_open_strikes(_HOLDING_DAYS)
        k_ = k.loc[self.get_pricing_date_paths(_HOLDING_DAYS).index]
        return pd.concat([k_] * (_HOLDING_DAYS + 1), axis=1)

    def get_sig_paths(self, _HOLDING_DAYS):

        if ('v', _HOLDING_DAYS) not in self._cache.keys():

            p = self.get_pricing_date_paths(_HOLDING_DAYS).values.flatten()
            k = pd.concat([self.get_strikes(_HOLDING_DAYS)] * (_HOLDING_DAYS + 1), axis=1)
            t = self.get_maturity_paths(_HOLDING_DAYS)

            sig_fit = self._vs.get_ivols(p, StrikeReference.STRIKE_PRICE, np.ravel(k.values), np.ravel(t), MaturityType.EXPIRY_DATE)

            self._cache[('v', _HOLDING_DAYS)] = pd.DataFrame(sig_fit['mid'].values.reshape(k.shape), index=k.index)
        return self._cache[('v', _HOLDING_DAYS)]

    def get_option_price_paths(self, _HOLDING_DAYS, option_type=-1):

        if ('o', _HOLDING_DAYS) not in self._cache.keys():

            f = self.get_f_paths(_HOLDING_DAYS).values
            k = self.get_k_paths(_HOLDING_DAYS).values
            t = self.get_ttm_paths(_HOLDING_DAYS).values
            r = self.get_rd_paths(_HOLDING_DAYS).values
            v = self.get_sig_paths(_HOLDING_DAYS)

            bsprice = blsprice(f,
                               t,
                               k,
                               r,
                               v.values,
                               option_type)

            locs = t == 0
            intr_val = (option_type * (f - k)).clip(0)
            bsprice[locs] = intr_val[locs]

            self._cache[('o', _HOLDING_DAYS)] = pd.DataFrame(bsprice, index=v.index)
        return self._cache[('o', _HOLDING_DAYS)]

    def get_option_pnls(self, _HOLDING_DAYS, option_type=-1, position=-1):
        prices = self.get_option_price_paths(_HOLDING_DAYS, option_type)
        return position * prices.diff(axis=1).dropna(axis=1, how='all')

    def get_account_pnl(self, _HOLDING_DAYS, option_type=-1, position=-1):
        prices = self.get_option_price_paths(_HOLDING_DAYS, option_type)
        premium = -1 * position * prices.get(0).to_frame()

        pricing_dates = self.get_pricing_date_paths(_HOLDING_DAYS)
        term = ((pricing_dates.iloc[:, -1] - pricing_dates.iloc[:, 0]) /
                np.timedelta64(1, 'D') / 365)

        rd = self._vs.interpolator.get_rd(pricing_dates.iloc[:, 0].values, term.values, True)
        daily_return = (-1 + np.exp(rd.values[:, None] * term.values[:, None]))
        return pd.DataFrame((1/_HOLDING_DAYS) * premium.values.repeat(_HOLDING_DAYS, 1)
                            * daily_return, index=premium.index)

    def get_delta_hedged_pnl(self, _HOLDING_DAYS, option_type=-1, position=-1):

        option_deltas = self.get_delta_paths(_HOLDING_DAYS, option_type)
        position_deltas = position * option_deltas

        ds = self.get_f_paths(_HOLDING_DAYS).diff(axis=1).dropna(axis=1)
        return -1 * position_deltas.iloc[:, 0:-1].values * ds

    def get_delta_hedged_option_pnls(self, _HOLDING_DAYS, option_type=-1, position=-1):
        contract_pnls = self.get_option_pnls(_HOLDING_DAYS, option_type, position)
        hedge_pnls = self.get_delta_hedged_pnl(_HOLDING_DAYS, position)
        balance_pnls = self.get_account_pnl(_HOLDING_DAYS, option_type, position)
        #return contract_pnls + hedge_pnls + balance_pnls.values
        return contract_pnls + hedge_pnls

    def get_delta_paths(self, _HOLDING_DAYS, option_type=-1):

        if ('d', _HOLDING_DAYS) not in self._cache.keys():

            f = self.get_f_paths(_HOLDING_DAYS).values
            k = self.get_k_paths(_HOLDING_DAYS).values
            t = self.get_ttm_paths(_HOLDING_DAYS).values
            r = self.get_rf_paths(_HOLDING_DAYS).values
            v = self.get_sig_paths(_HOLDING_DAYS)

            bsdelta = black_delta(f,
                                  t,
                                  k,
                                  r,
                                  v.values,
                                  2,
                                  option_type)

            self._cache[('d', _HOLDING_DAYS)] = pd.DataFrame(bsdelta, index=v.index)
        return self._cache[('d', _HOLDING_DAYS)]

    def get_sensitivity_matrix(self, period=21):
        X = self.get_independent_variables(period)
        _, res, e, _, _, _, _= self.estimate_market_price_of_risk(21)
        _r = FrameUtils.set_levels(e, 'e', 'var')
        return pd.concat((X, _r.swaplevel(2, 0, axis=1)), axis=1)

    def get_factor_model_pricing_residuals(self):
        _, res, _, _, _, _, _ = self.estimate_market_price_of_risk(21)
        return res

    def get_static_strategy_pnls(self, estimation_period=21, holding_period=1, strategy_type=5):

        _pnls = self.get_delta_hedged_option_pnls(_HOLDING_DAYS=holding_period)
        _wts = self.get_strategy_weights(estimation_period, strategy_type)

        _common = _pnls.index.intersection(_wts.index)
        pnls = _pnls.loc[_common].values * _wts.loc[_common]
        return pnls.groupby('date', axis=0).sum()

    def get_strategy_weights(self, period=21, type=5):

        risks = ['vega', 'mu', 'gamma', 'volga', 'vanna', 'e']
        H = self.get_sensitivity_matrix(period)

        d = np.zeros((6, 1))
        d[type] = 1

        T = H.shape[0]
        wts = np.empty((self.N, T))
        for t in range(T):
            h_bar = H.iloc[t, :].unstack(level=0)[risks]
            try:
                wts[:, t] = (np.dot(h_bar, np.linalg.inv(np.dot(h_bar.T, h_bar))) @ d).flat


            except:
                pass

        wts_1 = pd.DataFrame(wts, index=h_bar.index, columns=H.index).T
        wts_2 = wts_1 / wts_1.abs().sum(axis=1).values.reshape(-1, 1)
        return pd.concat((wts_1.stack(level=[1, 0]).to_frame('fixed'),
                          wts_2.stack(level=[1, 0]).to_frame('dollar')), axis=1)

    def get_pca_loadings(self, type='level'):
        if type == 'level':
            I = np.log(self.get_I())
        else:
            I = self.get_dI()

        cov_mat = I.cov()
        L, V = np.linalg.eig(cov_mat)
        return pd.DataFrame(V, columns=L, index=I.columns).sort_index(axis=1, ascending=False)

    def get_principle_components(self, type='level'):

        if type == 'level':
            I = np.log(self.get_I())
        else:
            I = self.get_dI()

        L = self.get_pca_loadings(type)
        eigvecs = pd.concat([pd.DataFrame(I.values @ L.values[:, x]) for x in np.array(range(L.shape[1]))], axis=1)
        eigvecs.columns = np.array(range(L.shape[1]))
        return eigvecs


    def extract_common_risk_neutral_moments(self):

        I = self.get_I()
        F = self.get_F()
        K = pd.DataFrame(black_strike(F.values, self.t, self.get_rf().values, -1, -1 * self.x,
                          2, I.values), index = I.index, columns=I.columns)

        zp = np.log(K / F.values) + self.t * np.power(I, 2)
        zm = np.log(K / F.values) - self.t * np.power(I, 2)

        X1 = 2 * np.power(I, 2) * self.t
        X2 = I/I.values
        X3 = 2 * zp
        X4 = zp * zm
        y = np.power(I, 2)

        N = y.shape[0]
        B = np.full((N, 4), np.nan)

        for t in range(N):
            y_ = y.values[t].reshape(-1, 1)

            x1 = X1.values[t]
            x2 = X2.values[t]
            x3 = X3.values[t]
            x4 = X4.values[t]

            Xmat = np.column_stack((x1, x2, x3, x4))

            XtX = np.dot(Xmat.T, Xmat)
            #XtY = np.dot(Xmat.T, y_)
            YtY = np.dot(y_.T, y_)

            result = lsq_linear(Xmat,
                                y_.flatten(),
                                bounds=( [-np.inf, 0, -np.inf, 0],  [np.inf, np.inf, np.inf, np.inf]), lsmr_maxiter=None,
                                method='trf', lsq_solver=None, lsmr_tol=None, max_iter=200, tol=1e-8)

            #betas = np.linalg.solve(XtX, XtY)
            B[t, :] = result.x

        return pd.DataFrame(B, columns=['vega', 'gamma', 'vanna', 'volga'], index=y.index)




if __name__ == "__main__":

    import pandas as pd
    import numpy as np
    from epsilonPhi.core.utils.DateUtils import DateUtils

    SD = '24-Jan-1996'
    ED = '13-Oct-2022'

    self = SmileStatArb('GBPUSD', SD, ED)
    rp, r_df, e, b_df, c_df, p_df, q_df = self.estimate_market_price_of_risk(period=21)
    wts = self.get_strategy_weights(21, 5)


