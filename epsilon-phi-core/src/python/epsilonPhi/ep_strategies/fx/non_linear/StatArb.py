from epsilonPhi.core.dataModel.dataSources.impliedVolatility.VolSurface import AbstractVolSurface
from epsilonPhi.core.dataModel.enums.ImpliedVolatility import *
from epsilonPhi.core.lib.smoothing.GaussianKernel import smooth_2d
from epsilonPhi.core.utils.DateUtils import DateUtils
from epsilonPhi.core.utils.OptionUtils import *
import numpy as np
from numba import njit
import pandas as pd
from scipy.stats import t as tstudent

class SmileStatArb(object):
    _DELTAS = [-0.1, -0.25, -0.5, -0.75, -0.9]
    _MATURITIES = np.array([1, 3, 6, 9, 12]) / 12

    def __init__(self, underlier, start_date=None, end_date=None):

        # Set the underlier in the object
        self._underlier = underlier

        # Properties of the strategy class
        self._vs = AbstractVolSurface(self._underlier,
                   strike_reference=StrikeReference.DELTA,
                   interpolation_method=Interpolator.CUBIC_SPLINE)

        # Set dates
        self.set_dates(start_date, end_date)
        self._ivols = None
        self._load_ivols()

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
        f = self._vs.interpolator.get_f(self._I.date, self._I.t, True)
        zp_ = np.log(self._I.k / f.values) + 0.5 * np.power(self._I.mid, 2)
        self._z = zp_ / (self._I.mid * np.array(np.sqrt(self._I.t)))

    def _load_ivols(self):

        self._I = self.get_floating_strike_ivols()

        # Get the fixed strike-expiry contract ivol t+1
        t = self.shift_dates(self._I.date)
        ttm = self._I.t - DateUtils.get_date_delta(self._I.date, t, True)
        iv_t = self.get_fixed_strike_and_maturity_ivols(t, self._I.k, ttm)

        # Get fixed strike, expiry dI.
        dI = self._I[['date','t', 'x']].set_index('date')
        dI['fy'] = np.array(np.log(iv_t.mid) - np.log(self._I.mid))
        dI['z'] = self.z.values
        dI['y'] = np.log(np.array(self._I.t))
        _dI = dI.pivot(columns=['x', 't'])

        _dI_S = self.smooth_dataframe(_dI)
        self._dI = pd.DataFrame(_dI_S, columns=_dI.get('z').columns, index=_dI.index).replace(0, np.nan)
        self._dI = self._dI.dropna(axis=0).stack(level=[0, 1]).to_frame('dI').reset_index(drop=False)
        self._I = self._I[self._I.date.isin(self._dI.date)]

        self._dI['x'] = np.abs(self._dI['x'])
        self._I['x'] = np.abs(self._I['x'])

        self._dI = self._dI.set_index('date').pivot(columns=['x', 't']).get('dI')
        self._I = self._I.set_index('date').pivot(columns=['x', 't']).get('mid')

        # Set dates in object
        self._dates = pd.to_datetime(np.unique(self._dI.index.values))

    def load_ks(self):
        _k = nb_strike(self.get_S_prime().values,
                       self.t,
                       self.get_rd().values,
                       self.get_rf().values,
                       -1,
                       -1 * self.x,
                       2,
                       self.get_I().values)
        self._k = pd.DataFrame(_k, columns=self.cols, index=self.dates)

    def shift_dates(self, dates, periods=1):
        return DateUtils.shift_dates_in_range(np.array(dates),
                                              np.unique(dates),
                                              periods)

    def get_fixed_strike_and_maturity_ivols(self, dates, strikes, maturities):
        return self._vs.get_ivols(strikes, maturities, MaturityType.YEARFRAC, dates, StrikeReference.STRIKE_PRICE)

    def get_floating_strike_ivols(self, dates=None):
        return self._vs.get_ivols(self._DELTAS, self._MATURITIES, MaturityType.YEARFRAC, dates)

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

    def get_sig(self, _HOLDING_DAYS):
        ED = np.max(self.get_open_dates(_HOLDING_DAYS))
        return self.get_I().loc[:ED].stack([0, 1])

    def get_k(self):
        if not hasattr(self, '_k'):
           self.load_ks()
        return self._k.get(self.cols)

    def get_S(self):
        return self._vs.get_spot_prices(self.dates).to_frame(self._underlier)

    def get_S_prime(self):
        return pd.concat([self.get_S()] * self.N, axis=1) / self.get_S().values

    def get_dS(self):
        return np.log(self._vs.spot_prices).diff().shift(-1).loc[self.dates].to_frame(self._underlier)

    def get_dSdS(self):
        return np.power(self.get_dS(), 2)

    def get_rf(self):
        return self._vs.get_funding_rate(self.dates, np.unique(self.t_cols)).get(self.t_cols)

    def get_rd(self):
        return self._vs.get_risk_free(self.dates, np.unique(self.t_cols)).get(self.t_cols)

    def get_F(self):
        return self._vs.interpolator.get_f(np.unique(self._vs.dates),
                                           np.unique(self.t)).loc[self.dates][self.t_cols]

    def get_F_prime(self):
        return (np.exp(-self.t*self.get_rf()) / np.exp(-self.t*self.get_rd()))

    def get_dF(self):
        if not hasattr(self, '_dF'):
            f0 = self.get_F().iloc[:, ~self.get_F().columns.duplicated()].unstack()
            t = self.shift_dates(f0.index.get_level_values(1))
            ttm = np.array(f0.index.get_level_values(0)) - DateUtils.get_date_delta(f0.index.get_level_values(1), t, True)
            fT = self._vs.interpolator.get_f(t, ttm, True)
            self._dF = (np.log(fT).values - np.log(f0)).unstack(level=0)
        return self._dF.loc[self.dates].get(self.t_cols)

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

    def bsm_delta(self, option_type=-1):
        return nb_delta(self.get_S_prime().values,
                        self.t,
                        self.get_k().values,
                        self.get_rd().values,
                        self.get_rf().values,
                        self._I,
                        2,
                        option_type)

    def black_delta(self, option_type=-1):
        return black_delta(self.get_F_prime().values,
                           self.t,
                           self.get_k().values,
                           self.get_rf().values,
                           self._I,
                           2,
                           option_type)

    def delta_bump(self, option_type=-1):
        return delta_bump(self.get_F_prime().values,
                          self.t,
                          self.get_k().values,
                          self.get_rf().values,
                          self._I,
                          option_type)

    def bsm_gamma(self):
        return bs_gamma(self.get_S_prime().values,
                        self.t,
                        self.get_k().values,
                        self.get_rd().values,
                        self.get_rf().values,
                        self.get_I())

    def black_gamma(self):
        return black_gamma(self.get_F_prime().values,
                           self.t,
                           self.get_k().values,
                           self.get_rd().values,
                           self._I)

    def cash_gamma(self):
        return np.power(self.get_F_prime().values, 2) * self.get_I() * self.black_gamma().get(self.cols)

    def gamma_bump(self, option_type=-1):
        return gamma_bump(self.get_F_prime().values,
                          self.t,
                          self.get_k().values,
                          self.get_rf().values,
                          self._I,
                          -2,
                          option_type)

    def bsm_theta(self, option_type=-1):
        return bs_theta(self.get_S_prime().values,
                        self.t,
                        self.get_k().values,
                        self.get_rd().values,
                        self.get_rf().values,
                        self.get_I(),
                        option_type)

    def black_theta(self, option_type=-1):
        return black_theta(self.get_F_prime().values,
                           self.t,
                           self.get_k().values,
                           self.get_rd().values,
                           self.get_rf().values,
                           self.get_I(),
                           option_type)

    def theta_bump(self, option_type=-1):
        return theta_bump(self.get_F_prime().values,
                          self.t,
                          self.get_k().values,
                          self.get_rf().values,
                          self.get_I(),
                          option_type)

    def bsm_vega(self):
        return bs_vega(self.get_S_prime().values,
                       self.t,
                       self.get_k().values,
                       self.get_rd().values,
                       self.get_rf().values,
                       self.get_I())

    def black_vega(self):
        return black_vega(self.get_F_prime().values,
                          self.t,
                          self.get_k().values,
                          self.get_rd().values,
                          self.get_I())

    def vega_bump(self, option_type=-1):
        return vega_bump(self.get_F_prime().values,
                          self.t,
                          self.get_k().values,
                          self.get_rf().values,
                          self.get_I(),
                          option_type)

    def bsm_vanna(self):
        return bs_vanna(self.get_S_prime().values,
                        self.t,
                        self.get_k().values,
                        self.get_rd().values,
                        self.get_rf().values,
                        self.get_I())

    def black_vanna(self):
        return black_vanna(self.get_F_prime().values,
                          self.t,
                          self.get_k().values,
                          self.get_rf().values,
                          self.get_I())

    def vanna_bump(self):
        return vanna_bump(self.get_F_prime().values,
                          self.t,
                          self.get_k().values,
                          self.get_rf().values,
                          self.get_I())

    def bsm_volga(self):
        return bs_volga(self.get_S_prime().values,
                        self.t,
                        self.get_k().values,
                        self.get_rd().values,
                        self.get_rf().values,
                        self.get_I())

    def black_volga(self):
        return black_volga(self.get_F_prime().values,
                          self.t,
                          self.get_k().values,
                          self.get_rd().values,
                          self.get_I())

    def volga_bump(self):
        return volga_bump(self.get_F_prime().values,
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

        beta_0 = pd.concat([self.get_rho_hat() * 0,
                  self.get_rho_hat() * 0,
                  self.get_rho_hat() / self.get_rho_hat(),
                  self.get_rho_hat() / self.get_rho_hat(),
                  self.get_rho_hat()], axis=1)

        rp = b_df - beta_0.reindex(b_df.index).values
        return rp, e, b_df, c_df, p_df, q_df

    def get_strikes(self, _HOLDING_DAYS=21):
        res = nb_strike(self.get_S().values.repeat(self.N, 1),
                        self.t,
                        self.get_rd().values,
                        self.get_rf().values, -1, -1 * self.x,
                       2,
                        self.get_I().values)

        ED = self.get_last_open_date(_HOLDING_DAYS)
        return pd.DataFrame(res, index=self.dates, columns=self.cols).loc[:ED].stack([0, 1])

    def get_open_dates(self, HOLDING_DAYS=21):
        return self.dates[:-HOLDING_DAYS]

    def get_open_vols(self, HOLDING_DAYS=21):
        return self.get_sig(HOLDING_DAYS)

    def get_open_strikes(self, HOLDING_DAYS=21):
        return self.get_strikes(HOLDING_DAYS)

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
                            index=ks.index,
                            columns=ks.columns)


    def get_s_paths(self, _HOLDING_DAYS):
        idx = self.get_path_idxs(_HOLDING_DAYS)
        return pd.DataFrame(self.get_S().values.flatten()[idx],
                            index=self.get_open_vols(_HOLDING_DAYS).index)

    def get_rf_paths(self):
        pass

    def get_rd_paths(self):
        pass

    def get_f_paths(self):
        pass

    def get_k_paths(self, _HOLDING_DAYS):
        pass

    def get_sig_paths(self):
        pass

    def get_ttm_paths(self):
        pass

    def get_delta_paths(self):
        pass



    def get_option_vols(self, _HOLDING_DAYS=21):

        # Implied Vols
        sig = self.get_sig(_HOLDING_DAYS)

        # Maturities
        t = np.array(sig.index.get_level_values(2)).reshape(-1, 1)

        # Strike Prices
        k  = self.get_strikes(_HOLDING_DAYS)
        ks = pd.concat([k] * (_HOLDING_DAYS + 1), axis=1)

        # All Pricing Dates
        idx = self.get_path_idxs(_HOLDING_DAYS)
        _PD = pd.DataFrame(self.dates.values[idx], index=ks.index, columns=ks.columns)

        # Time to maturity
        elapsed = np.asarray((_PD - _PD.values[:,0].reshape(-1, 1))/np.timedelta64(1, 'D'), dtype=int)/365
        ttm = pd.DataFrame(np.maximum(t - elapsed, 0), index=ks.index, columns=ks.columns)


        ivols = self.get_fixed_strike_and_maturity_ivols(_PD.values.flatten(),
                                                         ks.values.flatten(),
                                                         ttm.values.flatten())

        _vols = pd.DataFrame(ivols['mid'].values.reshape(_PD.shape), index=ks.index, columns=ks.columns)

        f = self._vs.interpolator.get_f(np.unique(ivols.get('date')),
                                        np.unique(ivols.get('t'))).stack(level=0)
        _f = f.loc[ivols.set_index(['date', 't']).index].values.reshape(_PD.shape)


        rf = self._vs.get_risk_free(np.unique(ivols.get('date')),
                                    np.unique(ivols.get('t'))).stack(level=0)
        rf_ = rf.loc[ivols.set_index(['date', 't']).index].values.reshape(_PD.shape)

        bsprice = blsprice(_f,
                           ttm.values,
                           ks.values,
                           rf_,
                           _vols.values,
                           -1)



if __name__ == "__main__":


    SD = pd.to_datetime('31-Dec-1996')
    ED = pd.to_datetime('31-Dec-2022')
    self = SmileStatArb('GBPUSD', SD, ED)
    self.get_option_vols(21)

    import matplotlib.pyplot as plt
    # Assuming self.get_I() returns a DataFrame
    ax = rp.get('volga').dropna().plot()
    # Adjust the x-axis to span from 0 to 0.6
    ax.set_ylim(-10, 20)
    # Add grid
    ax.grid(True)
    # Add a dashed yellow line across the zero point of the y-axis
    ax.axhline(y=0, color='yellow', linestyle='--')
    # Display the plot
    plt.show()





