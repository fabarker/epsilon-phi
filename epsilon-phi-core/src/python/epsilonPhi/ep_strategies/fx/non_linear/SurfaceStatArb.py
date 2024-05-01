import numpy as np
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

class SurfaceStatArb(object):

    _DELTAS = [-0.10, -0.25, -0.5, -0.75, -0.90]
    _MATURITIES = np.array([1, 3, 6, 9, 12]) / 12

    def __init__(self,
                 underlier,
                 start_date,
                 end_date,
                 interpolator=Interpolator.CUBIC_SPLINE):

        self._underlier = underlier
        self._vs = AbstractVolSurface.get_volatility_surface(self._underlier,
                                                             interpolation_method=interpolator)

        self._start_date = pd.to_datetime(start_date)
        self._end_date = pd.to_datetime(end_date)
        self.load_data()

    def get_ivols(self, dates, strike_reference, relative_strike, maturities, maturity_type):
        return self._vs.get_ivols(dates, strike_reference, relative_strike, maturities, maturity_type).dropna()

    def get_fixed_strike_implied_vols(self, dates, k, expiry_date):
        return self.get_ivols(dates, StrikeReference.STRIKE_PRICE, k, expiry_date, MaturityType.EXPIRY_DATE)

    def get_option_implied_vols(self, dates, strike_reference, relative_strike, maturity_type, maturities):

        # Get open ivols
        vols = self.get_ivols(dates, strike_reference, relative_strike, maturities, maturity_type).dropna()
        capped_pricing_dates = self._vs.interpolator.get_expiry_dates_from_settlement_dates_tenor(vols.date, '1m')
        dates, k, open_date, expiry_date, t = self._vs.get_contract_pricing_dates(vols.date, vols.expiry, vols.k, capped_pricing_dates)

        # Get vols to price remaining life of options
        idx = (dates >= self._start_date) & (dates <= self._end_date) & (dates != open_date)
        i_df = self.get_fixed_strike_implied_vols(dates[idx],
                                                  k[idx],
                                                  expiry_date[idx])

        ivols = pd.concat((vols, i_df), axis=0).reset_index(drop=True)
        sorted_ivols = ivols.sort_values(['expiry', 'k', 'date']).reset_index(drop=True)

        sorted_ivols['mid'] = sorted_ivols.groupby(['expiry', 'k'], group_keys=False, dropna=False).apply(lambda x: x.mid.ffill())
        sorted_ivols['open'] = sorted_ivols.groupby(['expiry', 'k'])['date'].transform('first')
        return sorted_ivols

    def load_option_data(self, dates, strike_reference, relative_strike, maturity_type, maturities, option_type):

        ivols = self.get_option_implied_vols(dates, strike_reference, relative_strike, maturity_type, maturities)
        ivols['q'] = self._vs.get_funding_rate(ivols.date, ivols.t, True).values
        ivols['r'] = self._vs.get_risk_free(ivols.date, ivols.t, True).values
        ivols['s'] = self._vs.get_spot_prices(ivols.date).values
        ivols['f'] = self._vs.get_forward_prices(ivols.date, ivols.t.values, True).values
        ivols['h'] = self._vs.get_hedge_instrument_prices(ivols.date, ivols.t.values).values

        # Add greeks to the dataset
        ivols['delta'] = ivols['x']
        ivols['gamma'] = black_gamma(ivols['f'], ivols['t'], ivols['k'], ivols['r'], ivols['mid'])
        ivols['vega']  = black_vega(ivols['f'], ivols['t'], ivols['k'], ivols['r'], ivols['mid'])
        ivols['vanna'] = black_vanna(ivols['f'], ivols['t'], ivols['k'], ivols['q'], ivols['mid'])
        ivols['volga'] = black_volga(ivols['f'], ivols['t'], ivols['k'], ivols['r'], ivols['mid'])
        ivols['theta'] = black_theta(ivols['f'], ivols['t'], ivols['k'], ivols['r'], ivols['q'], ivols['mid'], option_type)

        # Cache the data as a property
        self._data = ivols.copy()

        ##### Sample Option Dates #####

        idxs = ivols.open == ivols.date
        self._I = ivols[idxs].set_index(['date', 'x', 't']).get('mid')
        self._F = ivols[idxs].set_index(['date', 'x', 't']).get('f')
        self._r = ivols[idxs].set_index(['date', 'x', 't']).get('r')
        self._q = ivols[idxs].set_index(['date', 'x', 't']).get('q')

        self._delta = ivols[idxs].set_index(['date', 'x', 't']).get('delta')
        self._gamma = ivols[idxs].set_index(['date', 'x', 't']).get('gamma')
        self._vega =  ivols[idxs].set_index(['date', 'x', 't']).get('vega')
        self._vanna = ivols[idxs].set_index(['date', 'x', 't']).get('vanna')
        self._volga = ivols[idxs].set_index(['date', 'x', 't']).get('volga')
        self._theta = ivols[idxs].set_index(['date', 'x', 't']).get('theta')

        ########## Get dI ##########
        dI = ivols.groupby(['expiry', 'k']).agg(
            fy=('mid', lambda x: x.pct_change().iloc[1] if len(x) > 1 else None),
            x=('x', 'first'),
            t=('t', 'first'),
            f=('f', 'first'),
            i=('mid', 'first'),
            date=('date', lambda x: x.iloc[1] if len(x) > 1 else None),
        ).reset_index().dropna()

        ######### GET dF ############
        dF = ivols.groupby(['expiry', 'k']).agg(
            df=('f', lambda x: x.pct_change().iloc[1] if len(x) > 1 else None),
            x=('x', 'first'),
            t=('t', 'first'),
            date=('date', lambda x: x.iloc[1] if len(x) > 1 else None),
        ).reset_index().dropna()

        self._dF = dF.set_index(['date', 'x', 't']).get('df').reindex(self._I.index)

        dI['z'] = (np.log(dI.k / dI.f) + 0.5 * np.power(dI.i, 2) * dI.t) / (dI.i * np.sqrt(dI.t))
        dI['y'] = np.log(np.array(dI.t))

        df_array = dI.set_index('date').get(['t', 'x', 'fy', 'z', 'y']).pivot(columns=['x', 't'])
        _dI = pd.DataFrame(self.smooth_dataframe(df_array), index=df_array.index, columns=df_array.get('fy').columns)
        self._dI = _dI.stack(level=[0, 1]).reindex(self._I.index)

    def smooth_dataframe(self, df):

        x = np.asarray(df.get('z'), dtype=np.float64)
        y = np.asarray(df.get('y'), dtype=np.float64)
        z = np.asarray(df.get('fy'), dtype=np.float64)

        @njit
        def smooth(x, y, z):

            res = np.empty(shape=z.shape, dtype=np.float64)
            for t in range(res.shape[0]):
                for k in range(res.shape[1]):
                    res[t, k] = smooth_2d(x[t, :], y[t, :], x[t, k], y[t, k], z[t, :])
            return res
        return smooth(x, y, z)


    def load_data(self):

        # Load Market Data
        locs = ((self._vs.common_dates >= self._start_date) &
                (self._vs.common_dates <= self._end_date))

        self.load_option_data(self._vs.common_dates[locs],
                              StrikeReference.DELTA,
                              self._DELTAS,
                              MaturityType.YEARFRAC,
                              self._MATURITIES,
                              -1)

    def get_I(self):
        return self._I.unstack(level=[1, 2])

    def get_F(self):
        return self._F.unstack(level=[1, 2])

    def get_r(self):
        return self._r.unstack(level=[1, 2])

    def get_q(self):
        return self._q.unstack(level=[1, 2])

    def get_dF(self):
        return self._dF.unstack(level=[1, 2])

    def get_dI(self):
        return self._dI.unstack(level=[1, 2])

    def get_dFdF(self):
        return np.power(self.get_dF(), 2)

    def get_dIdI(self):
        return np.power(self.get_dI(), 2)

    def get_dIdF(self):
        return self.get_dI().mul(self.get_dF())

    def get_slope(self):
        return self.get_I().get(0.5).get(1) - self.get_I().get(0.5).get(1/12)

    def get_skew(self):
        return self.get_I().get(-0.9).get(3/12) - self.get_I().get(-0.1).get(3/12)

    def get_nu(self, period=21):
        return self.get_dFdF().rolling(window=period).mean() * 252

    def get_mu(self, period=21):
        return self.get_dI().rolling(window=period).mean() * 252

    def get_gamma(self, period=21):
        return self.get_dIdF().rolling(window=period).mean() * 252

    def get_omega(self, period=21):
        return self.get_dIdI().rolling(window=period).mean() * 252

    def get_rho(self, period=252):
        return self.get_gamma(period) / self.get_sqrt_nu_omega(period)

    def get_rho_hat(self, period=252):
        return self.get_rho(period).mean(axis=1).to_frame()

    def get_sqrt_nu_omega(self, period=21):
        return np.sqrt(self.get_nu(period).values * self.get_omega(period))

    ################  Greeks ################

    def get_bsdelta(self):
        return self._delta.unstack(level=[1, 2])

    def get_bsgamma(self):
        return self._gamma.unstack(level=[1, 2])

    def get_bsvega(self):
        return self._vega.unstack(level=[1, 2])

    def get_bsvanna(self):
        return self._vanna.unstack(level=[1, 2])

    def get_bsvolga(self):
        return self._volga.unstack(level=[1, 2])

    def get_bstheta(self):
        return self._theta.unstack(level=[1, 2])

    ###################### FACTORS FOR FACTOR MODEL #####################

    def get_vega_factor(self, period=21):
        return self.get_bsvega() * self.get_I() * self.get_omega(period)

    def get_drift_factor(self, period=21):
        return self.get_bsvega() * self.get_I() * self.get_mu(period)

    def get_gamma_factor(self, period=21):
        return 0.5 * self.get_bsgamma() * self.get_dFdF() * self.get_nu(period)

    def get_volga_factor(self, period=21):
        return 0.5 * self.get_bsvolga() * np.power(self.get_I(), 2) * self.get_omega(period)

    def get_vanna_factor(self, period=21):
        return self.get_bsvanna() * self.get_F() * self.get_I() * self.get_sqrt_nu_omega(period)

    def get_cash_gamma(self):
        return np.power(self.get_F(), 2) * self.get_I() * self.get_bsgamma()

    def estimate_market_price_of_risk(self, period=21):

        Y_ = -1 * self.get_bstheta()
        X1 = self.get_vega_factor(period).get(Y_.columns).reindex(Y_.index).values
        X2 = self.get_drift_factor(period).get(Y_.columns).reindex(Y_.index).values
        X3 = self.get_gamma_factor(period).get(Y_.columns).reindex(Y_.index).values
        X4 = self.get_volga_factor(period).get(Y_.columns).reindex(Y_.index).values
        X5 = self.get_vanna_factor(period).get(Y_.columns).reindex(Y_.index).values

        g = self.get_cash_gamma().get(Y_.columns).reindex(Y_.index)

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





if __name__ == "__main__":

    import pandas as pd
    import numpy as np
    from epsilonPhi.core.utils.DateUtils import DateUtils

    SD = pd.to_datetime('31-Dec-2021')
    ED = pd.to_datetime('31-Dec-2023')
    self = SurfaceStatArb('GBPUSD', SD, ED)
    self.get_skew()


