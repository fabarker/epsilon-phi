import numpy as np
from epsilonPhi.core.dataModel.dataSources.impliedVolatility.VolSurface import AbstractVolSurface
from epsilonPhi.core.dataModel.enums.ImpliedVolatility import *
from epsilonPhi.core.lib.smoothing.GaussianKernel import smooth_2d
from epsilonPhi.core.lib.Decorators import auto_repr
from scipy.optimize import lsq_linear
from epsilonPhi.core.utils.DateUtils import DateUtils
from epsilonPhi.core.utils.OptionUtils import *
from epsilonPhi.core.utils.FrameUtils import FrameUtils
import numpy as np
from numba import njit
import pandas as pd
from scipy.stats import t as tstudent


@auto_repr
class Estimates:
    def __init__(self, index, risk_cols, option_cols):
        self._index = index
        self._risk_cols = risk_cols
        self._cols = option_cols

        self._errors = None
        self._residuals = None
        self._variance_contributions = None
        self._betas = None
        self._rho = None
        self._rsq = None
        self._horizon = None

    @property
    def errors(self):
        return self._errors

    @errors.setter
    def errors(self, val):
        self._errors = (
            pd.DataFrame(val, index=self._index, columns=self._cols))

    @property
    def residuals(self):
        return self._residuals

    @residuals.setter
    def residuals(self, val):
        self._residuals = (
            pd.DataFrame(val, index=self._index, columns=self._cols))

    @property
    def variance_contributions(self):
        return self._variance_contributions

    @variance_contributions.setter
    def variance_contributions(self, val):
        self._variance_contributions = (
            pd.DataFrame(val, index=self._index, columns=self._risk_cols))

    @property
    def betas(self):
        return self._betas

    @betas.setter
    def betas(self, val):
        self._betas = (
            pd.DataFrame(val, index=self._index, columns=self._risk_cols))

    @property
    def rsq(self):
        return self._rsq

    @rsq.setter
    def rsq(self, val):
        self._rsq = val

    @property
    def horizon(self):
        return self._horizon

    @horizon.setter
    def horizon(self, val):
        self._horizon = val

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, val):
        self._rho = val

    def get_errors(self, risk=None):
        return self.errors.get(risk, self.errors)

    def get_residuals(self, cols=None):
        return self.residuals.get(cols, self.residuals)

    def get_variance_contribution(self, risk=None):
        return self.variance_contributions.get(risk, self.variance_contributions)

    def get_betas(self, risk=None):
        return self.betas.get(risk, self.betas)

    def get_nulls(self):
        T = self.betas.shape[0]
        _nulls = [[0] * T, [0] * T, [1] * T, [1] * T, np.ravel(self.rho)]
        df_ = pd.DataFrame(_nulls, columns=self.betas.index, index=self.betas.columns)
        return df_.T

    def get_premiums(self, risk=None):
        b = self.get_betas().sub(self.get_nulls())
        return b.get(risk, b)

    def get_rsq(self):
        return pd.Series(self.rsq.flat, index=self._index)


class SurfaceStatArb(object):
    _DELTAS = [-0.10, -0.25, -0.5, -0.75, -0.90]
    _MATURITIES = np.array([1, 3, 6, 9, 12]) / 12
    _RISK_FACTORS = ['vega', 'trend', 'gamma', 'volga', 'vanna']
    _estimates = dict()

    def __init__(self,
                 underlier,
                 start_date,
                 end_date,
                 interpolator=Interpolator.CUBIC_SPLINE):

        self._underlier = underlier
        self._dates = None
        self._roll_frequency = '1d'

        self._vega_factor = {}
        self._drift_factor = {}
        self._gamma_factor = {}
        self._volga_factor = {}
        self._vanna_factor = {}

        self._risk_dimensions = self._RISK_FACTORS
        self._vs = AbstractVolSurface.get_volatility_surface(self._underlier,
                                                             interpolation_method=interpolator)
        self._start_date = pd.to_datetime(start_date)
        self._end_date = pd.to_datetime(end_date)

    @property
    def dates(self):
        return self._dates

    @property
    def number_of_options(self):
        return len(self._DELTAS) * len(self._MATURITIES)

    @property
    def roll_frequency(self):
        return self._roll_frequency

    @roll_frequency.setter
    def roll_frequency(self, val):
        self._roll_frequency = val

    def get_ivols(self, dates, strike_reference, relative_strike, maturities, maturity_type):
        return self._vs.get_ivols(dates, strike_reference, relative_strike, maturities, maturity_type).dropna()

    def get_fixed_strike_implied_vols(self, dates, k, expiry_date):
        return self.get_ivols(dates, StrikeReference.STRIKE_PRICE, k, expiry_date, MaturityType.EXPIRY_DATE)

    def get_option_implied_vols(self, dates, strike_reference, relative_strike, maturity_type, maturities):

        # Get open ivols
        vols = self.get_ivols(dates, strike_reference, relative_strike, maturities, maturity_type).dropna()
        capped_pricing_dates = self._vs.interpolator.get_expiry_dates_from_settlement_dates_tenor(vols.date,
                                                                                                  self.roll_frequency)
        dates, k, open_date, expiry_date, t = self._vs.get_contract_pricing_dates(vols.date, vols.expiry, vols.k,
                                                                                  capped_pricing_dates)

        # Get vols to price remaining life of options
        idx = (dates >= self._start_date) & (dates <= self._end_date) & (dates != open_date)
        i_df = self.get_fixed_strike_implied_vols(dates[idx],
                                                  k[idx],
                                                  expiry_date[idx])

        ivols = pd.concat((vols, i_df), axis=0).reset_index(drop=True)
        sorted_ivols = ivols.sort_values(['expiry', 'k', 'date']).reset_index(drop=True)

        #sorted_ivols['mid'] = sorted_ivols.groupby(['expiry', 'k'], group_keys=False, dropna=False).apply(lambda x: x.mid.ffill())
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
        ivols['vega'] = black_vega(ivols['f'], ivols['t'], ivols['k'], ivols['r'], ivols['mid'])
        ivols['vanna'] = black_vanna(ivols['f'], ivols['t'], ivols['k'], ivols['q'], ivols['mid'])
        ivols['volga'] = black_volga(ivols['f'], ivols['t'], ivols['k'], ivols['r'], ivols['mid'])
        ivols['theta'] = black_theta(ivols['f'], ivols['t'], ivols['k'], ivols['r'], ivols['q'], ivols['mid'],
                                     option_type)

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
        self._vega = ivols[idxs].set_index(['date', 'x', 't']).get('vega')
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
        self._dates = pd.to_datetime(self._I.index.get_level_values('date').unique())

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
        return self.get_I().get(0.5).get(1) - self.get_I().get(0.5).get(1 / 12)

    def get_skew(self):
        return self.get_I().get(-0.9).get(3 / 12) - self.get_I().get(-0.1).get(3 / 12)

    def get_nu(self, period=21):
        return self.get_dFdF().rolling(window=int(period)).mean() * 252

    def get_mu(self, period=21):
        return self.get_dI().rolling(window=int(period)).mean() * 252

    def get_gamma(self, period=21):
        return self.get_dIdF().rolling(window=int(period)).mean() * 252

    def get_omega(self, period=21):
        return self.get_dIdI().rolling(window=int(period)).mean() * 252

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
    def load_vega_factor(self, period=21):
        self._vega_factor[int(period)] = self.get_bsvega() * self.get_I() * self.get_omega(period)

    def get_vega_factor(self, period=21):
        if period not in self._vega_factor.keys():
            self.load_vega_factor(period)
        return self._vega_factor.get(period)

    def load_drift_factor(self, period=21):
        self._drift_factor[int(period)] = self.get_bsvega() * self.get_I() * self.get_mu(period)

    def get_drift_factor(self, period=21):
        if period not in self._drift_factor.keys():
            self.load_drift_factor(period)
        return self._drift_factor.get(period)

    def load_gamma_factor(self, period=21):
        self._gamma_factor[int(period)] = 0.5 * self.get_bsgamma() * np.power(self.get_F(), 2) * self.get_nu(period)

    def get_gamma_factor(self, period=21):
        if period not in self._gamma_factor.keys():
            self.load_gamma_factor(period)
        return self._gamma_factor.get(period)

    def load_volga_factor(self, period=21):
        self._volga_factor[int(period)] = 0.5 * self.get_bsvolga() * np.power(self.get_I(), 2) * self.get_omega(period)

    def get_volga_factor(self, period=21):
        if period not in self._volga_factor.keys():
            self.load_volga_factor(period)
        return self._volga_factor.get(period)

    def load_vanna_factor(self, period=21):
        self._vanna_factor[int(period)] = self.get_bsvanna() * self.get_F() * self.get_I() * self.get_sqrt_nu_omega(
            period)

    def get_vanna_factor(self, period=21):
        if period not in self._vanna_factor.keys():
            self.load_vanna_factor(period)
        return self._vanna_factor.get(period)

    def get_cash_gamma(self):
        return np.power(self.get_F(), 2) * self.get_I() * self.get_bsgamma()

    def run_market_price_of_risk_estimates(self):
        periods = np.array(np.array([1 / 12, 3 / 12, 6 / 12, 1]) * 252, dtype=np.int32)
        for period in periods:
            self.estimate_market_price_of_risk(period)

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

                b[t, :] = betas
                r[t, :] = residuals
                q[t] = 1 - (np.dot(residuals.T, residuals) / YtY)

                dof = X.shape[0] - X.shape[1]
                cov_matrix = np.linalg.inv(XtX) * np.dot(residuals.T, residuals) / dof
                tstat = (betas - 0.5) / np.sqrt(np.diag(cov_matrix))

                p[t, :] = 2 * tstudent.sf(np.abs(tstat), dof)
                c[t, :] = (betas * np.dot(XtX, betas)) / YtY
            return b, r, p, c, q

        b, r, p, c, q = estimate_betas(Y_.values, X1, X2, X3, X4, X5)

        estimates = Estimates(index=g.index, risk_cols=self._risk_dimensions, option_cols=g.columns)
        estimates.rsq = q
        estimates.variance_contributions = c
        estimates.horizon = period
        estimates.residuals = r
        estimates.betas = b
        estimates.errors = r / g
        estimates.rho = self.get_rho_hat()

        SurfaceStatArb._estimates[(self._underlier, period)] = estimates

    def get_risk_factor_premium(self, risk_factor=None):

        if len(self._estimates.keys()) == 0:
            self.run_market_price_of_risk_estimates()

        fits = pd.concat([self._estimates.get(x).get_rsq().to_frame(x[1])
                          for x in self._estimates.keys()], axis=1)

        opt_fits = fits.idxmax(axis=1)
        nu = list()
        for date, val in opt_fits.items():
            if not np.isnan(val):
                nu.extend([self._estimates.get((self._underlier, int(val))).get_premiums().loc[date]])
        p = pd.concat(nu, axis=1).T
        return p.get(risk_factor, p).reindex(opt_fits.index)

    def get_sensitivity_matrix(self, date):

        if len(self._estimates.keys()) == 0:
            self.run_market_price_of_risk_estimates()

        fits = pd.concat([self._estimates.get(x).get_rsq().to_frame(x[1])
                          for x in self._estimates.keys()], axis=1)

        opt_fits = fits.idxmax(axis=1)
        tau = opt_fits.loc[date]

        if np.isnan(tau):
            return None
        else:
            tau = int(tau)

        # Get the model residuals
        R = self._estimates.get((self._underlier, tau)).residuals.loc[date]

        H = pd.concat((self.get_vega_factor(tau).loc[date],
                       self.get_drift_factor(tau).loc[date],
                       self.get_gamma_factor(tau).loc[date],
                       self.get_volga_factor(tau).loc[date],
                       self.get_vanna_factor(tau).loc[date], R), axis=1)

        H.columns = self._estimates.get((self._underlier, tau))._risk_cols + ['r']
        return H.copy()

    def get_stat_arb_signals(self):

        d = np.zeros((6, 1))
        d[0] = 1

        T = self.dates.shape[0]
        wts = np.full((T, self.number_of_options), np.nan)

        for t in range(len(self.dates)):
            h_bar = self.get_sensitivity_matrix(self.dates[t])

            if h_bar is not None:
                wts[t] = np.ravel(h_bar @ np.linalg.inv(h_bar.T @ h_bar) @ d)

        wts_df = pd.DataFrame(wts, index=self.dates, columns=self.get_I().columns)
        return wts_df / wts_df.abs().sum(axis=1, skipna=False).values.reshape(-1, 1)

    def get_risk_target_weights(self, risk_factor='gamma'):

        assert risk_factor.lower() in self._risk_dimensions, 'Error - risk factor {} not recognised'.format(risk_factor)
        idx = self._risk_dimensions.index(risk_factor)

        d = np.zeros((6, 1))
        d[idx] = 1

        T = self.dates.shape[0]
        wts = np.full((T, self.number_of_options), np.nan)

        for t in range(len(self.dates)):
            h_bar = self.get_sensitivity_matrix(self.dates[t])

            if h_bar is not None:
                wts[t] = np.ravel(h_bar @ np.linalg.inv(h_bar.T @ h_bar) @ d)

        wts_df = pd.DataFrame(wts, index=self.dates, columns=self.get_I().columns)
        return wts_df / wts_df.abs().sum(axis=1, skipna=False).values.reshape(-1, 1)

    def get_time_varying_single_risk_factor_targeting_weights(self, risk_factor='gamma'):

        # Get the weights for targeting fixed unit notional exposure to risk factor
        risk_target_weights = self.get_risk_target_weights(risk_factor)

        # Get the market pricing for the risk factor in question at t
        premium = self.get_risk_factor_premium(risk_factor)

        # We go long or short the factor based on the market premium pricing
        return risk_target_weights * np.sign(premium).values.reshape(-1, 1)

    def get_time_varying_multi_factor_weights(self):

        premiums = self.get_risk_factor_premium()


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from epsilonPhi.core.utils.DateUtils import DateUtils

    SD = pd.to_datetime('31-Dec-1998')
    ED = pd.to_datetime('31-Dec-2023')
    self = SurfaceStatArb('GBPUSD', SD, ED)
    self.roll_frequency = '1d'
    self.load_data()

    wts = self.get_risk_factor_premium()
