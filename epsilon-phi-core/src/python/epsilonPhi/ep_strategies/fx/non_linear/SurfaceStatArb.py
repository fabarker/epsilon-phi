import numpy as np
from epsilonPhi.core.dataModel.dataSources.impliedVolatility.VolSurface import AbstractVolSurface
from epsilonPhi.core.dataModel.enums.ImpliedVolatility import *
from epsilonPhi.core.lib.smoothing.GaussianKernel import smooth_2d
from epsilonPhi.core.lib.Decorators import auto_repr
from scipy.optimize import lsq_linear
from epsilonPhi.core.utils.DateUtils import DateUtils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
        return self._vs.get_ivols(dates, strike_reference, relative_strike, maturities, maturity_type)

    def get_fixed_strike_implied_vols(self, dates, k, expiry_date):
        return self.get_ivols(dates, StrikeReference.STRIKE_PRICE, k, expiry_date, MaturityType.EXPIRY_DATE)

    def get_option_implied_vols(self, dates, strike_reference, relative_strike, maturity_type, maturities):

        # Get open ivols
        vols = self.get_ivols(dates, strike_reference, relative_strike, maturities, maturity_type).dropna()
        vols['open'] = vols.date
        capped_pricing_dates = self._vs.interpolator.get_expiry_dates_from_settlement_dates_tenor(vols.date,
                                                                                                  self.roll_frequency)
        dates, k, open_date, expiry_date, t = self._vs.get_contract_pricing_dates(vols.date, vols.expiry, vols.k,
                                                                                  capped_pricing_dates)

        # Get vols to price remaining life of options
        idx = (dates >= self._start_date) & (dates <= self._end_date) & (dates != open_date)
        i_df = self.get_fixed_strike_implied_vols(dates[idx],
                                                  k[idx],
                                                  expiry_date[idx])
        i_df['open'] = open_date[idx]
        return pd.concat((vols, i_df), axis=0).reset_index(drop=True)


    def load_option_data(self, dates, strike_reference, relative_strike, maturity_type, maturities, option_type):

        ivols = self.get_option_implied_vols(dates, strike_reference, relative_strike, maturity_type, maturities)
        idxs = ivols.open == ivols.date
        self._I = ivols[idxs].set_index(['date', 'x', 't']).get('mid')

        # Get dI
        It = ivols[ivols.open != ivols.date].sort_values('open').set_index(['k', 'expiry'])
        Ft = self._vs.get_forward_prices(It.date, np.array(It.t), True).values

        I0 = ivols[ivols.open == ivols.date].set_index(['k', 'expiry']).loc[It.index]
        F0 = self._vs.get_forward_prices(I0.date, np.array(I0.t), True).values

        assert np.all(It.date > I0.date), 'Error - dates not aligned correctly'
        assert np.all(It.index == I0.index), 'Error - index not matched'

        dI = pd.DataFrame(-1 + ( It.get('mid').values / I0.get('mid').values ),
                          index=[It.date, I0.x.values, I0.t.values], columns=['dI']).reset_index()
        dI.columns = ['date', 'x', 't', 'fy']
        dI['z'] = np.array((np.log(np.array(I0.index.get_level_values('k')) / F0) + 0.5 * np.power(I0.mid, 2) * I0.t) / (I0.mid * np.sqrt(I0.t)))
        dI['y'] = np.array(np.log(np.array(I0.t)))
        df_array = dI.set_index('date').get(['t', 'x', 'fy', 'z', 'y']).pivot(columns=['x', 't'])
        _dI = pd.DataFrame(self.smooth_dataframe(df_array),
                           index=df_array.index,
                           columns=df_array.get('fy').columns).stack(level=[0, 1])

        _dF = pd.Series(-1 + ( Ft / F0 ), index=[It.date, I0.x, I0.t])

        common_dates = self._I.index.intersection(_dI.index)

        self._I = self._I.reindex(common_dates)
        self._dI = _dI.reindex(common_dates)
        self._dF = _dF.reindex(common_dates)

        ivols['q'] = self._vs.get_funding_rate(ivols.date, ivols.t, True).values
        ivols['r'] = self._vs.get_risk_free(ivols.date, ivols.t, True).values
        ivols['s'] = self._vs.get_spot_prices(ivols.date).values
        ivols['f'] = self._vs.get_forward_prices(ivols.date, ivols.t.values, True).values
        ivols['h'] = self._vs.get_hedge_instrument_prices(ivols.date, ivols.t.values).values

        # Add greeks to the dataset
        ivols['delta'] = black_delta(ivols['f'], ivols['t'], ivols['k'], ivols['q'], ivols['mid'], 2, option_type)
        ivols['gamma'] = black_gamma(ivols['f'], ivols['t'], ivols['k'], ivols['r'], ivols['mid'])
        ivols['vega'] = black_vega(ivols['f'], ivols['t'], ivols['k'], ivols['r'], ivols['mid'])
        ivols['vanna'] = black_vanna(ivols['f'], ivols['t'], ivols['k'], ivols['q'], ivols['mid'])
        ivols['volga'] = black_volga(ivols['f'], ivols['t'], ivols['k'], ivols['r'], ivols['mid'])
        ivols['theta'] = black_theta(ivols['f'], ivols['t'], ivols['k'], ivols['r'], ivols['q'], ivols['mid'], option_type)
        ivols['premium'] = self._vs.get_mid_premium(ivols['s'], ivols['t'], ivols['k'], ivols['r'], ivols['q'], ivols['mid'], option_type)

        # Cache the data as a property
        _data = ivols.set_index(['date', 'x', 't']).sort_index(level=0)

        ##### Sample Option Dates #####

        self._F = _data.loc[common_dates].get('f')
        self._r = _data.loc[common_dates].get('r')
        self._q = _data.loc[common_dates].get('q')

        self._delta = _data.loc[common_dates].get('delta')
        self._gamma = _data.loc[common_dates].get('gamma')
        self._vega = _data.loc[common_dates].get('vega')
        self._vanna = _data.loc[common_dates].get('vanna')
        self._volga = _data.loc[common_dates].get('volga')
        self._theta = _data.loc[common_dates].get('theta')

        self._dates = pd.to_datetime(self._I.index.get_level_values('date').unique())

        # Update the t and x
        I0 = ivols[ivols.date == ivols.open].copy()
        IT = ivols[ivols.date != ivols.open].copy()
        IT['t'] = I0.set_index(['k','expiry']).loc[zip(IT.k.values, IT.expiry.values)].t.values
        IT['x'] = I0.set_index(['k', 'expiry']).loc[zip(IT.k.values, IT.expiry.values)].x.values
        self._data = pd.concat((I0, IT), axis=0).reset_index(drop=True)

    def get_option_pnl(self):

        df_ = self._data.reset_index(drop=True).set_index(['open', 'expiry', 'k'])

        idx = df_.index.get_level_values('open') == df_.date
        p0 = df_[idx]
        pT = df_[~idx]

        _common = p0.index.intersection(pT.index)
        pT_ = pT.loc[_common]
        p0_ = p0.loc[_common]

        pnl = (pT_['premium'] - p0_['premium'].values).to_frame()
        pnl['date'] = pT_.date.values
        pnl['t'] = np.array(p0_.t)
        pnl['x'] = np.array(p0_.x)

        return pnl.reset_index().set_index('date').get(['premium', 't', 'x']).pivot(columns=['x', 't']).get('premium')

    def get_delta_hedge_pnl(self):

        df_ = self._data.reset_index(drop=True).set_index(['open', 'expiry', 'k'])
        idx = df_.index.get_level_values('open') == df_.date
        p0 = df_[idx]
        pT = df_[~idx]

        _common = p0.index.intersection(pT.index)
        pT_ = pT.loc[_common]
        p0_ = p0.loc[_common]

        f_pnl = ((pT_['f'] - p0_.get('f').values) / np.array(pT_.index.get_level_values('k'))).to_frame()
        f_pnl['f'] = f_pnl['f'] * p0_.x.mul(-1).values
        f_pnl['date'] = df_[~idx].date
        f_pnl['t'] = df_[idx].t
        f_pnl['x'] = df_[idx].x

        return f_pnl.reset_index().set_index('date').get(['f', 't', 'x']).pivot(columns=['x', 't']).get('f')

    def get_hedged_contract_pnl(self):
        return -1 * (self.get_option_pnl() + self.get_delta_hedge_pnl())

    def get_strike_prices(self):
        return (self._data[self._data.index.get_level_values('date') == self._data.open].
                get('k').unstack(level=[1,2]).sort_index())

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
        return self._I.unstack(level=[1, 2]).sort_index()

    def get_F(self):
        return self._F.unstack(level=[1, 2]).sort_index()

    def get_r(self):
        return self._r.unstack(level=[1, 2]).sort_index()

    def get_q(self):
        return self._q.unstack(level=[1, 2]).sort_index()

    def get_dF(self):
        return self._dF.unstack(level=[1, 2]).sort_index()

    def get_dI(self):
        return self._dI.unstack(level=[1, 2]).sort_index()

    def get_dFdF(self):
        return np.power(self.get_dF(), 2)

    def get_dIdI(self):
        return np.power(self.get_dI(), 2)

    def get_dIdF(self):
        return self.get_dI().mul(self.get_dF())

    def get_slope(self):
        return self.get_I().get(-0.5).get(1) - self.get_I().get(-0.5).get(1 / 12)

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
        return self.get_rho(period).mean(axis=1)

    def get_sqrt_nu_omega(self, period=21):
        return np.sqrt(self.get_nu(period).values * self.get_omega(period))

    ################  Greeks ################

    def get_bsdelta(self):
        return self._delta.unstack(level=[1, 2]).sort_index()

    def get_bsgamma(self):
        return self._gamma.unstack(level=[1, 2]).sort_index()

    def get_bsvega(self):
        return self._vega.unstack(level=[1, 2]).sort_index()

    def get_bsvanna(self):
        return self._vanna.unstack(level=[1, 2]).sort_index()

    def get_bsvolga(self):
        return self._volga.unstack(level=[1, 2]).sort_index()

    def get_bstheta(self):
        return self._theta.unstack(level=[1, 2]).sort_index()

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

    def plot_pricing_errors(self, horizon):

        self.estimate_market_price_of_risk(int(horizon))
        errors = SurfaceStatArb._estimates.get((self._underlier, int(horizon))).get_errors()
        mae = errors.abs().mean().unstack().T

        X, Y = np.meshgrid(mae.columns, mae.index)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        Z = mae.values * 100
        surface = ax.plot_surface(X, Y, Z, cmap='viridis')
        fig.colorbar(surface, shrink=0.5, aspect=5)

        # Labels and title
        ax.set_xlabel('Put Delta')
        ax.set_ylabel('Maturity')
        ax.set_zlabel('Pricing Error')
        ax.set_title('Mean absolute pricing errors')


    def run_market_price_of_risk_estimates(self):
        periods = np.array(np.array([1 / 12, 3 / 12, 6 / 12, 9 / 12, 1]) * 252, dtype=np.int32)
        for period in periods:
            self.estimate_market_price_of_risk(period)

    def estimate_market_price_of_risk(self, period=21):

        if (self._underlier, int(period)) not in SurfaceStatArb._estimates.keys():

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

    def get_market_price_of_risk_estimates(self, period=21):
        self.estimate_market_price_of_risk(int(period))
        return SurfaceStatArb._estimates.get((self._underlier, int(period)))

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
        d[-1] = 1

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

    SD = '24-Jan-1996'
    ED = '13-Oct-2022'
    self = SurfaceStatArb('GBPUSD', SD, ED)
    self.roll_frequency = '1d'
    self.load_data()

    res = self.get_hedged_contract_pnl()
    wts = self.get_stat_arb_signals()
