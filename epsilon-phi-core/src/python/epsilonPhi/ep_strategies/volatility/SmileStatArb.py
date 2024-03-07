from epsilonPhi.core.dataModel.dataSources.impliedVolatility.VolSurface import AbstractVolSurface
from epsilonPhi.core.dataModel.enums.ImpliedVolatility import *
from scipy.optimize import lsq_linear
import numpy as np
from tqdm import tqdm
import pandas as pd

class SmileStatArb(object):

    _STRATEGY_HOLDING_PERIODS = 21
    _HISTORICAL_ROLLING_PERIODS = 21

    _STRATEGY_ESTIMATION_WINDOW = 4 * 252
    _STRATEGY_STARTING_WINDOW = 253 * 4
    _TRUNCATE_SPREAD = np.abs(1)

    _Z_SCORES = np.arange(-2, 2.5, 0.5)
    _MATURITIES = [1 / 12, 2 / 12, 3 / 12, 6 / 12, 12 / 12]

    def __init__(self, underlier):

        # Set the underlier in the object
        self._underlier = underlier

        # Properties of the strategy class
        self._vol_surface = None
        self._ivols = None
        self._s = None
        self._f = None
        self._rd = None
        self._rf = None

    @property
    def T(self):
        return len(self.dates)
    @property
    def unique_mats(self):
        return np.unique(self.ivols.columns.get_level_values(1))
    @property
    def NM(self):
        return len(self.unique_mats)
    @property
    def unique_x(self):
        return np.unique(self.ivols.columns.get_level_values(0))
    @property
    def NX(self):
        return len(self.unique_x)
    @property
    def dates(self):
        return self._ivols.index
    @property
    def ivols(self):
        return self._ivols.copy()
    @property
    def t(self):
        return np.array(self.ivols.columns.get_level_values(1)).reshape(1, -1).repeat(self.T, 0)
    @property
    def x(self):
        return np.array(self.ivols.columns.get_level_values(0)).reshape(1, -1).repeat(self.T, 0)
    @property
    def s(self):
        return self._s.copy()
    @property
    def ds(self):
        return np.log(self._s).diff().shift(-1)
    @property
    def ds_sq(self):
        return self.ds ** 2
    @property
    def f(self):
        return self._f.copy()
    @property
    def rd(self):
        return self._rd.copy()
    @property
    def rf(self):
        return self._rf.copy()
    @property
    def sig(self):
        return self._ivols.copy()
    @property
    def sig_sq(self):
        return np.power(self.sig, 2)
    @property
    def dsig(self):
        return np.log(self.sig).diff().shift(-1)
    @property
    def dsig_sq(self):
        return self.dsig ** 2
    @property
    def atm_sig(self):
        return self.sig.get(0)
    @property
    def atm_sig_sq(self):
        return self.atm_sig ** 2
    @property
    def atm_dsig(self):
        return np.log(self.atm_sig).diff().shift(-1)
    @property
    def lnm(self):
        return self.x * (self.sig * np.sqrt(self.t)) - 0.5 * self.sig_sq * self.t
    @property
    def moneyness(self):
        return np.exp(self.lnm)
    @property
    def z_plus(self):
        return self.lnm + 0.5 * self.sig_sq * self.t
    @property
    def z_minus(self):
        return self.z_plus - self.sig_sq * self.t
    @property
    def omega(self):
        return self.dsig_sq

    @property
    def gamma(self):
        return self.dsig * self.ds.values.reshape(-1, 1)
    @property
    def mu(self):
        return self.get_mu()
    @property
    def omega_cs(self):
        if not hasattr(self, '_cross_sectional_estimates'):
            self.run_cross_sectional_spread_regressions()
        return self._cross_sectional_estimates.get('omega')
    @property
    def gamma_cs(self):
        if not hasattr(self, '_cross_sectional_estimates'):
            self.run_cross_sectional_spread_regressions()
        return self._cross_sectional_estimates.get('gamma')
    def get_gamma(self):
        pass
    def get_mu(self, maturity=None):

        Asq = self.atm_sig_sq
        Asq_tau = Asq * np.array(Asq.columns).reshape(1, -1)

        mu_ = ((1 / 2) * Asq.diff(axis=1) / Asq_tau.diff(axis=1)).dropna(axis=1, how='all')
        X = 0.5 * Asq_tau.columns.values[0:-1] + 0.5 * Asq_tau.columns.values[1:]
        mu_.columns = X

        mu_[[Asq_tau.columns]] = np.nan
        mu = mu_.sort_index(axis=1).interpolate(axis=1)
        mu[mu.columns[0]] = mu_[mu_.columns[0]]

        if maturity is None:
            return mu.get(Asq.columns)
        else:
            return mu.get(maturity)
    def get_variance_spreads(self):
        return self.sig_sq - self.atm_sig_sq[self.sig_sq.columns.get_level_values(1)].values

    def run_cross_sectional_spread_regressions(self):

        # For each day, we compute the observed spread
        # of each IV to ATM and then regress this on
        # The following:
        # I^2 - A^2 = gamma * 2 * z_plus + omega^2 * z_plus * z_minus

        lower_bounds = [-np.inf, 0]
        upper_bounds = [np.inf, np.inf]

        S = self.get_variance_spreads()
        z_p = self.z_plus
        z_m = self.z_minus

        X = pd.concat((2 * z_p.stack(level=[0,1]), (z_p * z_m).stack(level=[0, 1])), axis=1)
        Y_prime = S.stack(level=[0,1]).loc[X.index].values
        X_prime = X.values

        all_dates = X.index.get_level_values('date')
        all_mats = X.index.get_level_values('t')
        x_locs = X.index.get_level_values(1).isin([-1, -0.5, 0.5, 1])

        unique_dates = np.unique(all_dates)
        unique_mats = np.unique(all_mats)

        reg = list()
        for t in tqdm(range(self.T), desc="Processing"):
            for mat in unique_mats:

                idx = ((all_dates == unique_dates[t]) &
                       (all_mats == mat) & (x_locs))

                result = lsq_linear(X_prime[idx, :],
                                    Y_prime[idx],
                                    bounds=(lower_bounds, upper_bounds), lsmr_maxiter=None,
                                    method='trf', lsq_solver=None, lsmr_tol=None, max_iter=200, tol=1e-8)

                reg.extend([(unique_dates[t], mat, result.x[0], result.x[1])])

                #omega, gamma = result.x
                #e = Y_prime[idx] - X_prime[idx, :] @ result.x
                #rsq = 1 - (np.mean(np.power(e, 2)) / np.var(Y_prime[idx]))
                #reg.extend([(unique_dates[t], mat, omega, gamma, rsq)])
        _df = pd.DataFrame(reg, columns=['date', 't', 'omega', 'gamma']).set_index(['date','t'])
        self._cross_sectional_estimates = _df.unstack(level=1)

    def estimate_time_series_omega(self, rolling_window=1):
        return self.omega.rolling(window=rolling_window).mean()
    def estimate_time_series_gamma(self, rolling_window=1):
        return self.gamma.rolling(window=rolling_window).mean()
    def estimate_time_series_mu(self, rolling_window=1):
        return self.mu.rolling(window=rolling_window).mean()

    def get_spot_prices(self, dates):
        return self._vol_surface.get_spot_prices(dates)

    def get_forward_prices(self, dates, maturities):

        # Get maturities
        _mats = np.unique(maturities)

        # get risk free and funding rates
        rd = self.get_risk_free_rate(self.dates, maturities)
        rf = self.get_funding_rate(self.dates, maturities)

        rd_t = rd * np.array(rd.columns).reshape(1, -1).repeat(rd.shape[0], 0)
        rf_t = rf * np.array(rd.columns).reshape(1, -1).repeat(rd.shape[0], 0)

        s_ = self.get_spot_prices(dates)

        return s_.values.reshape(-1, 1) * np.exp(-rf_t) / np.exp(-rd_t)

    def get_risk_free_rate(self, dates, maturities):
        return self._vol_surface.get_risk_free(dates, maturities)

    def get_funding_rate(self, dates, maturities):
        return self._vol_surface.get_funding_rate(dates, maturities)

    def set_vol_surface_parameters(self, interpolation_method):

        print('Loading volatility surface....')
        self._vol_surface = AbstractVolSurface(self._underlier,
                                               strike_reference=StrikeReference.CONVEXITY_MN,
                                               interpolation_method=interpolation_method)

        # Set the ivols in the object
        ivols = self._vol_surface.get_ivols(relative_strike=self._Z_SCORES,
                                            maturity=self._MATURITIES)

        self._ivols = ivols.get('vol').unstack(level=1).sort_index()
        # Set the implied strike prices
        self._k = ivols.get('k').unstack(level=1).sort_index()
        # Set spot rates
        self._s = self.get_spot_prices(self.dates)
        # Set the forward rates
        self._f = self.get_forward_prices(self.dates, self.unique_mats)
        # Set the funding rates
        self._rf = self.get_funding_rate(self.dates, self.unique_mats)
        # Set the risk free rates
        self._rd = self.get_risk_free_rate(self.dates, self.unique_mats)

    def predict_implied_moments(self):

        # For each day, predict the 1 month ahead realized covariance
        # and variance for the ATM based on the cross sectional and historical
        # realized moments.
        # The realized moments are the one month rolling values
        # The estimated moments are the cross sectional regression coefficents

        # gamma(t+1) = alpha + b(0) * gamma(t, cs) + b(1) * gamma(t, ts) + e(t+1)
        # omega(t+1) = alpha + b(0) * omega(t, cs) + b(1) * omega(t, ts) + e(t+1)

        pass

    def get_stat_arb_weights(self):
        if not hasattr(self, 'weights'):
            self.load_strategy_weights()
        return self._weights.get('sa')

    def get_risk_return_weights(self):
        if not hasattr(self, 'weights'):
           self.load_strategy_weights()
        return self._weights.get('rr')

    def load_strategy_weights(self):

        # The strategy forms weights on spread portfolios from
        # rolling estimates of the conditional moments
        # Realized Moments

        # Get the estimation dates and build indexed panels
        T0 = self._STRATEGY_STARTING_WINDOW
        L = self._STRATEGY_ESTIMATION_WINDOW
        LL = self._HISTORICAL_ROLLING_PERIODS

        pass

if __name__ == "__main__":

    self = SmileStatArb('SPX')
    self.set_vol_surface_parameters(Interpolator.GAUSSIAN_KERNEL_SMOOTHING)
    self.run_cross_sectional_spread_regressions()
