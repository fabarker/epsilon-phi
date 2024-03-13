from epsilonPhi.core.dataModel.dataSources.impliedVolatility.VolSurface import AbstractVolSurface
from epsilonPhi.core.utils.FrameUtils import FrameUtils
from epsilonPhi.core.utils.OptionUtils import *
from epsilonPhi.core.dataModel.enums.ImpliedVolatility import *
from scipy.optimize import lsq_linear
import numpy as np
from tqdm import tqdm
import pandas as pd
import itertools


class SmileStatArb(object):

    _STRATEGY_HOLDING_PERIODS = 21
    _HISTORICAL_ROLLING_PERIODS = 21

    _STRATEGY_ESTIMATION_WINDOW = 4 * 252
    _STRATEGY_STARTING_WINDOW = 253 * 4
    _TRUNCATE_SPREAD = np.abs(1)

    _STRATEGY_STRIKE_CUTOFF = 1
    _Z_SCORES = np.arange(-2, 2.5, 0.5)
    _MATURITIES = [1 / 12, 2 / 12, 3 / 12, 6 / 12, 12 / 12]

    def __init__(self, underlier, start_date=None, end_date=None):

        self._start_date = start_date
        self._end_date = end_date

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
    def dates(self):
        return self._ivols.index
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
    def strategy_strikes(self):
        return self.unique_x[(np.abs(self.unique_x) <= self._STRATEGY_STRIKE_CUTOFF) &
                             (self.unique_x != 0)]
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
        return (self.lnm + 0.5 * self.sig_sq * self.t).reindex(self.dates)
    @property
    def z_minus(self):
        return (self.z_plus - self.sig_sq * self.t).reindex(self.dates)
    @property
    def omega_ts(self):
        return self.dsig_sq.reindex(self.dates)
    @property
    def gamma_ts(self):
        return (self.dsig * self.ds.values.reshape(-1, 1)).reindex(self.dates)
    @property
    def mu_ts(self):
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
        for t in tqdm(range(self.T), desc="Estimating Cross Sectional Moments"):
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
        _df = pd.DataFrame(reg, columns=['date', 't', 'gamma', 'omega']).set_index(['date', 't'])
        self._cross_sectional_estimates = _df.unstack(level=1).reindex(self.dates)

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

        s_ = self.get_spot_prices(rf_t.index)

        return s_.values.reshape(-1, 1) * np.exp(-rf_t) / np.exp(-rd_t)

    def get_risk_free_rate(self, dates, maturities):
        return self._vol_surface.get_risk_free(dates, maturities)

    def get_funding_rate(self, dates, maturities):
        return self._vol_surface.get_funding_rate(dates, maturities)

    def set_vol_surface_parameters(self, interpolation_method):

        print('Loading volatility surface....')
        self._vol_surface = AbstractVolSurface(self._underlier,
                                               strike_reference=StrikeReference.Z_SCORE,
                                               interpolation_method=interpolation_method)

        # Set the ivols in the object
        ivols = self._vol_surface.get_ivols(relative_strike=self._Z_SCORES,
                                            maturity=self._MATURITIES)

        self._ivols = ivols.get('sig').unstack(level=[1, 2]).sort_index().loc[self._start_date:self._end_date]
        self._ivols.columns.names = ['k', 't']
        self._ivols = self._ivols.dropna(how='any', axis=0)
        # Set the implied strike prices
        self._k = ivols.get('k').unstack(level=[1, 2]).sort_index()
        self._k.columns.names = ['k', 't']
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

    def get_predictors(self):

        _rolling_window = self._HISTORICAL_ROLLING_PERIODS
        _ocs = self.omega_cs.values.reshape(self.T, 1, self.NM)
        _gcs = self.gamma_cs.values.reshape(self.T, 1, self.NM)
        _ots = self.omega_ts.get(0).rolling(_rolling_window).mean().values.reshape(self.T, 1, self.NM) * 252
        _gts = self.gamma_ts.get(0).rolling(_rolling_window).mean().values.reshape(self.T, 1, self.NM) * 252
        return np.hstack((np.ones((_ocs.shape)), _ocs, _ots, _gcs, _gts))

    def get_strategy_weights(self, strategy):
        if not hasattr(self, '_weights'):
            self.load_strategy_weights()
        return self._weights.get(strategy)

    def load_strategy_weights(self):

        # The strategy forms weights on spread portfolios from
        # rolling estimates of the conditional moments
        # Realized Moments

        # Get the estimation dates and build indexed panels
        T0 = self._STRATEGY_STARTING_WINDOW
        L = self._STRATEGY_ESTIMATION_WINDOW
        LL = self._HISTORICAL_ROLLING_PERIODS

        # Get the predictors for estimating 1 period ahead gamma and omega.
        x = self.get_predictors()

        z_p = 2 * self.z_plus.get(self.strategy_strikes).reorder_levels([1, 0], axis=1)
        z_p_ = np.stack([z_p.get(x).sort_index().values for x in self.unique_mats], axis=2)

        z_pm = (self.z_plus * self.z_minus).get(self.strategy_strikes).reorder_levels([1, 0], axis=1)
        z_pm_ = np.stack([z_pm.get(x).sort_index().values for x in self.unique_mats], axis=2)

        s = self.get_variance_spreads().reindex(self.dates).get(self.strategy_strikes).reorder_levels([1, 0], axis=1)
        s_ = np.stack([s.get(x).sort_index().values for x in self.unique_mats], axis=2)

        atms = np.power(self.atm_sig, 2).reindex(self.dates).values

        weights = np.full((self.T-LL-T0+1, len(self.strategy_strikes), self.NM, 2), np.nan)
        for j in tqdm(range(self.NM), desc="Building Strategy Weights"):
            for t in range(T0-1, self.T-LL):

                idx_1 = np.arange(np.maximum(t-L-LL, 0), t-LL+1)
                idx_2 = idx_1 + LL

                # 1. Predict 1 Period ahead omega: omega(t+1) = alpha + b(0) * omega(t, cs) + b(1) * omega(t, ts) + e(t+1)
                # Intercept = 0, Omega CS = 1, Gamma TS = 2
                x_0 = x[idx_1, :3, j]
                _nan_loc = ~np.any(np.isnan(x_0), axis=1)
                x_o = x_0[_nan_loc, :]
                B_omega = np.round(np.linalg.solve(x_o.T @ x_o + np.eye(3), x_o.T @ x[idx_2[_nan_loc], 2, j]), 5)
                omega_predict = [1, x[t, 1, j], x[t, 2, j]] @ B_omega

                #2 . Predict 1 Periof ahead gamma: gamma(t+1) = alpha + b(0) * gamma(t, cs) + b(1) * gamma(t, ts) + e(t+1)
                # Intercept = 0, Gamma CS = 3, Gamma TS = 4
                x_G = x[idx_1, :, j][:, [0, 3, 4]]
                _nan_loc = ~np.any(np.isnan(x_G), axis=1)
                x_g = x_G[_nan_loc, :]
                B_gamma = np.linalg.solve(x_g.T @ x_g + np.eye(3), x_g.T @ x[idx_2[_nan_loc], 4, j])
                gamma_predict = [1, x[t, 3, j], x[t, 4, j]] @ B_gamma

                # 3. From Omega and Gamma Estimates, Predict the Implied Variance Spread
                _bp = [gamma_predict, omega_predict]
                pred_spr_ts = np.vstack((z_p_[t, :, j], z_pm_[t, :, j])).T @ _bp

                # 4. From The Cross Section Estimates for Omega and Gamma, Get Residuals
                pred_spr_cs = np.vstack((z_p_[t, :, j], z_pm_[t, :, j])).T @ x[t, [3, 1], j]

                # 4. Positions are proportional to the difference between observed and predicted spreads
                weights[t-T0+1, :, j, 0] = (s_[t, :, j] - pred_spr_ts) / atms[t, j]
                weights[t-T0+1, :, j, 1] = (10/atms[t, j]) * (s_[t, :, j] - pred_spr_cs)

        NX_ = len(self.strategy_strikes)

        rr = [pd.DataFrame(weights[:, :, x, 0], columns=FrameUtils.multiindex(['rr'] * NX_, self.strategy_strikes, [self.unique_mats[x]] * NX_)) for x in range(self.NM)]
        rr = pd.concat(rr, axis=1)
        rr.index = self.dates[np.arange(T0-1, self.T-LL)]

        sa = [pd.DataFrame(weights[:, :, x, 1], columns=FrameUtils.multiindex(['sa'] * NX_, self.strategy_strikes, [self.unique_mats[x]] * NX_)) for x in range(self.NM)]
        sa = pd.concat(sa, axis=1)
        sa.index = self.dates[np.arange(T0 - 1, self.T - LL)]
        self._weights = pd.concat((rr, sa), axis=1)

    def load_strategy_paths(self):

        HP = self._STRATEGY_HOLDING_PERIODS
        _paths = np.array(range(0, self.T - HP + 1)).reshape(1, self.T - HP + 1).repeat(HP, 0) + \
                 np.array(range(HP)).reshape(-1, 1).repeat(self.T - HP + 1, 1)

        sim_paths = pd.DataFrame(_paths.T)

        dates_ = pd.DataFrame(self.dates.values[sim_paths.values])
        dates_['start'] = dates_.get(0)
        dates_['end'] = dates_.get(HP - 1)

        sim_paths['start'] = dates_.get(0)
        sim_paths['end'] = dates_.get(HP - 1)

        self._paths = sim_paths.set_index(['start', 'end'], drop=True)

    @property
    def strategy_open_dates(self):
        return pd.to_datetime(self.get_paths().index.get_level_values('start'))
    @property
    def strategy_open_vols(self):
        return self.ivols.stack(level=[0, 1]).loc[self.strategy_open_dates]
    @property
    def strategy_strike_prices(self):
        return 100 * pd.concat([self.moneyness.stack(level=[0, 1])] *
                                self.get_paths().shape[1], axis=1).loc[self.strategy_open_dates]

    def get_strategy_path_dates(self):
        _paths = self.get_paths()
        _d = pd.DataFrame(self.dates.values[_paths.values], index=self.strategy_open_dates)
        __d = _d.loc[self.strategy_open_vols.index.get_level_values(0)]
        __d.index = self.strategy_open_vols.index
        return __d.copy()
    def get_ttm_paths(self):

        str_dates = self.get_strategy_path_dates()
        days_elapsed = (str_dates - str_dates.get(0).to_frame().values) / np.timedelta64(1, 'D')
        tau = np.array(days_elapsed.index.get_level_values('t')).reshape(-1, 1) - (days_elapsed / 365.25)
        return tau.clip(lower=0)

    def get_log_moneyness_paths(self):
        _strikes = self.strategy_strike_prices
        _fwds = self.get_forward_paths()
        return np.log(_strikes.values / _fwds)

    def get_ivol_paths(self):
        ivols = pd.concat([self.strategy_open_vols] * self._STRATEGY_HOLDING_PERIODS, axis=1)

        #_d = self.get_strategy_path_dates().stack()
        #_k = self.get_strategy_log_moneyness().stack()
        #_t = self.get_strategy_ttm().stack()

        #_idx = list(zip(_d, _k, _t))

        #ivols = self._vol_surface.get_ivols(pricing_dates=_d,
        #                                    strike_reference=StrikeReference.LOG_MONEYNESS,
        #                                    relative_strike=_k,
        #                                    maturity=_t)
        return ivols

    def get_forward_paths(self):

        if not hasattr(self, '_fwd_paths'):

            _rf = self.get_funding_rate_paths()
            _rd = self.get_risk_free_rate_paths()
            _s = self.get_spot_paths()
            _t = self.get_ttm_paths()
            self._fwd_paths = _s * np.exp(_rf * _t) / np.exp(_rd * _t)
        return self._fwd_paths

    def get_spot_paths(self):
        idx = self.get_paths()

        spts = self.s.values[idx.values].reshape(idx.shape[0], idx.shape[1])
        _spts = pd.DataFrame(spts, index=self.strategy_open_dates)
        norm = _spts / _spts[[0]].values

        spts = norm.loc[self.strategy_open_vols.index.get_level_values(0)]
        spts.index = self.strategy_open_vols.index
        return spts.copy() * 100

    def get_paths(self):
        if not hasattr(self, '_paths'):
            self.load_strategy_paths()
        return self._paths.copy()

    def get_risk_free_rate_paths(self):

        if not hasattr(self, '_rd_paths'):
            _t = self.get_ttm_paths().stack()
            _d = self.get_strategy_path_dates().stack()

            rd = self.get_risk_free_rate(_d.values, _t.values).stack()
            _rd = rd.loc[zip(_d, _t)]
            _rd.index = _d.index
            self._rd_paths = _rd.unstack(level=3)

        return self._rd_paths.copy()


    def get_funding_rate_paths(self):

        if not hasattr(self, '_rf_paths'):
            _t = self.get_ttm_paths().stack()
            _d = self.get_strategy_path_dates().stack()

            rf = self.get_funding_rate(_d.values, _t.values).stack()
            _rf = rf.loc[list(zip(_d, _t))]
            _rf.index = _d.index
            self._rf_paths = _rf.unstack(level=3)
        return self._rf_paths.copy()

    def blsprice(self, f, t, k, rf, v, option_type):
        return blsprice(f, t, k, rf, v, option_type)

    def blsdelta(self, option_type):

        _s = self.get_spot_paths()
        _t = self.get_ttm_paths()
        _k = self.strategy_strike_prices
        _rd = self.get_risk_free_rate_paths()
        _rf = self.get_funding_rate_paths()
        _vol = self.get_ivol_paths()

        delta = fast_delta(_s,
                          _t,
                          _k,
                          _rd,
                          _rf,
                          _vol,
                          1,
                          option_type)
        return delta

    def get_bsvega(self):
        _s = self.get_spot_paths()
        _k = self.strategy_strike_prices
        _t = self.get_ttm_paths()
        _v = self.get_ivol_paths()
        _r = self.get_risk_free_rate_paths()
        _q = self.get_funding_rate_paths()
        return bs_vega(_s, _t, _k, _r, _q, _v)

    def get_option_prices(self, option_type=-1):

        _f = self.get_forward_paths()
        _t = self.get_ttm_paths()
        _k = self.strategy_strike_prices
        _rf = self.get_risk_free_rate_paths()
        _v = self.get_ivol_paths()

        prices = self.blsprice(_f, _t, _k, _rf, _v, option_type=option_type)

        locs = _t == 0
        intr_val = (option_type * (_f - _k)).clip(0)
        prices.values[locs] = intr_val.values[locs]
        return prices

    def get_path_option_pnls(self, put_calls=-1, position=-1):
        prices = position * self.get_option_prices(put_calls)
        return prices.diff(axis=1).dropna(axis=1, how='all')

    def get_option_cumulative_pnl(self, put_call=-1, position=-1):
        return self.get_path_option_pnls(put_call, position).cumsum(axis=1)

    def get_account_pnl(self, put_call=-1, position=-1):
        prices = self.get_option_prices(put_call)
        premium = position * prices.get(0).to_frame()

        rates = self.get_risk_free_rate_paths()
        t = self.get_ttm_paths()
        return (np.exp(rates * t.diff(axis=1).abs()) - 1).dropna(axis=1) * premium.values

    def get_delta_hedge_pnl(self, put_call=-1, position=-1):

        option_deltas = self.blsdelta(put_call)
        position_deltas = position * option_deltas

        ds = self.get_forward_paths().diff(axis=1).dropna(axis=1)
        return -1 * position_deltas.iloc[:, 0:-1].values * ds

    def get_delta_hedge_account_pnl(self, put_call, position):

        option_deltas = self.blsdelta(put_call)
        position_deltas = position * option_deltas
        balance = position_deltas * self.get_forward_paths()

        rates = self.get_risk_free_rate_paths()
        t = self.get_ttm_paths()
        return balance.values[:, 0:-1] * (np.exp(rates * t.diff(axis=1).abs()) - 1).dropna(axis=1)

    def get_vega_weights(self):
        bsvega = self.get_bsvega()
        bsvega_atm = bsvega.iloc[bsvega.index.get_level_values(1) == 0, :].droplevel(1, axis=0)
        bsvega_atm = bsvega_atm.loc[bsvega.index.droplevel(1)]
        return (bsvega_atm.values / bsvega).get(0).to_frame()

    def get_delta_hedged_single_option_strategy_pnl(self, put_call=-1, position=-1):

        pnl_opt = self.get_path_option_pnls(put_call, position)
        pnl_int = self.get_account_pnl(put_call, position)
        pnl_hdg = self.get_delta_hedge_pnl(put_call, position)
        pnl_acc = self.get_delta_hedge_account_pnl(put_call, position)

        return pnl_opt + pnl_int + pnl_hdg + pnl_acc

    def get_short_put_spread_pnls(self):
        pnls = self.get_delta_hedged_single_option_strategy_pnl(put_call=-1, position=-1)

        vega_neutral_wts = self.get_vega_weights()
        pnls_vega_weighted = pnls * vega_neutral_wts.values

        atm_pnls = pnls.iloc[pnls.index.get_level_values(1) == 0, :].droplevel(1, axis=0)
        atm_pnls = atm_pnls.loc[pnls.index.droplevel(1)]
        return pnls_vega_weighted - atm_pnls.values

    def run_stat_arb_strategy(self):

        # Get Strategy Weights
        _wts = self.get_strategy_weights('sa')
        _wts = _wts.unstack().reorder_levels([2, 1, 0]).sort_index(level=0)

        # Get Asset PnLs
        _pnls = self.get_short_put_spread_pnls().reorder_levels([0, 2, 1])
        _pnls = _pnls.reindex(_wts.index)

        # Compute Strategy PnL
        str_pnl = _wts.values.reshape(-1, 1).repeat(_pnls.shape[1], 1) * _pnls
        str_pnl = str_pnl[str_pnl.index.get_level_values(1) != 0]

        stk_pnls = str_pnl.stack(0).to_frame()
        stk_pnls.index.names = ['start', 'mat', 'x', 'days']

        stk_pnls['pricing_dates'] = (stk_pnls.index.get_level_values('start')
                                     + pd.to_timedelta(stk_pnls.index.get_level_values('days'), unit='D'))

        pnls = stk_pnls.reset_index(drop=False).set_index(['pricing_dates', 'start', 'mat', 'x']).drop(columns=['days'])
        return pnls.unstack(level=[1, 2, 3])

    def run_risk_return_strategy(self):

        # Get Strategy Weights
        _wts = self.get_strategy_weights('rr')
        _wts = _wts.unstack().reorder_levels([2, 1, 0]).sort_index(level=0)

        # Get Asset PnLs
        _pnls = self.get_short_put_spread_pnls().reorder_levels([0, 2, 1])
        _pnls = _pnls.reindex(_wts.index)

        # Compute Strategy PnL
        str_pnl = _wts.values.reshape(-1, 1).repeat(_pnls.shape[1], 1) * _pnls
        str_pnl = str_pnl[str_pnl.index.get_level_values(1) != 0]

        stk_pnls = str_pnl.stack(0).to_frame()
        stk_pnls.index.names = ['start', 'mat', 'x', 'days']

        stk_pnls['pricing_dates'] = (stk_pnls.index.get_level_values('start')
                                     + pd.to_timedelta(stk_pnls.index.get_level_values('days'), unit='D'))

        pnls = stk_pnls.reset_index(drop=False).set_index(['pricing_dates', 'start', 'mat', 'x']).drop(columns=['days'])
        return pnls.unstack(level=[1, 2, 3])

    def get_returns_table_for_strategy(self, strategy='sa'):

        if strategy.lower() == 'sa':
            _pnls = self.run_stat_arb_strategy()
        else:
            _pnls = self.run_risk_return_strategy()

        _pnls_M = _pnls.droplevel(0, axis=1).sum(axis=0)
        cml_pnls = _pnls_M.to_frame().unstack([1, 2]).droplevel(0, axis=1)
        ann_rtns = cml_pnls.mean(axis=0).unstack() * 12
        return ann_rtns






if __name__ == "__main__":

    _start_date = pd.to_datetime('12-Dec-1996')
    _end_date = pd.to_datetime('29-Apr-2024')

    self = SmileStatArb('SPX', start_date=_start_date, end_date=_end_date)
    self.set_vol_surface_parameters(Interpolator.GAUSSIAN_KERNEL_SMOOTHING)

    rr = self.get_returns_table_for_strategy('rr')
    sa = self.get_returns_table_for_strategy('sa')

    _sig = self.sig.stack(level=[0,1]).to_frame('sig')
    _dsig = self.dsig.stack(level=[0,1]).to_frame('dsig')
    _sigs = pd.concat((_sig, _dsig), axis=1)

    _dates = _sig.index.get_level_values(0)
    _s = self.s.loc[_dates].to_frame('s')
    _ds = self.ds.loc[_dates].to_frame('ds')

    _sigs['s'] = _s.values
    _sigs['ds'] = _ds.values
    _sigs.to_csv('For Matlab.csv')


    _pnls = self.run_risk_return_strategy()

    # Convert to cumulative PnLs
    pnls = _pnls.get(0).stack(level=['start', 'x', 'mat']).reorder_levels([1,0,2,3])
    pnls_ = pnls.unstack(level=['x','mat'])

    _prices = prices.reorder_levels([1,2,0]).loc[-1].loc[np.round(1/12, 10)].stack().to_frame('p')
    _prices['pricing_dates'] = (_prices.index.get_level_values(0)
                                 + pd.to_timedelta(_prices.index.get_level_values(1), unit='D'))

