import pandas as pd
import numpy as np
from scipy.optimize import lsq_linear
from epsilonPhi.core.utils.FrameUtils import FrameUtils
from concurrent.futures import ProcessPoolExecutor
import copy
from statsmodels.api import add_constant
from scipy.stats import norm
from tqdm import tqdm

class OptionAttributer(object):

    _DATA_PATH = '/Users/francisbarker/Desktop/SPX Options 1.csv'
    _DAYS_PER_YEAR = 365.25
    _STRATEGY_HOLDING_PERIODS = 21
    _HISTORICAL_ROLLING_PERIODS = 21
    _STRATEGY_ESTIMATION_WINDOW = 4 * 252

    def __init__(self, start_date, end_date):
        self._process_data(start_date,
                           end_date)

    def _process_data(self, start_date, end_date):

        _df = pd.read_csv('/Users/francisbarker/Desktop/SPX Options 1.csv')
        _df['date'] = pd.to_datetime(_df['date'])
        _df = _df.set_index(['date', 'Z', 'T'], drop=True)

        keep_rows = np.logical_and(_df.index.get_level_values('date') >= pd.to_datetime(start_date),
                                   _df.index.get_level_values('date') <= pd.to_datetime(end_date))

        self._ds = _df.iloc[keep_rows].copy()
        self._start_date = self._ds.index.get_level_values('date').min()
        self._end_date = self._ds.index.get_level_values('date').max()

    @property
    def T(self):
        return len(self.dates)
    @property
    def X(self):
        return pd.DataFrame(np.array(self.sig.columns.get_level_values('Z')).reshape(1, -1).repeat(self.T, 0),
                            index=self.dates, columns=self.sig.columns)
    @property
    def dates(self):
        return self._ds.index.get_level_values('date').unique()
    @property
    def unique_strikes(self):
        return np.array(self._ds.index.get_level_values('Z').unique()).reshape(1, -1)
    @property
    def unique_maturities(self):
        return np.array(self._ds.index.get_level_values('T').unique()).reshape(1, -1)
    @property
    def maturities(self):
        return pd.DataFrame(np.array(self.sig.columns.get_level_values('T')).reshape(1, -1).repeat(self.T, 0),
                            index=self.dates, columns=self.sig.columns)
    @property
    def log_moneyness(self):
        return self.z_plus - 0.5 * self.sig_sq * self.maturities
    @property
    def strike_prices(self):
        return pd.concat([self.moneyness.unstack()] *
                         OptionAttributer._STRATEGY_HOLDING_PERIODS, axis=1) * 100
    @property
    def moneyness(self):
        return np.exp(self.log_moneyness)
    @property
    def spot(self):
        return self._ds.get('s').droplevel([1, 2]).drop_duplicates().reindex(self.dates).ffill().to_frame().copy()
    @property
    def dS(self):
        return self._ds.get('rx').droplevel([1, 2]).drop_duplicates().reindex(self.dates).fillna(0).to_frame().copy()
    @property
    def sig(self):
        return self._ds.get('sig').to_frame().unstack([1, 2]).get('sig').reindex(self.dates).sort_index(axis=1, level=1).copy()
    @property
    def dsig(self):
        return self._ds.get('dsig').to_frame().unstack([1, 2]).get('dsig').reindex(self.dates).sort_index(axis=1, level=1).copy()
    @property
    def sig_sq(self):
        return np.power(self.sig, 2).copy()
    @property
    def atm_sig(self):
        return self.sig.get(0).sort_index(axis=1).copy()
    @property
    def atm_dsig(self):
        return self.dsig.get(0).sort_index(axis=1).copy()
    @property
    def z_plus(self):
        return self.X * self.sig * np.sqrt(self.maturities)
    @property
    def z_minus(self):
        return self.z_plus - self.sig_sq * self.maturities
    @property
    def atm_sig_sq(self):
        return np.power(self.sig, 2).get(0).copy()

    def interpolate(self, strikes, maturities):
        pass
    def get_cross_sectional_spreads(self):
        return self.sig_sq - self.sig_sq.get(0)[self.sig_sq.columns.get_level_values('T')].values
    def get_variance_spread(self, dates):
        return self.get_cross_sectional_spreads().loc[dates].reorder_levels([1, 0])
    def get_omega(self):
        return np.power(self.dsig, 2)
    def get_gamma(self):
        return self.dsig * self.dS.values

    def get_mu(self, maturity=None):

        if not hasattr(self, '_mu'):

            Asq = self.atm_sig_sq
            Asq_tau = Asq * Asq.columns.to_numpy().reshape(1, -1)

            mu_ = ((1/2) * Asq.diff(axis=1) / Asq_tau.diff(axis=1)).dropna(axis=1)
            X = 0.5 * Asq_tau.columns.values[0:-1] + 0.5 * Asq_tau.columns.values[1:]
            mu_.columns = X

            mu_[[Asq_tau.columns]] = np.nan
            mu = mu_.sort_index(axis=1).interpolate(axis=1)
            mu[mu.columns[0]] = mu_[mu_.columns[0]]
            self._mu = mu.get(Asq.columns)

        return self._mu.get(maturity, self._mu.copy())

    def get_realized_moments(self, rolling_window=None):

        # Get the volatility return
        mu_ = self.get_mu()
        mu_.columns = pd.MultiIndex.from_tuples(list(zip(['mu']*mu_.shape[1], mu_.columns)))
        mu_.columns = FrameUtils.add_level(mu_.columns, 0, 'Z')
        mu_.columns.names = ['measure','T','Z']

        # Get Omega
        omega_ = self.get_omega()
        omega_.columns = FrameUtils.add_level(omega_.columns, 'omega', 'measure')
        omega_ = omega_.reorder_levels([2, 0, 1], 1)

        # Get Gamma
        gamma_ = self.get_gamma()
        gamma_.columns = FrameUtils.add_level(gamma_.columns, 'gamma', 'measure')
        gamma_ = gamma_.reorder_levels([2, 0, 1], 1)

        mmnts = pd.concat((pd.concat((omega_, gamma_), axis=1), mu_), axis=1)

        if rolling_window is not None:
           return mmnts.rolling(rolling_window).mean()
        else:
           return mmnts.copy()

    def run_implied_vol_prediction_regressions(self):

        # X is the implied vol drift term
        # Y is the change in implied vol oevr the next time step
        # Therefore these are prediction regressions over 1 day

        mu = self.get_mu()
        intercept = pd.DataFrame(np.ones(mu.shape[0]), columns=['constant'], index=mu.index)

        reg_stats = [pd.DataFrame()]
        for col in mu.columns:

            mu_col_ = pd.concat((intercept, mu.get(col)), axis=1)
            y_ = self.atm_dsig.get(col)

            common_dates = np.intersect1d(mu_col_.index, y_.index)
            reg = np.linalg.lstsq(mu_col_.loc[common_dates],
                                  y_.loc[common_dates], rcond=None)

            res = y_ - mu_col_ @ reg[0]
            reg_stats.extend([pd.DataFrame([reg[0][0], reg[0][1],  1 - np.var(res) / np.var(y_)],
                               index=['alpha', 'beta', 'rsq'], columns=[col])])
        return pd.concat(reg_stats, axis=1)

    def get_cross_sectional_fitted_moments(self):
        if not hasattr(self, '_fitted_moments'):
            self._run_cross_sectional_spread_regressions()
        return self._fitted_moments

    def _run_cross_sectional_spread_regressions(self):

        # For each day, we compute the observed spread
        # of each IV to ATM and then regress this on
        # The following:
        # I^2 - A^2 = gamma * 2 * z_plus + omega^2 * z_plus * z_minus

        lower_bounds = [-np.inf, 0]
        upper_bounds = [np.inf, np.inf]

        S = self.get_cross_sectional_spreads()
        keep_cols = np.abs(S.columns.get_level_values(0)) <= 1

        S = S.iloc[:, keep_cols]
        z_p = 2 * self.z_plus.get(np.unique(S.columns.get_level_values(0)))
        z_m = (self.z_plus.get(np.unique(S.columns.get_level_values(0))) *
               self.z_minus.get(np.unique(S.columns.get_level_values(0))))

        X = pd.concat((z_p.unstack(), z_m.unstack()), axis=1).reorder_levels([2, 1, 0], 0).sort_index()
        Y = S.unstack().reorder_levels([2, 1, 0]).reindex(X.index)

        obs = X.index.droplevel('Z').unique()
        reg = np.ones((len(obs), 3)) * np.nan
        for t in tqdm(range(len(obs)), desc="Processing"):

            result = lsq_linear(X.loc[obs[t]].values,
                                Y.loc[obs[t]].values,
                                bounds=(lower_bounds, upper_bounds),
                                verbose=0)

            omega, gamma = result.x
            e = Y.loc[obs[t]] - X.loc[obs[t]] @ result.x
            rsq = 1 - (np.power(e, 2).mean() / np.var(Y.loc[obs[t]].values))

            reg[t, :] = omega, gamma, rsq

        self._fitted_moments = pd.DataFrame(reg, index=obs, columns=['gamma', 'omega', 'rsq'])

    def run_pca(self, start_date=None, end_date=None):

        # X is the implied vol drift term
        # Y is the change in implied vol oevr the next time step
        # Therefore these are prediction regressions over 1 day

        cov_mat = self.dsig[start_date:end_date].cov()
        L, V = np.linalg.eig(cov_mat)
        return pd.DataFrame(V, columns=L, index=self.dsig.columns).sort_index(axis=1, ascending=False)
    def bsdelta(self, F, K, r, t, st, sig, call_put=1):
        sv = sig * st
        d1 = (np.log(F / K) + 0.5 * sv ** 2) / sv
        B = np.exp(-r * t)

        if call_put < 0:
           return B * (norm.cdf(d1) - 1)
        else:
           return B * norm.cdf(d1)

    def bsvega(self, F, K, st, sig):
        """
        Dollar vega.
        """

        sv = sig * st
        d1 = (np.log(F / K) + 0.5 * sv ** 2) / sv
        return F * st * norm.pdf(d1)
    def blsprice(self, F, K, r, t, st, sig, call_put=1):

        """
        Call option pricing formula in forward space.

        Parameters:
        F : Forward price of the underlying asset
        K : Strike price
        r : Risk-free interest rate
        t : Time to expiration
        st : Time steps
        sig : Volatility of the underlying asset

        Returns:
        P : Call option price

        """

        sv = sig * st
        d1 = (np.log(F / K) + 0.5 * sv ** 2) / sv
        d2 = d1 - sv
        B = np.exp(-r * t)

        prices = call_put * B * (F *  norm.cdf(call_put * d1) - K * norm.cdf(call_put * d2))

        # Replace expired options with their payoff
        locs = t == 0
        intr_val = (call_put * (F - K)).clip(0)
        prices.values[locs] = intr_val.values[locs]
        return prices

    def get_strategy_paths(self):

        if not hasattr(self, '_paths'):

            HP = self._STRATEGY_HOLDING_PERIODS
            _paths = np.array(range(0, self.T - HP + 1)).reshape(1, self.T - HP + 1).repeat(HP, 0) + \
                     np.array(range(HP)).reshape(-1, 1).repeat(self.T - HP + 1, 1)

            sim_paths = pd.DataFrame(_paths.T)
            dt = pd.DataFrame(self.dates.values[sim_paths.values])
            sim_paths['start'] = dt.get(0)
            sim_paths['end'] = dt.get(HP-1)

            self._paths = sim_paths.set_index(['start', 'end'], drop=True)
        return self._paths.copy()

    def get_strategy_dates(self):
        dates = pd.DataFrame(self.dates.values[self.get_strategy_paths().values], index=self.get_strategy_open_dates())

        str_paths = self.get_strike_paths()
        str_dates = dates.loc[str_paths.index.get_level_values('date')]
        str_dates.index = str_paths.index
        return str_dates

    def get_strategy_open_dates(self):
        return self.get_strategy_paths().index.get_level_values('start')

    def get_spot_paths(self):
        idx = self.get_strategy_paths()
        spts = pd.DataFrame(100 * (self.spot.get('s').values[idx.values] /
                self.spot.values[idx.values[:,0]]), index=self.get_strategy_open_dates())

        strike_paths = self.get_strike_paths()
        spts = spts.loc[strike_paths.index.get_level_values(0)]
        spts.index = strike_paths.index
        return spts.copy()

    def get_strike_paths(self):
        if not hasattr(self, '_strike_paths'):
            locs = self.get_strategy_open_dates()
            self._strike_paths = self.strike_prices.reorder_levels([2,1,0]).loc[locs]
        return self._strike_paths

    def get_time_to_maturity_paths(self):
        str_dates = self.get_strategy_dates()
        days_elapsed = (str_dates - str_dates.get(0).to_frame().values) / np.timedelta64(1, 'D')
        tau = np.array(days_elapsed.index.get_level_values('T')).reshape(-1, 1) - (days_elapsed/self._DAYS_PER_YEAR)
        return tau.clip(lower=0)

    def get_vol_paths(self):
        sig = pd.concat([self.sig.unstack()] * 21, axis=1).reorder_levels([2,1,0])
        return sig.loc[self.get_strike_paths().index]

    def get_bsvega(self):
        return self.bsvega(self.get_spot_paths(),
                           self.get_strike_paths(),
                           np.sqrt(self.get_time_to_maturity_paths()),
                           self.get_vol_paths())

    def get_option_deltas(self, put_call=1):
        return self.bsdelta(self.get_spot_paths(),
                            self.get_strike_paths(),
                            0,
                            self.get_time_to_maturity_paths(),
                            np.sqrt(self.get_time_to_maturity_paths()),
                            self.get_vol_paths(), put_call)

    def get_option_prices(self, put_call=-1):
        return self.blsprice(self.get_spot_paths(),
                             self.get_strike_paths(),
                             0,
                             self.get_time_to_maturity_paths(),
                             np.sqrt(self.get_time_to_maturity_paths()),
                             self.get_vol_paths(), put_call)

    def get_option_pnls(self, put_calls=-1, position=-1):
        prices = position * self.get_option_prices(put_calls)
        return prices.diff(axis=1).dropna(axis=1, how='all')

    def get_option_cumulative_pnl(self, put_call=-1, position=-1):
        return self.get_option_pnls(put_call, position).cumsum(axis=1)

    def get_rates(self):
        return self.get_time_to_maturity_paths() * 0

    def get_account_pnl(self, put_call=-1, position=-1):
        prices = self.get_option_prices(put_call)
        premium = position * prices.get(0).to_frame()

        rates = self.get_rates()
        t = self.get_time_to_maturity_paths()
        return (np.exp(rates * t.diff(axis=1).abs()) - 1).dropna(axis=1) * premium.values

    def get_delta_hedge_pnl(self, put_call, position):

        option_deltas = self.get_option_deltas(put_call)
        position_deltas = position * option_deltas

        ds = self.get_spot_paths().diff(axis=1).dropna(axis=1)
        return -1 * position_deltas.iloc[:,0:-1].values * ds

    def get_delta_hedge_account_pnl(self, put_call, position):

        option_deltas = self.get_option_deltas(put_call)
        position_deltas = position * option_deltas
        balance = position_deltas * self.get_spot_paths()

        rates = self.get_rates()
        t = self.get_time_to_maturity_paths()
        return balance.values[:,0:-1] * (np.exp(rates * t.diff(axis=1).abs()) - 1).dropna(axis=1)

    def get_delta_hedged_single_option_strategy_pnl(self, put_call=-1, position=-1):

        pnl_opt = self.get_option_pnls(put_call, position)
        pnl_int = self.get_account_pnl(put_call, position)
        pnl_hdg = self.get_delta_hedge_pnl(put_call, position)
        pnl_acc = self.get_delta_hedge_account_pnl(put_call, position)

        return pnl_opt + pnl_int + pnl_hdg + pnl_acc

    def get_delta_hedged_single_option_strategy_cumulative_pnl(self, put_call=-1, position=-1):
        return self.get_delta_hedged_single_option_strategy_pnl(put_call, position)

    def run_spread_forecasting_regression(self):

        # For each day, predict the 1 month ahead realized covariance
        # and variance for the ATM based on the cross sectional and historical
        # realized moments.
        # The realized moments are the one month rolling values
        # The estimated moments are the cross sectional regression coefficents

        # gamma(t+1) = alpha + b(0) * gamma(t, cs) + b(1) * gamma(t, ts) + e(t+1)
        # omega(t+1) = alpha + b(0) * omega(t, cs) + b(1) * omega(t, ts) + e(t+1)

        # Realized Moments
        r_mnts = self.get_realized_moments(self._HISTORICAL_ROLLING_PERIODS)
        omega_ts = r_mnts.get('omega').get(0) * 252
        gamma_ts = r_mnts.get('gamma').get(0) * 252

        # Cross Sectional Moments
        c_mnts = self.get_cross_sectional_fitted_moments()
        omega_cs = c_mnts.get('omega').unstack()
        gamma_cs = c_mnts.get('gamma').unstack()

        b_cov = list()
        b_var = list()
        for mat in self.unique_maturities.flatten():

            # Forecast Omega
            y_o = omega_ts.dropna().shift(-21).get(mat).dropna()
            X_o = pd.concat((omega_cs.get(mat), omega_ts.get(mat)), axis=1).reindex(y_o.index)
            X_o_prime = pd.concat((X_o.iloc[:, 0] * 0 + 1, X_o), axis=1)
            B_o, _, _, _ = np.linalg.lstsq(X_o_prime, y_o, rcond=None)
            res_o = y_o - X_o_prime @ B_o
            rsq_o = 1 - np.var(res_o) / np.var(y_o)
            b_var.extend([np.append(B_o, rsq_o)])

            # Forecast Gamma
            y_g = gamma_ts.dropna().shift(-21).get(mat).dropna()
            X_g = pd.concat((gamma_cs.get(mat), gamma_ts.get(mat)), axis=1).reindex(y_o.index)
            X_g_prime = pd.concat((X_g.iloc[:, 0] * 0 + 1, X_g), axis=1)
            B_g, _, _, _ = np.linalg.lstsq(X_g_prime, y_g, rcond=None)
            res_g = y_g - X_g_prime @ B_g
            rsq_g = 1 - np.var(res_g) / np.var(y_g)
            b_cov.extend([np.append(B_g, rsq_g)])

        res_var = pd.DataFrame(b_var, index=self.unique_maturities.flatten(), columns=['alpha', 'beta_cs', 'beta_ts', 'rsq'])
        res_cov = pd.DataFrame(b_cov, index=self.unique_maturities.flatten(), columns=['alpha', 'beta_cs', 'beta_ts', 'rsq'])

        print('Table 7: Predicted Realized Variance/Covariance with Cross-Sectional and Time Series Estimators')
        print()
        print('Panel A; Covariance')
        print(res_cov)
        print()
        print('Panel B; Variance')
        print(res_var)

    def get_vega_weights(self):
        bsvega = self.get_bsvega()
        bsvega_atm = bsvega.iloc[bsvega.index.get_level_values(2) == 0, :].droplevel(2, axis=0)
        bsvega_atm = bsvega_atm.loc[bsvega.index.droplevel(2)]
        return (bsvega_atm.values / bsvega).get(0).to_frame()

    def get_short_put_spread_pnls(self):
        pnls = self.get_delta_hedged_single_option_strategy_pnl(put_call=-1, position=-1)

        vega_neutral_wts = self.get_vega_weights()
        pnls_vega_weighted = pnls * vega_neutral_wts.values

        atm_pnls = pnls.iloc[pnls.index.get_level_values(2) == 0, :].droplevel(2, axis=0)
        atm_pnls = atm_pnls.loc[pnls.index.droplevel(2)]
        return pnls_vega_weighted - atm_pnls.values

    def get_short_call_spread_pnls(self):
        pnls = self.get_delta_hedged_single_option_strategy_pnl(put_call=1,position=-1)

        vega_neutral_wts = self.get_vega_weights()
        pnls_vega_weighted = pnls * vega_neutral_wts.values

        atm_pnls = pnls.iloc[pnls.index.get_level_values(2) == 0, :].droplevel(2, axis=0)
        atm_pnls = atm_pnls.loc[pnls.index.droplevel(2)]
        return pnls_vega_weighted - atm_pnls.values

    def get_2_z_plus(self):
        return FrameUtils.set_levels(2 * self.z_plus, 'g', 'measure')

    def get_z_plus_z_minus(self):
        return FrameUtils.set_levels(self.z_plus * self.z_minus, 'g', 'measure')

    def get_time_series_moments(self, moneyness=0):

        ts = self.get_realized_moments(self._HISTORICAL_ROLLING_PERIODS)
        ts = FrameUtils.set_levels(ts, level_values='ts', level_name='estimator')
        ts.columns.names = ['measure', 'x', 'mat', 'estimator']
        ts = ts.reorder_levels(['measure', 'mat', 'x', 'estimator'], axis=1)
        if moneyness:
            return ts.iloc[:, ts.columns.get_level_values('x') == moneyness]
        else:
            return ts.copy()

    def get_cross_sectional_moments(self):

        ts = self.get_cross_sectional_fitted_moments()
        ts['estimator'] = 'cs'
        ts = ts.reset_index(drop=False).set_index(['date', 'T', 'estimator']).unstack(level=[1, 2])
        ts = FrameUtils.set_levels(ts, level_values=0, level_name='x')
        ts.columns.names = ['measure', 'T', 'estimator', 'x']
        ts = ts.reorder_levels(['measure', 'T', 'x', 'estimator'], axis=1)
        ts.columns.names = ['measure', 'mat', 'z', 'estimator']
        return ts[['gamma', 'omega']]

    def get_cross_sectional_and_time_series_moments(self):
        return pd.concat((self.get_cross_sectional_moments(),
                          252 * self.get_time_series_moments()), axis=1)

    def _build_idxs(self, T, N):
        return (np.array(range(T)).reshape(-1, 1).repeat(repeats=N, axis=1) +
                np.array(range(0, N)).reshape(1, -1).repeat(T, 0))

    def _load_strategy_weights(self):
        pass

    def get_strategy_weights(self, strategy):
        if strategy not in self._weights.keys():
            self._load_strategy_weights()
        return self._weights.get(strategy)

    def run_risk_return_strategy(self):

        # The strategy forms weights on spread portfolios from
        # rolling estimates of the conditional moments
        # Realized Moments

        # Get the data that forms our regression set and the forward values
        ts_est = self.get_cross_sectional_and_time_series_moments()
        ts_fwd = ts_est.dropna().shift(-21)

        # Get the smile coefficients
        zp = self.get_2_z_plus()
        zm = self.get_z_plus_z_minus()
        X = pd.concat((zp, zm), axis=1).reorder_levels([1, 0, 2], axis=1)

        # Variables for regressions - Omegas
        y_o = ts_fwd.get('omega')
        X_o = ts_est.get('omega')

        # Variables for regressions - Gamma
        y_g = ts_fwd.get('gamma')
        X_g = ts_est.get('gamma')

        # Get the estimation dates
        win_size = 253 * 4 - self._STRATEGY_HOLDING_PERIODS
        N = self.T - self._STRATEGY_HOLDING_PERIODS
        idxs = self._build_idxs(win_size, N)

        for mat in self.unique_maturities.flatten():
            for t in range(N):
                t_ = t + win_size + self._STRATEGY_HOLDING_PERIODS - 1

                ############# Estimate it for Omega - Vol of Vol ############################
                X_prime_o = X_o.iloc[idxs[:, t], :].dropna().get(mat)
                Xo = add_constant(X_prime_o.get(mat))
                yo_prime = y_o.loc[Xo.index].get(mat)

                # Fit Model Over History - 4 Years Minus One Month
                B_omega = np.linalg.solve(Xo.T @ Xo + np.eye(3), Xo.T @ yo_prime)
                # Using The Fitted Model - Forecast One Month Ahead
                omega_predict = (X_o.iloc[t_].loc[mat] @ B_omega[1:] + B_omega[0]).item()

                ############# Estimate it for Gamma - Spot/Vol Covariance ###################
                X_prime_g = X_g.iloc[idxs[:, t], :].dropna()
                Xg = add_constant(X_prime_g.get(mat))
                yg_prime = y_g.loc[Xg.index].get(mat)

                # Fit Model Over History - 4 Years Minus One Month
                B_gamma = np.linalg.solve(Xg.T @ Xg + np.eye(3), Xg.T @ yg_prime)
                # Using The Fitted Model - Forecast One Month Ahead
                gamma_predict = (X_g.iloc[t_].loc[mat] @ B_gamma[1:] + B_gamma[0]).item()

                # Predict Spread
                _bp = [gamma_predict, omega_predict]
                _x = X.iloc[t_].get(mat).unstack()[['g','o']]
                prd_spd = _x @ _bp

                obs_spd = self.get_variance_spread(self.dates[t_]).get(mat)
                atm = self.atm_sig_sq.loc[self.dates[t_]].get(mat)
                wts = (obs_spd - prd_spd).to_frame('rr') * (1/atm)
                wts['T'] = mat
                wts['date'] = self.dates[t_]

                bp_ = [gamma_cs.loc[self.dates[t_]].get(mat).values.item(),
                       omega_cs.loc[self.dates[t_]].get(mat).values.item()]
                prd_res = _x @ bp_
                wts_arb = 10 * (obs_spd - prd_res).to_frame('arb') * (1/atm)
                wts = pd.concat((wts, wts_arb), axis=1)







    def run_stat_arb_strategy(self):
        pass

    def _print_table(self, table_number):

        assert table_number in [5, 6], 'Error - table number {} not supported'.format(table_number)
        if table_number == 5:

            vars_ = self.get_realized_moments(21) * 252

            # Implied Vol Level
            df = self.sig.describe().loc['mean'].unstack().T
            df.columns.names = ['Strike']
            df.index.names = ['Maturity']
            print('Table 5: A : Mean Implied Vol Smile')
            print(df)

            # Covariance Estimates
            gamma = vars_.get('gamma').describe().loc['mean'].unstack().T
            gamma.columns.names = ['Strike']
            gamma.index.names = ['Maturity']
            print('Table 5: B : Historical Covariance (Skewness) Estimates')
            print(gamma)

            # Variance Estimates
            omega = vars_.get('omega').describe().loc['mean'].unstack().T
            omega.columns.names = ['Strike']
            omega.index.names = ['Maturity']
            print('Table 5: C : Historical Variance (Vol of Vol) Estimates')
            print(omega)

        if table_number == 6:
            vars_ = self.get_cross_sectional_fitted_moments()

            # Estimated Cross-Sectional Gamma
            gamma = vars_.get('gamma').unstack().describe().loc[['mean', 'std', 'min', 'max']]
            gamma.columns.names = ['Maturity']
            print('Table 6: Cross-Sectional Regression Estimates of Variance and Covariance Rates')
            print()
            print('Panel A: Covariance Estimates')
            print()
            print(gamma)

            # Estimated Cross-Sectional Omega
            omega = vars_.get('omega').unstack().describe().loc[['mean', 'std', 'min', 'max']]
            omega.columns.names = ['Maturity']
            print('Panel B: Variance Estimates')
            print()
            print(omega)


            # R-Squared
            rsq = vars_.get('rsq').unstack().describe().loc[['mean', 'std', 'min', 'max']]
            rsq.columns.names = ['Maturity']
            print('Panel C: R-Squared')
            print()
            print(rsq)




if __name__ == "__main__":
    self = OptionAttributer('31-Dec-1990', '31-Dec-2025')
    self.run_risk_return_strategy()

