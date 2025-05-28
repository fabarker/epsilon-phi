from epsilonPhi.core.dataModel.enums.Rates import RateType
from epsilonPhi.core.schema.Schema import ContextCreator
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import CubicSpline
from sklearn.linear_model import LinearRegression

class SimulationHelper:

    @staticmethod
    def get_inflation_mu_path(
            forward_rates,
            r_star,
            dt,
            T,
            convergence_target=0.95,
            convergence_years=5,
    ):

        rend = forward_rates[-1]
        a_decay = -np.log(1 - convergence_target) / convergence_years

        yearly_years = np.arange(0, T + 1)
        start_decay = len(forward_rates) - 2
        decay_years = yearly_years[start_decay + 1:]
        mu_decay = rend * np.exp(-a_decay * (decay_years - start_decay)) + r_star * (
                    1 - np.exp(-a_decay * (decay_years - start_decay)))
        mu_yearly = np.concatenate([forward_rates, mu_decay])

        # Step 1: Convert yearly % returns to log returns
        log_returns = np.log(1 + mu_yearly)  # shape (N,)

        # Step 2: Construct log cumulative return
        log_cumulative = np.cumsum(log_returns)  # shape (N,)

        # Step 3: Fit spline over actual years (0, 1, 2, ..., T)
        years = np.arange(len(log_cumulative)) - 1
        spline = CubicSpline(years, log_cumulative, bc_type='natural')

        # Step 4: Evaluate at fine grid and convert back to monthly % returns
        monthly_time = np.array(range(-int(1 / dt), T * int(1 / dt))) * dt
        log_cum_interp = spline(monthly_time)
        mu_vals = np.exp(np.diff(log_cum_interp)) - 1

        dmu_dt_vals = np.diff(mu_vals) / dt
        dmu_dt_vals = np.append(dmu_dt_vals, dmu_dt_vals[-1])

        mu_vals = mu_vals[monthly_time[1:] >= 0]
        dmu_dt_vals = dmu_dt_vals[monthly_time[1:] >= 0]

        return mu_vals, dmu_dt_vals

    @staticmethod
    def get_rfr_mu_path(
            forward_rates,
            r_star,
            dt,
            T,
            convergence_target=0.95,
            convergence_years=5,
    ):

        rend = forward_rates[-1]
        a_decay = -np.log(1 - convergence_target) / convergence_years

        yearly_years = np.arange(0, T + 1)
        start_decay = len(forward_rates) - 1
        decay_years = yearly_years[start_decay + 1:]
        mu_decay = rend * np.exp(-a_decay * (decay_years - start_decay)) + r_star * (
                    1 - np.exp(-a_decay * (decay_years - start_decay)))
        mu_yearly = np.concatenate([forward_rates, mu_decay]) * dt

        mu_spline = CubicSpline(yearly_years, mu_yearly, bc_type='natural')
        time = np.linspace(0, T, int(T / dt))
        mu_vals = mu_spline(time)
        dmu_dt_vals = mu_spline(time, 1)
        return mu_vals, dmu_dt_vals




    @staticmethod
    def get_mu_path(
            forward_rates,
            r_star,
            dt,
            T,
            rate_type,
            convergence_target=0.95,
            convergence_years=5,
    ):
        if rate_type == RateType.INFLATION:
           return SimulationHelper.get_inflation_mu_path(
                        forward_rates,
                        r_star,
                        dt,
                        T,
                        convergence_target,
                        convergence_years)
        elif rate_type == RateType.CASH:
            return SimulationHelper.get_rfr_mu_path(
                forward_rates,
                r_star,
                dt,
                T,
                convergence_target,
                convergence_years)
        else:
            raise Exception("Rate type not supported")



    @staticmethod
    def extract_shocks(
            asset,
            mu_vals,
            dmu_vals,
            curr_idx=None):

        from epsilonPhi.core.simulation.Bootstrap import SAABootstrapper

        ctr = 0
        panel = np.full((asset.shape[0], mu_vals.shape[0], 3), np.nan)
        for mu, dmu in zip(mu_vals, dmu_vals):
            panel[:, ctr, 0], panel[:, ctr, 2] = SimulationHelper.extract_shocks_single_params(asset.values, mu, dmu)
            if curr_idx is not None:
                panel[:, ctr, 1] = SAABootstrapper.demean_values_in_blocks(
                                        panel[:, ctr, 0],
                                        curr_idx
                )
            ctr += 1
        return panel

    @staticmethod
    def extract_shocks_single_params(
            vals,
            mu,
            dmu,
    ):


        """
        Calibrate static Hull-White model and extract shocks.

        Parameters:
        -----------
        mu : float
            Constant mean reversion level θ
        dmu_dt : float
            Time derivative of mean (should be 0 for static model)
        """


        r_t = vals[:-1]
        r_tp1 = vals[1:]

        # Static Hull-White: r(t+1) - r(t) - dμ = -a×[r(t) - μ] + shock
        y = r_tp1 - r_t - dmu
        x = (mu - r_t)

        a_est = np.dot(x, y) / np.dot(x, x)
        shocks = np.insert(y - (a_est * x), 0, 0)
        return shocks, a_est.repeat(shocks.shape)

    @staticmethod
    def simulate_paths(
            mu_vals,
            dmu_vals,
            shocks,
            mean_reversion,
            rate_floor
    ):

        paths = np.zeros(shocks.shape)
        paths[0, :] = mu_vals[0]

        for t in range(1, paths.shape[0]):
            drift = dmu_vals[t - 1] + mean_reversion[t] * (mu_vals[t - 1] - paths[t - 1])
            paths[t, shocks[t] != 0] = paths[t - 1, shocks[t] != 0] + drift[shocks[t] != 0] + shocks[t, shocks[t] != 0]
            paths[t, shocks[t] == 0] = paths[t - 1, shocks[t] == 0]

            # enforce the lower bound and mean correction
            paths[t] = np.maximum(rate_floor, paths[t])
            excess = np.mean(paths[t]) - mu_vals[t]
            paths[t, shocks[t] != 0] = np.maximum(
                rate_floor,
                paths[t, shocks[t] != 0] - excess / np.mean((shocks[t] != 0).astype(int))
            )

        return paths

