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
    def get_mu_path(
            forward_rates,
            r_star,
            dt,
            T,
            rate_type,
            convergence_target=0.95,
            convergence_years=5,
    ):

        rend = forward_rates[-1]
        a_decay = -np.log(1 - convergence_target) / convergence_years

        yearly_years = np.arange(0, T+1)
        start_decay = len(forward_rates) - 1
        decay_years = yearly_years[start_decay+1:]
        mu_decay = rend * np.exp(-a_decay * (decay_years - start_decay)) + r_star * (1 - np.exp(-a_decay * (decay_years - start_decay)))
        mu_yearly = np.concatenate([forward_rates, mu_decay])

        mu_spline = CubicSpline(yearly_years, np.cumprod(1 + mu_yearly), bc_type='natural')
        time = np.linspace(0, T, int(T / dt))
        mu_lvls = mu_spline(time)
        mu_vals = -1 + (mu_lvls[1:] / mu_lvls[:-1])
        dmu_dt_vals = mu_spline(time, 1)
        return mu_vals, dmu_dt_vals

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

