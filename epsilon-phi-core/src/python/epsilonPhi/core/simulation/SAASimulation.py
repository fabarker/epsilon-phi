import scipy.stats.mstats

from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries
from epsilonPhi.core.simulation.Bootstrap import SAABootstrapper
from epsilonPhi.core.asset.proxies.LendingRate import CLendingRate
from epsilonPhi.core.simulation.SimStructs import ReturnsPanel, FactorStressTestLosses, HistoricalStressTestLosses, \
    PerformanceVaRMetrics, WealthFlows, WealthProjections
from epsilonPhi.core.config.appConfig import CAppConfig
from collections import OrderedDict
from scipy.stats import zscore
import logging
import pandas as pd
import numpy as np
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
config_util = CAppConfig.get_config_util()


class StressMultiplier:
    def __init__(self, beta_mult_ts):
        self._stress_mult_ts = beta_mult_ts

    def get_stress_multi_ts(self):
        return self._stress_mult_ts


class SAASimulation:
    """
    SAASimulation manages the setup and execution of stress testing, scenario analysis,
    and portfolio simulation for asset allocation studies.

    This class integrates portfolio-level data with historical factor returns and crisis definitions,
    and produces stressed return panels, historical crisis loss estimates, and factor-based
    stress tests. It also bootstraps scenarios for simulation and estimates appropriate stress
    multipliers for use in systematic portfolio stress testing.

    Attributes:
        _portfolio_mgr (SAAPortfolioManager): Portfolio manager object used for retrieving portfolio-level data.
        _bootstrap (SAABootstrapper): Object responsible for resampling and generating bootstrapped return paths.
        _factor_crisis_map (dict): A mapping of factor names to crisis definitions used for stress testing.
        _beta_mult_ts (CTimeSeries): Cached time series of stress multipliers used in simulation.
        _is_setup (bool): Flag indicating whether the simulation setup has been completed.

    Notes:
        - The `ReturnsPanel` object built in `get_stressed_returns_panel` contains normalized factor returns,
          factor contributions, and real return series for use in multi-dimensional analysis.
        - Stress multiplier calibration can be tuned to match historical loss behavior using predefined
          benchmark portfolios (e.g., 50/50 equity/bond).
        - Requires proper configuration of schema and portfolio manager interfaces for full functionality.
    """

    _beta_mult_ts = None

    def __init__(self, portfolio_mgr):

        self._portfolio_mgr = portfolio_mgr
        self._factor_crisis_map = None
        self._bootstrap = None
        self._is_setup = False
        self.setup()

    @property
    def nbstraps(self):
        return self.bootstrap.nbstraps

    @property
    def long_horizon(self):
        return self.bootstrap.long_horizon

    @property
    def schema(self):
        return self._portfolio_mgr.get_context()

    @property
    def portfolio_mgr(self):
        return self._portfolio_mgr

    @property
    def bootstrap(self):
        return self._bootstrap

    @property
    def start_date(self):
        return self.config.simStartDate if self.config.simStartDate else self.schema.start_date

    @property
    def end_date(self):
        return self.config.simEndDate if self.config.simEndDate else self.schema.end_date

    @property
    def config(self):
        return self.schema.get_simulation_config()

    def setup(self):

        if self._is_setup:
            return

        if self._bootstrap is None:
            self._bootstrap = SAABootstrapper(self.schema)
            self._is_setup = True

    def get_stress_coeff_ts(self):
        if SAASimulation._beta_mult_ts is None:
            self.set_beta_multipliers()

            dates = self.schema.dates
            crises = self.get_factor_crisis_map()

            arr = np.zeros((len(dates), 1))
            for crisis in crises:
                mask = ((dates > crises[crisis]._start_date) &
                        (dates <= crises[crisis]._end_date))
                arr[mask] = crises[crisis]._stress_coefficient
            SAASimulation._beta_mult_ts = CTimeSeries(arr, index=dates, columns=['stress_coeff'])

        return SAASimulation._beta_mult_ts

    def get_factor_crisis_map(self):
        return self.schema.get_factor_crisis_map()

    def get_historical_stress_tests(self):

        # 1. Get realized portfolio return
        nominal_index = self.portfolio_mgr.get_historical_cuml_return_series()
        real_index = self.portfolio_mgr.get_historical_real_cuml_return_series()
        rfr_index = self.schema.get_risk_free_rate_asset().add(1).cumprod().reindex(nominal_index.dates)

        crises = self.get_factor_crisis_map()
        dates = nominal_index.index
        stress_losses = {}
        for crisis_name, crisis in crises.items():
            if crisis.start_date not in dates or crisis.end_date not in dates:
                continue

            idx_start = dates.get_loc(crisis.start_date)
            idx_end = dates.get_loc(crisis.end_date)

            nominal_loss = -1 + (nominal_index[idx_end] / nominal_index[idx_start])
            real_loss = -1 + (real_index[idx_end] / real_index[idx_start])
            rfr = -1 + (rfr_index[idx_end] / rfr_index[idx_start])

            losses = HistoricalStressTestLosses(
                total=nominal_loss,
                cash_other=rfr,
                real=real_loss,
                start=crisis.start_date,
                end=crisis.end_date,
            )

            stress_losses[crisis_name] = losses
        return stress_losses

    def get_factor_stress_tests(self):

        # Get stressed returns panel
        rtns_panel = self.get_stressed_returns_panel()
        nominal_index = rtns_panel.get_ptf_systematic_index()
        real_index = rtns_panel.get_real_ptf_systematic_index()
        factor_index = rtns_panel.cumulative_factor_panel()

        crises = self.get_factor_crisis_map()
        dates = rtns_panel.dates
        stress_losses = {}
        for crisis_name, crisis in crises.items():
            if crisis.start_date not in dates or crisis.end_date not in dates:
                continue

            idx_start = dates.get_loc(crisis.start_date)
            idx_end = dates.get_loc(crisis.end_date)

            loss_n = -1 + (nominal_index[idx_end] / nominal_index[idx_start])
            loss_r = -1 + (real_index[idx_end] / real_index[idx_start])
            loss_f = -1 + (factor_index[idx_end, :] / factor_index[idx_start, :])

            losses = FactorStressTestLosses(
                total=loss_n,
                factors=loss_f,
                cash_other=loss_n - np.sum(loss_f),
                real=loss_r
            )

            stress_losses[crisis_name] = losses

        return stress_losses

    def get_factor_panel_normalized(self):
        return self.get_stressed_returns_panel().factor_panel_normalized

    def get_stressed_returns_panel(self,
                                   start_date=None,
                                   end_date=None
                                   ):

        if start_date is None:
            start_date = self.start_date

        if end_date is None:
            end_date = self.end_date

        # Get the risk free rate asset
        rfr = self.schema.get_risk_free_rate_asset()
        cpi = self.schema.get_price_deflator()

        # Instantiate returns panel
        rtns_panel = ReturnsPanel()

        # Get the factor panels
        factor_panel = self.schema.get_factor_panels()

        # Get return factor panels
        return_factors_panel = self.schema.get_return_factors_panel()

        # Get the factor sharpe ratios
        factor_sharpes = np.array(self.schema.get_return_factors_sharpe_ratios())

        # Get historical factor vols
        factor_vols = factor_panel.get_factors_historical_stds(
            return_factors_panel.columns
        )

        # Get portfolio betas
        betas = np.array(self.portfolio_mgr.get_return_betas())

        # get the beta multiplier
        beta_mult = self.get_stress_coeff_ts().loc[start_date:end_date]
        assert np.all(beta_mult.index == return_factors_panel.index), 'Error - panels dont match'

        factor_to_stress = 'EQUITY_GLOBAL_ISG'  # self.schema.get_factor_names_to_stress()
        stress_factor_index = return_factors_panel.columns.get_loc(factor_to_stress)
        stress_coef_panel = np.ones(return_factors_panel.shape)
        stress_coef_panel[:, stress_factor_index] = beta_mult[beta_mult.columns[0]].values

        # Get indices where the stress multiplier is zero for the target factor
        idx = stress_coef_panel[:, stress_factor_index] == 0
        stress_coef_panel[idx, stress_factor_index] = 1

        ptf_vol = self.portfolio_mgr.get_risk()
        LB = 0.12
        RB = 0.16
        if ptf_vol > LB:
            if ptf_vol > RB:
                stress_coef_panel = np.ones(return_factors_panel.shape)
            else:
                stress_coef_panel = (RB - ptf_vol) / (RB - LB) * stress_coef_panel + \
                                    (ptf_vol - LB) / (RB - LB) * np.ones(return_factors_panel.shape)

        # Demean the panel and add the correct returns
        z_score = np.asarray(zscore(return_factors_panel, ddof=1))
        tiled_sharpes = np.tile(factor_sharpes.conj().T, (return_factors_panel.shape[0], 1))
        tiled_betas = np.tile(betas, (return_factors_panel.shape[0], 1))

        # Normalize the factor panels
        rtns_panel.factor_panel_normalized = z_score + tiled_sharpes / math.sqrt(self.schema.frequency.obs_per_year())
        T = return_factors_panel.shape[0]
        factor_contr_panel = rtns_panel.factor_panel_normalized * stress_coef_panel * tiled_betas

        # Check if there's any non-zero contribution in the zero-stressed rows
        has_contribution_in_zero_rows = np.mean(np.sum(factor_contr_panel[idx, :], axis=1)) != 0
        if has_contribution_in_zero_rows:
            # Calculate the target average contribution from this factor
            mean_factor_contribution = np.mean(
                rtns_panel.factor_panel_normalized[:, stress_factor_index] * betas[stress_factor_index]
            )

            # Total contribution already made by non-zero-stressed rows
            total_contributed_by_non_zero = np.sum(
                factor_contr_panel[~idx, stress_factor_index]
            ) / T

            # Total contribution made by zero-stressed rows (before correction)
            total_contributed_by_zero = np.sum(
                factor_contr_panel[idx, stress_factor_index]
            ) / T

            correction_multiplier = (mean_factor_contribution - total_contributed_by_non_zero) / total_contributed_by_zero

            # Apply the correction to only the zero-stressed rows for the stressed factor
            stress_coef_panel[idx, stress_factor_index] = correction_multiplier
        else:
            # No contribution to correct, fallback to neutral multiplier
            stress_coef_panel[idx, stress_factor_index] = 1

        # put values into returns panel object
        rtns_panel.dates = return_factors_panel.index
        rtns_panel.factor_contributions_panel = rtns_panel.factor_panel_normalized * stress_coef_panel * betas.conj().T
        rtns_panel.rfr_values = rfr.reindex(rtns_panel.dates).values
        rtns_panel.cpi_values = cpi.reindex(rtns_panel.dates).values
        rtns_panel.return_factor_panel = return_factors_panel
        rtns_panel.betas = betas
        rtns_panel.beta_mult = beta_mult
        rtns_panel.factor_sharpes = factor_sharpes
        rtns_panel.factor_vols = factor_vols
        rtns_panel.stress_coeff_panels = stress_coef_panel
        rtns_panel.alpha_total_monthly = self.portfolio_mgr.get_alpha() / self.schema.frequency.obs_per_year()

        return rtns_panel

    def initialize_stress_multiplier(self):
        SAASimulation._beta_mult_ts = self.get_stress_coeff_ts()

    def load_stress_multiplier(self):

        # If we are loading then initalize state
        self.initialize_stress_multiplier()

        equity_bmk = 'MSUSAML'
        bond_bmk = 'LHAGGBD'

        # equity_bmk = self.schema.get_currency_config().equity_stress_ticker
        # bond_bmk   = self.schema.get_currency_config().bond_stress_ticker

        from epsilonPhi.core.portfolio.SAAPortfolio import SAAPortfolio
        ptf = SAAPortfolio('Calibration', self.schema)
        ptf.add_asset_by_name(bond_bmk, 0.5, 0)
        ptf.add_asset_by_name(equity_bmk, 0.5, 0)
        ptf.setup()

        sim = SAASimulation(ptf.get_portfolio_mgr())

        bonds_rng = [0.2, 0.3, 0.5, 0.7, 0.8]
        stress_rng = np.linspace(1.0, 3.0, num=41)
        crisis_map = self.get_factor_crisis_map()

        error_arr = np.full((len(stress_rng), len(crisis_map.keys()), len(bonds_rng)), np.nan)
        for i, wt in enumerate(bonds_rng):
            # set weight in the portfolio
            ptf.set_weights([wt, 1 - wt])
            # calculate the historical stressed performance
            historical_stressed = sim.get_historical_stress_tests()

            # Iterate through each crisis
            for j, crisis_name in enumerate(crisis_map.keys()):

                # Iterate through each candidate stress coefficient
                for k, stress in enumerate(stress_rng):
                    logger.info(
                        "Computing factor stress losses for portfolio {}, crisis {} and stress mult {}".format(wt,
                                                                                                               crisis_name,
                                                                                                               stress))
                    crisis_map[crisis_name].set_stress_coefficient(stress)

                    # Compute the factor based stress tests
                    factor_stressed = sim.get_factor_stress_tests()
                    # set the error in the error array
                    error_arr[k, j, i] = np.power(
                        factor_stressed[crisis_name].total - historical_stressed[crisis_name].total, 2)

    def set_beta_multipliers(self):

        # Get the factor metrics
        return_factors_panel = self.schema.get_return_factors_panel()
        dates = return_factors_panel.index
        factor_sharpes = np.array(self.schema.get_return_factors_sharpe_ratios())
        normalized_return_factor_panel = np.asarray(zscore(return_factors_panel, ddof=1))
        rfr_values = self.schema.get_risk_free_rate_asset().reindex(dates).values

        from epsilonPhi.core.portfolio.SAAPortfolio import SAAPortfolio
        ptf = SAAPortfolio('Calibration', self.schema)
        ptf.add_asset_by_name('LHAGGBD', 0.5, 0)
        ptf.add_asset_by_name('MSUSAML', 0.5, 0)
        ptf.setup()

        # Look through portfolios
        bonds_rng = [0.2, 0.3, 0.5, 0.7, 0.8]
        stress_coefs = np.linspace(1.0, 3.0, num=1000)
        crises = self.get_factor_crisis_map()
        error_arr = np.full((len(stress_coefs), len(crises.keys()), len(bonds_rng)), np.nan)
        for i, wt in enumerate(bonds_rng):

            # Construct a portfolio with the benchmark assets
            ptf.set_weights([wt, 1 - wt])

            # Get Historical Crisis Performance
            historical_stressed = ptf.get_historical_stress_tests()

            # Get betas and stressing panel
            sqrt_t = math.sqrt(self.schema.frequency.obs_per_year())
            betas = np.array(ptf._portfolio_mgr.get_return_betas())

            # 2. Build normalized factor panel (shape: [T, F])
            factor_panel_normalized = normalized_return_factor_panel + factor_sharpes.flatten() / sqrt_t  # shape: (T, F)

            # 3. Unstressed contributions (shape: [T, F])
            unstressed_contrib = factor_panel_normalized * betas  # shape: (T, F)

            # 4. Expand unstressed contributions for stress grid (shape: [T, F, 100])
            stressed_contrib = np.repeat(unstressed_contrib[:, :, np.newaxis], repeats=1000, axis=2)

            # 5. Apply stress to the first factor (dimension 1 index 0)
            # This avoids tiling a giant array and instead broadcasts efficiently
            stressed_contrib[:, 0, :] *= stress_coefs  # broadcasting over T and stress scenarios

            # Get portfolio systematic index after stressing
            lvls = np.cumprod(1 + rfr_values.reshape(-1, 1) + np.sum(stressed_contrib, axis=1), axis=0)

            for j, crisis_name in enumerate(crises.keys()):
                logger.info("Computing factor stress losses for portfolio {}, crisis {}".format(wt, crisis_name))

                if (crises[crisis_name].start_date not in return_factors_panel.dates
                        or crises[crisis_name].end_date not in return_factors_panel.dates):
                    continue

                idx_s = dates.get_loc(crises[crisis_name].start_date)
                idx_e = dates.get_loc(crises[crisis_name].end_date)

                losses = -1 + (lvls[idx_e] / lvls[idx_s])
                error_arr[:, j, i] = np.power(losses - historical_stressed[crisis_name].total, 2)

        # Sum across portfolios
        errors = error_arr.sum(axis=2)
        # beta multiplier is the min errors
        beta_mult = stress_coefs[(errors.min(axis=0) == errors).argmax(axis=0)]

        # Set beta multiplier in crisis map
        for j, crisis_name in enumerate(crises.keys()):
            crises[crisis_name].set_stress_coefficient(beta_mult[j])

    def get_portfolio_paths(self):
        idx = self.bootstrap.get_bootstrap_indicies()
        return self.bootstrap.prepare_portfolio_paths(
            self.portfolio_mgr,
            idx,
            0
        )

    def get_portfolio_current_env_risk_premia(self):
        return self.get_portfolio_paths().get_returns_panel().medium_term_risk_premia

    def get_inflation_panels(self, frequency=Frequency.MONTHLY):

        # Get panel of periodic inflation returns
        inflation_panel = self.bootstrap.prepare_inflation_paths(
            self.bootstrap.get_bootstrap_indicies()
        ).get_panel()

        # convert to inflation price index
        inflation_levels = np.cumprod(1 + inflation_panel, axis=0)

        # simulation frequency, in schema
        base_freq = self.portfolio_mgr._context.frequency

        # Downsample monthly data to yearly if needed
        if frequency in [Frequency.YEARLY, Frequency.BUSINESS_YEARLY] and base_freq in [Frequency.MONTHLY,
                                                                                        Frequency.BUSINESS_MONTHLY]:
            # Use index 11 to get end-of-year (i.e., December)
            inflation_levels = inflation_levels[11::12, :]

        # Compute cumulative inflation over time
        return inflation_levels

    def get_simulated_portfolio_returns(self, long_term_shocks=0, frequency=Frequency.MONTHLY):

        lending_wt = 0  # This seems to be hardcoded. Should this be dynamically retrieved?

        idxs = self.bootstrap.get_bootstrap_indicies()

        # Prepare core paths
        cp = self.bootstrap.prepare_cash_paths(idxs, long_term_shocks)
        pp = self.bootstrap.prepare_portfolio_paths(self.portfolio_mgr, idxs, long_term_shocks)

        if abs(lending_wt) < self.portfolio_mgr._sqrt_epsilon:
            # No lending case
            total_returns = (
                    cp.get_panel() +
                    pp.systematic_panel +
                    pp.idiosyncratic_panel +
                    pp.alpha_monthly_total
            )
        else:
            # Lending case
            lending_paths = []
            lending_weights = []
            for asset_name, weight in zip(self.portfolio_mgr.get_asset_names(), self.portfolio_mgr.get_weights()):
                asset = self.portfolio_mgr.get_asset(asset_name)
                if isinstance(asset, CLendingRate):
                    lending_weights.append(weight)
                    lending_paths.append(self.bootstrap.prepare_lending_paths(asset, idxs, long_term_shocks))

            # Aggregate lending panel
            lending_panel = np.zeros_like(cp.get_panel())
            for weight, lp in zip(lending_weights, lending_paths):
                lending_panel += weight * lp.get_panel()

            base_returns = (
                    cp.get_panel() +
                    pp.systematic_panel +
                    pp.idiosyncratic_panel +
                    pp.alpha_monthly_total
            )
            total_returns = (1 - lending_wt) * base_returns + lending_panel

        # Conver to levels/cumulative return in schema frequency
        indexed_panel = np.cumprod(1 + total_returns, axis=0)

        # simulation frequency, in schema
        base_freq = self.portfolio_mgr._context.frequency

        # Downsample monthly data to yearly if needed
        if frequency in [Frequency.YEARLY, Frequency.BUSINESS_YEARLY] and base_freq in [Frequency.MONTHLY,
                                                                                        Frequency.BUSINESS_MONTHLY]:
            # Use index 11 to get end-of-year (i.e., December)
            indexed_panel_with_start = np.vstack((np.ones((1, self.nbstraps)), indexed_panel))
            indexed_panel = indexed_panel_with_start[::12, :]
            total_returns = indexed_panel[1:, :] / indexed_panel[:-1, :] - 1

        # Compute cumulative inflation over time
        return total_returns, indexed_panel

    def get_portfolio_var_pol_exc_ss(self, confidence=0.99, loss=0):

        _, lvls_panel = self.get_simulated_portfolio_returns(long_term_shocks=1)
        real_panel = lvls_panel / self.get_inflation_panels()

        # Value At Risk
        var_n = np.clip(-(np.quantile(lvls_panel, 1 - confidence, axis=1) - 1), 0, np.inf)
        var_r = np.clip(-(np.quantile(real_panel, 1 - confidence, axis=1) - 1), 0, np.inf)

        # PoL
        pol_n = np.mean(lvls_panel < (1 - loss), axis=1)
        pol_r = np.mean(real_panel < (1 - loss), axis=1)

        # Conditional Value at Risk
        lvls_panel[lvls_panel > np.quantile(lvls_panel, 1 - confidence, axis=1).reshape(-1, 1)] = np.nan
        real_panel[real_panel > np.quantile(real_panel, 1 - confidence, axis=1).reshape(-1, 1)] = np.nan
        cvar_n = np.clip(-(np.nanmean(lvls_panel, axis=1) - 1), 0, np.inf)
        cvar_r = np.clip(-(np.nanmean(real_panel, axis=1) - 1), 0, np.inf)

        return PerformanceVaRMetrics(
            np.stack((var_n, var_r)).T,
            np.stack((cvar_n, cvar_r)).T,
            np.stack((pol_n, pol_r)).T,
            confidence,
            loss,
        )

    def get_portfolio_var_pol(self, confidence=0.99, loss=0):

        if self.portfolio_mgr.has_single_stock():
            ptf_x_ss, ss_wt = self.portfolio_mgr.get_portfolio_excl_single_stock()
            risk = ptf_x_ss.get_portfolio_var_pol_exc_ss(confidence=confidence, loss=loss)

            # Compute the single stock losses
            ss_loss = risk * (1 - ss_wt) + confidence * ss_wt
            return ss_loss
        return self.get_portfolio_var_pol_exc_ss(confidence=confidence, loss=loss)

    def get_portfolio_wealth_projection(self,
                                        ws_inflows=None,
                                        ws_outflows=None,
                                        ptf_sim_order=None,
                                        quantiles=None,
                                        ptf_list=None,
                                        frequency=Frequency.YEARLY
                                        ):

        if self.portfolio_mgr.get_current_value() is None or self.portfolio_mgr.get_current_value() <= 0:
            raise ValueError('Error - not current value set in portfolio')

        if not quantiles:
            quantiles = [0.01, 0.1, 0.5, 0.9]

        # Multiplier to deal with differing freq/periodcities
        freq_mult = frequency.obs_per_year()
        long_horizon = self.long_horizon * freq_mult

        if not ws_inflows:
            ws_inflows = [WealthFlows() for _ in range(long_horizon)]

        if not ws_outflows:
            ws_outflows = [WealthFlows() for _ in range(long_horizon)]

        if ptf_sim_order is None:
            ptf_sim_order = np.zeros(long_horizon, dtype=int)
        elif len(ptf_sim_order) != long_horizon:
            raise ValueError('Error - number of points to simulate is inconsistent')

        # If we only have a single ptf
        if ptf_list is None:
            ptf_list = [self.portfolio_mgr.portfolio]

        # tax bill panel
        tax_bill = [None] * len(ptf_list)

        ptf_panel = np.full((self.long_horizon, self.nbstraps, len(ptf_list)), np.nan)
        for i, ptf in enumerate(ptf_list):
            rtns, _ = self.get_simulated_portfolio_returns(long_term_shocks=0, frequency=frequency)

            if ptf.is_taxable:
                effective_tax_rate = 1 - (ptf.get_return() / ptf.get_return_pre_tax())
                rtns = rtns - effective_tax_rate * np.mean(rtns, axis=1, keepdims=True)
            else:
                effective_tax_rate = 0

            ptf_panel[..., i] = rtns
            if frequency == Frequency.YEARLY:
                tax_bill[i] = effective_tax_rate * np.mean(rtns, axis=1, keepdims=True)

        tax_panel = []
        rtns_panel = np.full((self.long_horizon, self.nbstraps), np.nan)
        for i in range(self.long_horizon):
            rtns_panel[i, :] = ptf_panel[i, :, ptf_sim_order[i]]

            if i % 12 == 0 and frequency == Frequency.MONTHLY:
                tax = tax_bill[ptf_sim_order[i]]
                tax_panel.append(tax[i // 12])

        # Inflation index
        ip = self.get_inflation_panels(frequency=frequency)

        # Pre-Allocate Arrays
        nominal_value_of_real_inflows = np.full((self.long_horizon, self.nbstraps), np.nan)
        nominal_value_of_real_outflows = np.full((self.long_horizon, self.nbstraps), np.nan)

        # Prepare flows - convert real to nominal
        for i in range(self.long_horizon):
            if i == 0:
                nominal_value_of_real_inflows[0] = ws_inflows[0].real
                nominal_value_of_real_outflows[0] = ws_outflows[0].real
            else:
                nominal_value_of_real_inflows[i] = ws_inflows[i].real * ip[i - 1]
                nominal_value_of_real_outflows[i] = ws_outflows[i].real * ip[i - 1]

        nominal_inflow_panel = np.full((self.long_horizon, self.nbstraps), np.nan)
        nominal_outflow_panel = np.full((self.long_horizon, self.nbstraps), np.nan)
        value_paths = np.full((self.long_horizon, self.nbstraps), np.nan)
        for i in range(self.long_horizon):

            # prepare flows for year

            if i == 0:
                nominal_inflow_panel[i] = (nominal_value_of_real_inflows[i] +
                                           ws_inflows[i].nominal +
                                           self.portfolio_mgr.get_current_value() * ws_inflows[i].percent)

                nominal_outflow_panel[i] = (nominal_value_of_real_outflows[i] +
                                            ws_outflows[i].nominal +
                                            self.portfolio_mgr.get_current_value() * ws_outflows[i].percent)

                value_pre_flows = self.portfolio_mgr.get_current_value() * (1 + rtns_panel[i])
            else:
                nominal_inflow_panel[i] = (nominal_value_of_real_inflows[i] +
                                           ws_inflows[i].nominal +
                                           value_paths[i - 1] * ws_inflows[i].percent)

                nominal_outflow_panel[i] = (nominal_value_of_real_outflows[i] +
                                            ws_outflows[i].nominal +
                                            value_paths[i - 1] * ws_outflows[i].percent)
                value_pre_flows = value_paths[i - 1] * (1 + rtns_panel[i])

            value_paths[i] = np.maximum(value_pre_flows + nominal_inflow_panel[i] - nominal_outflow_panel[i], 0)

            # Tax-Accounting to add here

        # Compute real value of simullated paths
        value_paths_with_current_value = np.vstack(
            (self.portfolio_mgr.get_current_value() * np.ones((1, self.nbstraps)), value_paths))

        ip = np.vstack((np.ones((1, self.nbstraps)), ip))
        ws = WealthProjections(
            value_paths_with_current_value,
            rtns_panel,
            ip,
            nominal_inflow_panel,
            nominal_outflow_panel,
            quantiles,
            frequency
        )

        return ws

    def get_private_assets_distributions(self):
        pass


if __name__ == "__main__":
    from epsilonPhi.core.asset.AssetMgr import CAssetMgr
    from epsilonPhi.core.schema.Schema import ContextCreator
    from epsilonPhi.core.portfolio.SAAPortfolio import SAAPortfolio

    schema = ContextCreator(currency='USD',
                            start_date='30-Nov-1983',
                            end_date='31-Dec-2022').create_context()

    assetMgr = CAssetMgr(schema)
    asset = assetMgr.get_asset_by_name('MSUSAML')

    ptf = SAAPortfolio('portfolio', schema)
    ptf.add_asset_by_name('MSUSAML', 1.0, 0)
    ptf.add_asset_by_name('LHAGGBD', 0.0, 0)
    mgr = ptf.get_portfolio_mgr()

    self = SAASimulation(mgr)
    ws = self.get_portfolio_wealth_projection()
