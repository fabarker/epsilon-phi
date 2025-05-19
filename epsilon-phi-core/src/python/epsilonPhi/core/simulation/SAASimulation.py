from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries
from epsilonPhi.core.simulation.Bootstrap import SAABootstrapper
from epsilonPhi.core.simulation.SimStructs import ReturnsPanel, FactorStressTestLosses, HistoricalStressTestLosses
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
        return self.config.simEndDate if self.config.simEndDate else  self.schema.end_date

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

        factor_to_stress = 'EQUITY_GLOBAL_ISG' #self.schema.get_factor_names_to_stress()
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

        #equity_bmk = self.schema.get_currency_config().equity_stress_ticker
        #bond_bmk   = self.schema.get_currency_config().bond_stress_ticker

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
            ptf.set_weights([wt, 1-wt])
            # calculate the historical stressed performance
            historical_stressed = sim.get_historical_stress_tests()

            # Iterate through each crisis
            for j, crisis_name in enumerate(crisis_map.keys()):

                # Iterate through each candidate stress coefficient
                for k, stress in enumerate(stress_rng):
                    logger.info("Computing factor stress losses for portfolio {}, crisis {} and stress mult {}".format(wt, crisis_name, stress))
                    crisis_map[crisis_name].set_stress_coefficient(stress)

                    # Compute the factor based stress tests
                    factor_stressed = sim.get_factor_stress_tests()
                    # set the error in the error array
                    error_arr[k, j, i] = np.power(factor_stressed[crisis_name].total - historical_stressed[crisis_name].total, 2)


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
            ptf.set_weights([wt, 1-wt])

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

    def get_factor_backfilled_asset_returns(self):
        pass

    def get_factor_backfilled_portfolio_returns(self):
        pass

    def get_portfolio_var_pol_exc_ss(self):
        pass

    def get_portfolio_var_pol(self):
        pass

    def get_simulated_portfolio_returns(self):
        pass

    def get_portfolio_wealth_projection(self):
        pass

    def get_private_assets_distributions(self):
        pass












if __name__ == "__main__":

    from epsilonPhi.core.asset.AssetMgr import CAssetMgr
    from epsilonPhi.core.schema.Schema import ContextCreator
    from epsilonPhi.core.portfolio.Portfolio import CPortfolio
    schema = ContextCreator(currency='USD',
                            start_date='30-Nov-1983',
                            end_date='31-Dec-2022').create_context()

    assetMgr = CAssetMgr(schema)
    asset = assetMgr.get_asset_by_name('MSUSAML')

    ptf = CPortfolio('portfolio', schema)
    ptf.add_asset_by_name('MSUSAML', 1.0, 0)
    ptf.add_asset_by_name('LHAGGBD', 0.0, 0)
    mgr = ptf.get_portfolio_mgr()

    self = SAASimulation(mgr)
    factor_stress = self.get_factor_stress_tests()
    historical_stressed = ptf.get_historical_stress_tests()











