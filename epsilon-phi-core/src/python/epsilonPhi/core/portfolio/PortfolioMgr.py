import math
import numpy as np
import logging
from epsilonPhi.core.schema.Schema import CContext
from epsilonPhi.core.simulation.SAASimulation import SAASimulation
from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries, CSlice
from epsilonPhi.core.asset.Asset import CAsset
from numpy.typing import NDArray
import sys

logger = logging.getLogger(__name__)

def nans(rows=0, cols=0):
    return np.full((rows, cols), np.nan)

class CPortfolioMgr(object):
    _sqrt_epsilon = math.sqrt(sys.float_info.epsilon)

    def __init__(
            self,
            context: CContext,
    ):
        self._context = context
        self._portfolio = None

        self._reference_string = []
        self._low_vol_bond_reference = None
        self._bond_reference = None
        self._equity_reference = None
        self._cash_reference = None
        self._hedge_fund_reference = None

        self._simulation = None


    @property
    def num_assets(self):
        return self.portfolio.num_assets

    @property
    def portfolio(self):
        return self._portfolio if self._portfolio is not None else None

    ############# Setter Methods ##############

    def reset_properties(self):
        self.portfolio.reset_properties()
        self._reference_string = []
        self._low_vol_bond_reference = None
        self._bond_reference = None
        self._equity_reference = None
        self._cash_reference = None
        self._hedge_fund_reference = None
        self._simulation = None

    def set_portfolio(self, portfolio) -> None:
        from epsilonPhi.core.portfolio.Portfolio import CPortfolio
        from epsilonPhi.core.portfolio.SAAPortfolio import SAAPortfolio
        assert isinstance(portfolio, (CPortfolio, SAAPortfolio)), 'Error'
        self._portfolio = portfolio

    def set_weights(self, weights):

        self.check_weights(weights)
        for i, asset_name in enumerate(self.get_asset_names()):
            self.get_asset(asset_name).set_weight(weights[i])
        self.portfolio.reset_properties()

    def set_reference_assets(
            self,
            low_vol_reference,
            bond_reference,
            equity_reference,
            cash_reference=None,
            hedge_fund_reference=None
    ) -> None:

        self._low_vol_bond_reference = low_vol_reference
        self._bond_reference = bond_reference
        self._equity_reference = equity_reference
        self._cash_reference = cash_reference
        self._hedge_fund_reference = hedge_fund_reference

    def set_reference_portfolio_name(self, name='Reference Portfolio'):
        self._reference_portfolio_name = name


    def set_single_asset_hedge_ratio(self,
                                     asset_name,
                                     hedge_ratio
                                     ) -> None:

        if asset_name in self.get_asset_names():
           self.get_asset(asset_name).set_hedge_ratio(hedge_ratio)
        # Reset the risk metrics in the portfolio
        self.portfolio.reset_properties()

    def add_asset_by_name(self,
                          asset_name,
                          weight,
                          hedging_ratio=0.0,
                          asset_time_series=None) -> None:

        asset = self.get_asset_mgr().get_asset_by_name(asset_name)
        self.add_asset(asset, weight, hedging_ratio)

    def add_asset(self,
                  asset: CAsset,
                  weight: float,
                  hedging_ratio: float
                  ) -> None:
        # Add the asset into the portfolio
        self.portfolio.add_asset(asset, weight, hedging_ratio)

    def get_simulator(self):
        if self._simulation is None:
           self._simulation = SAASimulation(self)
           self._simulation.setup()
        return self._simulation

    def get_context(self):
        return self.portfolio.context

    def get_portfolio(self):
        return self._portfolio

    def get_tax_region(self):
        return self.portfolio.tax_region

    def get_tax_info(self):
        return self.portfolio.get_tax_info()

    def is_amt(self):
        return self.portfolio.is_amt()

    def get_create_date(self):
        return self.portfolio.create_date

    def get_asset_names(self):
        return self.portfolio.get_asset_names()

    def get_asset(self, asset_name):
        return self.portfolio.get_asset(asset_name)

    def get_assets(self):
        return self.portfolio.get_assets()

    def get_portfolio_name(self):
        return self.portfolio.name

    def get_name(self):
        return self.portfolio.name

    def get_current_value(self):
        return self.portfolio.current_value

    def get_weights(self):
        return self._portfolio.get_weights()

    def get_flattened_weights(self):
        return self._portfolio.get_weights().flatten()

    def get_risk_factor_number(self):
        return self._context.get_risk_factors_panel().shape[1]

    def get_return_factor_number(self):
        return self._context.get_return_factors_panel().shape[1]

    def get_weight(self, asset_name):
        if asset_name in self.get_asset_names():
           return self.get_asset(asset_name).weight
        else:
            raise Exception("Asset {} doesn't exist in portfolioMgr".format(asset_name))

    def check_weights(self, weights=None):
        self.portfolio.check_weights(weights)

    def get_portfolio_excl_single_stocks(self):
        pass

    def get_portfolio_excl_lending_asset(self):
        pass

    def get_asset_mgr(self):
        return self._context.get_asset_manager()

    ################### Hedging Related ###################

    def get_hedge_ratios_from_hedging_option(self, hedging_option):

        # Get each assets hedging ratio for the specified hedging option
        ratios = []
        for asset_name in self.get_asset_names():
            ratios.append(
                SAAHedging.get_hedge_ratio_for_asset(
                    self.get_asset(asset_name),
                    hedging_option
                )
            )
        return ratios

    def get_hedging_ratios(self):
       return np.asarray([x.hedging_ratio for x in self.get_assets()])

    def get_hedging_option(self):
        return self.portfolio.hedging_option

    ################### Tax Related ###################

    #def get_tax_rates(self):
    #    return PortfolioTax.get_tax_rates(self)

    #def set_tax_rates(self, tax_rates=[], income_tax_rates=[]):
    #    PortfolioTax.set_tax_rates(self, tax_rates, income_tax_rates)

    #def set_asset_tax_rate(self, asset_name, tax_rate=None, income_tax_rate=None):
    #    PortfolioTax.set_asset_tax_rate(self, asset_name, tax_rate, income_tax_rate)

    ################### Factor Model Return Metrics ###################

    def get_current_risk_free_rate(self):
        pass

    def get_risk_free_rate(self) -> float:
        """
        Retrieves the long-term/equilibrium risk-free rate for the currency region as specified in the schema

        Returns:
            float: ann risk free rate
        """
        return self.portfolio.risk_free_rate()

    def get_asset_total_return(self, asset_name, after_tax=False) -> float:
        asset = self.get_asset(asset_name)
        asset_return = asset.get_total_return()

        if after_tax:
            if self.portfolio.is_taxable:
                #tax_rate, _ = PortfolioTax.get_tax_rates(self)
                asset_idx = self.portfolio.asset_names.index(asset_name)
                #asset_return = asset_return * (1 - tax_rate[asset_idx])
            else:
                raise Exception(
                    "Warning ... trying get after-tax return for asset {} for non-taxable portfolio ".format(
                        asset_name))
        return asset_return

    def get_assets_total_return(self, after_tax=False) -> NDArray[float]:
        total_returns = []
        for asset_name in self.get_asset_names():
            total_returns.append(self.get_asset_total_return(asset_name, after_tax))
        return np.asarray(total_returns)

    def get_asset_risk_premias(self, asset_name):
        return self.get_asset(asset_name).get_risk_premia()

    def get_assets_risk_premias(self):
        assets_risk_premias = nans(self.num_assets, self.get_return_factor_number())
        for i, asset_name in enumerate(self.get_asset_names()):
            assets_risk_premias[i] = self.get_asset(asset_name).get_risk_premia()
        return assets_risk_premias

    def get_asset_total_risk_premia(self):
        return self.get_assets_total_return() - self.get_risk_free_rate()

    def get_assets_total_risk_premia(self):
        return self.get_assets_total_return() - self.get_risk_free_rate()

    def get_assets_total_risk_premia_(self):
        return self.get_assets_risk_premias().sum(axis=1, keepdims=True)

    def get_total_return(self):
        return (self.get_weights().T @ self.get_assets_total_return()).flatten()

    def get_risk_premias(self):
        return (self.get_weights().T @ self.get_assets_risk_premias()).flatten()

    def get_risk_premia(self):
        return self.get_risk_premias().sum(axis=1)

    def get_asset_current_env_total_return(self, asset_name):
        return self.get_asset(asset_name).get_risk_premia_in_current_environment() + self.get_current_risk_free_rate()

    def get_assets_current_env_total_return(self):
        return self.get_assets_current_env_risk_premias().sum(axis=1) + self.get_current_risk_free_rate()

    def get_asset_current_env_risk_premias(self, asset_name):
        return self.get_asset(asset_name).get_risk_premia_in_current_environment()

    def get_assets_current_env_risk_premias(self):
        risk_premias = nans(self.num_assets, self.get_return_factor_number())
        for i, asset_name in enumerate(self.get_asset_names()):
            risk_premias[i] = self.get_asset(asset_name).get_risk_premia_in_current_environment()
        return risk_premias

    def get_asset_total_current_env_risk_premia(self, asset_name):
        return self.get_asset_current_env_risk_premias(asset_name).sum(axis=1)

    def get_assets_total_current_env_risk_premia(self):
        return self.get_assets_current_env_risk_premias().sum(axis=1)

    def get_current_env_total_return(self):
        return self.get_alpha() + self.get_current_env_risk_premia() + self.get_current_risk_free_rate()

    def get_current_env_risk_premias(self):
        return self.get_assets_current_env_risk_premias() @ self.get_weights()

    def get_current_env_risk_premia(self):
        return self.get_assets_total_current_env_risk_premia() @ self.get_weights()

    def get_asset_sharpe_ratio(self, asset_name):
        if asset_name not in self.get_asset_names():
            return None
        else:
            return self.get_asset(asset_name).get_Sharpe_ratio()

    def get_assets_sharpe_ratios(self):
        sharpes = nans(self.num_assets)
        for i, asset_name in enumerate(self.get_asset_names()):
            sharpes[i] = self.get_asset(asset_name).get_sharpe_ratio()
        return sharpes

    def get_assets_curr_env_sharpe_ratio(self):
        sharpes = nans(self.num_assets)
        for i, asset_name in enumerate(self.get_asset_names()):
            sharpes[i] = self.get_asset(asset_name).get_curr_env_sharpe_ratio()
        return sharpes

    def get_asset_curr_env_sharpe_ratio(self, asset_name):
        return self.get_asset(asset_name).get_curr_env_sharpe_ratio()

    def get_sharpe_ratio(self):
        return self.get_risk_premia() / self.get_risk()

    def get_curr_env_sharpe_ratio(self):
        return self.get_risk_premia_in_current_environment() / self.get_risk()

    def get_current_environment_risk_premia_5yr(self):
        pass

    def get_asset_return_medium(self):
        pass

    def get_medium_return(self):
        pass

    def get_return_betas(self):
        risk_premia = self.get_risk_premias()
        factor_sharpes = np.array(self._context.get_return_factors_sharpe_ratios()).flatten()
        betas = risk_premia / factor_sharpes / math.sqrt(self._context.frequency.obs_per_year())

        return betas

    def get_alpha(self) -> float:
        return self.get_flattened_weights() @ self.get_asset_alphas()

    def get_asset_alphas(self) -> np.array:

        alphas = []
        for i, asset_name in enumerate(self.get_asset_names()):
            alphas.append(self.get_asset(asset_name).get_alpha())
        return np.array(alphas)

    ################### Factor Model Risk Metrics ###################

    def get_asset_risk(self, asset_name):
        return self.get_asset(asset_name).get_risk()

    def get_asset_systematic_variance(self, asset_name):
        return self.get_asset(asset_name).get_systematic_variance()

    def get_asset_idio_variance(self, asset_name):
        return self.get_asset(asset_name).get_idio_variance()

    def get_assets_risk(self):
        risks = nans(self.num_assets)
        for i, asset_name in enumerate(self.get_asset_names()):
            risks[i] = self.get_asset(asset_name).get_risk()
        return risks

    def get_assets_systematic_variances(self):
        risks = nans(self.num_assets)
        for i, asset_name in enumerate(self.get_asset_names()):
            risks[i] = self.get_asset(asset_name).get_systematic_variance()
        return risks

    def get_assets_idio_variances(self):
        risks = nans(self.num_assets)
        for i, asset_name in enumerate(self.get_asset_names()):
            risks[i] = self.get_asset(asset_name).get_idiosyncratic_variance()
        return risks

    def get_sigma(self):

        # Validate wieights sum to 1
        self.check_weights()

        # Get assets risk betas
        betas = self.get_assets_risk_betas()

        # Get assets idio risk
        idio = self.get_assets_idio_variances()

        # Get Factor Covariance Matrix
        factor_cov = self._context.get_risk_factor_covariance()

        betas_mult_cov = np.matmul(betas, factor_cov.values)
        return np.matmul(betas_mult_cov, betas.transpose()) + np.diag(idio)

    def get_systematic_sigma(self):
        betas = self.get_assets_risk_betas()
        factor_cov = self._context.get_risk_factor_covariance()
        return np.matmul(np.matmul(betas, factor_cov.values), betas.transpose())

    def get_risk(self):
        return math.sqrt(
            np.matmul(
                np.matmul(
                    self.get_weights().transpose(), self.get_sigma()
                ),
                self.get_weights())
        )

    def get_asset_risk_betas(self, asset_name):
        return self.get_asset(asset_name).get_risk_betas()

    def get_assets_risk_betas(self):
        risk_betas = nans(self.num_assets, self.get_risk_factor_number())
        for i, asset_name in enumerate(self.get_asset_names()):
            betas = self.get_asset(asset_name).get_risk_betas()
            risk_betas[i] = betas.conj().T
        return risk_betas

    def get_risk_betas(self):
        return self.get_weights() @ self.get_assets_risk_betas()

    def get_systematic_risk(self):
        return math.sqrt(
            np.matmul(
                np.matmul(
                    self.get_weights().transpose(), self.get_systematic_sigma()
                ),
                self.get_weights())
        )

    def get_systematic_variance(self):
        return self.get_systematic_risk() ** 2

    def get_idio_variance(self):
        return (self.get_risk() ** 2) - (self.get_systematic_risk() ** 2)

    def get_total_variance(self):
        return self.get_systematic_variance() + self.get_idio_variance()

    def get_risk_decomposition_factor(self):

        fac_cov = self._context.get_risk_factor_covariance()
        wtd_risk = np.matmul(fac_cov.transpose(), self.get_risk_betas()).T
        wtd_fac_cov = fac_cov @ wtd_risk
        fact_cont = wtd_risk * wtd_fac_cov
        idio = self.get_idio_variance()
        return fact_cont, idio

    def get_fx_risk_decomposition(self):

        factor_cov = self._context.get_risk_factor_covariance()
        wts = self.get_weights()

        # Get the current risk parameters
        betas = self.get_risk_betas()
        idio = self.get_idio_variance()

        # Get the asset (Hedged) risk parameters
        betas_H = self.get_risk_betas()
        idio_H = self.get_idio_variance()

        ccy_betas = betas - betas
        ccy_idio = idio - idio_H

        asset_cov = betas_H @ factor_cov @ betas_H.T + np.diag(idio_H)
        ccy_cov = ccy_betas @ factor_cov @ ccy_betas.T + 2 * ccy_betas @ factor_cov @ betas_H + np.diag(ccy_idio)

        asset_risk = wts * (asset_cov @ wts) / self.get_risk()
        fx_risk = wts * (ccy_cov @ wts) / self.get_risk()
        return asset_risk, fx_risk

    def get_marginal_asset_risk_contribution(self):
        return np.matmul(self.get_sigma(), self.get_weights())

    def get_total_asset_risk_contribution(self):
        return self.get_weights() * self.get_marginal_asset_risk_contribution()

    def get_risk_decomposition(self):
        return self.get_total_asset_risk_contribution() / self.get_total_variance()

    def get_factor_stress_tests(self):
        return self._simulation.get_factor_stress_tests()

    def get_factor_stress_tests_extended(self):
        return self._simulation.get_factor_stress_tests_extended()

    def get_portfolio_var_pol(self):
        return self._simulation.get_portfolio_var_pol()

    def get_portfolio_var_pol_exc_ss(self):
        pass

    def get_stress_multiplier(self):
        return self.get_simulator().get_stress_coeff_ts()

    def get_tracking_error(self):
        pass

    def get_single_stock_risk_decomposition(self):
        pass



    ################### Optimization Related ###################

    def set_uncertainty_matrix(self, matrix):
        pass

    def get_uncertainty_matrix(self):
        pass

    def get_portfolio_uncertainty(self):
        pass

    def get_asset_data_length(self):
        pass

    def optimize(self, target_vol, contstraints):
        pass

    ################### Simulation Related ###################

    def get_factor_panels(self):
        return self._context.get_factor_panels()

    def get_stressed_returns_panel(self):
        return self.get_simulator().get_stressed_returns_panel()

    def get_stressed_risk_panel(self):
        return self.get_simulator().get_stressed_risk_panel()

    def get_portfolio_simulated_returns_panel(self):
        return self.get_simulator().get_portfolio_simulated_returns_panel()

    def get_portfolio_wealth_projection(self):
        pass

    ################### Historical Related #######################

    def get_historical_stress_tests(self):
        return self.get_simulator().get_historical_stress_tests()

    def get_realized_asset_return_panel(self):

        panel = CTimeSeries()
        for i, asset_name in enumerate(self.get_asset_names()):
            panel = panel.concat(self.get_asset(asset_name).get_realized_return_time_series())
        return panel.reindex(self._context.dates).dropna()

    def get_historical_return_time_series(self):
        return self.get_realized_asset_return_panel() @ self.get_flattened_weights()

    def get_historical_cuml_return_series(self):
        return self.get_historical_return_time_series().add(1).cumprod()

    def get_historical_real_cuml_return_series(self):
        return self.get_historical_cuml_return_series() / self._context.get_price_deflator().values

    def get_factor_backfilled_returns_panel(self):
        pass

    def get_historical_worst_peak_to_trough_loss(self):
        pass

    def get_historical_max_drawdown(self):
        pass

    def get_get_worst_periodic_return(self):
        pass

    def get_worst_periodic_real_return(self):
        pass

    def get_historical_excess_return(self):
        pass

    def get_historical_beta(self):
        pass

    ################### Public Portfolio Methods ##################

    def get_income_summary(self, assumption_version=None, income_version=None, df=None):
        pass

    def check_for_unhedged_put_writing(self):
        pass

    def get_asset_reporting_names(self, weights=False, category_dict_flag=False):
        pass

    def get_lending_weight(self):
        pass

    def has_single_stock(self):
        pass


if __name__ == "__main__":

    from epsilonPhi.core.asset.AssetMgr import CAssetMgr
    from epsilonPhi.core.schema.Schema import ContextCreator, CContext
    from epsilonPhi.core.portfolio.SAAPortfolio import SAAPortfolio
    schema = ContextCreator(currency='USD',
                            start_date='30-Nov-1983',
                            end_date='31-Dec-2022').create_context()


    ptf = SAAPortfolio('portfolio', schema)
    ptf.add_asset_by_name('MSUSAML', 0.5, 0)
    ptf.add_asset_by_name('LHAGGBD', 0.5, 0)
    ptf.get_weights()
    self = ptf.get_portfolio_mgr()
    self.get_asset_risk_premia_in_current_environment()