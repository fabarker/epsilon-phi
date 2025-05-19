from epsilonPhi.core.portfolio.Portfolio import CPortfolio
from epsilonPhi.core.config.configUtil import CAppConfig
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.logging import *

class TaxInfo:
    def __init__(self, is_taxable, tax_region, is_amt):
        if not isinstance(is_taxable, bool) or not isinstance(is_amt, bool):
            raise Exception("Expecting boolean value True/False for is_taxable/is_amt.")
        self._isTaxable = is_taxable
        self._taxRegion = tax_region
        self._isAMT = is_amt

    def get_tax_info(self):
        return self._isTaxable, self._taxRegion, self._isAMT


class SAAPortfolio(CPortfolio):
    def __init__(self, 
                 portfolio_name, 
                 context,
                 hedging_option=None, 
                 tax_info=None
                 ):
        
        # Call Constructor
        super(SAAPortfolio, self).__init__(portfolio_name, context)

        # Portfolio
        self._reporting_name = None
        self._hedging_option = None

        # Tax Related Attributes
        self._is_taxable = False
        self._tax_region = None
        self._is_amt = False
        self._tax_rates = []
        self._income_tax_rates = []

        ###### Factor-Based Return Attributes
        self._return_betas = None
        self._risk_premias = None
        self._risk_premias_curr_env = None

        self._alphas = None

        ###### Factor-Based Risk Attributes
        self._risk_betas = None
        self._sigma = None
        self._systematic_sigma = None

        # Historical Risk and Return Related
        self._historical_risk_premia = None

        # Leverage
        self._lending_weight = 0

        # Optimization Related Attributes
        self._kappa = None
        self._markowitz = None
        self._t12 = None

        # Reporting
        self._reportingName = None

        SAAPortfolio.check_tax_info_type(tax_info)
        if isinstance(tax_info, TaxInfo) and tax_info is not None:
            self.set_tax_info(*tax_info.get_tax_info())

        # Set up the portfolio manager
        self._hedgingOption = hedging_option if hedging_option else self.get_default_hedging_option()
        self.setup()

    def reset_properties(self):
        super().reset_properties()
        self.reset_risk()
        self.reset_return()

    def reset_risk(self):
        self._risk_betas = None
        self._sigma = None
        self._systematic_sigma = None
        self._historical_risk_premia = []
        self._kappa = None
        self._markowitz = None
        self._t12 = None

    def reset_return(self):
        self._return_betas = None
        self._risk_premias = None
        self._risk_premias_curr_env = None
        self._alphas = None

    def setup(self):
        from epsilonPhi.core.portfolio.SAAPortfolioMgr import SAAPortfolioMgr
        if not self._is_setup and self._portfolio_mgr is None:
            self._portfolio_mgr = SAAPortfolioMgr(self._context)
            self._portfolio_mgr.set_portfolio(self)
            self._is_setup = True

    ########## Public Properties ############

    @property
    def portfolio_mgr(self):
        return self._portfolio_mgr

    @property
    def reporting_name(self):
        return self._reportingName

    @reporting_name.setter
    def reporting_name(self, value):
        self._reportingName = value


    ############## Setter Methods #################

    def set_taxable(self, taxable):
        self._is_taxable = taxable

    def set_kappa(self, kappa):
        self._kappa = kappa

    def set_markowitz(self, m):
        self._markowitz = m

    def set_lending_weight(self, weight):
        self.get_portfolio_mgr().set_lending_weight(weight)

    def set_tax_info(self, is_taxable, tax_region, is_amt):
        if is_taxable is False or is_taxable == 'N':
            self._is_taxable = False
        elif is_taxable is True or is_taxable == 'Y':
            if self._context.currency == 'USD':
                self._is_taxable = True
                self._tax_region = tax_region
                self._is_amt = is_amt
            else:
                logger.error("Tax modeling only supported for USD perspective")
                raise Exception("Invalid tax region for currency.")
        else:
            raise Exception("Unrecognized tax flag.")

    def set_tax_rates(self, **kwargs):
        pass


    ################### Hedging Related ###################

    def get_hedge_ratios_from_hedging_option(self, hedging_option):
        return self.portfolio_mgr.get_hedge_ratios_from_hedging_option(hedging_option)

    def get_hedging_option(self):
        return self._hedgingOption

    def get_default_hedging_option(self):
        if self._context:
            return 'Hedged'
        else:
            raise Exception("Context not set in Portfolio")

    def set_hedging_ratios_from_hedging_option(self, hedging_option):
        # Get each assets hedging ratio for the specified hedging option
        ratios = self.get_hedge_ratios_from_hedging_option(hedging_option)

        # Set hedge ratios in the portfolios
        self.set_hedging_ratios(ratios)
        self._hedging_option = hedging_option
        return ratios

    def set_hedging_option(self, hedging_option):
        self.set_hedging_ratios_from_hedging_option(hedging_option)

    ###################### Tax Related #######################

    def get_tax_info(self):
        return self._is_taxable, self._tax_region, self._isAMT

    def get_tax_region(self):
        return self._tax_region

    ################### Factor Model Return Metrics ###################

    def get_asset_total_return(self, asset_name, after_tax=False):
        return self.get_portfolio_mgr().get_asset_total_return(asset_name, after_tax)

    def get_assets_total_return(self):
        return self.get_portfolio_mgr().get_assets_total_return()

    def get_asset_risk_premias(self, asset_name):
        return self.get_portfolio_mgr().get_asset_risk_premias(asset_name)

    def get_assets_risk_premias(self):
        return self.get_portfolio_mgr().get_assets_risk_premias()

    def get_asset_total_risk_premia(self, asset_name):
        return self.get_portfolio_mgr().get_asset_total_risk_premia(asset_name)

    def get_assets_total_risk_premia(self):
        return self.get_portfolio_mgr().get_assets_total_risk_premia()

    def get_total_return(self):
        return self.get_portfolio_mgr().get_total_return()

    def get_risk_premias(self):
        return self.get_portfolio_mgr().get_risk_premias()

    def get_risk_premia(self):
        return self.get_portfolio_mgr().get_risk_premia()

    def get_asset_current_env_total_return(self, asset_name):
        return self.get_portfolio_mgr().get_asset_current_env_total_return(asset_name)

    def get_assets_current_env_total_return(self):
        return self.get_portfolio_mgr().get_assets_current_env_total_return()

    def get_asset_current_env_risk_premias(self, asset_name):
        return self.get_portfolio_mgr().get_asset_current_env_risk_premias(asset_name)

    def get_assets_current_env_risk_premias(self):
        return self.get_portfolio_mgr().get_assets_current_env_risk_premias()

    def get_asset_total_current_env_risk_premia(self, asset_name):
        return self.get_portfolio_mgr().get_asset_total_current_env_risk_premia(asset_name)

    def get_assets_total_current_env_risk_premia(self):
        return self.get_portfolio_mgr().get_assets_total_risk_premias()

    def get_current_env_total_return(self):
        return self.get_portfolio_mgr().get_current_env_total_return()

    def get_current_env_risk_premias(self):
        return self.get_portfolio_mgr().get_current_env_risk_premias()

    def get_current_env_risk_premia(self):
        return self.get_portfolio_mgr().get_current_env_risk_premia()

    def get_asset_sharpe_ratio(self, asset_name):
        return self.get_portfolio_mgr().get_asset_sharpe_ratio(asset_name)

    def get_assets_sharpe_ratios(self):
        return self.get_portfolio_mgr().get_asset_sharpe_ratio()

    def get_assets_curr_env_sharpe_ratio(self):
        return self.get_portfolio_mgr().get_assets_curr_env_sharpe_ratio()

    def get_asset_curr_env_sharpe_ratio(self):
        return self.get_portfolio_mgr().get_asset_curr_env_sharpe_ratio()

    def get_sharpe_ratio(self):
        return self.get_portfolio_mgr().get_sharpe_ratio()

    def get_curr_env_sharpe_ratio(self):
        pass

    def get_current_environment_risk_premia_5yr(self):
        return self.get_portfolio_mgr().get_current_environment_risk_premia_5yr()

    def get_asset_return_medium(self):
        return self.get_portfolio_mgr().get_asset_return_medium()

    def get_medium_return(self):
        return self.get_portfolio_mgr().get_medium_return()

    def get_return_betas(self):
        return self.get_portfolio_mgr().get_return_betas()

    def get_alpha(self):
        return self.get_portfolio_mgr().get_alpha()

    def get_asset_alphas(self):
        return self.get_portfolio_mgr().get_asset_alphas()


    ################### Factor Model Risk Metrics ###################

    def get_asset_risk(self):
        return self.get_portfolio_mgr().get_asset_risk()

    def get_asset_systematic_variance(self, asset_name):
        return self.get_portfolio_mgr().get_asset_systematic_variance(asset_name)

    def get_asset_idio_variance(self, asset_name):
        return self.get_portfolio_mgr().get_asset_idio_variance(asset_name)

    def get_assets_idio_variance(self):
        return self.get_portfolio_mgr().get_assets_idio_variance()

    def get_assets_risk(self):
        return self.get_portfolio_mgr().get_assets_risk()

    def get_assets_systematic_variances(self):
        return self.get_portfolio_mgr().get_assets_systematic_variances()

    def get_assets_idio_variances(self):
        return self.get_portfolio_mgr().get_assets_idio_variances()

    def get_sigma(self):
        return self.get_portfolio_mgr().get_sigma()

    def get_systematic_sigma(self):
        return self.get_portfolio_mgr().get_systematic_sigma()

    def get_risk(self):
        return self.get_portfolio_mgr().get_risk()

    def get_asset_risk_betas(self, asset_name):
        return self.get_portfolio_mgr().get_asset_risk_betas(asset_name)

    def get_assets_risk_betas(self):
        return self.get_portfolio_mgr().get_assets_risk_betas()

    def get_risk_betas(self):
        return self.get_portfolio_mgr().get_risk_betas()

    def get_systematic_risk(self):
        return self.get_portfolio_mgr().get_systematic_risk()

    def get_systematic_variance(self):
        return self.get_portfolio_mgr().get_systematic_variance()

    def get_idio_variance(self):
        return self.get_portfolio_mgr().get_idio_variance()

    def get_total_variance(self):
        return self.get_portfolio_mgr().get_total_variance()

    def get_risk_decomposition_factor(self):
        return self.get_portfolio_mgr().get_risk_decomposition_factor()

    def get_fx_risk_decomposition(self):
        return self.get_portfolio_mgr().get_fx_risk_decomposition()

    def get_marginal_asset_risk_contribution(self):
        return self.get_portfolio_mgr().get_marginal_asset_risk_contribution()

    def get_total_asset_risk_contribution(self):
        return self.get_portfolio_mgr().get_total_asset_risk_contribution()

    def get_risk_decomposition(self):
        return self.get_portfolio_mgr().get_risk_decomposition()

    def get_factor_stress_tests(self):
        return self.get_portfolio_mgr().get_factor_stress_tests()

    def get_stress_multiplier(self):
        return self.get_portfolio_mgr().get_stress_multiplier()

    def get_factor_stress_tests_extended(self):
        return self.get_portfolio_mgr().get_factor_stress_tests_extended()

    def get_portfolio_var_pol(self):
        return self.get_portfolio_mgr().get_portfolio_var_pol()

    def get_portfolio_var_pol_exc_ss(self):
        return self.get_portfolio_mgr().get_portfolio_var_pol_exc_ss()

    def get_stress_multiplier(self):
        return self.get_portfolio_mgr().get_stress_multiplier()

    def get_tracking_error(self):
        return self.get_portfolio_mgr().get_tracking_error()

    def get_single_stock_risk_decomposition(self):
        return self.get_portfolio_mgr().get_single_stock_risk_decomposition()


    ################### Optimization Related ###################

    def set_uncertainty_matrix(self, matrix):
        return self.get_portfolio_mgr().set_uncertainty_matrix(matrix)

    def get_uncertainty_matrix(self):
        return self.get_portfolio_mgr().get_uncertainty_matrix()

    def get_portfolio_uncertainty(self):
        return self.get_portfolio_mgr().get_portfolio_uncertainty()

    def get_asset_data_length(self):
        return self.get_portfolio_mgr().get_asset_data_length()

    def optimize(self, target_vol, contstraints):
        return self.get_portfolio_mgr().optimize(target_vol, contstraints)


    ################### Simulation Related ###################

    def get_factor_panels(self):
        return self.get_portfolio_mgr().get_factor_panels()

    def get_portfolio_simulated_returns_panel(self):
        return self.get_portfolio_mgr().get_portfolio_simulated_returns_panel()

    def get_stressed_returns_panel(self):
        return self._portfolio_mgr.get_stressed_returns_panel()

    def get_stressed_risk_panel(self):
        return self.get_portfolio_mgr().get_stressed_risk_panel()

    def get_portfolio_wealth_projection(self):
        return self.get_portfolio_mgr().get_portfolio_wealth_projection()

    def get_factor_backfilled_returns_panel(self):
        return self.get_portfolio_mgr().get_factor_backfilled_returns_panel()


    ################### Public Portfolio Methods ##################

    def get_income_summary(self, assumption_version=None, income_version=None, df=None):
        pass


    def check_for_unhedged_put_writing(self):
        self.get_portfolio_mgr().check_for_unhedged_put_writing()

    def get_asset_reporting_names(self, weights=False, category_dict_flag=False):
        pass

    def get_lending_weight(self):
        return self.get_portfolio_mgr().get_lending_weight()

    def has_single_stock(self):
        return self.get_portfolio_mgr().has_single_stock()




    def get_private_equity_distribution(
            self,
            liquid_current_asset_total,
            pe_target_weight,
            pe_annual_commitment,
            pe_current_asset,
            wealth_flows=None,
            num_years=20
    ):
        pass

    def get_private_equity_distribution_sub(
            self,
            liquid_current_asset_total,
            subset_class_annual_commitments,
            pe_current_assets,
            pe_target_weights=None,
            wealth_flows=None,
            num_years=20,
            multiplier=2,
            shocks=None,
            use_total_mv=False,
            target_system=None
    ):
        pass

    def get_private_equity_distribution_sub_standalone(
            self,
            liquid_current_asset_total,
            subset_class_annual_commitments,
            pe_curr,
            pe_target_weights=None,
            wealth_flows=None,
            num_years=20,
            multiplier=2,
            shocks=None,
            use_total_mv=False
    ):
        pass

    def get_portfolio_wealth_projection(
            self,
            inflows,
            outflows,
            frequency=Frequency.YEARLY,
            ptf_array=None,
            ptf_simulation_order=None,
            quantiles=[]
    ):
        pass

    def get_portfolio_wealth_projection_custom(
            self,
            inflows,
            outflows,
            frequency=Frequency.YEARLY,
            ptf_array=None,
            ptf_simulation_order=None,
            quantiles=[],
            custom_class=None,
            *args,
            **kwargs):
        pass

    @staticmethod
    def _copy_portfolio(portfolio_mgr, name):

        """
        This method is used from SAAPortfolioMgr to make a copy of portfolio and copying assets with reference.
        By copying asset reference, we are preserving the asset parameters set on the asset object.
        If we create new asset object, properties like alpha/idioVar will get wiped out, if set on original asset.
        :param name: portfolio name
        :return: SAAPortfolio
        """

        ptf = SAAPortfolio(
            name,
            portfolio_mgr.get_p_context(),
            portfolio_mgr.get_hedging_option()
        )

        ptf.set_tax_info(
            *portfolio_mgr.get_tax_info()
        )

        for i, asset_name in enumerate(portfolio_mgr.get_asset_names()):
            asset = portfolio_mgr.get_asset(asset_name)
            ptf.add_asset(asset, asset.weight, asset.hedge_ratio)

        if portfolio_mgr.is_taxable():
            ptf.set_tax_rates(*portfolio_mgr.get_tax_rates())

        return ptf


    @staticmethod
    def create_equal_weighted_portfolio(
            asset_list,
            portfolio_name="EqualWeightedPortfolio",
            context=None,
            asset_time_series=None,
            tax_info=None
    ):

        if context is None:
            raise Exception("Passed in None context...")

        SAAPortfolio.check_tax_info_type(tax_info)
        ptf = SAAPortfolio(
            portfolio_name,
            context
        )

        n = len(asset_list)
        for asset_name in asset_list:
            asset_ts = asset_time_series.get(asset_name, None)
            ptf.add_asset_by_name(
                asset_name,
                1 / n,
                0,
                asset_time_series=asset_ts
            )

        if isinstance(tax_info, TaxInfo) and tax_info is not None:
            ptf.set_tax_info(*tax_info.get_tax_info())

        return ptf

    @staticmethod
    def clone_portfolio(portfolio, portfolio_name="Custom Portfolio", equal_weighted=False):
        if portfolio is None or portfolio.get_p_context() is None:
            raise Exception("None portfolio or context passed in")

        ptf = SAAPortfolio(portfolio_name, portfolio.get_p_context(), portfolio.get_hedging_option())
        if equal_weighted:
            equal_weight = 1 / len(portfolio.get_assets())
        for i, (asset_name, asset_param) in enumerate(portfolio.get_asset_params().items()):
            asset_ts = portfolio.get_assets()[asset_name]
            weight = asset_param.get_weight() if not equal_weighted else equal_weight
            #from epsilonPhi.core.asset.proxies CCustomAsset
            #from cyrus.isg.saa.assets.csingle_stock import CSingleStock
            #if isinstance(asset_ts, (CCustomAsset, CSingleStock)):
            #    ptf.add_asset_by_name(asset_name, weight, passed_asset_ts=asset_ts)
            #else:
            ptf.add_asset_by_name(asset_name, weight)
        ptf.set_tax_info(*portfolio.get_tax_info())
        return ptf


    @staticmethod
    def check_tax_info_type(tax_info, type_to_verify=TaxInfo):
        if tax_info is not None:
            if isinstance(tax_info, type_to_verify) is False:
                raise TypeError("Expected type of tax_info is {} got {}".format(type_to_verify, type(tax_info)))

if __name__ == "__main__":

    from epsilonPhi.core.asset.AssetMgr import CAssetMgr
    from epsilonPhi.core.schema.Schema import ContextCreator
    schema = ContextCreator(
        currency='USD',
        start_date='30-Nov-1983',
        end_date='31-Dec-2022'
    ).create_context()

    assetMgr = CAssetMgr(schema)
    asset = assetMgr.get_asset_by_name('MSUSAML')

    self = SAAPortfolio('portfolio', schema)
    self.add_asset_by_name('MSUSAML', 1, 0)
    self.get_portfolio_mgr()
    self.get_sigma()
