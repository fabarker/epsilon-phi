from epsilonPhi.core.portfolio.Portfolio import CPortfolio
from epsilonPhi.core.config.configUtil import CAppConfig
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.logging import *
import copy

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

    @property
    def is_taxable(self):
        return self._is_taxable


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
        return self.get_portfolio_mgr().get_hedge_ratios_from_hedging_option(hedging_option)

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
        return self._is_taxable, self._tax_region, self._is_amt

    def get_tax_region(self):
        return self._tax_region

    ################### Factor Model Return Metrics ###################

    def get_current_risk_free_rate(self):
        return self.get_portfolio_mgr().get_current_risk_free_rate()

    def get_risk_free_rate(self) -> float:
        return self.get_portfolio_mgr().get_risk_free_rate()

    def get_asset_total_return(self, asset_name, after_tax=False):
        return self.get_portfolio_mgr().get_asset_total_return(asset_name, after_tax)

    def get_assets_return_betas(self):
        return self.get_portfolio_mgr().get_assets_return_betas()

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
        return self.get_portfolio_mgr().get_assets_total_current_env_risk_premia()

    def get_current_env_total_return(self):
        return self.get_portfolio_mgr().get_current_env_total_return()

    def get_current_env_risk_premias(self):
        return self.get_portfolio_mgr().get_current_env_risk_premias()

    def get_current_env_risk_premia(self):
        return self.get_portfolio_mgr().get_current_env_risk_premia()

    def get_asset_sharpe_ratio(self, asset_name):
        return self.get_portfolio_mgr().get_asset_sharpe_ratio(asset_name)

    def get_assets_sharpe_ratios(self):
        return self.get_portfolio_mgr().get_assets_sharpe_ratios()

    def get_assets_curr_env_sharpe_ratio(self):
        return self.get_portfolio_mgr().get_assets_curr_env_sharpe_ratio()

    def get_asset_curr_env_sharpe_ratio(self):
        return self.get_portfolio_mgr().get_asset_curr_env_sharpe_ratio()

    def get_sharpe_ratio(self):
        return self.get_portfolio_mgr().get_sharpe_ratio()

    def get_curr_env_sharpe_ratio(self):
        return self.get_portfolio_mgr().get_curr_env_sharpe_ratio()

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
        return self.get_portfolio_mgr().get_assets_idio_variances()

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

    def get_factor_stress_tests_extended(self):
        return self.get_portfolio_mgr().get_factor_stress_tests_extended()

    def get_stress_multiplier(self):
        return self.get_portfolio_mgr().get_stress_multiplier()

    def get_portfolio_var_pol(self, confidence=0.99, loss=0):
        return self.get_portfolio_mgr().get_portfolio_var_pol(confidence, loss)

    def get_portfolio_var_pol_exc_ss(self, confidence=0.99, loss=0):
        return self.get_portfolio_mgr().get_portfolio_var_pol_exc_ss(confidence, loss)

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

    def optimize(self, target_vol=None, contstraints=None, lower_bounds=None, upper_bounds=None):
        return self.get_portfolio_mgr().optimize(
            target_vol or self.get_risk(),
            contstraints,
            lower_bounds,
            upper_bounds
        )


    ################### Simulation Related ###################

    def get_factor_panels(self):
        return self.get_portfolio_mgr().get_factor_panels()

    def get_portfolio_simulated_returns_panel(self):
        return self.get_portfolio_mgr().get_portfolio_simulated_returns_panel()

    def get_stressed_returns_panel(self):
        return self._portfolio_mgr.get_stressed_returns_panel()

    def get_stressed_risk_panel(self):
        return self.get_portfolio_mgr().get_stressed_risk_panel()

    def get_portfolio_wealth_projection(self,
                                        ws_inflows=None,
                                        ws_outflows=None,
                                        ptf_sim_order=None,
                                        quantiles=None,
                                        ptf_list=None,
                                        frequency=Frequency.YEARLY
                                        ):
        return self.get_portfolio_mgr().get_portfolio_wealth_projection(
            ws_inflows=ws_inflows,
            ws_outflows=ws_outflows,
            ptf_sim_order=ptf_sim_order,
            quantiles=quantiles,
            ptf_list=ptf_list,
            frequency=frequency
        )

    def get_factor_backfilled_assets_returns_panel(self):
        return self.get_portfolio_mgr().get_factor_backfilled_assets_returns_panel()

    def get_factor_backfilled_return_series(self):
        return self.get_portfolio_mgr().get_factor_backfilled_return_series()


    ################### Public Portfolio Methods ##################

    def get_income_summary(self, assumption_version=None, income_version=None, df=None):
        pass

    def check_for_unhedged_put_writing(self):
        self.get_portfolio_mgr().check_for_unhedged_put_writing()

    def get_asset_reporting_names(self, weights=False, category_dict_flag=False):
        pass


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


    @staticmethod
    def create_equal_weighted_portfolio(
            asset_list,
            portfolio_name="EqualWeightedPortfolio",
            context=None,
            tax_info=None
    ):

        if context is None:
            raise Exception("Passed in None context...")

        SAAPortfolio.check_tax_info_type(tax_info)
        ptf = SAAPortfolio(
            portfolio_name,
            context,
            tax_info=tax_info
        )

        n = len(asset_list)
        for asset_name in asset_list:
            ptf.add_asset_by_name(
                asset_name,
                1 / n,
                0,
            )

        if isinstance(tax_info, TaxInfo) and tax_info is not None:
            ptf.set_tax_info(*tax_info.get_tax_info())

        # setup the portfolio
        ptf.setup()
        return ptf

    #TODO - Fix this function so it works
    def deepcopy(self, name: str = None):

        """
            Create a deep copy of the object. Optionally assign a new name.

            Args:
                name (str, optional): New name for the copied object.

            Returns:
                A fully independent deep copy of the object.
        """

        copyobj = super().deepcopy()
        if name is not None:
            copyobj.reporting_name = name
        return copyobj

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

    import numpy as np
    import pandas as pd

    # Load weights from excel
    raw = pd.read_excel(
        '/Users/francisbarker/repo/epsilon-psi/epsilon-phi-core/src/python/epsilonPhi/core/reporting/formatted_excel.xlsx',
        sheet_name='RAW WEIGHTS',
        index_col=0
    )

    last_row = mask = raw.astype(str).apply(lambda row: row.str.contains("total", case=False)).any(axis=1)
    names = np.setdiff1d(raw.columns, ['Hedge Ratios', 'Name', 'Category', 'Reporting Name'])

    ptfs = []

    hedge_ratios = raw['Hedge Ratios'][~last_row].astype(float)
    categories = raw['Category'][~last_row].astype(str)
    reporting_names = raw['Reporting Name'][~last_row].astype(str)

    for col in names:
        weights = raw[col][~last_row].replace("-", 0).astype(float)
        weights /= weights.sum()

        ptf = SAAPortfolio.create_equal_weighted_portfolio(
            weights.index,
            portfolio_name=col,
            context=schema,
        )

        ptf.set_weights(weights.values)
        ptf.set_hedging_ratios(hedge_ratios.values)

        for asset in ptf.get_assets():
            asset.set_reporting_info(
                reporting_names.loc[asset.name[0]],
                categories.loc[asset.name[0]]
            )

        ptfs.append(ptf)


    # 1. Create the portfolio allocations page

    all_ptfs = []
    for p in ptfs:

        asset_categories = np.array([x.category for x in p.get_assets()])
        reporting_names = np.array([x.reporting_name for x in p.get_assets()])
        unique_categories = np.unique(asset_categories)

        dfs = []
        for cat in unique_categories:
            idx = asset_categories == cat

            wts = p.get_flattened_weights()[idx]
            wts = np.concatenate(([np.sum(wts)], wts * 100))
            names = [cat] + ["  " + x for x in reporting_names[idx]]
            tmp = pd.DataFrame(
                wts, columns=[p.name], index=names
            )

            dfs.extend([tmp])

        res = pd.concat(dfs)
        res = pd.concat((res, pd.DataFrame(1, columns=[p.name], index=["Total"])), axis=0)
        all_ptfs.extend([res])
    output = pd.concat(all_ptfs, axis=1)

    # 2. Create the assumptions page
    assets = []
    unique_cats = set([x.category for y in ptfs for x in y.get_assets()])
    for cat in unique_cats:

        cat_list = []
        for p in ptfs:
            for a in p.get_assets():

                if a.category == cat:

                    asmpt = {('','reporting_name'): "    " + a.reporting_name,
                             ('Long-Term Estimates', 'Lower Range'): a.get_risk_premia() - a.get_uncertainty(),
                             ('Long-Term Estimates', 'Risk Premia\nwith Estimated Range'): a.get_risk_premia(),
                             ('Long-Term Estimates', 'Upper Range'): a.get_risk_premia() + a.get_uncertainty(),
                             ('Long-Term Estimates', 'Volatility'): a.get_volatility(),
                             ('Long-Term Estimates', 'Sharpe Ratio'): a.get_sharpe_ratio(),
                             ('Long-Term Estimates', 'Estimated Mean Return\n(2.5% Risk Free Rate)'): a.get_total_return(),
                             ('', 'Hedging Ratio'): a.hedging_ratio,
                             ('Modelling Dates', 'From'): a.index.min(),
                             ('Modelling Dates', 'To'): a.index.max()
                             }

                    tmp = pd.DataFrame(asmpt.values(), index=pd.MultiIndex.from_tuples(asmpt.keys())).T
                    cat_list.extend([tmp])

        cat_list = pd.concat(cat_list, axis=0)
        tmp = pd.DataFrame([cat] + list(np.full(cat_list.shape[1] - 1, np.nan)), index=cat_list.columns).T
        assets.extend([pd.concat((tmp, cat_list), axis=0)])

    assets = pd.concat(assets, axis=0)
    assets = assets[~assets.duplicated(keep='first')].set_index(("", "reporting_name"), drop=True)
    assets.index.names = [None]

    import xlwings as xw

    # Open Excel (if not already running) and create a new workbook
    wb = xw.Book()  # This opens a new Excel workbook
    sheet = wb.sheets[0]

    # Write to Excel live
    sheet.range("A1").value = "Hello from Python!"
    sheet.range("B1:B5").value = [[i ** 2] for i in range(1, 6)]

    # Optional: keep Excel visible and interactive
    wb.app.visible = True

    wb = xw.Book()
    sheet = wb.sheets[0]
    sheet.name = "Estimates"

    # Clear previous content (optional)
    sheet.clear()

    # === WRITE DATAFRAME ===
    start_row = 2
    sheet.range((start_row, 1)).value = assets

    # === HEADERS ===
    # Write the merged header "Risk Premium with Estimated Range"
    sheet.range("B2:G2").merge()
    sheet.range("I2:J2").merge()
    sheet.range("B2").color = (255, 255, 255)

    # === WRITE DATAFRAME ===
    start_row = 2
    sheet.range((start_row, 1)).value = assets

    # === SECTION HEADERS FORMATTING ===
    for row_idx, (index_label, row_data) in enumerate(df.iterrows()):
        if pd.isna(row_data).all():
            cell = sheet.range((start_row + row_idx, 1))
            cell.value = index_label
            cell.api.Font.Bold = True
            cell.color = (242, 242, 242)

    # === CONDITIONAL FORMATTING ===
    # Highlight positive (green) and negative (red) in column B (Risk Premium Low End)
    n_rows = len(df)

    for i in range(n_rows):
        cell = sheet.range((start_row + i, 2))  # column B
        value = cell.value
        if isinstance(value, (float, int)):
            if value < 0:
                cell.color = (192, 0, 0)  # red
            elif value > 0:
                cell.color = (0, 112, 0)  # green

    # === COLUMN WIDTHS ===
    sheet.range("A:A").column_width = 35
    for col in range(2, 9):
        sheet.range((1, col)).column_width = 12

    # === OPTIONAL: Freeze header row ===
    sheet.api.Application.ActiveWindow.SplitRow = start_row - 1
    sheet.api.Application.ActiveWindow.FreezePanes = True

    for col in range(2, 9):  # B to H → column numbers 2 to 8
        rng = sheet.range((1, col), (1000, col))  # rows 1–1000 (adjust as needed)
        rng.number_format = '0.0%'  # or '0.00%' for two decimal places

    from openpyxl import load_workbook
    from openpyxl.styles import Alignment

    wb = load_workbook("your_file.xlsx")
    ws = wb.active

    # Center align B5 to H50
    for row in ws.iter_rows(min_row=5, max_row=50, min_col=2, max_col=8):
        for cell in row:
            cell.alignment = Alignment(horizontal="center")

    ISG_FACTOR_SHARPES = [0.37, 0.36, 0.58, 0.33, 0.28, 0.12]

    # Load the uploaded workbook
    wb = load_workbook("/Users/francisbarker/repo/epsilon-psi/epsilon-phi-core/src/python/epsilonPhi/core/reporting/Book2.xlsx")
    ws = wb.active

    # We'll extract formatting from a few representative cells in the range B5:H5 as a sample
    sample_range = ws["B5:H5"][0]

    # Extract styles from these cells
    style_summary = []
    for cell in sample_range:
        style_summary.append({
            "cell": cell.coordinate,
            "font": {
                "name": cell.font.name,
                "size": cell.font.size,
                "bold": cell.font.bold,
                "italic": cell.font.italic,
                "color": cell.font.color.rgb if cell.font.color else None
            },
            "fill": {
                "type": cell.fill.fill_type,
                "fgColor": cell.fill.fgColor.rgb if cell.fill.fgColor else None
            },
            "alignment": {
                "horizontal": cell.alignment.horizontal,
                "vertical": cell.alignment.vertical,
                "wrap_text": cell.alignment.wrap_text
            },
            "number_format": cell.number_format,
            "border": {
                "top": cell.border.top.style,
                "bottom": cell.border.bottom.style,
                "left": cell.border.left.style,
                "right": cell.border.right.style
            }
        })

    column_widths = {}
    for col_letter in ['B', 'C', 'D', 'E', 'F', 'G', 'H']:
        width = ws.column_dimensions[col_letter].width
        column_widths[col_letter] = width

    # Extract row heights for relevant rows (let's check rows 1 to 10)
    row_heights = {}
    for row in range(1, 11):
        height = ws.row_dimensions[row].height
        row_heights[row] = height

    # Combine both into a single DataFrame for display
    col_df = pd.DataFrame(list(column_widths.items()), columns=['Column', 'Width'])
    row_df = pd.DataFrame(list(row_heights.items()), columns=['Row', 'Height'])

    # Merge for viewing
    combined_info = {
        'Column Widths': col_df,
        'Row Heights': row_df
    }