from typing import Union, Iterable
from epsilonPhi.core.portfolio.SAAPortfolio import SAAPortfolio
import pandas as pd
import numpy as np
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl import load_workbook
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.chart import LineChart, Reference
from openpyxl.chart.layout import Layout, ManualLayout
from openpyxl.drawing.line import LineProperties
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.chart.text import RichText
from openpyxl.drawing.text import Paragraph, ParagraphProperties, CharacterProperties
from openpyxl.drawing.text import Font as ChartingFont
from epsilonPhi.core.utils.ExcelUtils import ExcelUtils

import yaml
import os
import re
import math

#yaml_assumptions = "/Users/francisbarker/Repositories/Python/epsilon-phi/epsilon-phi-core/src/python/epsilonPhi/core/reporting/formatting_assumptions.yaml"
#yaml_portfolios = "/Users/francisbarker/Repositories/Python/epsilon-phi/epsilon-phi-core/src/python/epsilonPhi/core/reporting/formatting_portfolios.yaml"
#yaml_risk = "/Users/francisbarker/Repositories/Python/epsilon-phi/epsilon-phi-core/src/python/epsilonPhi/core/reporting/formatting_risk.yaml"


yaml_assumptions = "/Users/francisbarker/repo/epsilon-psi/epsilon-phi-core/src/python/epsilonPhi/core/reporting/formatting_assumptions.yaml"
yaml_portfolios = "/Users/francisbarker/repo/epsilon-psi/epsilon-phi-core/src/python/epsilonPhi/core/reporting/formatting_portfolios.yaml"
yaml_risk = "/Users/francisbarker/repo/epsilon-psi/epsilon-phi-core/src/python/epsilonPhi/core/reporting/formatting_risk.yaml"


class Reporting(object):

    def __init__(self,
                 report_directory: str,
                 report_name: str,
                 sort_by_vol: bool = False
                 ):

        self._report_dir = report_directory
        self._name = report_name
        self._portfolios = []
        self._wb = None
        self._sort_by_vol = sort_by_vol

    @property
    def output_path(self):
        return os.path.join(self._report_dir, self._name + '.xlsx')

    @property
    def portfolio_names(self):
        if len(self._portfolios) > 0:
            return [x.name for x in self._portfolios]

    def load_workbook(self):

        workbook_path = self.output_path

        if os.path.isfile(workbook_path):
            self._wb = load_workbook(workbook_path)
            print(f"Loaded existing workbook: {workbook_path}")
        else:
            self._wb = Workbook()
            default_sheet = self._wb.active
            self._wb.remove(default_sheet)
            print("Created new workbook (default sheet removed)")

    def get_workbook(self):
        if self._wb is None:
            self.load_workbook()
        return self._wb

    def save_workbook(self):
        if self._wb is not None:
            self._wb.save(self.output_path)

    def close_workbook(self):
        if self._wb is not None:
            self._wb.close()

    def get_worksheet(self, sheet_name):
        wb = self.get_workbook()
        if sheet_name in wb.sheetnames:
            self.delete_worksheet(sheet_name)
        return wb.create_sheet(title=sheet_name)

    def delete_worksheet(self, sheet_name):
        wb = self.get_workbook()
        if sheet_name in wb.sheetnames:  # safety check
            wb.remove(wb[sheet_name])

    def generate_report(self, include_wealth_simulations=True):

        if len(self._portfolios) == 0:
            return

        # Generate portfolio analytics
        self.write_portfolios()
        # Generate risk dashboard
        self.write_risk_dashboard()
        # Generate Assumtpions
        self.write_assumptions()
        # Generate Wealth Simulations

        if include_wealth_simulations:
            self.write_wealth_simulations()

        # Close Workbook
        self.close_workbook()

    def add_portfolios(self, portfolios: Union[SAAPortfolio, Iterable[SAAPortfolio]]) -> None:
        if not isinstance(portfolios, Iterable) or isinstance(portfolios, (str, bytes)):
            portfolios = [portfolios]

        self._portfolios.extend(portfolios)

    def get_unique_categories(self):
        return list(dict.fromkeys([x.category for y in self._portfolios for x in y.get_assets()]))

    def get_assumptions(self):
        assumption_blocks = []

        for category in self.get_unique_categories():
            category_rows = []

            for portfolio in self._portfolios:
                for asset in portfolio.get_assets():
                    if asset.category != category:
                        continue

                    # Cache repeated calls
                    risk_premia = asset.get_risk_premia()
                    uncertainty = asset.get_uncertainty()

                    assumptions = {
                        ("", "reporting_name"): "    " + asset.reporting_name,
                        ("Long-Term Estimates", "Risk Premia\n with Estimated Range"): risk_premia - uncertainty,
                        ("Long-Term Estimates", "Mean"): risk_premia,
                        ("Long-Term Estimates", "Upper Range"): risk_premia + uncertainty,
                        ("Long-Term Estimates", "Volatility"): asset.get_volatility(),
                        ("Long-Term Estimates", "Sharpe Ratio"): asset.get_sharpe_ratio(),
                        ("Long-Term Estimates",
                         "Estimated Mean Return\n(2.5% Risk Free Rate)"): asset.get_total_return(),
                        ("", "Hedging Ratio"): asset.hedging_ratio,
                        ("Modelling Dates", "From"): asset.index.min(),
                        ("Modelling Dates", "To"): asset.index.max()
                    }

                    df = pd.DataFrame([assumptions], columns=pd.MultiIndex.from_tuples(assumptions.keys()))
                    category_rows.append(df)

            if category_rows:
                category_df = pd.concat(category_rows, axis=0)

                # Add a category header row (merged-style)
                header_row = pd.DataFrame(
                    [[category] + [np.nan] * (category_df.shape[1] - 1)],
                    columns=category_df.columns
                )

                full_block = pd.concat([header_row, category_df], axis=0)
                assumption_blocks.append(full_block)

        # Concatenate all category blocks into one DataFrame
        if assumption_blocks:
            final_df = pd.concat(assumption_blocks, axis=0)
            final_df = final_df[~final_df.duplicated(keep="first")]
            final_df.set_index(("", "reporting_name"), drop=True, inplace=True)
            final_df.index.names = [None]
            return final_df
        else:
            return pd.DataFrame()

    def write_assumptions(self):

        # get the assumptions table
        df = self.get_assumptions()

        # get the workbook
        ws = self.get_worksheet("assumptions")

        # Load formatting from YAML
        with open(yaml_assumptions, "r") as f:
            fmt = yaml.safe_load(f)

        style_dict = fmt.get("row_styles", {})

        df.reset_index(inplace=True)
        for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), 1):

            if r_idx == 1:
               row = list(df.columns.get_level_values(0))
               row = [x if x != "index" else None for x in row]

            if r_idx == 2:
               row = list(df.columns.get_level_values(1))

            if r_idx in [1, 2, 3, 4]:
                row_style = style_dict.get(r_idx)
            elif isinstance(row[0], str) and row[0].startswith("    "):
                row_style = style_dict.get(4)
            else:
                row_style = style_dict.get(5)

            for c_idx, value in enumerate(row, 1):
                cell = ws.cell(row=r_idx, column=c_idx, value=value)

                style = row_style.get(
                    get_column_letter(
                        cell.column
                    )
                )

                # row height
                ws.row_dimensions[int(r_idx)].height = style.get("height", {})

                # Font
                font_cfg = style.get("font", {})
                cell.font = Font(
                    name=font_cfg.get("name"),
                    size=font_cfg.get("size"),
                    bold=font_cfg.get("bold"),
                    italic=font_cfg.get("italic"),
                    color=font_cfg.get("color")
                )

                # Fill
                fill_cfg = style.get("fill", {})
                if fill_cfg.get("type") and fill_cfg.get("fgColor"):
                    cell.fill = PatternFill(
                        fill_type=fill_cfg["type"],
                        fgColor=fill_cfg["fgColor"]
                    )

                # Alignment
                align_cfg = style.get("alignment", {})
                cell.alignment = Alignment(
                    horizontal=align_cfg.get("horizontal"),
                    vertical=align_cfg.get("vertical"),
                    wrap_text=align_cfg.get("wrap_text"),
                    indent=align_cfg.get("indent", 0)
                )

                # Number format
                if style.get("number_format"):
                    cell.number_format = style["number_format"]

                # Borders
                border_cfg = style.get("border", {})
                sides = {}
                for side in ["top", "bottom", "left", "right"]:
                    side_def = border_cfg.get(side, {})
                    sides[side] = Side(
                        style=side_def.get("style"),
                        color=side_def.get("color")
                    )
                cell.border = Border(**sides)

        # Apply column widths
        col_widths = fmt.get("column_widths", {})
        for col in ws.iter_cols(min_row=1, max_row=ws.max_row):
            if any(cell.value is not None for cell in col):
                col_letter = get_column_letter(col[0].column)
                ws.column_dimensions[col_letter].width = col_widths.get(col_letter)

        # Post Processing of SR
        for row in ws.iter_rows(min_row=1, min_col=6, max_col=6):
            for cell in row:
                if isinstance(cell.value, (int, float)):  # Only format numeric cells
                    cell.number_format = '0.00'

        ws.merge_cells("B1:G1")
        ws.merge_cells("I1:J1")
        ws.merge_cells("B2:D2")

        cell = ws["B2"]
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True
        )

        # Find the last active row (non-empty)
        last_row = ws.max_row

        # Get the max number of columns used
        max_col = ws.max_column

        # Apply bottom border to each cell in the last active row
        for col in range(1, max_col + 1):
            cell = ws.cell(row=last_row, column=col)
            cell.border = Border(
                top=cell.border.top,
                left=cell.border.left,
                right=cell.border.right,
                bottom=Side(style='thin')  # Add bottom border
            )

        # Save Edits in the workbook
        self.save_workbook()

    def get_portfolios_table(self):

        all_portfolios = []

        for portfolio in self._portfolios:
            portfolio_name = portfolio.name
            assets = portfolio.get_assets()
            weights = portfolio.get_flattened_weights()

            # Gather asset info
            categories = np.array([a.category for a in assets])
            names = np.array([a.reporting_name for a in assets])
            unique_categories = list(dict.fromkeys(categories))

            category_blocks = []

            for cat in unique_categories:
                # Mask for current category
                mask = categories == cat

                # Extract weights and names for this category
                cat_weights = weights[mask]
                cat_names = names[mask]

                # Calculate weights: category total + asset-level weights (scaled to %)
                cat_total = np.sum(cat_weights)
                all_weights = np.concatenate([[cat_total], cat_weights * 100])

                # Prepare index: category name + indented asset names
                index = [cat] + ["  " + name for name in cat_names]

                df = pd.DataFrame(all_weights, index=index, columns=[portfolio_name])
                category_blocks.append(df)

            # Combine all category blocks and add a "Total" row
            portfolio_df = pd.concat(category_blocks)
            total_row = pd.DataFrame([1], index=["TOTAL"], columns=[portfolio_name])
            risk_and_return = [np.nan, portfolio.get_total_return(), portfolio.get_sharpe_ratio(), None,
                               portfolio.get_risk()]
            df_rr = pd.DataFrame(risk_and_return, index=["", "Estimated Mean Return", "Sharpe Ratio", "", "Volatility"],
                                 columns=[portfolio.name])
            portfolio_df = pd.concat([portfolio_df, total_row, df_rr])
            all_portfolios.append(portfolio_df)

        # Combine all portfolios side-by-side
        final_df = pd.concat(all_portfolios, axis=1)
        return final_df

    def write_portfolios(self):

        df = self.get_portfolios_table()
        ws = self.get_worksheet("portfolios")

        # Load formatting from YAML
        with open(yaml_portfolios, "r") as f:
            fmt = yaml.safe_load(f)

        style_dict = fmt.get("styles", {})

        max_rows = df.shape[0] + 2
        for r_idx, row in enumerate(dataframe_to_rows(df, index=True, header=True), 1):

            if r_idx == 2:
                continue

            if r_idx > 2:
                r_idx = r_idx - 1

            for c_idx, value in enumerate(row, 1):
                cell = ws.cell(row=r_idx, column=c_idx, value=value)
                coord = cell.coordinate

                if coord not in style_dict:
                    # it is either, standard, category or last
                    if re.fullmatch(r"[A-Za-z]+1", coord):
                        coord = "P1"
                    elif isinstance(value, str) and value.upper() == "TOTAL":
                        coord = "AT"
                    elif isinstance(row[0], str) and row[0].upper() == "TOTAL":
                        coord = "PT"
                    elif row[0] in [None, "", '']:
                        coord = "E"
                    elif re.fullmatch(r"A\d+", coord) and row[0].startswith(" "):
                        coord = "AS"
                    elif re.fullmatch(r"A\d+", coord) and r_idx < max_rows - 4:
                        coord = "AC"
                    elif isinstance(row[0], str) and row[0].startswith(" "):
                        coord = "PS"
                    elif isinstance(row[0], str) and isinstance(value, float) and row[0] not in [
                        "Estimated Mean Return", "Sharpe Ratio", "Volatility"]:
                        coord = "PC"
                    elif value in ["Estimated Mean Return", "Sharpe Ratio", "Volatility"]:
                        coord = "O"
                    else:
                        coord = "O1"

                style = style_dict[coord]

                # row height
                ws.row_dimensions[int(r_idx)].height = style.get("height", {})

                # Font
                font_cfg = style.get("font", {})
                cell.font = Font(
                    name=font_cfg.get("name"),
                    size=font_cfg.get("size"),
                    bold=font_cfg.get("bold"),
                    italic=font_cfg.get("italic"),
                    color=font_cfg.get("color")
                )

                # Fill
                fill_cfg = style.get("fill", {})
                if fill_cfg.get("type") and fill_cfg.get("fgColor"):
                    cell.fill = PatternFill(
                        fill_type=fill_cfg["type"],
                        fgColor=fill_cfg["fgColor"]
                    )

                # Alignment
                align_cfg = style.get("alignment", {})
                cell.alignment = Alignment(
                    horizontal=align_cfg.get("horizontal"),
                    vertical=align_cfg.get("vertical"),
                    wrap_text=align_cfg.get("wrap_text"),
                    indent=align_cfg.get("indent", 0)
                )

                # Number format
                if style.get("number_format"):
                    cell.number_format = style["number_format"]

                # Borders
                border_cfg = style.get("border", {})
                sides = {}
                for side in ["top", "bottom", "left", "right"]:
                    side_def = border_cfg.get(side, {})
                    sides[side] = Side(
                        style=side_def.get("style"),
                        color=side_def.get("color")
                    )
                cell.border = Border(**sides)

        # Apply column widths
        col_widths = fmt.get("column_widths", {})
        for col in ws.iter_cols(min_row=1, max_row=ws.max_row):
            if any(cell.value is not None for cell in col):
                col_letter = get_column_letter(col[0].column)
                ws.column_dimensions[col_letter].width = col_widths.get(col_letter, col_widths.get("B"))

        # Post Processing of SR
        loc = df.index.get_loc("Sharpe Ratio") + 2
        for col in range(2, ws.max_column + 1):
            cell = ws.cell(row=loc, column=col)
            cell.number_format = "0.00"  # set to 1 decimal place

        # Save Workbook
        self.save_workbook()

    def get_crisis_tbl(self):
        return pd.concat([
            pd.DataFrame({
                period: {"Nominal": s.total, "Real": s.real}
                for period, s in ptf.get_factor_stress_tests().items()
            }).T
            .set_axis(pd.MultiIndex.from_product([["Nominal", "Real"], [ptf.name]]), axis=1)
            .rename(index=lambda x: "  " + str(x))
            for ptf in self._portfolios
        ], axis=1)

    def get_portfolio_tbl(self):

        # Get and clean portfolio table
        tb = self.get_portfolios_table().dropna()

        # Split into categories and totals
        categories = self.get_unique_categories()
        category_rows = [idx for idx in tb.index if idx in categories]
        total_start = tb.index.get_loc("TOTAL")
        total_rows = tb.index[total_start:]

        # Filter and reorder rows
        tb = tb.loc[category_rows + list(total_rows)]

        # Create a MultiIndex column: (portfolio_name, type)
        types = ["Nominal"] * tb.shape[1] + ["Real"] * tb.shape[1]
        multi_cols = pd.MultiIndex.from_tuples(
            [(t, col) for col, t in zip(list(tb.columns) * 2, types)]
        )

        # Duplicate data across types (Nominal, Real)
        tb2 = pd.concat([tb, tb], axis=1)
        tb2.columns = multi_cols

        # Add annotations
        tb2_T = tb2.T.copy()
        tb2_T["Factor Based Risk Analytics"] = None
        tb2_T["Predicted Performance Over Stress Periods"] = tb2_T.index.get_level_values(0)

        # Final tidy dataframe
        return tb2_T.T

    def get_var_pol_tbl(self):
        def build_risk_df(name, title, fn):
            periods = [0, 11, 33]
            labels = [title, "  Over 1 Month", "  Over 1 Year", "  Over 3 Years"]
            data = [[None, None]] + [fn(p) for p in periods]
            df = pd.DataFrame(data, index=labels, columns=["Nominal", "Real"])
            df.columns = pd.MultiIndex.from_product([["Nominal", "Real"], [name]])
            return df

        risk_sections = []
        for ptf in self._portfolios:
            risk = ptf.get_portfolio_var_pol()
            ptf_name = ptf.name

            metrics = [
                ("Value at Risk with 99% Confidence", risk.get_VaR),
                ("Conditional Value at Risk with 99% Confidence", risk.get_CVaR),
                ("Probability of Loss", risk.get_PoL)
            ]

            ptf_df = pd.concat(
                [build_risk_df(ptf_name, title, fn) for title, fn in metrics],
                axis=0
            )

            ptf_df = ptf_df.T.copy()
            ptf_df["Portfolio Risk Premia"] = None
            risk_sections.append(ptf_df.T)

        return pd.concat(risk_sections, axis=1)

    def get_risk_dashboard(self, sort_by_vol=False):

        risk = pd.concat([
            self.get_portfolio_tbl(),
            self.get_crisis_tbl(),
            self.get_var_pol_tbl()
        ])

        risk = risk.drop(index="TOTAL", errors="ignore")  # safer drop

        if sort_by_vol:
            # Ensure 'Volatility' is present before sorting
            if "Volatility" in risk.index:
                return risk.sort_values(by="Volatility", axis=1)
            else:
                raise KeyError("'Volatility' row not found for sorting.")
        else:
            # Build column ordering from portfolio names
            tuples = [(t, n) for n in self.portfolio_names for t in ['Nominal', 'Real']]
            return risk.loc[:, tuples]

    def write_risk_dashboard(self, sort_by_vol=False):

        df = self.get_risk_dashboard(sort_by_vol)
        ws = self.get_worksheet("risk_dashboard")

        # Load formatting from YAML
        with open(yaml_risk, "r") as f:
            fmt = yaml.safe_load(f)

        style_dict = fmt.get("row_styles", {})
        for r_idx, row in enumerate(dataframe_to_rows(df, index=True, header=True), 1):

            if r_idx == 1:
                row = [None] + list(df.columns.get_level_values(0))

            if r_idx == 2:
                row = [None] + list(df.columns.get_level_values(1))

            # get the row formatting
            all_nan = all(
                x is None or (isinstance(x, float) and math.isnan(x))
                for x in row
            )

            all_floats_no_nans = all(
                isinstance(x, float) and not math.isnan(x)
                for x in row[1:]
            )

            all_none_nan_or_str = all(
                x is None or
                (isinstance(x, float) and math.isnan(x)) or
                isinstance(x, str)
                for x in row[1:]
            )

            if r_idx in [1, 2] or row[0] in ["Factor Based Risk Analytics", "Portfolio Risk Premia"]:
                row_style = style_dict.get(1)
            elif isinstance(row[0], str) and row[0].startswith("  "):
                row_style = style_dict.get(4)
            elif all_nan:
                row_style = style_dict.get(5)
            elif all_floats_no_nans:
                row_style = style_dict.get(2)
            elif all_none_nan_or_str:
                row_style = style_dict.get(3)

            for c_idx, value in enumerate(row, 1):
                cell = ws.cell(row=r_idx, column=c_idx, value=value)
                coord = cell.coordinate

                is_first_col = re.fullmatch(r"A\d+", coord)

                if is_first_col:
                    style = row_style.get("A")
                else:
                    style = row_style.get("B")

                # row height
                ws.row_dimensions[int(r_idx)].height = style.get("height", {})

                # Font
                font_cfg = style.get("font", {})
                cell.font = Font(
                    name=font_cfg.get("name"),
                    size=font_cfg.get("size"),
                    bold=font_cfg.get("bold"),
                    italic=font_cfg.get("italic"),
                    color=font_cfg.get("color")
                )

                # Fill
                fill_cfg = style.get("fill", {})
                if fill_cfg.get("type") and fill_cfg.get("fgColor"):
                    cell.fill = PatternFill(
                        fill_type=fill_cfg["type"],
                        fgColor=fill_cfg["fgColor"]
                    )

                # Alignment
                align_cfg = style.get("alignment", {})
                cell.alignment = Alignment(
                    horizontal=align_cfg.get("horizontal"),
                    vertical=align_cfg.get("vertical"),
                    wrap_text=align_cfg.get("wrap_text"),
                    indent=align_cfg.get("indent", 0)
                )

                # Number format
                if style.get("number_format"):
                    cell.number_format = style["number_format"]

                # Borders
                border_cfg = style.get("border", {})
                sides = {}
                for side in ["top", "bottom", "left", "right"]:
                    side_def = border_cfg.get(side, {})
                    sides[side] = Side(
                        style=side_def.get("style"),
                        color=side_def.get("color")
                    )
                cell.border = Border(**sides)

        # Apply column widths
        col_widths = fmt.get("column_widths", {})
        for col in ws.iter_cols(min_row=1, max_row=ws.max_row):
            if any(cell.value is not None for cell in col):
                col_letter = get_column_letter(col[0].column)
                ws.column_dimensions[col_letter].width = col_widths.get(col_letter, col_widths.get("B"))

        # Post Processing of SR
        loc = df.index.get_loc("Sharpe Ratio") + 4
        for col in range(2, ws.max_column + 1):
            cell = ws.cell(row=loc, column=col)
            cell.number_format = "0.00"  # set to 1 decimal place

        light_bottom = Border(top=Side(style="dotted", color="A9A9A9"))
        row_num = next(i for i, x in enumerate(df.index) if "Estimated" in x) + 4
        for col in range(1, ws.max_column + 1):
            ws.cell(row=row_num, column=col).border = light_bottom

        ws.delete_rows(3)
        ws.delete_rows(1)
        ws.row_dimensions[1].height = 35

        def find_row(val):
            for row in ws.iter_rows(min_col=1, max_col=1):  # Only column A
                cell = row[0]
                if cell.value == val:
                    return cell.row

        # Loop through each row from 2 to 12
        for row in range(1, find_row("Factor Based Risk Analytics")):
            col = 2
            while col <= ws.max_column:
                col_letter_1 = get_column_letter(col)
                col_letter_2 = get_column_letter(col + 1)

                # Merge col1 and col2 in this row
                cell_range = f"{col_letter_1}{row}:{col_letter_2}{row}"
                ws.merge_cells(cell_range)

                col += 2  # Move to next pair

        red_font = Font(color="9C0006")  # Dark red
        green_font = Font(color="006100")  # Dark green

        # Apply rules to range B14:I21
        from_row = find_row("Predicted Performance Over Stress Periods") + 1
        to_row = find_row("Value at Risk with 99% Confidence") - 1
        cell_range = "B" + str(from_row) + ":" + get_column_letter(ws.max_column) + str(to_row)

        # Rule for values less than 0 → red font\
        from openpyxl.formatting.rule import CellIsRule

        ws.conditional_formatting.add(
            cell_range,
            CellIsRule(operator='lessThan', formula=['0'], font=red_font)
        )

        # Rule for values greater than 0 → green font
        ws.conditional_formatting.add(
            cell_range,
            CellIsRule(operator='greaterThan', formula=['0'], font=green_font)
        )

        self.save_workbook()

    def write_wealth_simulations(self):

        for p in self._portfolios:

            # Get wealth simulations
            wsim = p.get_portfolio_wealth_projection()

            # Get the metrics from the wsim structs
            metrics = ["nominal", "real", "net_flows", "real_net_flows"]
            df = pd.concat(
                [wsim.get_quantile_dataframe(metric) for metric in metrics],
                axis=1,
                verify_integrity=True  # raise error if duplicate columns
            )

            # Create new sheet for this portfolio
            sheet_name = ExcelUtils.sanitize_sheet_name('ws_' + p.name)
            ws = self.get_worksheet(sheet_name)

            # Write data to worksheet
            for r_idx, row in enumerate(dataframe_to_rows(df, index=True, header=True), 1):

                if r_idx == 1:
                    row = [None] + list(df.columns.get_level_values(0))

                if r_idx == 2:
                    row = [None] + [str(int(x * 100)) + "st %ile" if x == 0.01 else str(int(x * 100)) + "th %ile" for x in df.columns.get_level_values(1)]

                for c_idx, value in enumerate(row, 1):
                    cell = ws.cell(row=r_idx, column=c_idx, value=value)
                    cell.alignment = Alignment(horizontal="center", vertical="center")

            # Remove the empty
            ws.delete_rows(3)

            # build charts across the worksheet
            data_types = np.unique(df.columns.get_level_values(0))
            for id, col_type in enumerate(data_types):

                nominal_cols = []
                for col in range(1, ws.max_column + 1):
                    cell_value = ws.cell(row=1, column=col).value
                    if isinstance(cell_value, str) and col_type == cell_value.lower():
                        nominal_cols.append(col)

                s_col = min(nominal_cols)
                e_col = max(nominal_cols)
                e_row = ws.max_row

                start_col_letter = get_column_letter(s_col+1)
                cell_range = f"{start_col_letter}{e_row + 5}"

                # Create chart
                chart = LineChart()

                data = Reference(ws, min_col=s_col, max_col=e_col, min_row=2, max_row=e_row)
                cats = Reference(ws, min_col=1, min_row=3, max_row=e_row)
                chart.add_data(data, titles_from_data=True)
                chart.set_categories(cats)

                # set the chart height
                chart.height = 3 * 2.65
                chart.width = 3 * 4

                ccy = {
                    "USD": "$",
                    "EUR": "¢",
                    "GBP": "£"
                }

                fmt_title = col_type.replace("_", " ").title()
                currency = " (" + ccy.get(p.schema.currency) + ")"
                chart.y_axis.title = fmt_title + currency if "Flows" in fmt_title else fmt_title + " Portfolio Values" + currency
                chart.x_axis.title = wsim.frequency.name.capitalize().replace("ly", "")
                chart.y_axis.majorTickMark = 'cross'
                chart.title = None

                # Define chart and font
                font_test = ChartingFont(typeface='Aptos')
                cp = CharacterProperties(latin=font_test, sz=900, b=False)
                chart.x_axis.txPr = RichText(p=[Paragraph(pPr=ParagraphProperties(defRPr=cp), endParaRPr=cp)])
                chart.y_axis.txPr = RichText(p=[Paragraph(pPr=ParagraphProperties(defRPr=cp), endParaRPr=cp)])

                # Add this for the axis titles
                chart.x_axis.title.txPr = RichText(p=[Paragraph(pPr=ParagraphProperties(defRPr=cp), endParaRPr=cp)])
                chart.y_axis.title.txPr = RichText(p=[Paragraph(pPr=ParagraphProperties(defRPr=cp), endParaRPr=cp)])
                chart.y_axis.title.tx.rich.p[0].r[0].rPr = cp
                chart.x_axis.title.tx.rich.p[0].r[0].rPr = cp

                chart.graphical_properties = GraphicalProperties()
                chart.graphical_properties.line.noFill = True
                chart.graphical_properties.line.prstDash = None
                chart.y_axis.minorGridlines = None  # Disable minor gridlines
                chart.y_axis.majorGridlines = None  # Disable major gridlines
                chart.x_axis.tickMarkSkip = 5
                chart.x_axis.tickLblSkip = 5
                chart.x_axis.crosses = "autoZero"
                chart.x_axis.tickLblPos = "nextTo"  # ✅ Ensures labels are on the tick marks

                # --- Legend: top, horizontal layout ---
                chart.legend.position = "t"
                chart.legend.layout = Layout(
                    manualLayout=ManualLayout(
                        x=0.25, y=0.0, w=1, h=0.1,
                        xMode="factor", yMode="factor", wMode="factor", hMode="factor"
                    )
                )

                chart.legend.txPr = RichText(p=[
                    Paragraph(pPr=ParagraphProperties(defRPr=cp), endParaRPr=cp)
                ])

                # Define colors similar to screenshot
                colors = ["C00000", "8064A2", "376092", "77933C"]  # red, purple, blue, olive green

                for i, ser in enumerate(chart.series):
                    line = LineProperties()
                    line.solidFill = colors[i % len(colors)]
                    line.width = 12700
                    ser.graphicalProperties.line = line

                    if hasattr(ser, 'dLbls') and ser.dLbls:
                        ser.dLbls.textProperties = CharacterProperties(typeface="Aptos")

                # Add chart to worksheet
                ws.add_chart(chart, cell_range)
                self.save_workbook()


if __name__ == "__main__":
    from epsilonPhi.core.portfolio.SAAPortfolio import SAAPortfolio

    path = '/Users/francisbarker/Repositories/Python/epsilon-phi/epsilon-phi-core/src/python/epsilonPhi/core/reporting/formatted_excel.xlsx'
    #path = '/Users/francisbarker/repo/epsilon-psi/epsilon-phi-core/src/python/epsilonPhi/core/reporting/formatted_excel.xlsx'

    ptfs = SAAPortfolio.get_portfolios_from_template(
        "USD",
        path
    )

    report = Reporting(
        os.getcwd(),
        'risk_report_new_ext'
    )

    res = pd.DataFrame(ptfs[-1].get_assets_risk_premias(), index=ptfs[-1].get_asset_names())
    report.add_portfolios(ptfs)
    report.generate_report(True)
