import pandas as pd
import numpy as np
import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl import load_workbook
import os, sys
import math


class ExcelUtils(object):
    pass

    @staticmethod
    def extract_formatting(fullfile_path):

        # Load the uploaded workbook
        wb = load_workbook(fullfile_path)
        ws = wb.active

        def extract_side_info(side):
            return {
                "style": side.style,
                "color": side.color.rgb if side.color and side.color.type == "rgb" else None
            }

        # Get the range of active cells
        min_row = ws.min_row
        max_row = ws.max_row
        min_col = ws.min_column
        max_col = ws.max_column

        # Convert column numbers to letters
        start_col_letter = get_column_letter(min_col)
        end_col_letter = get_column_letter(max_col)

        # Build the A1-style range string
        cell_range = f"{start_col_letter}{min_row}:{end_col_letter}{max_row}"
        sample_range = ws[cell_range]

        style_summary = {}
        for row in sample_range:
            for cell in row:
                cell_fmt = {
                    "cell": cell.coordinate,
                    "font": {
                        "name": cell.font.name,
                        "size": cell.font.size,
                        "bold": cell.font.bold,
                        "italic": cell.font.italic,
                        "color": cell.font.color.rgb if cell.font.color and cell.font.color.type == "rgb" else None
                    },
                    "fill": {
                        "type": cell.fill.fill_type,
                        "fgColor": cell.fill.fgColor.rgb if cell.fill.fgColor and cell.fill.fgColor.type == "rgb" else None
                    },
                    "alignment": {
                        "horizontal": cell.alignment.horizontal,
                        "vertical": cell.alignment.vertical,
                        "wrap_text": cell.alignment.wrap_text
                    },
                    "number_format": cell.number_format,
                    "border": {
                        "top": extract_side_info(cell.border.top),
                        "bottom": extract_side_info(cell.border.bottom),
                        "left": extract_side_info(cell.border.left),
                        "right": extract_side_info(cell.border.right)
                    }
                }
                style_summary[cell.coordinate] = cell_fmt

        import yaml

        # Extract column widths
        column_widths_all = {}
        for col in range(min_col, max_col + 1):
            col_letter = get_column_letter(col)
            width = ws.column_dimensions[col_letter].width
            column_widths_all[col_letter] = width

        # Extract row heights
        row_heights_all = {}
        for row in range(min_row, max_row + 1):
            height = ws.row_dimensions[row].height
            row_heights_all[row] = height

        res = {
            "styles": style_summary,
            "column_widths": column_widths_all,
            "row_heights": row_heights_all
        }



    # Method to load all sheets in a workbook
    @staticmethod
    def xlsread_sheets(fullfile_path, sheet_names=None, dtype=None):

        # load the Excel workbook
        if os.path.isfile(fullfile_path) is False:
            return None

        excel_data = pd.read_excel(fullfile_path,
                                   sheet_name=None,
                                   engine='openpyxl',
                                   header=None,
                                   dtype=dtype)

        if sheet_names is None:
            return excel_data

        # loop through all spreadsheets in the workbook
        res = dict()
        for sheet_name in sheet_names:
            res[sheet_name] = excel_data.get(sheet_name, pd.DataFrame())
        return res

    @staticmethod
    def dict_to_excel(dataframes_dict, filename, include_index=False):

        """
        Write a dictionary of dataframes to different sheets in an Excel workbook.

        Parameters:
        dataframes_dict (dict): Dictionary where keys are sheet names and values are DataFrames
        filename (str): Output Excel filename
        include_index (bool): Whether to include the DataFrame index in the output
        """

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            for sheet_name, df in dataframes_dict.items():
                # Clean sheet name for Excel compatibility (max 31 chars, no special chars)
                clean_sheet_name = str(sheet_name)[:31]
                clean_sheet_name = clean_sheet_name.replace('/', '_').replace('\\', '_').replace('?', '_').replace('*',
                                                                                                                   '_').replace(
                    '[', '_').replace(']', '_').replace(':', '_')

                df.to_excel(writer, sheet_name=clean_sheet_name, index=include_index)

    @staticmethod
    def extract_formatting_grouped_by_row(path, sheetname=None):
        wb = load_workbook(path)
        ws = wb[sheetname] if sheetname else wb.active

        row_formatting = {}

        for row in ws.iter_rows():
            row_number = row[0].row
            row_formatting[row_number] = {}

            for cell in row:
                col_letter = get_column_letter(cell.column)

                row_formatting[row_number][col_letter] = {
                    "font": {
                        "name": cell.font.name,
                        "size": cell.font.size,
                        "bold": cell.font.bold,
                        "italic": cell.font.italic,
                        "color": cell.font.color.rgb if cell.font.color and cell.font.color.type == "rgb" else None,
                    },
                    "fill": {
                        "type": cell.fill.fill_type,
                        "fgColor": cell.fill.fgColor.rgb if cell.fill.fgColor and cell.fill.fgColor.type == "rgb" else None,
                    },
                    "alignment": {
                        "horizontal": cell.alignment.horizontal,
                        "vertical": cell.alignment.vertical,
                        "wrap_text": cell.alignment.wrap_text,
                        "indent": cell.alignment.indent,
                    },
                    "number_format": cell.number_format,
                    "height": ws.row_dimensions[cell.row].height,
                    "border": {
                        side: {
                            "style": getattr(cell.border, side).style,
                            "color": getattr(cell.border, side).color.rgb
                            if getattr(cell.border, side).color and getattr(cell.border, side).color.type == "rgb"
                            else None
                        }
                        for side in ["top", "bottom", "left", "right"]
                    }
                }

        # Column widths and row heights
        column_widths = {
            col: ws.column_dimensions[col].width
            for col in ws.column_dimensions
        }

        return {
            "row_styles": row_formatting,
            "column_widths": column_widths,
        }


if __name__ == "__main__":

    import yaml
    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    import os
    import re

    path = '/Users/francisbarker/repo/epsilon-psi/epsilon-phi-core/src/python/epsilonPhi/core/reporting/formatted_excel.xlsx'

    #res = ExcelUtils.extract_formatting_grouped_by_row(path, "Sheet4 (2)")

    #with open('formatting_assumptions.yaml', 'w') as f:
    #    yaml.dump(res, f)


    fullfile_path = '/Users/francisbarker/repo/epsilon-psi/epsilon-phi-core/src/python/epsilonPhi/core/reporting/formatted_excel.xlsx'
    yaml_assumptions = '/Users/francisbarker/repo/epsilon-psi/epsilon-phi-core/src/python/epsilonPhi/core/reporting/formatting_assumptions.yaml'
    df = pd.read_excel(fullfile_path, "Sheet4", index_col=0, header=[0, 1])

    # get the workbook
    wb = Workbook()
    default_sheet = wb.active
    wb.remove(default_sheet)
    ws = wb.create_sheet("assumptions")

    # Load formatting from YAML
    with open(yaml_assumptions, "r") as f:
        fmt = yaml.safe_load(f)

    style_dict = fmt.get("row_styles", {})

    for r_idx, row in enumerate(dataframe_to_rows(df, index=True, header=True), 1):

        if r_idx in [1, 2, 3, 4]:
            row_style = style_dict.get(r_idx)
        elif isinstance(row[0], str) and row[0].startswith("    "):
            row_style = style_dict.get(4)
        else:
            row_style = style_dict.get(5)


        for c_idx, value in enumerate(row, 1):
            cell = ws.cell(row=r_idx, column=c_idx, value=value)
            coord = cell.coordinate

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

    # Define a bottom border style
    bottom_border = Border(bottom=Side(style='thin'))

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
    wb.save("assumps.xlsx")
    wb.close()