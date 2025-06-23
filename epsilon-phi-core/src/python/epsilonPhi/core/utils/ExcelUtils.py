import pandas as pd
import numpy as np
import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl import load_workbook
import os, sys

class ExcelUtils(object):
    pass

    @staticmethod
    def extract_formatting(fullfile_path, sheet_name):

        # Load the uploaded workbook
        wb = load_workbook("/mnt/data/Book2.xlsx")
        ws = wb.active

        # We'll extract formatting from a few representative cells in the range B5:H5 as a sample
        sample_range = ws["B5:H5"][0]

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

        # Convert to DataFrames for viewing
        col_df_all = pd.DataFrame(list(column_widths_all.items()), columns=['Column', 'Width'])
        row_df_all = pd.DataFrame(list(row_heights_all.items()), columns=['Row', 'Height'])


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

if __name__ == "__main__":
    fullfile_path = '/Users/francisbarker/Library/Mobile Documents/com~apple~CloudDocs/Data/FX/Linear/Spot Rates.xlsx'
    res = ExcelUtils.xlsread_sheets(fullfile_path)