import pandas as pd
import numpy as np
import openpyxl
import os, sys

class ExcelUtils(object):
    pass


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