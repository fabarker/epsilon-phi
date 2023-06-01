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

if __name__ == "__main__":
    fullfile_path = '/Users/francisbarker/Library/Mobile Documents/com~apple~CloudDocs/Data/FX/Linear/Spot Rates.xlsx'
    res = ExcelUtils.xlsread_sheets(fullfile_path)