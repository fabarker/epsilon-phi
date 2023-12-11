import pandas as pd
import numpy as np
import os

folder_path = r'C:\Users\fabar\OneDrive\Desktop\data\factors'
filename = 'AQR and FF Factors.xlsx'
file_path = os.path.join(folder_path, filename)

df_ = pd.read_excel(file_path, sheet_name=None)

exclusion = ['Data Sources','Definition','Sources and Definitions']

factor_data = pd.DataFrame()
for sheetname in df_.keys():
    if sheetname not in exclusion:
        sheet = pd.read_excel(file_path, sheet_name=sheetname, header=[0,1], index_col=0)
        sheet.index = pd.to_datetime(sheet.index)
        factor_data = pd.concat((factor_data, sheet), axis=1)

factor_data.to_clipboard()

