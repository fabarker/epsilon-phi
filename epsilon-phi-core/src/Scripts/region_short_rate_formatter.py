import os, sys
import pandas as pd

path = r'C:\Users\fabar\Documents\Data\rates\short rates\Regional'
workbook = 'Interest Rates Regions.xlsx'

df = pd.read_excel(os.path.join(path, workbook), sheet_name=None)

#%%

time_series = pd.DataFrame()
for key in df.keys():
    _tmp = df.get(key)
    time_series = pd.concat((time_series, _tmp), axis=0)