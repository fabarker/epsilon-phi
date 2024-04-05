import pandas as pd
import numpy as np

_paths = '/Users/francisbarker/Library/Mobile Documents/com~apple~CloudDocs/Data/Bloomberg/FX/Vols/GBPUSD.xlsx'
info = pd.read_excel(_paths, sheet_name=['Asks', 'Bids'], index_col=0, header=[0, 1])

_B = info.get('Bids').dropna()
_A = info.get('Asks').dropna()

_B.index = pd.to_datetime(_B.index)
_A.index = pd.to_datetime(_A.index)

_B = _B.sort_index().droplevel(1, axis=1)
_A = _A.sort_index().droplevel(1, axis=1)

common_index = np.intersect1d(_B.index, _A.index)
common_cols =  np.intersect1d(_B.columns, _A.columns)

S = 0.01 * (_A.loc[common_index].get(common_cols) - _B.loc[common_index].get(common_cols))