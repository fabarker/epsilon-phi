import numpy as np

from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.dataModel.dataSources.vendor.Bloomberg import Bloomberg
from epsilonPhi.core.dataModel.dataSources.vendor.Datastream import pyDatastream
from epsilonPhi.core.utils.FrameUtils import FrameUtils
import pandas as pd
import datetime as dt

path = '/Users/francisbarker/Desktop/GBP Rate Info.xlsx'

fx_info = pd.read_excel(path, sheet_name='Spot', index_col=0)

spts = pd.read_csv('GBP Spot Rates.csv', index_col=[0,1])
fwd_1 = pd.read_csv('GBP 1m fwds.csv', index_col=[0,1])
fwd_3 = pd.read_csv('GBP 3m fwds.csv', index_col=[0,1])

rates = pd.concat((pd.concat((spts, fwd_1), axis=0), fwd_3), axis=0)
rates = rates[['EB','ER','EO','X']].dropna(how='all')

unique_labels = np.intersect1d(fx_info.index.unique(), rates.index.get_level_values(0).unique())

df = pd.DataFrame()
for label in unique_labels:
    label_data = rates.loc[label]
    label_info = fx_info.loc[label]

    idx = pd.concat([label_info] * 4, axis=1)
    dtype = pd.DataFrame(label_data.columns, columns=['dataType']).T
    idx.columns = dtype.columns

    label_data.columns = pd.MultiIndex.from_frame(pd.concat((idx, dtype), axis=0).T)
    label_data.columns = FrameUtils.add_index_to_multi_index(label_data.columns, label, 'ticker')
    label_data.index = pd.to_datetime(label_data.index)

    df = pd.concat((df, label_data.sort_index()), axis=1)

df.index = pd.to_datetime(df.index)
df = df.sort_index()

cols = df.columns
GBPUSD_spt_idx = np.logical_and(cols.get_level_values('domestic_currency') == 'USD', cols.get_level_values('maturity') == '0m')
GBPUSD_1m_idx  = np.logical_and(cols.get_level_values('domestic_currency') == 'USD', cols.get_level_values('maturity') == '1m')
GBPUSD_3m_idx  = np.logical_and(cols.get_level_values('domestic_currency') == 'USD', cols.get_level_values('maturity') == '3m')


GBPUSD_spt = df.iloc[:,GBPUSD_spt_idx].copy()
GBPUSD_1m  = df.iloc[:,GBPUSD_1m_idx].copy()
GBPUSD_3m  = df.iloc[:,GBPUSD_3m_idx].copy()

rate_df = {}
rate_df['0m'] = GBPUSD_spt
rate_df['1m'] = GBPUSD_1m
rate_df['3m'] = GBPUSD_3m

inv_type = {}
inv_type['ER'] = 'ER'
inv_type['EB'] = 'ER'
inv_type['EO'] = 'ER'
inv_type['X'] = 'X'

mat_ln = {}
mat_ln['0m'] = 'Spot'
mat_ln['1m'] = '1 Month Forward'
mat_ln['3m'] = '3 Month Forward'



USD_rates = pd.DataFrame()
for col in df.columns:

    print(col)
    subset = df.get([col]).dropna()
    mat = subset.columns.get_level_values('maturity')[0]
    type = subset.columns.get_level_values('dataType')[0]

    xrate = rate_df.get(mat)
    xrate = xrate.iloc[:, xrate.columns.get_level_values('dataType') == inv_type.get(type)].dropna()

    if subset.size > 0:

        common_dates = np.intersect1d(subset.index, xrate.index)
        mult = subset.loc[common_dates] / xrate.loc[common_dates].values

        col_idx = pd.DataFrame(mult.columns.to_frame().T.values.flatten(),
                               index=mult.columns.to_frame().T.index)

        # Udpdate values in the old columns to the new columns
        col_idx.loc['To Currency'] = 'United States Dollar'
        col_idx.loc['foreign_currency'] = 'USD'
        col_idx.loc['Name'] = col_idx.loc['From Currency'].values[0] + ' to ' + col_idx.loc['To Currency'].values[0] + ' ' + mat_ln.get(mat)
        col_idx.loc['long_name'] = col_idx.loc['Name'] + ' (' + col[-1] + '/USDOLLR)'
        col_idx.loc['Name'] = col_idx.loc['Name'][0].upper()

        mult.columns = pd.MultiIndex.from_frame(col_idx.T)
        USD_rates = pd.concat((USD_rates, mult), axis=1)

USD_rates.to_csv('USD Rates.csv')





