import pandas as pd
import numpy as np

res_1706 = pd.read_excel('/Users/francisbarker/Repositories/Python/epsilon-phi/epsilon-phi-core/src/python/epsilonPhi/core/dataModel/dataSources/futures/futures_info/Results/results_1706.xlsx', sheet_name=None, index_col=0)
res_new  = pd.read_excel('/Users/francisbarker/Repositories/Python/epsilon-phi/epsilon-phi-core/src/python/epsilonPhi/core/dataModel/dataSources/futures/futures_info/Results/results_polars.xlsx', sheet_name=None, index_col=0)

unique_symbols = np.intersect1d(
    list(res_1706.keys()),
    list(res_new.keys())
)

for sym in unique_symbols:

    df_1 = res_1706.get(sym).sort_values('date_0').dropna()
    df_2 = res_new.get(sym).sort_values('date_0').dropna()
    idx = (~df_1.eq(df_2)).any(axis=1)
    df_1[idx].compare(df_2[idx])




    df_1.index.names = ['week']
    df_1 = df_1[['spread_0', 'spread_t', 'f0', 'f1', 'f0_days_t', 'f1_days_t', 'date_t', 'f1_ret_1', 'f0_ret_1']]
    df_1[['spread_0', 'spread_t']] = -1 * df_1[['spread_0', 'spread_t']]
    df_1.index = df_1.index + 1

    df_2 = res_new.get(sym).sort_values('date')
    df_2.columns = df_1.columns


    df_1_prime = df_1.loc[df_2.index]

    diffs = df_1_prime[~df_2.eq(df_1_prime).all(axis=1)]