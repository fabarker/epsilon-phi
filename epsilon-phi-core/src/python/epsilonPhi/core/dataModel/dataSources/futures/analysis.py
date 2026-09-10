import pandas as pd
import numpy as np
import statsmodels.api as sm
from epsilonPhi.core.dataModel.dataSources.futures.full_polars_impl import *

instrument_list = ['ZS', 'NG', 'OF', 'RR', 'MS', 'HG', 'CL', 'LG', 'KK', 'SB', 'CC', 'SY',
                    'SL', 'CT', 'KC', 'GC', 'HO', 'JO', 'FD', 'NR', 'CF', 'SN', 'RB', 'LD']

strat = Strategy(instrument_list)
res_1 = strat.get_results()


# Load data
res = pd.read_excel('futures_info/Results/results_1706.xlsx', sheet_name=None, index_col=0)
df = pd.concat(res.values(), axis=0).reset_index()

df['quartile'] = pd.qcut(df['spread_0'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
return_cols = ['spread_t', 'f0_ret_1', 'f1_ret_1']

nw_stats = []
for q in df['quartile'].unique():
    df_q = df[df['quartile'] == q].copy()

    row = {'quartile': q}

    for col in return_cols:
        y = df_q[col].dropna()
        if y.empty:
            row.update({
                f'mean_{col}': None,
                f'se_{col}': None,
                f'ci_lower_{col}': None,
                f'ci_upper_{col}': None
            })
            continue

        X = sm.add_constant(pd.Series(1, index=y.index))  # Intercept-only model

        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})

        mean_annualised = model.params[0] * 52
        se_annualised = model.bse[0] * 52
        ci_lower = mean_annualised - 1.96 * se_annualised
        ci_upper = mean_annualised + 1.96 * se_annualised

        row.update({
            f'mean_{col}': mean_annualised,
            f'se_{col}': se_annualised,
            f'ci_lower_{col}': ci_lower,
            f'ci_upper_{col}': ci_upper
        })

    nw_stats.append(row)

# Final output
nw_df = pd.DataFrame(nw_stats)
nw_df = nw_df.set_index('quartile').loc[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']]

# Optional: include confidence intervals
for col in return_cols:
    nw_df[f'{col}_error_plus'] = nw_df[f'ci_upper_{col}'] - nw_df[f'mean_{col}']
    nw_df[f'{col}_error_minus'] = nw_df[f'mean_{col}'] - nw_df[f'ci_lower_{col}']

keep_cols = ['mean_spread_t', 'mean_f0_ret_1', 'mean_f1_ret_1', 't_error_plus', 't_error_minus', 'back_sub_error_plus', 'back_sub_error_minus', 'front_sub_error_plus', 'front_sub_error_minus']
nw_df[keep_cols].to_clipboard()


# Quick Backtest
df['position'] = -1 * np.sign(df['t0'])
df['strategy'] = df['t'] * df['position']
df['code'] = [ x[:3] for x in df['short_symbol'] ]
df = df.set_index(['code', 'date']).get('strategy').unstack(level=0).sort_index()
