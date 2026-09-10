from epsilonPhi.core.dataModel.dataSources.futures.Futures import Futures
import os, glob
import pandas as pd
import numpy as np
from epsilonPhi.core.utils.ExcelUtils import ExcelUtils

# Instantiates futures object
fut = Futures()

# Get list of available futures series
files = glob.glob(os.path.join('futures_info/', "*.pkl"))
codes = [os.path.basename(f).replace('.pkl', '') for f in files]

codes = ['CL', 'ZO', 'ZM', 'ZL', 'LE', 'OJ', 'SB', 'GF', 'HO', 'KC', 'RB', 'GC', 'SI', 'HG', 'HE', 'ZS', 'ZR', 'CC', 'CT', 'ZW', 'NG', 'ZC']

results = {}
for code in codes:
    # Load for instrument
    df = fut.load_futures_prices_from_pickle(code).get(['PS']).reset_index()
    df['week'] = pd.to_datetime(df['date'].values).to_period('W')

    # compute daily returns
    df = pd.concat((
        df,
        df.sort_values(['symbol', 'date']).groupby('symbol')['PS'].pct_change().to_frame('rets')
    ), axis=1)

    # the period ends define the period for signal extraction and forecast horizon
    period_ends = pd.date_range(
        start=df.date.min(),
        end=df.date.max(),
        freq='W-FRI'
    )

    symbol_res = []
    for i in range(len(period_ends)-1):
        print('symbol {}, weeks left: {}'.format(code,len(period_ends)-i))

        # get the end of period dates for current and next periods
        prev_p = df[df.date == period_ends[i]]
        next_p = df[(df.date == period_ends[i + 1]) & (df.days > 30)].sort_values('days')

        # keep only the contracts that have prices in both current and previous periods
        common_symbols = np.intersect1d(
            next_p.symbol,
            prev_p.symbol,
        )

        # select subset of the common symbols
        next_p = next_p[next_p.symbol.isin(common_symbols)]

        # remove duplicate expiring contracts
        candidates = next_p[~next_p['days'].duplicated(keep='first')]

        if candidates.shape[0] > 1:

            # get the near contracts
            near_fut = candidates.iloc[0].symbol
            next_fut = candidates.iloc[1].symbol

            # week_returns
            prev_locs = (df.week == period_ends[i].to_period('W'))
            curr_locs = (df.week == period_ends[i + 1].to_period('W'))

            near_locs = (df.symbol == near_fut)
            next_locs = (df.symbol == next_fut)

            # get the contract returns over the last week
            near_fut_ret_prev = df[prev_locs & near_locs].get('rets').add(1).prod() - 1
            next_fut_ret_prev = df[prev_locs & next_locs].get('rets').add(1).prod() - 1

            # get the contract returns over the later week
            near_fut_ret_sub = df[curr_locs & near_locs].get('rets').add(1).prod() - 1
            next_fut_ret_sub = df[curr_locs & next_locs].get('rets').add(1).prod() - 1

            # compute the spread returns
            sprd_0 = near_fut_ret_prev - next_fut_ret_prev
            sprd_t = near_fut_ret_sub - next_fut_ret_sub

            # collect all the info
            res = [sprd_0,
                   sprd_t,
                   near_fut_ret_prev,
                   next_fut_ret_prev,
                   near_fut_ret_sub,
                   next_fut_ret_sub,
                   near_fut,
                   next_fut,
                   prev_p[prev_p.symbol == near_fut].days.item(),
                   next_p[next_p.symbol == near_fut].days.item(),
                   prev_p[prev_p.symbol == next_fut].days.item(),
                   next_p[next_p.symbol == next_fut].days.item(),
                   period_ends[i],
                   period_ends[i + 1],
                   code
                   ]

        else:

            res = [np.nan,
                   np.nan,
                   np.nan,
                   np.nan,
                   np.nan,
                   np.nan,
                   np.nan,
                   np.nan,
                   np.nan,
                   np.nan,
                   np.nan,
                   np.nan,
                   period_ends[i],
                   period_ends[i + 1],
                   code
                   ]

        symbol_res.extend([res])

    df_res = pd.DataFrame(
              symbol_res,
              columns=['spread_0',
                     'spread_t',
                     'f0_ret_0',
                     'f1_ret_0',
                     'f0_ret_1',
                     'f1_ret_1',
                     'f0',
                     'f1',
                     'f0_days_0',
                     'f0_days_t',
                     'f1_days_0',
                     'f1_days_t',
                     'date_0',
                     'date_t',
                     'code']
    )

    results[code] = df_res.copy()

ExcelUtils.dict_to_excel(
    results,
    os.path.join('futures_info/results_1706.xlsx'),
    include_index=True,
)