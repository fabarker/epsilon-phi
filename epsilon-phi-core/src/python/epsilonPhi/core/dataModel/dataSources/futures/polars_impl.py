from epsilonPhi.core.dataModel.dataSources.futures.Futures import Futures
import polars as pl
import numpy as np

fut = Futures()

def process_code(code):

    df = fut.load_futures_prices_from_pickle(code).get(['PS']).reset_index()
    df = pl.from_pandas(df)

    # Parse and prep
    df = df.with_columns([
        pl.col('date').cast(pl.Date),
        pl.col('symbol'),
        pl.col('days'),
        pl.col('PS')
    ])

    # Compute daily returns per symbol
    df = df.sort(['symbol', 'date']).with_columns(
        pl.col('PS').pct_change().over('symbol').alias('rets')
    )

    # Compute week identifier and floor to period
    df = df.with_columns(
        (pl.col("date").dt.year() * 100 + pl.col("date").dt.week()).alias("week")
    )

    # Weekly return per (symbol, week)
    weekly = (
        df
        .group_by(['week', 'symbol'])
        .agg([
            ((pl.col('rets') + 1).product() - 1).alias('week'),
            pl.col('days').last().alias('days'),
            pl.col('date').last().alias('date')
        ])
        .sort(['week', 'symbol'])
    )

    # Convert weeks to a list of unique sorted weeks
    weeks = weekly.select('week').unique().sort('week')['week'].to_list()

    results = []
    for i in range(len(weeks) - 1):
        prev_w, next_w = weeks[i], weeks[i+1]

        prev_df = weekly.filter(pl.col('week') == prev_w)
        next_df = weekly.filter(
            (pl.col('week') == next_w) &
            (pl.col('days') > 30)
        ).sort('days')

        # Only keep symbols present in both periods
        common_symbols = set(prev_df['symbol']).intersection(set(next_df['symbol']))
        next_df = next_df.filter(pl.col('symbol').is_in(list(common_symbols)))

        # Drop duplicate days, keep first 2
        candidates = (
            next_df.unique(subset=['days'])
                   .sort('days')
                   .limit(2)
        )

        if candidates.height < 2:
            results.append([np.nan] * 14 + [prev_w, next_w, code])
            continue

        f0, f1 = candidates['symbol'][0], candidates['symbol'][1]

        # Get returns
        def get_ret(w, s):
            return weekly.filter((pl.col('week') == w) & (pl.col('symbol') == s))['weekly_return'][0]

        f0_ret_0 = get_ret(prev_w, f0)
        f1_ret_0 = get_ret(prev_w, f1)
        f0_ret_1 = get_ret(next_w, f0)
        f1_ret_1 = get_ret(next_w, f1)

        spread_0 = f0_ret_0 - f1_ret_0
        spread_t = f0_ret_1 - f1_ret_1

        # Extract days
        def get_days(w, s):
            return weekly.filter((pl.col('week') == w) & (pl.col('symbol') == s))['days'][0]

        res = [
            spread_0,
            spread_t,
            f0_ret_0,
            f1_ret_0,
            f0_ret_1,
            f1_ret_1,
            f0,
            f1,
            get_days(prev_w, f0),
            get_days(next_w, f0),
            get_days(prev_w, f1),
            get_days(next_w, f1),
            prev_w,
            next_w,
            code
        ]
        results.append(res)

    return pl.DataFrame(
        results,
        schema=[
            'spread_0', 'spread_t',
            'f0_ret_0', 'f1_ret_0',
            'f0_ret_1', 'f1_ret_1',
            'f0', 'f1',
            'f0_days_0', 'f0_days_t',
            'f1_days_0', 'f1_days_t',
            'date_0', 'date_t', 'code'
        ]
    )


if __name__ == "__main__":

    codes = ['CL', 'ZO', 'ZM', 'ZL', 'LE', 'OJ', 'SB', 'GF', 'HO', 'KC', 'RB', 'GC', 'SI', 'HG', 'HE', 'ZS', 'ZR', 'CC', 'CT', 'ZW', 'NG', 'ZC']

    for code in codes:
        res = process_code(code)