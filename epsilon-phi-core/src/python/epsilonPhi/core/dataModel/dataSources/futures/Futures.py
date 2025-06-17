import numpy as np
from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.lib.Decorators import SingletonDecorator
from epsilonPhi.core.timeSeries.timeSeriesMain import *
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.ep_strategies.utils.StrategyUtils import StrategyUtils
from epsilonPhi.core.utils.FrameUtils import FrameUtils
from sqlalchemy import desc
import pandas as pd
import warnings
import os
import polars as pl
import glob

warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')

sessionMgr = SessionMgr()
session = sessionMgr.getSessionFactory()

EXCLUSION_LIST = [
    'CFD0106',
    'CFD0106',
    'CFD1006',
    'CHE0215',
    'CW.0709',
    'CC.0710',
    'CCF1210',
    'CCF1213',
    'CHE1011',
    'CZR0197',
    'CZR0397',
    'CZR0597',
    'CZR0797',
    'CZR0997',
    'CRR0109',
    'CRR0709',
    'CZR0313',
    'CZR0306',
    'CZR0106',
    'CZR0903',
    'CZR0103',
    'CZR0902',
    'CZR0502',
    'CZR0702',
    'CZR0302',
    'CZR0102',
    'CZR0700',
    'CZR0500',
    'CZR1199',
    'CZR0999',
    'CZR0799',
    'CZR1197',
    'CZR0198',
    'CZR0398',
    'CZR0998',
    'CZR0798',
    'CZR1198',
    'CZR0399',
    'CZR0599',
    'CZR1100',
    'CZR0900',
    'CZR0101',
    'CZR0501',
    'CZR1110',
    'CRR1110',
    'CZC0912',
    'CLC0216',
    'CLD0615',
    'CLC1000',
    'CLC1001',
]

map = {
    'CL': 'NCL',
    'LE': 'CLD',
    'ZO': 'COF',
    'ZW': 'KKK', # New
    'ZC': 'CCF',
    'NG': 'NNG',
    'CT': 'NCT',
    'CC': 'NCC',
    'ZR': 'CRR', # New
    'ZS': 'CSY', # New
    'HE': 'CLG', # New
    'HG': 'NHG',
    'SI': 'NSL',
    'GC': 'NGC',
    'RB': 'NRB',
    'KC': 'NKC',
    'HO': 'NHO',
    'GF': 'CFD',
    'SB': 'NSB',
    'OJ': 'NJO',
    'ZL': 'CSN', # New
    'ZM': 'CMS', # New
}

@SingletonDecorator
class Futures(object):

    _cache = pd.DataFrame()
    _columns = ([column.name for column in FutureSpec.__table__.columns] +
                [column.name for column in TimeSeriesSpec.__table__.columns])


    def __init__(self):
        pass

    def get_field_value_for_ticker(self, ticker, field_name):

        if field_name not in self._columns:
            raise ValueError("Invalid field name")

            # Use 'getattr' to dynamically get the field based on 'field_name'
        field = getattr(FutureSpec, field_name, None)

        return session.query(field).filter(FutureSpec.ticker == ticker).scalar()

    def get_field_value_for_uid(self, uid, field_name):

        if field_name not in self._columns:
            raise ValueError("Invalid field name")

            # Use 'getattr' to dynamically get the field based on 'field_name'
        field = getattr(FutureSpec, field_name, None)

        return session.query(field).filter(FutureSpec.uid == uid).scalar()

    def get_field_values_for_mnemonic(self, mnemonic, field_name):

        if field_name not in self._columns:
            raise ValueError("Invalid field name")

            # Use 'getattr' to dynamically get the field based on 'field_name'
        field = getattr(FutureSpec, field_name, None)
        res = session.query(field).filter(FutureSpec.future == mnemonic).all()

        if len(res) > 0:
           return [x[0] for x in res]
        else:
           return None

    def get_uids_from_instrument_mnemonic(self, mnemonic):
        return self.get_field_values_for_mnemonic(mnemonic, 'uid')

    def get_series_position_forward_from_uid(self, uid: int):
        return int(self.get_field_value_for_uid(uid, 'position_forward'))

    def get_tick_size_from_uid(self, uid: int):
        return self.get_field_value_for_uid(uid, 'tick_size')

    def get_tick_value_from_uid(self, uid: int):
        return self.get_field_value_for_uid(uid, 'tick_value')

    def get_contract_size_from_uid(self, uid: int):
        return self.get_field_value_for_uid(uid, 'contract_size')

    def get_denominated_currency_from_uid(self, uid: int):
        return self.get_field_value_for_uid(uid, 'denominated_currency')

    def get_security_type_from_uid(self, uid: int):
        return self.get_field_value_for_uid(uid, 'security_type')

    def get_futures_mnemonic_from_uid(self, uid: int):
        return self.get_field_value_for_uid(uid, 'future')

    def get_name_from_uid(self, uid: int):
        return self.get_field_value_for_uid(uid, 'name')

    def get_security_from_uid(self, uid: int):
        return self.get_field_value_for_uid(uid, 'security')

    def get_ticker_from_uid(self, uid: int):
        return self.get_field_value_for_uid(uid, 'ticker')

    def get_available_forward_positions_from_instrument_menomic(self, mnemonic):
        return np.unique(self.get_field_values_for_mnemonic(mnemonic, 'position_forward')).astype(int)

    def get_continuous_series_single_ticker(self, ticker):
        uid = self.get_field_value_for_ticker(ticker, 'uid')
        return self.get_continuous_series_single_uid(uid)

    def get_continuous_series_single_uid(self, uid):
        from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource as gds

        res = gds().get_time_series_data_from_uid(uid, ts_type=TimeSeriesType.LEVELS, returnsType=ReturnsType.DIFFERENCE)
        res.set_attribute_single('forward', self.get_series_position_forward_from_uid(uid))
        res.set_attribute_single('mnemonic', self.get_futures_mnemonic_from_uid(uid))
        res.set_attribute_single('ticker', self.get_ticker_from_uid(uid))
        res.set_attribute_single('name', self.get_name_from_uid(uid))
        res.set_attribute_single('security', self.get_security_from_uid(uid))
        res.columns = res.columns.swaplevel('mnemonic', res.columns.get_level_values(0).name)
        return res.deepcopy()

    def _resolve_ticker_list(self, list):
        return sorted(list, key=lambda x: x.replace('.', '~'))

    def get_futures_continuous_series_forward(self, mnemonic, forward):

        # query the database for the information we need
        res = session.query(FutureSpec.ticker)\
                       .filter(FutureSpec.future == mnemonic,
                               FutureSpec.position_forward == int(forward))\
                       .order_by(desc(FutureSpec.ticker))\
                       .all()

        ts = CTimeSeries(ts_type=TimeSeriesType.LEVELS, returns_type=ReturnsType.DIFFERENCE)
        for ticker in res:
            tmp = self.get_continuous_series_single_ticker(ticker[0])
            tmp.drop_attributes(['ticker', 'uid', 'name'])
            ts = ts.backfill_levels(tmp)
        return ts.deepcopy()


    def get_futures_continuous_series(self, mnemonic):

        # Get the position forward for the mnemonic
        fwds = self.get_available_forward_positions_from_instrument_menomic(mnemonic)

        ts = CTimeSeries(ts_type=TimeSeriesType.LEVELS)
        for fwd in fwds:
            ts = ts.concat(self.get_futures_continuous_series_forward(mnemonic, fwd))
        return ts.deepcopy()

    def get_all_continuous_series(self, mnemonic):

        res = session.query(FutureSpec.ticker) \
            .filter(FutureSpec.future == mnemonic) \
            .order_by(desc(FutureSpec.start_date)) \
            .all()

        ts = CTimeSeries(ts_type=TimeSeriesType.LEVELS, returns_type=ReturnsType.DIFFERENCE)
        for ticker in res:
            tmp = self.get_continuous_series_single_ticker(ticker[0])
            ts = ts.concat(tmp)
        return ts.deepcopy()


    def get_front_futures_continuous_series_settlement_price(self, mnemonic):
        return (self.get_futures_continuous_series_settlement_price(mnemonic).
                select_subset_attribute('forward', 0))

    def get_back_futures_continuous_series_settlement_price(self, mnemonic):
        return (self.get_futures_continuous_series_settlement_price(mnemonic).
                select_subset_attribute('forward', 1))

    def get_futures_continuous_series_settlement_price(self, mnemonic):
        return self.get_futures_continuous_series(mnemonic).select_subset_attribute('field', 'PS')

    def get_futures_continuous_series_open_interest(self, mnemonic):
        return self.get_futures_continuous_series(mnemonic).select_subset_attribute('field', 'OI')

    def get_futures_continuous_volume(self, mnemonic):
        return self.get_futures_continuous_series(mnemonic).select_subset_attribute('field', 'VM')

    def get_OHLC_for_continuous_future(self, mnemonic):
        return self.get_futures_continuous_series(mnemonic).select_subset_attribute('field',
                                                                        ['PO','PH','PL','PS'])

    def get_futures_carry_continuous(self, mnemonic):
        front = self.get_front_futures_continuous_series_settlement_price(mnemonic)
        back = self.get_back_futures_continuous_series_settlement_price(mnemonic)
        return np.log(back.division_over_common_dates(front))

    def estimate_bid_ask_prices_for_futures_continuous(self, mnemonic):
        ohlc = self.get_OHLC_for_continuous_future(mnemonic).dropna(how='any', axis=0)
        bid, ask = StrategyUtils.estimate_bid_ask_prices(
            open=ohlc.select_subset_attribute('field', 'PO'),
            high=ohlc.select_subset_attribute('field', 'PH'),
            low=ohlc.select_subset_attribute('field', 'PL'),
            close=ohlc.select_subset_attribute('field', 'PS')
        )

        prices = bid.concat(ask)
        prices.set_attribute_single('price_quote_type', ['bid', 'ask'])
        return prices.copy()

    def display_futures_info(self):
        sessionMgr.show_table(FutureSpec)

    @staticmethod
    def select_symbols_polars_advanced(df_pandas):
        """
        Advanced Polars implementation with all operations vectorized
        """
        # Convert to Polars
        if isinstance(df_pandas.index, pd.MultiIndex):
            df_work = df_pandas.reset_index()
        else:
            df_work = df_pandas.copy()

        df = pl.from_pandas(df_work)

        # Add maturity column
        df = df.with_columns([
            pl.col('symbol').str.slice(-4).alias('mat')
        ])

        # All operations in single chain for maximum efficiency

        # For each mat and symbol, get

        # 1. the ps_count (number of prices, no nans)
        # 2. distance between the max pricing date and the settlement date

        selected_symbols = (
            df
            .group_by(['mat', 'symbol'])
            .agg([
                pl.col('PS').is_not_null().sum().alias('ps_count'),
                pl.col('settlement').max().alias('max_settlement'),
                pl.col('date').max().alias('max_date'),
                pl.col('symbol').first().alias('symbol_name')
            ])
            .with_columns([
                (pl.col('max_date') - pl.col('max_settlement')).dt.total_seconds().alias('distance'),
                (~pl.col('symbol_name').str.contains('.')).alias('no_dot')
            ])
            .with_columns([
                pl.col('ps_count').max().over('mat').alias('max_count_per_mat'),
                pl.col('distance').min().over('mat').alias('min_distance_per_mat')
            ])
            .filter(pl.col('ps_count') == pl.col('max_count_per_mat'))  # Keep max count symbols
            .filter(pl.col('distance') == pl.col('min_distance_per_mat'))  # Keep min distance symbols
            .with_columns([
                pl.col('no_dot').max().over('mat').alias('has_no_dot_candidate')
            ])
            .filter(
                pl.when(pl.col('has_no_dot_candidate'))
                .then(pl.col('no_dot'))
                .otherwise(pl.lit(True))
            )
            .group_by('mat')
            .agg([
                pl.col('symbol_name').first().alias('selected_symbol')
            ])
            .select('selected_symbol')
            .to_series()
            .to_list()
        )

        return df_pandas.loc[selected_symbols].copy()

    @staticmethod
    def process_symbols(df, short_code):

        keep_symbols = []
        df['mat'] = [ x[-4:] for x in df.index.get_level_values(0) ]
        for mat in df['mat'].unique():
            tmp = df[df['mat'] == mat].get('PS')

            unique_symbols = list(
                tmp.index.get_level_values(0).unique()
            )

            # if the number of signals, is greater than one, process
            if len(unique_symbols) == 1:
                keep_symbols.append(unique_symbols[0])
                continue

            # Step 1: Pick the series that has the longest/highest number of price observations
            symbol_counts = {symbol: tmp.loc[symbol].notna().sum().item() for symbol in unique_symbols}
            max_count = max(symbol_counts.values())
            candidates = np.array([symbol for symbol, count in symbol_counts.items() if count == max_count])

            if len(candidates) == 1:
                keep_symbols.append(candidates[0])
                continue

            mat_date = max(tmp.index.get_level_values('settlement'))
            distance = np.array([(tmp.loc[x].index.get_level_values('date').max() - mat_date).total_seconds() for x in candidates])
            candidates = candidates[distance == min(distance)]

            if len(candidates) == 1:
                keep_symbols.append(candidates[0])
                continue

            #assert np.all(tmp.loc[candidates].unstack(level=0).diff(axis=1).dropna(axis=1) == 0), 'Error in prices for {}'.format(candidates)
            price_diff = tmp.loc[candidates].unstack(level=0).diff(axis=1).dropna(axis=1)
            if not np.all(price_diff == 0):
                print(f'Price differences found for candidates: {candidates}')

                # Show the actual differences
                non_zero_diffs = price_diff[np.any(price_diff != 0, axis=1)].dropna()
                print(f'Dates and prices with differences:')
                for date_idx, row in non_zero_diffs.iterrows():
                    prices_at_date = tmp.loc[candidates].xs(date_idx, level='date')
                    prices_at_date = prices_at_date.get('PS').to_dict()
                    print(f'  Date: {date_idx}, Code: {short_code}, Prices: {prices_at_date}')

            final_candidate = [ x for x in candidates if '.' not in x ]
            if len(final_candidate) == 0:
                keep_symbols.append(candidates[0])
                continue

            keep_symbols.append(final_candidate[0])
        return df.loc[keep_symbols].copy()

    @staticmethod
    def load_futures_prices_raw(short_code):

        df = pd.read_pickle('futures_info/' + short_code + '.pkl')
        df = df[~df.index.duplicated(keep='first')]
        # keep only the instruments in the info
        df.index.names = ['symbol', 'date']
        df = df.reset_index().set_index('date')

        # Resample cols
        cols = np.setdiff1d(df.columns, 'VM')
        resampled = df.groupby('symbol', group_keys=False)[cols].resample('1B').ffill()
        resampled = resampled.reset_index().set_index(['symbol', 'date'])

        return pd.concat(
            (
                resampled,
                df.reset_index().set_index(['symbol', 'date']).get(np.setdiff1d(df.columns, cols))
             ), axis=1
        )
        #pl_df = pl.from_pandas(df.reset_index())
        #pl_df = pl_df.with_columns(pl.col('date').cast(pl.Date))

        #symbol_ranges = (
        #    pl_df
        #    .group_by('symbol')
        #    .agg([
        #        pl.col('date').min().alias('start'),
        #        pl.col('date').max().alias('end')
        #    ])
        #)

        #calendar = symbol_ranges.select([
        #    pl.col('symbol'),
        #    pl.date_ranges(
        #        pl.col('start'),
        #        pl.col('end'),
        #        "1d",  # business days
        #        closed='both'
        #    ).alias("date")
        #]).explode("date")

        #joined = (
        #    calendar
        #    .join(pl_df, on=["symbol", "date"], how="left")
        #    .sort(["symbol", "date"])
        #)

        # 4. Forward fill within each group
        # Fill all nulls in a group with `.with_columns(...).over('symbol')`
        #cols_to_fill = [c for c in joined.columns if c not in ['symbol', 'date', 'VM']]
        #filled = joined.with_columns([
        #    pl.col(c).fill_null(strategy="forward").over("symbol") for c in cols_to_fill
        #])

    @staticmethod
    def load_futures_prices_from_pickle(short_code):


        path = 'futures_info/' + short_code + '.pkl'
        if os.path.exists(path):
            info = pd.read_excel('futures_info/' + short_code + '_.xlsx', index_col=0, sheet_name='Sheet1')
            df = Futures().load_futures_prices_raw(short_code)
            df.index.names = ['symbol', 'date']
            df['settlement'] = info.reindex(df.index.get_level_values(0)).LTDT.values
            df['days'] = (df['settlement'] - df.index.get_level_values('date')).dt.days
            return df.reset_index().set_index(['symbol', 'date', 'settlement', 'days']).sort_index(level=[1, 2])
        else:
            return None

    @staticmethod
    def load_all_futures_from_directory(directory='futures_info/', pattern="*.pkl", columns=['PS']):

        """Load all pickle files matching pattern from directory."""

        files = glob.glob(os.path.join(directory, pattern))
        codes = [os.path.basename(f).replace('.pkl', '') for f in files]
        # Load and combine all DataFrames
        dataframes = []
        for code in codes:
            pd_df = Futures().load_futures_prices_from_pickle(code).get(columns).reset_index()
            pd_df['code'] = code
            dataframes.append(pl.from_pandas(pd_df))

        # Concatenate all at once
        return pl.concat(dataframes, how="vertical_relaxed", rechunk=True)




if __name__ == "__main__":

    files = glob.glob(os.path.join('futures_info/', "*.pkl"))
    codes = [os.path.basename(f).replace('.pkl', '') for f in files]

    results = {}
    for code in codes:

        # Load for instrument
        df = Futures().load_futures_prices_from_pickle(code).get(['PS']).reset_index()

        df['rets'] = df.sort_values(['symbol', 'date']).groupby('symbol')['PS'].pct_change()
        df['week'] = pd.Categorical(pd.to_datetime(df.get('date').values).to_period('W')).codes

        weekly_returns = (df.groupby(['week', 'symbol'], group_keys=False)
                       .agg({
                           'rets': lambda x: -1 + (x + 1).prod(),  # Weekly return
                           'date': 'last',        # Preserve max date
                           'days': 'last'
                       })
                       .rename(columns={'rets': 'weekly_return'})
                       ).reset_index().set_index('week')

        output = pd.DataFrame()
        for i in range(1, max(weekly_returns.index.get_level_values(0))):

            # Find the contracts we want
            candidates = weekly_returns.query(
                f'days > {30} and week == {i} and symbol in '
                f'{list(weekly_returns.query(f"week == {i - 1}")["symbol"].unique())}'
            ).drop_duplicates('days').nsmallest(2, 'days')

            if candidates.shape[0] > 1:

                tmp = weekly_returns.query(
                        f'symbol in {list(candidates.symbol)} and week in {[i, i-1]}'
                    ).sort_index()

                spd = pd.concat(
                    [
                        tmp.loc[x].sort_values('date').get('weekly_return').diff().dropna().to_frame()
                        for x in np.unique(tmp.index)
                    ]
                )

                spd.index = ['t0', 't']
                spd.columns = [i]
                spd = spd.T
                spd['short_symbol'] = candidates.sort_values('days').symbol.iloc[0]
                spd['long_symbol'] =  candidates.sort_values('days').symbol.iloc[1]
                spd['short_days'] = candidates.sort_values('days').days.iloc[0]
                spd['long_days'] = candidates.sort_values('days').days.iloc[1]
                spd['date'] = candidates.sort_values('days').date.iloc[0]
                output = pd.concat((output, spd), axis=0)
        results[code] = output.copy()

    #from epsilonPhi.core.utils.ExcelUtils import ExcelUtils
    #ExcelUtils.dict_to_excel(
    #    results,
    #    os.path.join('futures_info/results_old.xlsx'),
    #    include_index=True,
    #)

    res = pd.concat(results.values(), axis=0)

    res['quartile'] = pd.qcut(res['t0'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    # Compute means and standard errors per bucket
    grouped = res.groupby('quartile')[['t0', 't']]

    # Mean * 52 as in your original
    mean_annualised = grouped.mean() * 52

    # Standard error = std / sqrt(n) → also annualised
    standard_errors = grouped.std() / np.sqrt(grouped.count()) * 52

    # Combine results
    quartile_stats = mean_annualised.rename(columns=lambda c: f'{c}_mean')
    quartile_stats[[f'{col}_se' for col in standard_errors.columns]] = standard_errors.values

    # Optional: include confidence intervals
    for col in ['t0', 't']:
        quartile_stats[f'{col}_ci_lower'] = quartile_stats[f'{col}_mean'] - 1.96 * quartile_stats[f'{col}_se']
        quartile_stats[f'{col}_ci_upper'] = quartile_stats[f'{col}_mean'] + 1.96 * quartile_stats[f'{col}_se']
