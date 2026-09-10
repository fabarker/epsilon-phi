import glob
import os

import time
import timeit
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from typing import List, Optional
from dateutil import parser
from datetime import datetime
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm

cols = ['spread_0', 'spread_t', 'f0_ret_0', 'f1_ret_0', 'f0_ret_t', 'f1_ret_t',
        'f0', 'f1', 'f0_days_0', 'f0_days_t', 'f1_days_0', 'f1_days_t', 'date_0', 'date_t', 'symbol', 'period']


class SpreadStrategy(object):
    WEEKS_IN_YEAR = 52
    CI_Z_SCORE = 1.96

    def __init__(
            self,
            instruments: List[str],
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            freq_string: str = "W-SUN",
            maturity_cutoff: int = 30,
    ):

        if not instruments:
            raise ValueError("Instruments list cannot be empty.")

        # Store instrument list (deduplicated)
        self._instruments = np.unique(instruments)
        self._maturity_cutoff = maturity_cutoff

        # Validate and store frequency string
        self.validate_freq_string(freq_string)
        self._freq_string = freq_string

        # Parse start and end dates
        self._start_date = parser.parse(start_date) if isinstance(start_date, str) else datetime.min
        self._end_date = parser.parse(end_date) if isinstance(end_date, str) else datetime.max

        # Placeholder for computed results
        self.__results = None

        # Load and preprocess the data immediately
        self.__load_data()

    def __repr__(self):
        return f"<SpreadStrategy | {len(self._instruments)} instruments | {self._start_date.date()} to {self._end_date.date()}>"

    @property
    def df(self):
        return self.__results.copy()

    @staticmethod
    def validate_freq_string(freq_str: str) -> None:
        try:
            pd.tseries.frequencies.to_offset(freq_str)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid freq_string string: '{freq_str}' is not a valid pandas frequency.")

    def __load_data(self):

        """
            Load and preprocess futures data from a Parquet file.

            This method:
            - Reads raw futures data from disk
            - Filters it by date range and selected instruments
            - Computes daily returns per contract ('rets')
            - Assigns weekly period labels based on the selected frequency
            - Aggregates returns and metadata at the (period, ticker) level
            - Stores the processed dataset as internal attributes for further analysis
            - Automatically triggers the main computation
        """

        # Step 1: Load Parquet and filter by date range and instrument list
        self._df = pl.read_parquet(
            'futures_info/Parquet/futures_data.parquet'
        ).filter(
            (pl.col("date") >= self._start_date) &
            (pl.col("date") <= self._end_date) &
            (pl.col("symbol").is_in(self._instruments))
        )

        # Step 2: Sort data by ticker and date, and compute daily returns (percentage change in PS)
        self._df = self._df.sort(['ticker', 'date']).with_columns(
            pl.col('PS').pct_change().over('ticker').alias('rets')
        )

        # Step 3: Assign period codes based on the specified frequency string (e.g., "W-SUN")
        self.period_codes = pd.Categorical(
            pd.to_datetime(
                self._df.select('date').to_pandas().values.flatten()
            ).to_period(self._freq_string)
        ).codes

        # Step 4: Add the period codes back into the Polars DataFrame
        self._df = self._df.with_columns(
            pl.Series('period', self.period_codes)
        )

        # Step 5: Aggregate data at (period, ticker) level
        # Compute cumulative periodic return and retain the last values for 'days', 'date', and 'symbol'
        self._period_df = (
            self._df
            .group_by(['period', 'ticker'])
            .agg([
                ((pl.col('rets') + 1).product() - 1).alias('rets'),
                pl.col('days').last().alias('days'),
                pl.col('date').last().alias('date'),
                pl.col('symbol').last().alias('symbol')
            ])
            .sort(['period', 'ticker'])
        )

        # Step 6: Run the main computation on this processed data
        self.compute()

    def get_instruments_time_series_characteristics(self) -> pl.DataFrame:

        """
        Compute annualised return and volatility statistics per instrument.

        For each instrument (symbol), this method:
        - Computes the average return of the front and second-nearest contracts (`f0_ret_0`, `f1_ret_0`)
        - Computes the average spread between contracts (`spread_0`)
        - Computes the standard deviation (volatility) of the above metrics
        - Annualises returns by multiplying by the number of periods per year
        - Annualises volatility by scaling with the square root of the number of periods per year

        Returns:
            pl.DataFrame: A summary table with one row per instrument, containing:
                - mean_f0_ret_0
                - mean_f1_ret_0
                - mean_spread_0
                - vol_f0_ret_0
                - vol_f1_ret_0
                - vol_spread_0
        """
        return self.df.group_by("symbol").agg([
            # Annualised mean returns
            (pl.col("f0_ret_0").mean() * SpreadStrategy.WEEKS_IN_YEAR).alias("mean_f0_ret_0"),
            (pl.col("f1_ret_0").mean() * SpreadStrategy.WEEKS_IN_YEAR).alias("mean_f1_ret_0"),
            (pl.col("spread_0").mean() * SpreadStrategy.WEEKS_IN_YEAR).alias("mean_spread_0"),

            # Annualised volatility (std * sqrt(weeks))
            (pl.col("f0_ret_0").std(ddof=1) * np.sqrt(SpreadStrategy.WEEKS_IN_YEAR)).alias("vol_f0_ret_0"),
            (pl.col("f1_ret_0").std(ddof=1) * np.sqrt(SpreadStrategy.WEEKS_IN_YEAR)).alias("vol_f1_ret_0"),
            (pl.col("spread_0").std(ddof=1) * np.sqrt(SpreadStrategy.WEEKS_IN_YEAR)).alias("vol_spread_0"),
        ])

    def get_quantile_bins(self, nbins: int = 5, instrument: str = None, normalize: bool = False) -> pl.DataFrame:

        """
        Compute spread-based quantile bins and their predictive statistics.

        This method:
        - Divides `spread_0` into `nbins` quantile bins (Q1 to Qn)
        - Optionally normalizes `spread_0` and `spread_t` by rolling volatility
        - Computes 95% confidence interval errors using normal approximation (1.96 * SE)
        - Returns a tidy DataFrame with results for each quantile bin

        Args:
            nbins (int): Number of quantile bins (default: 5)
            instrument (str, optional): Instrument symbol to filter. If None, use all instruments.
            normalize (bool): If True, normalize spread metrics by `rolling_std`.

        Returns:
            pl.DataFrame: Table with columns:
                - quantiles (e.g., Q1 to Qn)
                - avg_spread_0, avg_spread_t, err_spread_t
                - avg_f0_ret_t, err_f0_ret_t
                - avg_f1_ret_t, err_f1_ret_t
        """

        # Step 1: Get raw results (filtered if instrument is specified)
        df0 = self.get_results(instrument)

        # Step 2: Optional signal normalization by rolling standard deviation
        if normalize:
            df0 = (
                df0.with_columns([
                    (pl.col("spread_0") / pl.col("rolling_std")).alias("spread_0"),
                    (pl.col("spread_t") / pl.col("rolling_std")).alias("spread_t"),
                ])
                .drop_nulls(["spread_0", "spread_t"])
            )

        # Step 3: Create quantile labels
        labels = [f"Q{x + 1}" for x in range(nbins)]

        # Step 4: Quantile-based aggregation
        quants = (
            df0.lazy()
            .with_columns(
                # Bin spread_0 into quantiles
                pl.col("spread_0")
                .qcut(quantiles=nbins, labels=labels, left_closed=True)
                .alias("quantiles")
            )
            .group_by("quantiles")
            .agg([
                # Aggregates for spread_0
                pl.col("spread_0").mean().alias("avg_spread_0"),
                pl.col("spread_0").std().alias("std_spread_0"),
                pl.col("spread_0").count().alias("n_spread_0"),

                # Aggregates for spread_t
                pl.col("spread_t").mean().alias("avg_spread_t"),
                pl.col("spread_t").std().alias("std_spread_t"),
                pl.col("spread_t").count().alias("n_spread_t"),

                # Aggregates for front contract return
                pl.col("f0_ret_t").mean().alias("avg_f0_ret_t"),
                pl.col("f0_ret_t").std().alias("std_f0_ret_t"),
                pl.col("f0_ret_t").count().alias("n_f0_ret_t"),

                # Aggregates for second contract return
                pl.col("f1_ret_t").mean().alias("avg_f1_ret_t"),
                pl.col("f1_ret_t").std().alias("std_f1_ret_t"),
                pl.col("f1_ret_t").count().alias("n_f1_ret_t"),
            ])
            .with_columns([
                # Compute standard error bounds (95% CI ~ 1.96 * SE)
                (pl.col("std_spread_0") / pl.col("n_spread_0").sqrt() * 1.96).alias("err_spread_0"),
                (pl.col("std_spread_t") / pl.col("n_spread_t").sqrt() * 1.96).alias("err_spread_t"),
                (pl.col("std_f0_ret_t") / pl.col("n_f0_ret_t").sqrt() * 1.96).alias("err_f0_ret_t"),
                (pl.col("std_f1_ret_t") / pl.col("n_f1_ret_t").sqrt() * 1.96).alias("err_f1_ret_t"),
            ])
            .select([
                "quantiles", "avg_spread_0",
                "avg_spread_t", "err_spread_t",
                "avg_f0_ret_t", "err_f0_ret_t",
                "avg_f1_ret_t", "err_f1_ret_t"
            ])
            .collect()
        )

        return pl.concat([
            quants.filter(pl.col("quantiles") == label) for label in labels
        ], how="vertical")

    def get_near_contract_continuous_returns_series_daily(self, instrument: str = None) -> pl.DataFrame:

        """
        Extract daily return series of the front (near) contract across time.

        This method:
        - Filters the original daily dataframe to include only tickers used as front contracts (`f0`)
        - Performs an inner join on ['ticker', 'period'] to retain matched near contracts
        - Returns the full daily return series (`rets`) for the selected contracts

        Args:
            instrument (str, optional): If specified, filters the analysis to a single instrument (symbol).
                                        If None, uses all available instruments.

        Returns:
            pl.DataFrame: Daily time series of near contract returns including:
                - ticker
                - date
                - days to maturity
                - symbol
                - rets (daily return)
                - period (weekly code)
        """

        # Step 1: From the computed results, extract the (f0, period) combinations
        front_contract_periods = (
            self.get_results(instrument).lazy()
            .select(["f0", "period"])  # Select front contract tickers and period
            .unique()  # Keep only unique combinations
            .rename({"f0": "ticker"})  # Match naming convention in the main DataFrame
        )

        # Step 2: Join on (ticker, period) to get daily returns for near contracts only
        near_contract_df = (
            self._df.lazy()
            .join(front_contract_periods, on=["ticker", "period"], how="inner")
            .select(["ticker", "date", "days", "symbol", "rets", "period"])
            .collect()
        )

        return near_contract_df

    def process_period(self, period_integer) -> pl.DataFrame:

        # Lazy frame of the periodic-level returns data
        weekly_lf = self._period_df.lazy()

        # Step 1: Filter for current and next period
        prev_df = weekly_lf.filter(pl.col("period") == period_integer)
        next_df = weekly_lf.filter(
            (pl.col("period") == period_integer + 1) &
            (pl.col("days") > self._maturity_cutoff)
        )

        # Step 2: Find tickers present in both periods
        common_tickers = (
            prev_df.select("ticker").unique()
            .join(next_df.select("ticker").unique(), on="ticker", how="inner")
        )

        # Step 3: Join to get only valid contracts in t+1
        candidates_lf = (
            next_df
            .join(common_tickers, on="ticker", how="inner")
            .sort(["symbol", "days"])
            .group_by("symbol", maintain_order=True)
            .agg([pl.all().head(2)])  # select front 2 contracts
            .filter(pl.col("days").list.len() == 2)
            .explode(["days", "rets", "date", "period", "ticker"])
        )

        # Step 4: Collect candidates to check for viability
        candidates = candidates_lf.collect()
        if candidates.shape[0] > 1:
            ticker_filter = candidates_lf.select("ticker").unique()

            # Step 5: Get rets/spreads across both periods for valid tickers
            spread_rtns_lf = (
                self._period_df.lazy()
                .join(ticker_filter, on="ticker", how="inner")
                .filter(pl.col("period").is_in([period_integer, period_integer + 1]))
                .sort(["symbol", "period", "days"])
                .with_columns(
                    (-1 * pl.col("rets").diff().over(["symbol", "period"])).alias("rets_diff")
                )
            )

            return (
                spread_rtns_lf.collect()
                .sort(["symbol", "period", "days"])
                .group_by("symbol")
                .agg([
                    # Week 0 (min week)
                    pl.col("rets").filter(pl.col("period") == pl.col("period").min()).first().alias("f0_ret_0"),
                    pl.col("rets").filter(pl.col("period") == pl.col("period").min()).last().alias("f1_ret_0"),
                    pl.col("ticker").filter(pl.col("period") == pl.col("period").min()).first().alias("f0"),
                    pl.col("ticker").filter(pl.col("period") == pl.col("period").min()).last().alias("f1"),
                    pl.col("days").filter(pl.col("period") == pl.col("period").min()).first().alias("f0_days_0"),
                    pl.col("days").filter(pl.col("period") == pl.col("period").min()).last().alias("f1_days_0"),
                    pl.col("date").filter(pl.col("period") == pl.col("period").min()).first().alias("date_0"),
                    pl.col("rets_diff").filter(pl.col("period") == pl.col("period").min()).last().alias("spread_0"),

                    # Week t (max week)
                    pl.col("rets").filter(pl.col("period") == pl.col("period").max()).first().alias("f0_ret_t"),
                    pl.col("rets").filter(pl.col("period") == pl.col("period").max()).last().alias("f1_ret_t"),
                    pl.col("days").filter(pl.col("period") == pl.col("period").max()).first().alias("f0_days_t"),
                    pl.col("days").filter(pl.col("period") == pl.col("period").max()).last().alias("f1_days_t"),
                    pl.col("date").filter(pl.col("period") == pl.col("period").max()).first().alias("date_t"),
                    pl.col("rets_diff").filter(pl.col("period") == pl.col("period").max()).last().alias("spread_t"),

                    # Use week 0's period value as reference
                    pl.col("period").filter(pl.col("period") == pl.col("period").min()).last().alias("period"),
                ])
            )

    def compute(self) -> None:
        """
        Run the core strategy computation across weekly period pairs.

        This method:
        - Iterates over all available period pairs (t and t+1)
        - For each period, extracts relevant contract return/spread observations using `process_period()`
        - Concatenates the per-period DataFrames into one unified result
        - Computes a 52-week rolling standard deviation of the front contract return (`f0_ret_0`) for each symbol
        - Stores the final output internally in `self.__results`

        Notes:
            - The method uses `tqdm` for progress tracking.
            - The output includes one row per instrument per period pair, enriched with volatility information.
        """

        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Step 1: Define the range of periods
        period_range = list(range(max(self.period_codes) - 1))

        # Step 2: Run in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_period, period) for period in period_range]
            period_dfs = [future.result() for future in
                          tqdm(as_completed(futures), total=len(futures), desc="Processing period")]

        # Step 1: Collect results for each period using process_period()
        #period_dfs = []
        #for period in tqdm(range(max(self.period_codes) - 1), desc="Processing period pairs"):
        #    period_dfs.append(self.process_period(period))  # Already returns a pl.DataFrame

        # Step 2: Concatenate all period-level DataFrames vertically
        results = pl.concat(period_dfs, how="vertical")

        # Ensure `cols` is defined
        if 'cols' in locals() or hasattr(self, 'cols'):
            results = results.select(cols)

        # Step 4: Sort results by instrument and date
        results = results.sort(["symbol", "date_0"])

        # Step 5: Compute rolling volatility (52-week rolling std) of near/front contract returns
        results = results.with_columns(
            pl.col("f0_ret_0")
            .rolling_std(window_size=self.WEEKS_IN_YEAR)
            .over("symbol")
            .alias("rolling_std")
        )

        # Step 6: Store the final result
        self.__results = results

    def get_results(self, instrument_code: str = None) -> pl.DataFrame:

        """
        Retrieve the computed strategy results.

        This method:
        - Ensures results are computed (calls `self.compute()` if not already done)
        - Optionally filters the results for a specific instrument if `instrument_code` is provided
        - Returns a clone of the result DataFrame to avoid unintentional mutation

        Args:
            instrument_code (str, optional): Instrument symbol to filter results by.
                                             If None, returns results for all instruments.

        Returns:
            pl.DataFrame: The processed results, with or without filtering by instrument.
                          Includes columns like `f0_ret_0`, `f1_ret_0`, `spread_0`, `spread_t`, etc.

        """
        # Step 1: Run compute() if results have not yet been generated
        if self.__results is None:
            self.compute()

        # Step 2: Clone to avoid modifying internal results directly
        results = self.__results.clone()

        # Step 3: Optional filtering by instrument symbol
        if instrument_code is not None:
            if instrument_code not in self._instruments:
                raise ValueError(f"Instrument '{instrument_code}' not found in the instrument list.")
            return results.filter(pl.col("symbol") == instrument_code)

        return results

    def get_single_instrument_results(self, instrument_code: str) -> pl.DataFrame:
        """
        Convenience method to retrieve strategy results for a single instrument.

        Args:
            instrument_code (str): The symbol of the instrument to retrieve results for.

        Returns:
            pl.DataFrame: A filtered DataFrame containing results only for the specified instrument.
        """
        return self.get_results(instrument_code)

    def fit_AR1_across_instruments(self) -> pd.DataFrame:

        """
        Fit a AR(1) regression of `spread_0` on `spread_t` for each instrument.

        This method:
        - Iterates through each instrument's time series
        - Fits a simple OLS regression: `spread_0 ~ spread_t` (with intercept)
        - Extracts the slope coefficient (beta) and its 95% confidence interval
        - Returns the results as a tidy pandas DataFrame

        Returns:
            pd.DataFrame: One row per instrument, with columns:
                - symbol: Instrument symbol
                - beta: Estimated AR(1) slope
                - ci_lower: Lower bound of 95% confidence interval
                - ci_upper: Upper bound of 95% confidence interval
        """

        results = []
        # Group by instrument and perform OLS regression
        for symbol, group in self.df.group_by("symbol"):
            y = group["spread_0"].to_numpy()
            X = group["spread_t"].to_numpy()

            # Ensure we have enough data points to fit a regression
            if len(X) > 1:
                X = sm.add_constant(X)  # Add intercept term
                try:
                    model = sm.OLS(y, X).fit()
                    beta = model.params[1]  # Slope coefficient
                    ci = model.conf_int(alpha=0.05)[1]  # 95% CI for the slope

                    results.append({
                        "symbol": symbol[0],  # unpack symbol tuple
                        "beta": beta,
                        "ci_lower": ci[0],
                        "ci_upper": ci[1]
                    })

                except Exception as e:
                    continue

        return pd.DataFrame(results)

    def get_first_observation_dates(self) -> pl.DataFrame:

        """
        Retrieve the first available observation date for each instrument in the dataset.

        This method:
        - Groups the raw data by instrument symbol
        - Extracts the earliest available date for each symbol (i.e. start of data history)

        Returns:
            pl.DataFrame: A table with one row per instrument, showing:
                - symbol
                - min_date (earliest date of observation)
        """

        # Group by symbol and return the minimum (first) date observed
        return self._df.group_by("symbol").agg(
            pl.col("date").min().alias("min_date")
        )

    def get_confusion_matrix(self, instrument_code=None):

        # Step 1: Get results (filtered if instrument specified)
        res = self.get_results(instrument_code)

        # Step 2: Extract prediction and actual labels as signs
        actual = np.sign(res.select("spread_t").to_numpy().flatten())  # Realised return direction
        predicted = -1 * np.sign(res.select("spread_0").to_numpy().flatten())  # Strategy prediction (mean-reversion)

        # Step 3: Compute raw confusion matrix
        cm = confusion_matrix(actual, predicted, labels=[-1, 1])

        # Step 4: Normalize by column (prediction) totals for interpretability
        col_sums = cm.sum(axis=0, keepdims=True)
        cm_normalized = cm / col_sums

        # Step 5: Return as labelled DataFrame
        return pd.DataFrame(
            cm_normalized,
            index=["True Negative", "True Positive"],
            columns=["Predicted Negative", "Predicted Positive"]
        )



if __name__ == "__main__":

    instrument_list = [
        "HO", "RB", "CL", "NG",
        "HG", "GC", "SI",
        "GF", "HE", "LE",
        "ZR", "ZC", "ZO", "ZW",
        "SB", "ZS", "ZM", "ZL",
        "CC", "KC", "OJ", "CT"
    ]

    strat = SpreadStrategy(
        instrument_list,
        start_date='31-Dec-1974',
        maturity_cutoff=31,
        freq_string='W-FRI'
    )

    res = strat.get_results()

    out = {}
    for i in instrument_list:
        path = os.path.join('futures_info/', i + "_.xlsx")
        tmp = pd.read_excel(path, index_col=0)

        tmp_list = res.select(["f0", "f1", "symbol"]).filter(pl.col("symbol") == i)

        unique_values = pl.concat([
            tmp_list["f0"],
            tmp_list["f1"]
        ]).unique().to_pandas()

        tmp_out = tmp.loc[unique_values.values][['Name', 'Code']]
        tmp_out['Code'] = tmp["Code"].apply(lambda x: x[1:])
        tmp_out['Month'] = [x[3:5] for x in tmp_out.index]
        tmp_out['Year'] = [x[5:] for x in tmp_out.index]
        tmp_out["Month"] = tmp_out["Month"].astype(str)
        tmp_out["Year"] = tmp_out["Year"].astype(str)
        out[i] = tmp_out.copy()

        from epsilonPhi.core.utils.ExcelUtils import ExcelUtils
        ExcelUtils.dict_to_excel(
            out,
            'futures.xlsx',
            True
        )






    ####### Simple Backtest ######

    df = res.with_columns(
        (pl.col("f1_ret_t")).alias("signed_spread_t")
    ).sort(by="period")

    df = (
        df.join(
            df.group_by("period")
            .agg(pl.count().alias("group_count")),
            on="period"
        )
        .with_columns(
            (1 / pl.col("group_count").cast(pl.Float64)).alias("weight")
        )
        .drop("group_count")
    ).sort(by="period")

    df = df.with_columns(
        (pl.col("signed_spread_t") * pl.col("weight")).alias("weighted_instr_return")
    )

    df_summary = (
        df.group_by("period")
        .agg([
            pl.col("weighted_instr_return").sum().alias("period_return"),
            pl.col("date_t").first().alias("period_date")
        ])
    )

    df_summary.select(["period_return", "period_date"]).to_pandas().set_index('period_date').add(
        1).cumprod().to_clipboard()
