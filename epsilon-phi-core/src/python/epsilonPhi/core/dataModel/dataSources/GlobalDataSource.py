import pandas as pd

from epsilonPhi.core.lib.Decorators import SingletonDecorator
from epsilonPhi.core.dataModel.dataSources.fxCurve.FXCurve import FXCurve
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.dataModel.enums.Asset import PrivateAsset
from epsilonPhi.core.timeSeries.timeSeriesMain import *
from epsilonPhi.core.utils.TimeSeriesUtils import TimeSeriesUtils
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType
from epsilonPhi.core.dataModel.enums.Database import PriceQuote


@SingletonDecorator
class GlobalDataSource(object):

    _session_mgr = SessionMgr()
    _session = _session_mgr.getSessionFactory()

    # Cache Time Series Objects
    _cache_df = dict()
    _cache_ts = dict()


    def __init__(self):
        self.initialize()

    def initialize(self):
        self._fx_curve = FXCurve()

    # Method Associated with loading DataFrames
    def load_dataframe_from_uid(self, uid: int):
        if uid not in self._cache_df.keys():
            df_uid = self._session_mgr.get_dataframe_from_uid(uid)
            self._cache_df[uid] = df_uid[np.setdiff1d(df_uid.columns, 'uid')].dropna(axis=1, how='all')

    def get_dataframe_from_uid(self, uid: int, cols=None, index_col=None):

        if uid not in self._cache_df.keys():
            self.load_dataframe_from_uid(uid)
        df = self._cache_df.get(uid)

        if index_col is not None and index_col in df.columns:
            df = df.set_index(index_col, drop=True)
        elif index_col is None and 'date' in df.columns:
            df = df.set_index('date', drop=True).sort_index()

        if cols is not None:
           keep_cols = np.intersect1d(cols, df.columns)
           df = df.get(keep_cols).dropna(how='all')
           df[np.setdiff1d(cols, df.columns)] = np.nan
        else:
           df = df.dropna(how='all')

        if isinstance(df, pd.Series):
           df = df.to_frame()

        df.columns = pd.MultiIndex.from_tuples(list(zip([uid] * df.shape[1], df.columns)))
        return df

    def get_dataframe_from_ticker(self, ticker: str, cols=None, index_col=None):

        uid = self._session_mgr.get_uid_from_ticker(ticker)
        df_ = self.get_dataframe_from_uid(uid, cols, index_col)

        df_.columns = pd.MultiIndex.from_tuples([(ticker, x) for x in df_.columns.get_level_values(1)])
        return df_.copy()

    def get_dataframe_from_tickers(self, tickers: list, cols=None, index_col=None):

        if not DateUtils.is_iterable(tickers) and isinstance(tickers, str):
           tickers = [tickers]

        df_ = pd.DataFrame()
        for ticker in tickers:
            df_ = pd.concat((df_, self.get_dataframe_from_ticker(ticker, cols, index_col)), axis=1)
        return df_.copy()


    # Methods associated with loading raw time series
    def get_time_series_data_from_uid(self, uid, cols=None):
        df = self.get_dataframe_from_uid(uid, cols=cols, index_col='date')
        spec = self._session_mgr.get_time_series_spec_from_uid(uid, True).set_index('uid', drop=True)

        ts_spec = pd.concat([spec] * df.shape[1])
        ts_spec.index = df.columns
        return CTimeSeries(df, attributes=ts_spec.T)

    def get_time_series_data_from_ticker(self, ticker, cols=None, ts_type=TimeSeriesType.LEVELS):

        df = self.get_dataframe_from_ticker(ticker, cols=cols, index_col='date')
        spec = self._session_mgr.get_time_series_spec_from_ticker(ticker, True).set_index('ticker', drop=True)

        ts_spec = pd.concat([spec] * df.shape[1])
        ts_spec.index = df.columns
        return CTimeSeries(df, attributes=ts_spec.T, ts_type=ts_type)

    def get_total_return_series_from_ticker(self, ticker, returns_type=TimeSeriesType.LEVELS):
        ts = self.get_time_series_data_from_ticker(ticker)

        if returns_type == TimeSeriesType.RETURNS:
            return TimeSeriesUtils.convert_timeseries_to_return_index(ts).get_returns()
        else:
            return TimeSeriesUtils.convert_timeseries_to_return_index(ts)

    def get_interest_rate_tickers(self, currency_region, maturities=None, type=None):
        return self._session_mgr.get_interest_rate_tickers(currency_region,
                                                           maturities,
                                                           type)

    def get_risk_free_rate_for_currency_region(self, currency):
        from epsilonPhi.core.dataModel.dataSources.riskFreeRates.RiskFreeRates import CRiskFreeRate
        return CRiskFreeRate.get_risk_free_rate_from_currency(currency)

    def get_risk_free_rate_time_series(self, region):
        from epsilonPhi.core.dataModel.dataSources.riskFreeRates.RiskFreeRates import CRiskFreeRate
        return CRiskFreeRate.get_risk_free_for_region(region)

    # Methods associated with currencies / FX
    def get_fx_forward_prices(self, currency_pairs, pricing_dates, maturity_dates, price_quotes):
        return self._fx_curve.get_forward_prices(currency_pairs, pricing_dates, maturity_dates, price_quotes)

    def get_forward_returns(self, currency_pairs, from_dates, to_dates, maturity_dates, price_quotes):
        prices_from = self._fx_curve.get_forward_prices(currency_pairs, from_dates, maturity_dates, price_quotes)
        prices_to = self._fx_curve.get_forward_prices(currency_pairs, to_dates, maturity_dates, price_quotes)
        return np.log(prices_to) - np.log(prices_from)

    def get_fx_forward_rates(self, currency_pairs, maturities, price_quotes):
        return self._fx_curve.get_forward_rates(currency_pairs, maturities, price_quotes)

    def get_fx_spot_rates(self, currency_pairs, price_quotes):
        return self._fx_curve.get_spot_rates(currency_pairs, price_quotes)

    def get_fx_carry(self, currency_pairs, maturities, price_quotes):
        return self._fx_curve.get_carry(currency_pairs, maturities, price_quotes)

    def get_interest_rate_differential(self, domestic_currency, foreign_currency):
        rd = self.get_risk_free_rate_for_currency_region(domestic_currency)
        rf = self.get_risk_free_rate_for_currency_region(foreign_currency)
        return rf.subtract_over_common_dates(rd)

    def fx_convert_timeseries_to_currency(self, timeseries, denominated_currency, target_currency, hedge_ratio):
        return

    def fx_convert_timeseries_to_currency_hedged(self, time_series, target_currency, target_hedge_ratio, denominated_currency, exposure_currency=None, current_hedge_ratio=None, hedge_frequency=Frequency.BUSINESS_MONTHLY):
        return self._fx_curve.hedge_time_series(time_series, target_currency, target_hedge_ratio, denominated_currency, exposure_currency, current_hedge_ratio, hedge_frequency)

    def fx_convert_timeseries_to_currency_unhedged(self, timeseries, denominated_currency, target_currency):
        return self._fx_curve.unhedged_time_series(timeseries, denominated_currency, target_currency)

    def get_factor_dataframe(self, factor, universe=None, region=None, provider=None):
        tickers = self._session_mgr.get_factor_ticker(factor, region=region, universe=universe, provider=provider)

        if not DateUtils.is_iterable(tickers):
           tickers = [tickers]

        fac_ts = pd.DataFrame()
        for ticker in tickers:
            df_ = self.get_dataframe_from_ticker(ticker, index_col='date')
            fac_ts = pd.concat((fac_ts, df_), axis=1)
        return fac_ts.copy()


    # Methods associated with querying datastream
    def get_time_series_data_from_datastream(self,
                                             symbols,
                                             fields=None,
                                             from_date=None,
                                             to_date=None,
                                             frequency='D'):

        from epsilonPhi.core.dataModel.dataSources.vendor.Datastream import pyDatastream
        return pyDatastream.fetch(symbols,
                                  fields=fields,
                                  from_date=from_date,
                                  to_date=to_date,
                                  frequency=frequency)




    # Methods associated with querying GSQuant
    def get_time_series_data_from_qsquant(self):
        pass




    #################### Method for loading private equity assumptions ###########################
    def _load_private_asset_cash_flow_assumptions(self):
        if not hasattr(self, '_private_asset_cash_flow_assumptions'):
           config = self._private_asset_cash_flow_assumptions = pd.read_sql_table('private_asset_flow_config', self._session.get_bind())
           self._private_asset_cash_flow_assumptions = config.set_index('strategy', drop=True)

    def get_private_asset_cash_flow_assumptions(self, strategy: PrivateAsset):
        if not hasattr(self, '_private_asset_cash_flow_assumptions'):
           self._load_private_asset_cash_flow_assumptions()
        return self._private_asset_cash_flow_assumptions.loc[strategy.value]

    def get_private_asset_capital_call_assumptions(self, strategy: PrivateAsset):
        df = self.get_private_asset_cash_flow_assumptions(strategy)
        return df[df.get('type') == 'C'].set_index('year').get('value')

    def get_private_asset_distribution_assumptions(self, strategy: PrivateAsset):
        df = self.get_private_asset_cash_flow_assumptions(strategy)
        return df[df.get('type') == 'D'].set_index('year').get('value')



if __name__ == "__main__":

    self = GlobalDataSource()
    session = self._session

    tickers = ['MSWRLDL', 'MSFXDW$', 'MSHWLD$', 'MSWRLD$']
    _dfs = CTimeSeries()
    for ticker in tickers:
        df = self.get_time_series_data_from_ticker(ticker, cols='PI')
        _dfs = _dfs.concat(df)



    df = CTimeSeries(df, ts_type=TimeSeriesType.LEVELS, returns_type=ReturnsType.SIMPLE)
    df = df.get_bmonthly_returns()
    df = df['31-Jan-1985':]

    US_EQ_AUD_HEDGED = self.fx_convert_timeseries_to_currency_hedged(df, 'AUD', 0.5, 'USD', 'USD')
    US_EQ_GBP_HEDGED_VIA_AUD = self.fx_convert_timeseries_to_currency_hedged(US_EQ_AUD_HEDGED, 'GBP', 1, 'AUD', 'USD', 0.5)
    US_EQ_GBP_HEDGED = self.fx_convert_timeseries_to_currency_hedged(df, 'GBP', 1, 'USD', 'USD')
