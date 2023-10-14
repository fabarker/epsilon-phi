import pandas as pd

from epsilonPhi.core.lib.Decorators import SingletonDecorator
from epsilonPhi.core.dataModel.dataSources.fxCurve.FXCurve import FXCurve
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.dataModel.enums.Asset import PrivateAsset
from epsilonPhi.core.timeSeries.timeSeriesMain import *
from epsilonPhi.core.utils.TimeSeriesUtils import TimeSeriesUtils
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType


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


    # Methods associated with loading raw time series
    def get_time_series_data_from_uid(self, uid, cols=None):
        df = self.get_dataframe_from_uid(uid, cols=cols, index_col='date')
        spec = self._session_mgr.get_time_series_spec_from_uid(uid, True).set_index('uid', drop=True)

        ts_spec = pd.concat([spec] * df.shape[1])
        ts_spec.index = df.columns
        return CTimeSeries(df, attributes=ts_spec.T)

    def get_time_series_data_from_ticker(self, ticker, cols=None):

        df = self.get_dataframe_from_ticker(ticker, cols=cols, index_col='date')
        spec = self._session_mgr.get_time_series_spec_from_ticker(ticker, True).set_index('ticker', drop=True)

        ts_spec = pd.concat([spec] * df.shape[1])
        ts_spec.index = df.columns
        return CTimeSeries(df, attributes=ts_spec.T)

    def get_total_return_series_from_ticker(self, ticker, returns_type=TimeSeriesType.LEVELS):
        ts = self.get_time_series_data_from_ticker(ticker)

        if returns_type == TimeSeriesType.RETURNS:
            return TimeSeriesUtils.convert_timeseries_to_return_index(ts).get_returns()
        else:
            return TimeSeriesUtils.convert_timeseries_to_return_index(ts)

    def get_interest_rate_tickers(self, currency, maturities=None, type=None):
        return self._session_mgr.get_interest_rate_tickers(currency,
                                                           maturities,
                                                           type)

    # Methods associated with currencies / FX
    def get_fx_forward_prices(self, currency_pairs, pricing_dates, maturity_dates, price_quotes):
        return self._fx_curve.get_forward_prices(currency_pairs, pricing_dates, maturity_dates, price_quotes)

    def get_fx_forward_rates(self, currency_pairs, maturities, price_quotes):
        return self._fx_curve.get_forward_rates(currency_pairs, maturities, price_quotes)

    def get_fx_spot_rates(self, currency_pairs, price_quotes):
        return self._fx_curve.get_spot_rates(currency_pairs, price_quotes)

    def get_fx_carry(self, currency_pairs, maturities, price_quotes):
        return self._fx_curve.get_forward_prices(currency_pairs, maturities, price_quotes)


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
    res = self.get_interest_rate_tickers('USD')

    tickers = ['MSHWLD$','MSWRLD$','MSWRLDL','MSFXDW$']

    df_ = pd.DataFrame()
    for ticker in tickers:
        df = self.get_dataframe_from_ticker(ticker, cols='PI')
        df_ = pd.concat((df_, df), axis=1)






