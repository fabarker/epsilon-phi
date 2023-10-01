from epsilonPhi.core.lib.Decorators import SingletonDecorator
from epsilonPhi.core.dataModel.dataSources.fxCurve.FXCurve import FXCurve
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.dataModel.enums.Asset import PrivateAsset
from epsilonPhi.core.timeSeries.timeSeriesMain import *


@SingletonDecorator
class GlobalDataSource(object):

    _session_mgr = SessionMgr()
    _session = _session_mgr.getSessionFactory()

    # Cache Time Series Objects
    _cache = dict()


    def __init__(self):
        self.initialize()

    def initialize(self):
        self._fx_curve = FXCurve()

    # Method Associated with loading DataFrames

    def get_dataframe_from_ticker(self, ticker: str, cols=None, index_col=None):

        uid = self._session_mgr.get_uid_from_ticker(ticker)
        df_ = self.get_dataframe_from_uid(uid, cols, index_col)
        df_.columns = pd.MultiIndex.from_tuples([(ticker, x) for x in df_.columns.get_level_values(1)])
        return df_.copy()

    def get_dataframe_from_uid(self, uid: int, cols=None, index_col=None):
        df = self._session_mgr.get_dataframe_from_uid(uid)

        if index_col is not None and index_col in df.columns:
            df = df.set_index(index_col, drop=True)

        if cols is not None:
           df = df.get(cols, pd.DataFrame()).dropna(how='all')
        else:
           df = df.copy().dropna(how='all')

        if isinstance(df, pd.Series):
           df = df.to_frame()
        df.columns = pd.MultiIndex.from_tuples(list(zip([uid] * df.shape[1], df.columns)))
        return df

    # Methods associated with loading raw time series

    def get_time_series_data_from_uid(self, uid, cols='X', ts_type=None):
        df = self.get_dataframe_from_uid(uid, cols=cols, index_col='date')
        spec = self._session_mgr.get_time_series_spec_from_uid(uid, True).set_index('uid', drop=True)
        return CTimeSeries(df, ts_type=ts_type, attributes=spec.T)

    def get_time_series_data_from_ticker(self, ticker, cols='X', ts_type=None):
        df = self.get_dataframe_from_ticker(ticker, cols=cols, index_col='date')
        spec = self._session_mgr.get_time_series_spec_from_ticker(ticker, True).set_index('ticker', drop=True)
        return CTimeSeries(df, ts_type=ts_type, attributes=spec.T)

    # Methods associated with currencies / FX

    def get_fx_forward_prices(self, currency_pairs, pricing_dates, maturity_dates, price_quotes):
        return self._fx_curve.get_forward_prices(currency_pairs, pricing_dates, maturity_dates, price_quotes)

    def get_fx_forward_rates(self, currency_pairs, maturities, price_quotes):
        return self._fx_curve.get_forward_rates(currency_pairs, maturities, price_quotes)

    def get_fx_spot_rates(self, currency_pairs, price_quotes):
        return self._fx_curve.get_spot_rates(currency_pairs, price_quotes)

    def get_fx_carry(self, currency_pairs, maturities, price_quotes):
        return self._fx_curve.get_forward_prices(currency_pairs, maturities, price_quotes)

    def get_risk_free_rate(self, currency, frequency):
        pass



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
    df = self.get_time_series_data_from_ticker('UKPRATE.', cols=['IR'])





