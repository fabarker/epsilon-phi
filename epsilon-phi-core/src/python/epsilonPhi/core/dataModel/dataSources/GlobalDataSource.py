from epsilonPhi.core.lib.Decorators import SingletonDecorator
from epsilonPhi.core.dataModel.dataSources.FXCurve import FXCurve
from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.dataModel.enums.Asset import PrivateAsset


@SingletonDecorator
class GlobalDataSource(object):

    _session_mgr = SessionMgr()
    _session = _session_mgr.getSessionFactory()
    _cache = dict()

    def __init__(self):
        self.initialize()

    def initialize(self):
        self._fx_curve = FXCurve()

    # Methods associated with currencies / FX

    def get_fx_forward_prices(self, currency_pairs, pricing_dates, maturity_dates, price_quotes):
        return self._fx_curve.get_forward_prices(currency_pairs, pricing_dates, maturity_dates, price_quotes)

    def get_fx_forward_rates(self, currency_pairs, maturities, price_quotes):
        return self._fx_curve.get_forward_rates(currency_pairs, maturities, price_quotes)

    def get_fx_spot_rates(self, currency_pairs, price_quotes):
        return self._fx_curve.get_spot_rates(currency_pairs, price_quotes)

    def get_fx_carry(self, currency_pairs, maturities, price_quotes):
        return self._fx_curve.get_forward_prices(currency_pairs, maturities, price_quotes)

    # Method Associated with DataFrames
    def get_risk_free_rate(self, currency, frequency):
        pass

    def get_dataframe_from_ticker(self, ticker: str, cols=None, index_col=None):
        uid = self._session_mgr.get_uid_from_ticker(ticker)
        return self.get_dataframe_from_uid(uid, cols, index_col)

    def get_dataframe_from_uid(self, uid: int, cols=None, index_col=None):
        df = self._session_mgr.get_dataframe_from_uid(uid)

        if index_col is not None and index_col in df.columns:
            df = df.set_index(index_col, drop=True)

        if cols is not None:
           return df.get(cols, pd.DataFrame()).dropna(how='all')
        else:
            return df.copy().dropna(how='all')

    # Methods associated with interest rates




    # Methods associated with implied volatilties




    # Methods associated with querying datastream




    # Methods associated with querying GSQuant


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
    df = self.get_dataframe_from_ticker('UKPRATE.', cols=['IR'], index_col='date')





