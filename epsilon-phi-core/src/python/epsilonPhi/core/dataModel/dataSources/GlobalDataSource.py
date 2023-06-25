from epsilonPhi.core.lib.Decorators import SingletonDecorator
from epsilonPhi.core.dataModel.dataSources.FXCurve import FXCurve
from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.dataModel.enums.Asset import PrivateAsset


@SingletonDecorator
class GlobalDataSource(object):

    _session = SessionMgr().getSessionFactory()
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
    self.get_private_equity_cash_flow_assumptions(PrivateAsset.BUYOUT)





