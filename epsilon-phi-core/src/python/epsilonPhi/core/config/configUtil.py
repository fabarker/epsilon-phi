from collections import OrderedDict, defaultdict
import datetime as dt
import pandas as pd

from epsilonPhi.core.config.appConfig import CAppConfig
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.alchemist.Configs import CurrencyConfig

class CConfigUtil(object):
    def __init__(self):
        self._setup()

    def setup(self):
        pass

    def get_config(self, config_name, config_keys):
        pass

class SAAConfigUtil(CConfigUtil):
    def __init__(self, currency):
        super(SAAConfigUtil, self).setup()

        self._currency = currency

        self._currency_config = None
        self._asset_config = None


        self._private_asset_config = None
        self._private_asset_cashflow_config = None

    def get_currency_config(self, currency, frequency):
        pass

    def get_private_asset_config(self, currency, frequency):
        pass

    def get_private_asset_cashflow_config(self, currency, frequency):
        pass

    def get_asset_config(self, asset_name, currency, frequency):
        pass

    def get_beta_public_to_private_proxy(self, asset_name):
        return 1.1

    def get_premia_liquid_to_total(self, asset_name):
        return (1/3)

    def get_factor_config(self):
        pass

    def get_factor_Sharpe_cap(self):
        pass

    def get_reference_portfolio_assets(self):
        pass

    def get_model_portfolio_assets(self):
        pass

    def get_stressed_scenario_config(self):
        pass















