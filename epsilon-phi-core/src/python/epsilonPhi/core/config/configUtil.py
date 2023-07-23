from collections import OrderedDict, defaultdict
from epsilonPhi.core.config.appConfig import CAppConfig
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.lib.Constants import InfinityTime
from epsilonPhi.core.dataModel.alchemist.Configs import *
import epsilonPhi
import datetime as dt
import pandas as pd

class CBaseConfig(object):

    def __init__(self):

        self._catalog = {}
        self._param_catalog = {}
        self._type_catalog = {}
        self._keys = {}
        self._access_methods = {}
        self._access_type = {}
        self._custom_config = {}

    def create_config(self,
                      config_name,
                      params,
                      keys=None,
                      access_methods=None,
                      access_type=None,
                      input_is_class=False,
                      output_is_class=False):

        config = OrderedDict()
        self._catalog[config_name] = config

    def add_param_to_config(self, config_name, params):
        pass

    def update_config(self, config_name, keys, values):
        pass

    def get_config(self, config_name, keys):
        pass

    def load_config(self, _cls, vars=None, values=None):
        return CBaseConfig.perform_query(_cls, vars=None, values=None)

    @staticmethod
    def perform_query(cls_name, vars, values):
        """Method for loading a specific config from the database.
        cls_name represents the ORM defined class mapped to the config table
        vars is
        values is """

        infinity = InfinityTime
        q = SessionMgr.instance.getSessionFactory().query(cls_name).\
            filter(cls_name.toTime == infinity)
        if not vars is None:
            for i in range(len(vars)):
                var = getattr(cls_name, vars[i])
                q = q.filter(var == values[i])
        res = q.all()
        if res is None:
            return []
        else:
            return res

class CConfigUtil(CBaseConfig):
    def __init__(self):
        super(CConfigUtil, self).__init__()

        self._currency_config = None
        self._asset_config = None
        self._private_asset_config = None
        self._private_asset_cashflow_config = None
        self._config_info = None
        self.setup()

    def setup(self):
        _config_info = self.get_config_info()
        for config_key in _config_info.keys():
            self.load_config(_config_info.get(config_key))

    def load_config_info(self):
        info = {}
        for mapper in Base.registry.mappers:
            if hasattr(mapper, 'class_') and 'config' in mapper.class_.__tablename__:
                info[mapper.class_.__tablename__] = mapper.class_
        self._config_info = info

    def get_config_info(self):
        if self._config_info is None:
           self.load_config_info()
        return self._config_info

    def load_config(self, config_name, access_methods=None, access_type='Eager'):

        keys = [x.name for x in config_name.__mapper__.primary_key]

        input_is_class = False
        output_is_class = False
        if isinstance(config_name, type):
            input_is_class = True
            output_is_class = True
            name = config_name
        else:
            name = config_name[0].upper() + config_name[1:]

        res = super(CConfigUtil, self).load_config(name)
        self.create_config(config_name, None, keys, access_methods, access_type, input_is_class, output_is_class)
        SessionMgr.instance.getSessionFactory().expunge_all()

    def get_config(self, config_name, config_keys):
        if config_name not in self._catalog:
            self.load_config(config_name)
        return self._catalog.get(config_name)

    def get_estimation_config(self):
        return self.get_config(EstimationConfig, None)

    def get_simulation_config(self):
        return self.get_config(SimulationConfig, None)

    def get_currency_config(self, currency, frequency):
        return self.get_config(CurrencyConfig, [currency, frequency])

    def get_private_asset_config(self, currency, frequency):
        return self.get_config(CurrencyConfig, [currency, frequency])

    def get_private_asset_cashflow_config(self, currency, frequency):
        return self.get_config(PrivatAssetFlowConfig, None)

    #def get_asset_config(self, asset_name, currency, frequency):
    #    return self.get_config(AssetConfig, [currency, frequency])

    def get_beta_public_to_private_proxy(self, asset_name):
        return 1.1

    def get_premia_liquid_to_total(self, asset_name):
        return (1/3)

    def get_factor_config(self):
        return self.get_config(FactorConfig, None)

    def get_factor_Sharpe_cap(self):
        return None

    def get_reference_portfolio_assets(self):
        pass

    def get_model_portfolio_assets(self):
        pass

    def get_stressed_scenario_config(self):
        pass







