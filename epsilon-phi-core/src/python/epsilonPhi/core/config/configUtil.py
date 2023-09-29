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

        if params is not None:
            i = 0
            pdict = OrderedDict()
            for p in params:
                pdict[p] = i
                i += 1

        self._param_catalog[config_name] = pdict
        if input_is_class and output_is_class:
            self._type_catalog = ('class', 'class')
        elif input_is_class and not output_is_class:
            self._type_catalog = ('class', 'list')
        elif not input_is_class and not output_is_class:
            self._type_catalog = ('list', 'list')
        else:
            raise Exception('Error - input as list and output as class is not supported')

        self._keys[config_name] = keys

        if access_type is not None:
            self._access_type[config_name] = access_type
        if access_methods is not None:
            self._access_methods[config_name] = access_methods
        return config

    def add_param_to_config(self, config_name, keys, values):
        self._catalog[config_name][keys] = values

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
        for r in res:
            self.create_config(config_name, r.__dict__.keys(), keys, access_methods, access_type, input_is_class, output_is_class)
            self.add_param_to_config(name, tuple([r.__dict__.get(x) for x in keys]), r)

        SessionMgr.instance.getSessionFactory().expunge_all()

    def get_config(self, config_name, config_keys):
        if config_name not in self._catalog.keys():
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







