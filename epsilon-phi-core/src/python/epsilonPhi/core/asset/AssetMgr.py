from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from abc import ABC, abstractmethod

class CAssetMgrInf(ABC):
    def __init__(self):
        pass

    @staticmethod
    def get_asset_info(asset_name):
        pass

    @abstractmethod
    def get_asset_config(self, asset_name):
        pass

    @abstractmethod
    def get_asset_currency(self, asset_name):
        pass


    @abstractmethod
    def get_asset_category(self, asset_name):
        pass

class CAssetMgr(CAssetMgrInf):
    _cache = {}

    def __init__(self, schema):
        super(CAssetMgr, self).__init__()
        self._schema = schema

    @staticmethod
    def get_asset_type(asset_name):
        pass

    def get_asset_config(self, asset_name):
        pass


    def get_asset_category(self, asset_name):
        pass


    def get_asset_currency(self, asset_name):
        pass


    def get_context(self):
        return self._schema


    def get_schema(self):
        return self._schema


    # Function returns SAA asset
    def get_asset_by_name(self):
        pass

    def  add_asset_to_cache(self, asset):
        CAssetMgr._cache[asset.getName()] = asset

    def remove_asset_from_cache(self, asset_name):
        del CAssetMgr._cache[asset_name]

    def get_dataframe_for_asset(self, asset_name, freq=Frequency.Daily):
        pass
