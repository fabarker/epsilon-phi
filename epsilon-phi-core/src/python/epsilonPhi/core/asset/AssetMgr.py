from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from abc import ABC, abstractmethod
import pandas as pd

class CAssetMgrInf(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_asset_config(self, asset_name):
        pass

    @abstractmethod
    def get_asset_currency(self, asset_name):
        pass

class CAssetMgr(CAssetMgrInf):
    _cache = {}

    def __init__(self, schema):
        super(CAssetMgr, self).__init__()
        self._schema = schema

    @property
    def schema(self):
        return self._schema
    @property
    def context(self):
        return self.schema

    def get_asset_config(self, asset_name):
        pass

    def get_asset_currency(self, asset_name):
        return GlobalDataSource()._session_mgr.get_time_series_currency(asset_name)

    def get_asset_key(self, asset_name):
        return (asset_name,
                self._schema.currency,
                self._schema.frequency)

    @staticmethod
    def _prepare_dataframe_for_asset(schema, df_, ts_type):

        if df_ is None:
           return None

        if schema is None:
           return df_

        from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries
        assert isinstance(df_, pd.DataFrame), 'Error - data must be dataframe or timeseries object'
        assert len(df_.columns) == 1, 'Error - dataframe must be single return time series'

        ts_ = CTimeSeries(df_, ts_type=ts_type)

        if ts_type == TimeSeriesType.LEVELS:
            return ts_.get_levels().reindex(schema.dates)
        elif ts_type in [TimeSeriesType.RETURNS, TimeSeriesType.GROWTH]:
            return ts_.get_levels().reindex(schema.dates).get_returns()

    def load_asset_by_name(self, asset_name):

        from epsilonPhi.core.asset.Asset import CAsset
        df_ = self.get_dataframe_for_asset(asset_name)
        denominated_currency, exposure_currency = self.get_asset_currency(asset_name)

        asset = CAsset(schema=self._schema,
                      dataframe=df_,
                      denominated_currency=denominated_currency,
                      exposure_currency=exposure_currency)

        key = self.get_asset_key(asset_name)
        CAssetMgr._cache[key] = asset._deepcopy()

    # Function returns SAA asset
    def get_asset_by_name(self, asset_name):
        key = self.get_asset_key(asset_name)
        if key not in self._cache.keys():
            self.load_asset_by_name(asset_name)
        return self._cache.get(key)._deepcopy()

    def add_asset_to_cache(self, asset):
        CAssetMgr._cache[(asset.getName(),
                          asset._schema.currency,
                          asset._schema.frequency)] = asset.deepcopy()

    def remove_asset_from_cache(self, asset_name):
        del CAssetMgr._cache[asset_name]

    def get_dataframe_for_asset(self, asset_name):
        return GlobalDataSource().get_total_return_series_from_ticker(asset_name,
                                                                      TimeSeriesType.RETURNS)

    def get_risk_free_asset(self, currency):
        risk_free = self._schema.get_risk_free_rate_ticker(currency,
                                                           self._schema.frequency,
                                                           self._schema.dataversion)
        return self.get_asset_by_name(risk_free)

if __name__ == "__main__":

    from epsilonPhi.core.schema.Schema import ContextCreator

    schema = ContextCreator(currency='GBP',
                            start_date='31-Dec-1999',
                            end_date='31-Dec-2022').create_context()

    assetMgr = CAssetMgr(schema)

    spx = assetMgr.get_asset_by_name('S&PCOMP')
    rfr = assetMgr.get_risk_free_asset('GBP')


