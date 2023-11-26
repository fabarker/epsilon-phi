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
    def get_time_series_currency_info(self, asset_name):
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

    def get_time_series_currency_info(self, asset_name):
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

        from epsilonPhi.core.timeSeries.timeSeriesMain import CSlice
        assert isinstance(df_, pd.DataFrame) or isinstance(df_, pd.Series), 'Error - data must be dataframe or timeseries object'

        if isinstance(df_, pd.DataFrame):
            assert len(df_.columns) == 1, 'Error - dataframe must be single return time series'
            ts_ = CSlice(df_.iloc[:, 0], ts_type=ts_type, returns_type=df_.returns_type)
        else:
            ts_ = CSlice(df_, ts_type=ts_type, returns_type=df_.returns_type)

        ts_.insert_dates(schema.dates)
        if ts_type == TimeSeriesType.LEVELS:
            return ts_.get_periodic_levels(schema.frequency)
        elif ts_type in [TimeSeriesType.RETURNS, TimeSeriesType.GROWTH]:
            return ts_.get_levels().get_periodic_returns(schema.frequency)

    def load_asset_by_name(self, asset_name):

        from epsilonPhi.core.asset.Asset import CAsset
        df_ = self.get_dataframe_for_asset(asset_name)
        denominated_currency, exposure_currency, hedge_ratio = self.get_time_series_currency_info(asset_name)

        asset = CAsset(schema=self._schema,
                      dataframe=df_,
                      denominated_currency=denominated_currency,
                      exposure_currency=exposure_currency,
                      ts_hedge_ratio=hedge_ratio)

        key = self.get_asset_key(asset_name)
        CAssetMgr._cache[key] = asset.deepcopy()

    # Function returns SAA asset
    def get_asset_by_name(self, asset_name):
        key = self.get_asset_key(asset_name)
        if key not in self._cache.keys():
            self.load_asset_by_name(asset_name)
        return self._cache.get(key).deepcopy()

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
        from epsilonPhi.core.asset.Asset import CAsset
        risk_free = GlobalDataSource().get_risk_free_rate_for_currency_region(currency)
        risk_free.columns = [currency + '_RFR']

        return CAsset(schema=self._schema,
                       dataframe=risk_free,
                       denominated_currency=currency,
                       exposure_currency=currency)

    @staticmethod
    def convert_asset_to_currency(asset, target_currency, hedging_ratio):
        asset_fx = GlobalDataSource().fx_convert_timeseries_to_currency_hedged(asset.deepcopy(),
                                                                               target_currency,
                                                                               hedging_ratio,
                                                                               asset.denominated_currency,
                                                                               asset.exposure_currency,
                                                                               asset._ts_hedge_ratio)
        asset_fx._ts_hedge_ratio = hedging_ratio
        asset_fx._denominated_currency = target_currency
        return asset_fx


    @staticmethod
    def asset_to_currency_hedged(self, asset, currency):
        pass

    @staticmethod
    def asset_to_currency_unhedged(self, asset, currency):
        pass




if __name__ == "__main__":

    from epsilonPhi.core.schema.Schema import ContextCreator

    schema = ContextCreator(currency='GBP',
                            start_date='31-Dec-1999',
                            end_date='31-Dec-2022').create_context()

    assetMgr = CAssetMgr(schema)
    asset = assetMgr.get_asset_by_name('MSUSAML')
    asset_GBP = assetMgr.convert_asset_to_currency(asset, target_currency='GBP', hedging_ratio=0)
    asset_EUR = assetMgr.convert_asset_to_currency(asset, target_currency='EUR', hedging_ratio=0)

    from epsilonPhi.core.dataModel.enums.TimeSeries import ReturnsType
    asset_GBP_LOG = asset_GBP.get_returns(ReturnsType.LOG)
    asset_GBP_LOG_SIMPLE = asset_GBP_LOG.get_returns(ReturnsType.SIMPLE)


