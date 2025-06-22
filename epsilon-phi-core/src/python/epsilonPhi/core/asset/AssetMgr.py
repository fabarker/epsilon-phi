from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from abc import ABC, abstractmethod
from epsilonPhi.core.dataModel.enums.Asset import CustomAssets
import pandas as pd

from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries
gds = GlobalDataSource()

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
        return gds._session_mgr.get_time_series_currency(asset_name)

    def get_asset_reporting_name(self, asset_name):
        return gds._session_mgr.get_asset_reporting_name(asset_name)

    def get_asset_key(self, asset_name):
        return (asset_name,
                self._schema.currency,
                self._schema.frequency)

    @staticmethod
    def _prepare_dataframe_for_asset(
            schema,
            df_
    ):

        if df_ is None:
           return None

        if schema is None:
           return df_

        from epsilonPhi.core.timeSeries.timeSeriesMain import CSlice, CTimeSeries
        assert isinstance(df_, CTimeSeries) or isinstance(df_, CSlice), 'Error - data must be dataframe or timeseries object'

        if isinstance(df_, CTimeSeries):
            assert len(df_.columns) == 1, 'Error - dataframe must be single return time series'
            ts_ = CSlice(df_.iloc[:, 0], ts_type=df_.type, returns_type=df_.returns_type)
        else:
            ts_ = CSlice(df_, ts_type=df_.type, returns_type=df_.returns_type)

        # Comment this out due to issues casting assets ts_.insert_dates(schema.dates)
        if df_.type == TimeSeriesType.LEVELS:
           return ts_.get_periodic_levels(schema.frequency)
        elif df_.type in [TimeSeriesType.RETURNS, TimeSeriesType.GROWTH]:
           return ts_.get_periodic_returns(schema.frequency)
        else:
            raise ValueError('Error - unknown time series type')

    def get_asset_type_from_name(self, asset_name):
        pass

    def get_custom_asset_class(self, asset_name):
        pass

    def load_custom_proxy(self, asset_name):

        """
        Load a custom proxy asset by name, set its reporting metadata,
        and cache a deep copy for future access.
        """

        from epsilonPhi.core.asset.proxies import get_proxy_object_from_name

        proxy_class = get_proxy_object_from_name(asset_name)
        asset = proxy_class(self._schema)

        asset.set_reporting_info(
            proxy_class._reporting_name,
            proxy_class._category
        )

        asset_key = self.get_asset_key(asset_name)
        CAssetMgr._cache[asset_key] = asset.deepcopy()



    def load_asset_by_name(self, asset_name):
        if CustomAssets.is_custom_asset(asset_name):
           self.load_custom_proxy(asset_name)
        else:
           self.load_standard_asset(asset_name)

    def load_standard_asset(self, asset_name):
        from epsilonPhi.core.asset.Asset import CAsset

        # Load data and metadata for the asset
        df = self.get_dataframe_for_asset(asset_name)
        denom_currency, exposure_currency, hedge_ratio = self.get_time_series_currency_info(asset_name)
        reporting_name, category = self.get_asset_reporting_name(asset_name)

        # Construct the asset
        asset = CAsset(
            schema=self._schema,
            data=df,
            denominated_currency=denom_currency,
            exposure_currency=exposure_currency,
            ts_hedge_ratio=hedge_ratio,
            returns_type=ReturnsType.SIMPLE,
            ts_type=TimeSeriesType.RETURNS,
        )

        # add the reporting meta data
        asset.set_reporting_info(
            reporting_name=reporting_name,
            category=category,
        )

        # Cache the asset using a unique key
        asset_key = self.get_asset_key(asset_name)
        CAssetMgr._cache[asset_key] = asset.deepcopy()

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

    def get_inflation_asset(self, currency):

        key = self.get_asset_key(currency + '_CPI')
        if key not in self._cache.keys():

            from epsilonPhi.core.asset.Asset import CAsset
            cpi = GlobalDataSource().get_consumer_price_index_for_currency(currency)
            cpi.columns = [currency + '_CPI']
            cpi.index = cpi.index + pd.offsets.BMonthEnd(0)

            cpi = cpi.get_returns().reindex(self.schema.dates)
            asset = CAsset(
                schema=self._schema,
                data=cpi,
                denominated_currency=currency,
                exposure_currency=currency,
                ts_hedge_ratio=0,
                returns_type=ReturnsType.SIMPLE,
                ts_type=TimeSeriesType.RETURNS
            )

            key = self.get_asset_key(currency + '_CPI')
            CAssetMgr._cache[key] = asset
        return self._cache.get(key).deepcopy()

    def get_risk_free_asset(self, currency):

        key = self.get_asset_key(currency + '_RFR')
        if key not in self._cache.keys():

            from epsilonPhi.core.asset.Asset import CAsset
            risk_free = GlobalDataSource().get_risk_free_rate_for_currency_region(currency)
            risk_free.columns = [currency + '_RFR']

            asset = CAsset(
                schema=self._schema,
                data=risk_free,
                denominated_currency=currency,
                exposure_currency=currency,
                ts_hedge_ratio=0,
                returns_type=ReturnsType.SIMPLE,
                ts_type=TimeSeriesType.RETURNS
            )

            key = self.get_asset_key(currency + '_RFR')
            CAssetMgr._cache[key] = asset
        return self._cache.get(key).deepcopy()



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

    schema = ContextCreator(
        currency='USD',
        start_date='30-Nov-1983',
        end_date='31-Dec-2022'
    ).create_context()

    assetMgr = CAssetMgr(schema)
    asset = assetMgr.get_asset_by_name('FRUSS2L')
    asset.set_currency_hedge_ratio(0)
    res = asset.get_return_betas()