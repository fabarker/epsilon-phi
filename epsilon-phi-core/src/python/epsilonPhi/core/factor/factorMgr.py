from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from abc import ABC, abstractmethod
from epsilonPhi.core.asset.Asset import CAssetMgr
import pandas as pd


class CFactorMgrInf(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_factor_config(self, asset_name):
        pass

class CFactorMgr(CFactorMgrInf):
    _cache = {}

    def __init__(self, schema):
        super(CFactorMgr, self).__init__()
        self._schema = schema
        self._assetMgr = CAssetMgr(schema)

    def get_factor_config(self):
        pass

    def get_asset_by_name(self, asset_name):
        return self._assetMgr.get_asset_by_name(asset_name)

    def construct_factor(self, name):
        pass

    def is_constructed(self, factor_name):
        pass

    def load_factor_by_name(self, factor_name):
        pass

    def get_factor_by_name(self, factor_name):
        pass

    @staticmethod
    def _prepare_dataframe_for_factor(schema, df_, ts_type):

        if df_ is None:
           return None

        if schema is None:
           return df_

        from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries
        assert isinstance(df_, pd.DataFrame), 'Error - data must be dataframe or timeseries object'
        assert len(df_.columns) == 1, 'Error - dataframe must be single return time series'

        ts_ = CTimeSeries(df_, ts_type=ts_type)
        ts_.insert_dates(schema.dates)
        return ts_.get_levels().reindex(schema.dates).get_returns()

if __name__ == "__main__":
    from epsilonPhi.core.schema.Schema import ContextCreator

    schema = ContextCreator(currency='GBP',
                            start_date='31-Dec-1999',
                            end_date='31-Dec-2022').create_context()

    factorMgr = CFactorMgr(schema)