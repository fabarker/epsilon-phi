from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.utils.PickleUtils import PickleUtils
from abc import ABC, abstractmethod
from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries
import os, pkgutil, importlib
import numpy as np
from epsilonPhi.core.asset.Asset import CAssetMgr
import pandas as pd
from epsilonPhi.core.dataModel.enums.Factor import FACTOR

_FACTOR_PACKAGE = 'epsilonPhi.core.factor.factors'
gds = GlobalDataSource()

class CFactorMgrInf(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_factor_config(self, asset_name):
        pass

class CFactorMgr(CFactorMgrInf):
    _cache = {}

    def __init__(self,
                 start_date,
                 end_date,
                 frequency):

        super(CFactorMgr, self).__init__()
        self._schema = _schema
        self._assetMgr = CAssetMgr(_schema)

        self.__risk_factors_df = None
        self.__return_factors_df = None

        self.__risk_factor_list = FACTOR.get_default_risk_factor_list()
        self.__return_factor_list = FACTOR.get_default_return_factor_list()

    def get_factor_config(self, asset_name):
        return self._schema.get_factor_config()

    @property
    def risk_factors_list(self):
        return [x.value for x in self.__risk_factor_list]
    @property
    def return_factors_list(self):
        return [x.value for x in self.__return_factor_list]

    @property
    def return_factors_df(self):
        return self.__return_factors_df

    @property
    def risk_factors_df(self):
        return self.__risk_factors_df

    @property
    def return_risk_factor_list(self):
        return list(set(self.return_factors_list + self.risk_factors_list))

    def _reset_risk_factors_df(self):
        self.__risk_factors_df = None

    def _reset_return_factors_df(self):
        self.__return_factors_df = None

    def set_risk_factor_list(self, risk_factor_list: list):
        self.__check_factor_list(risk_factor_list)
        self._reset_risk_factors_df()
        self.__risk_factor_list = risk_factor_list

    def set_return_factor_list(self, return_factors_list: list):
        self.__check_factor_list(return_factors_list)
        self._reset_return_factors_df()
        self.__return_factor_list = return_factors_list

    def __check_factor_list(self, factor_list):
        assert isinstance(factor_list, list), 'ERROR - Must be list of return factor enums'
        bool = [isinstance(x, FACTOR) for x in factor_list]
        assert np.all(bool), 'Error - factor list must be a list of FACTOR enumerators'

    def _load_factors(self):
        for factor in self.return_risk_factor_list:
            self.__load_single_factor(factor)

    def __load_factor_from_pickles(self, factor):
        if PickleUtils.is_factor_pickled(factor,
                                         self._schema.end_date,
                                         self._schema.frequency):
            self._cache[factor] = PickleUtils.load_factor_from_pickles(factor,
                                                                       self._schema.end_date,
                                                                       self._schema.frequency)
    def __load_single_factor(self, factor):

        if factor not in self._cache.keys():
            self.__load_factor_from_pickles(factor)

        if factor not in self._cache.keys():
            if self.__isConstructed(factor):
               self.__construct_factor(factor)
            else:
               df = gds.get_time_series_data_from_ticker(factor, ts_type=TimeSeriesType.RETURNS)
               df.columns = [factor]

               from epsilonPhi.core.factor.Factor import CFactor
               self._cache[factor] = CFactor(df, self._schema, TimeSeriesType.RETURNS)
               PickleUtils.pickle_factor(self._cache[factor],
                                         factor,
                                         self._schema.end_date,
                                         self._schema.frequency)

    def __isConstructed(self, factor_name: str):
        return factor_name + '.py' in pkgutil.get_loader(_FACTOR_PACKAGE).contents()

    def __construct_factor(self, factor):
        module = importlib.import_module(_FACTOR_PACKAGE + '.' + factor)
        constructor = getattr(module, factor)
        self._cache[factor] = constructor(self._schema)
        PickleUtils.pickle_factor(self._cache[factor],
                                  factor,
                                  self._schema.end_date,
                                  self._schema.frequency)

    def get_risk_factor_panel(self):
        return self.get_factor_panel().get(self.risk_factors_list)

    def get_return_factors_panel(self):
        return self.get_factor_panel().get(self.return_factors_list)

    def get_factor_panel(self):
        self._load_factors()

        panel = CTimeSeries(ts_type=TimeSeriesType.RETURNS)
        for factor in self._cache.keys():
            panel = panel.concat(self._cache.get(factor))
        return panel.select_subset_dates(self._schema.dates)


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
        return ts_.get_periodic_returns(schema.frequency)

if __name__ == "__main__":
    from epsilonPhi.core.schema.Schema import ContextCreator

    schema = ContextCreator(currency='GBP',
                            start_date='30-Nov-1983',
                            end_date='31-Dec-2022').create_context()

    self = CFactorMgr(schema)
    self._load_factors()
    factors = self.get_risk_factor_panel()