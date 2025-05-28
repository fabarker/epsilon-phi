import pandas as pd

from epsilonPhi.core.factor.factorPanelInf import CFactorPanelInf
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.utils.PickleUtils import PickleUtils
from epsilonPhi.core.dataModel.enums.Factor import FACTOR
from epsilonPhi.core.factor.Factor import CFactor
from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries
import pkgutil
import importlib
import numpy as np
import datetime as dt

_FACTOR_PACKAGE = 'epsilonPhi.core.factor.factors'
gds = GlobalDataSource()


class CFactorPanels(CFactorPanelInf):

    _cache = {}
    __DEFAULT_RISK_FACTORS = FACTOR.get_default_risk_factor_list()
    __DEFAULT_RETURN_FACTORS = FACTOR.get_default_return_factor_list()
    __DEFAULT_END_DATE = dt.date(year=2022, month=12, day=31)
    __DEFAULT_FREQUENCY = Frequency.BUSINESS_MONTHLY

    def __init__(self,
                 factor_list,
                 frequency=None,
                 end_date=None):

        super(CFactorPanels, self).__init__()

        self._factor_list = factor_list
        if frequency is None:
            self._frequency = CFactorPanels.__DEFAULT_FREQUENCY
        else:
            self._frequency = frequency

        if end_date is None:
           self._end_date = CFactorPanels.__DEFAULT_END_DATE
        else:
           self._end_date = end_date

    @property
    def index(self):
        return self.__dates if hasattr(self, '__dates') else None

    def _reset_cache(self):
        self._cache = {}

    def load_factors(self):
        for factor_name in self._factor_list:
            self.__load_single_factor(factor_name)
            self.__dates = self.get_factors_df(self._factor_list).index

    def __load_factor_from_pickles(self, factor):
        if PickleUtils.is_factor_pickled(factor, self._end_date, self._frequency):
            self._cache[factor] = PickleUtils.load_factor_from_pickles(factor, self._end_date, self._frequency)

    def __load_single_factor(self, factor_name):

        if factor_name not in self._cache.keys():
            self.__load_factor_from_pickles(factor_name)

        if factor_name not in self._cache.keys():
            if self.__is_constructed(factor_name):
                self.__construct_factor(factor_name)
            else:
                self.__load_factor_from_DB(factor_name)

    def __load_factor_from_DB(self, factor_name):
        df = gds.get_time_series_data_from_ticker(FACTOR[factor_name].value, ts_type=TimeSeriesType.RETURNS)
        df_ = df.get_periodic_returns(self._frequency)
        self._cache[factor_name] = CFactor(df_.values.flatten(), index=df_.index,
                                           name=str(factor_name), ts_type=TimeSeriesType.RETURNS, returns_type=ReturnsType.SIMPLE)
        PickleUtils.pickle_factor(self._cache[factor_name], factor_name, self._end_date, self._frequency)

    def __is_constructed(self, factor_name: str):
        return FACTOR[factor_name].value + '.py' in pkgutil.get_loader(_FACTOR_PACKAGE).contents()

    def __construct_factor(self, factor_name):
        module = importlib.import_module(_FACTOR_PACKAGE + '.' + FACTOR[factor_name].value)
        constructor = getattr(module, FACTOR[factor_name].value)
        df_ = constructor.construct_factor(frequency=self._frequency)
        self._cache[factor_name] = constructor(data=df_.loc[:self._end_date], ts_type=TimeSeriesType.RETURNS, returns_type=ReturnsType.SIMPLE)
        PickleUtils.pickle_factor(self._cache[factor_name], factor_name, self._end_date, self._frequency)

    def get_factor(self, factor_name):
        if factor_name not in self._cache.keys():
           self.__load_single_factor(factor_name)
        return self._cache.get(factor_name).deepcopy()

    def get_factor_df(self, factor):
        return self.get_factors_df(factor)

    def get_factors_df(self, factor_list):

        panel = CTimeSeries(ts_type=TimeSeriesType.RETURNS)
        for factor in self._factor_list:
            panel = panel.concat(self.get_factor(factor))
            panel.index = pd.to_datetime(panel.index)
        return panel.dropna(how='any', axis=0).sort_index().get(factor_list)

    def get_factor_historical_std(self, factor_name):
        return self.get_factor(factor_name).get_historical_volatility()

    def get_factors_historical_stds(self, factor_list: list):
        return [ self.get_factor(x).get_historical_volatility() for x in factor_list ]

    def get_factor_unorthogonalized(self):
        pass

    def get_factor_orthogonalized(self):
        pass

    def get_factors_unorthogonalized(self, factor_list, end_date=None):
        df = self.get_factors_df(factor_list).loc[:end_date]


