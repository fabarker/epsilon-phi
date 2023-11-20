from epsilonPhi.core.factor.factorPanelInf import CFactorPanelInf
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
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


import QuantLib as ql
ql.Actual366()

_FACTOR_PACKAGE = 'epsilonPhi.core.factor.factors'
gds = GlobalDataSource()


class CFactorPanels(CFactorPanelInf):

    _cache = {}
    __DEFAULT_RISK_FACTORS = FACTOR.get_default_risk_factor_list()
    __DEFAULT_RETURN_FACTORS = FACTOR.get_default_return_factor_list()
    __DEFAULT_END_DATE = dt.date(year=2022, month=12, day=31)
    __DEFAULT_FREQUENCY = Frequency.BUSINESS_MONTHLY

    def __init__(self,
                 frequency=None,
                 end_date=None,
                 LOAD_DEFAULT_FACTOR_MODEL=False):

        super(CFactorPanels, self).__init__()

        if frequency is None:
            self._frequency = CFactorPanels.__DEFAULT_FREQUENCY
        else:
            self._frequency = frequency

        if end_date is None:
           self._end_date = CFactorPanels.__DEFAULT_END_DATE
        else:
           self._end_date = end_date

        if LOAD_DEFAULT_FACTOR_MODEL:
            self.__risk_factor_list = CFactorPanels.__DEFAULT_RISK_FACTORS
            self.__return_factor_list = CFactorPanels.__DEFAULT_RETURN_FACTORS
            self._load_factors()
        else:
            self.__risk_factor_list = None
            self.__return_factor_list = None


    @property
    def risk_factors_list(self):
        return self.__risk_factor_list

    @property
    def return_factors_list(self):
        return self.__return_factor_list

    @property
    def return_risk_factor_list(self):
        return list(set(self.return_factors_list + self.risk_factors_list))

    def set_factors_orthogonalize(self, orthogonalize_list):
        pass

    def _reset_cache(self):
        self._cache = {}

    def set_risk_factor_list(self, risk_factor_list: list):
        self.__check_factor_list(risk_factor_list)
        self.__risk_factor_list = risk_factor_list
        self._reset_cache()

    def set_return_factor_list(self, return_factors_list: list):
        self.__check_factor_list(return_factors_list)
        self.__return_factor_list = return_factors_list
        self._reset_cache()

    def append_risk_factor_to_panels(self, factor):
        assert isinstance(factor, CFactor), 'Error - factor must by of "CFactor type'

    def append_return_factor_to_panels(self, factor):
        assert isinstance(factor, CFactor), 'Error - factor must by of "CFactor type'


    def __check_factor_list(self, factor_list):
        assert isinstance(factor_list, list), 'ERROR - Must be list of return factor enums'
        boolean = [isinstance(x, FACTOR) for x in factor_list]
        assert np.all(boolean), 'Error - factor list must be a list of FACTOR enumerators'

    def _load_factors(self):
        for factor in self.return_risk_factor_list:
            self.__load_single_factor(factor.value)

    def __load_factor_from_pickles(self, factor):
        if PickleUtils.is_factor_pickled(factor, self._end_date, self._frequency):
            self._cache[factor] = PickleUtils.load_factor_from_pickles(factor, self._end_date, self._frequency)
    def __load_single_factor(self, factor):

        if factor not in self._cache.keys():
            self.__load_factor_from_pickles(factor)

        if factor not in self._cache.keys():
            if self.__isConstructed(factor):
                self.__construct_factor(factor)
            else:
                df = gds.get_time_series_data_from_ticker(factor, ts_type=TimeSeriesType.RETURNS)
                df.columns = [factor]
                self._cache[factor] = CFactor(df.get_periodic_returns(self._frequency))
                PickleUtils.pickle_factor(self._cache[factor],
                                          factor,
                                          self._end_date,
                                          self._frequency)

    def __isConstructed(self, factor_name: str):
        return factor_name + '.py' in pkgutil.get_loader(_FACTOR_PACKAGE).contents()

    def __construct_factor(self, factor):
        module = importlib.import_module(_FACTOR_PACKAGE + '.' + factor)
        constructor = getattr(module, factor)
        df_ = constructor.construct_factor(frequency=self._frequency)
        self._cache[factor] = constructor(dataframe=df_)
        PickleUtils.pickle_factor(self._cache[factor], factor, self._end_date, self._frequency)

    def get_risk_factor_panel(self):
        return self.get_factors_panel().get([x.name for x in self.risk_factors_list])


    def get_return_factors_panel(self):
        return self.get_factors_panel().get([x.name for x in self.return_factors_list])

    def get_factor_df(self, factor):
        if factor not in self._cache.keys():
            self.__load_single_factor(factor)
        return self._cache.get(factor)

    def get_factors_panel(self):

        panel = CTimeSeries(ts_type=TimeSeriesType.RETURNS)
        for factor in self._cache.keys():
            panel = panel.concat(self.get_factor_df(factor))
        return panel.copy()

    def get_risk_factor_covariance(self, dates):
        return self.get_risk_factor_panel().loc[dates].cov() * 12


    def get_return_factor_Sharpe_ratios(self):
        pass

    def get_risk_factor_betas(self):
        pass

    def get_return_factor_betas(self):
        pass

    def cap_factor_Sharpe(self, factor, cap):
        pass


if __name__ == "__main__":

    self = CFactorPanels(frequency=Frequency.BUSINESS_MONTHLY,
                         LOAD_DEFAULT_FACTOR_MODEL=True)
    panel_ = self.get_factors_panel()
