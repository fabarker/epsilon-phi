from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.timeSeries.timeSeriesMain import CSlice, CTimeSeries
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.estimator.estimationMgr import EstimationMgr
from epsilonPhi.core.factor.factorPanel import CFactorPanels
from epsilonPhi.core.modelFactory.modelFactory import BaseModel
from epsilonPhi.core.utils.DateUtils import DateUtils
from epsilonPhi.core.config.appConfig import CAppConfig
from epsilonPhi.core.env.Env import DATAVERSION, Env
from collections import OrderedDict
from dateutil import parser
from datetime import datetime as dt
from scipy.stats import zscore
import datetime
import numpy as np
import pandas as pd

__author__ = 'Francis Barker'
__date__ = '01/07/2023'

_Env = Env.get_env()
_START_DATE = None
_END_DATE = None
_FACTOR_SHARPE_END_DATE = datetime.datetime(2018, 12, 31)

import hashlib
import json

class Hashable(object):

    def _hashable_state(self):
        raise NotImplementedError("Subclasses must implement _hashable_state()")

    def __hash__(self):
        try:
            state = self._hashable_state()
            state_str = json.dumps(state, sort_keys=True, default=str)
            return hashlib.sha256(state_str.encode("utf-8")).hexdigest()
        except Exception as e:
            raise TypeError(f"{self.__class__.__name__} instance is not hashable: {e}")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self._hashable_state() == other._hashable_state()

    def __repr__(self):
        return f"<CContext hash={self.__hash__()}>"

class CContext(Hashable):

    def __init__(self, currency: str,
                       frequency: Frequency.BUSINESS_MONTHLY,
                       data_version: int,
                       start_date: dt.date,
                       end_date: dt.date):

        self.__currency = currency
        self.__frequency = frequency
        self.__data_version = data_version
        self.__dates = None
        self.__asset_manager = None
        self.__crisis_map = None
        self.__model = None
        self.__beta_mult_ts = None

        self.start_date = start_date
        self.end_date = end_date

    def _hashable_state(self):

        """
        Return a dict of the key internal attributes to be used for hashing.
        """

        return {
            "currency": str(self.__currency),
            "data_version": str(self.__data_version),
            "frequency": str(self.__frequency),
            "start_date": str(self.start_date),
            "end_date": str(self.end_date),
            "model": self.__model.__hash__(),
        }


    @property
    def dataversion(self):
        return self.__data_version

    @property
    def start_date(self):
        return self.__start_date

    @start_date.setter
    def start_date(self, value):
        self.__start_date = parser.parse(value)

    @property
    def end_date(self):
        return self.__end_date

    @end_date.setter
    def end_date(self, value):
        self.__end_date = parser.parse(value)

    @property
    def frequency(self):
        return self.__frequency

    @property
    def annualizing_factor(self):
        return self.frequency.yearfrac()

    @property
    def obs_per_year(self):
        return self.frequency.obs_per_year()

    @property
    def dt(self):
        return 1 / self.obs_per_year

    @property
    def dates(self):
        if self.__dates is None:
            self.load_dates()
        return self.__dates

    @property
    def currency(self):
        return self.__currency

    @property
    def risk_free_rate_ticker(self):
        return CContext.get_risk_free_rate_ticker(self.currency,
                                                  self.frequency,
                                                  self.dataversion)
    @property
    def risk_free_rate(self):
        return CAppConfig._configUtil.get_currency_config(self.currency,
                                                          self.frequency,
                                                          self.dataversion).risk_free_rate
    @property
    def medium_term_risk_free_rate(self):
        return CAppConfig._configUtil.get_currency_config(self.currency,
                                                          self.frequency,
                                                          self.dataversion).medium_risk_free_rate

    @property
    def curr_risk_free_rate(self):
        return CAppConfig._configUtil.get_currency_config(self.currency,
                                                          self.frequency,
                                                          self.dataversion).medium_risk_free_rate

    @staticmethod
    def get_risk_free_rate_ticker(currency, frequency, dataversion):
        return CAppConfig._configUtil.get_currency_config(currency,
                                                          frequency,
                                                          dataversion).risk_free_ticker


    @property
    def inflation_rate_ticker(self):
        return CContext.get_inflation_rate_ticker(self.currency,
                                                  self.frequency,
                                                  self.dataversion)


    @property
    def BaseModel(self):
        return self.__model

    def set_model(self, model):
        self.__model = model

    def get_inflation_rate_asset(self):
        from epsilonPhi.core.asset.AssetMgr import CAssetMgr
        return CAssetMgr(self).get_inflation_asset(self.currency).reindex(self.dates)

    def get_price_deflator(self):
        infl = self.get_inflation_rate_asset()
        return infl.add(1).cumprod().div(infl.values[0] + 1)

    def get_risk_free_rate_asset(self):
        from epsilonPhi.core.asset.AssetMgr import CAssetMgr
        return CAssetMgr(self).get_risk_free_asset(self.currency).reindex(self.dates)

    @staticmethod
    def get_inflation_rate_ticker(currency, frequency, dataversion):
        return CAppConfig._configUtil.get_currency_config(currency,
                                                          frequency,
                                                          dataversion).inflation_ticker

    def get_risk_factor_covariance(self):
        return self.BaseModel.get_risk_factor_covariance(
            self.dates[self.dates <= _FACTOR_SHARPE_END_DATE],
            True) * self.obs_per_year

    def get_factor_panels(self):
        return self.BaseModel.factor_panels

    def get_return_factors_sharpe_ratios(self):
        return self.BaseModel.get_return_factor_Sharpe_ratios()

    def get_return_factors_sharpe_ratio_uncapped(self):
        return self.BaseModel.get_return_factor_Sharpe_ratios_uncapped()

    def get_risk_factors_panel(self):
        return self.BaseModel.get_risk_factor_df()\
            .select_subset_dates(self.dates)

    def get_return_factors_panel(self):
        return self.BaseModel.get_return_factor_df() \
            .select_subset_dates(self.dates)

    def get_orthog_return_factors_panel(self):
        return self.BaseModel.get_return_factor_df(orthogonalize=True) \
            .select_subset_dates(self.dates)

    def get_normalized_return_factors_panel(self):
        return zscore(self.get_return_factors_panel(), ddof=1)

    def _setup(self):
        self.__load_configs()
        self.load_dates()

    def __load_configs(self):
        CAppConfig.setup(app=_Env)

    def load_dates(self):
        self.__dates = DateUtils.get_date_range(self.start_date,
                                                self.end_date,
                                                self.frequency)

    def get_currency_config(self):
        return CAppConfig._configUtil.get_currency_config(self.currency,
                                                          self.frequency,
                                                          self.dataversion)
    def get_simulation_config(self):
        return CAppConfig._configUtil.get_simulation_config(self.currency,
                                                            self.dataversion)

    def get_estimation_config(self):
        return CAppConfig._configUtil.get_estimation_config()

    def get_factor_config(self):
        return CAppConfig._configUtil.get_factor_config()

    def get_factor_crisis_map(self):

        if self.__crisis_map is None:

            from epsilonPhi.core.simulation.SimStructs import Crisis

            # Map is an ordered dictionary
            map = OrderedDict()

            crises = self.get_factor_crises()
            for crisis in crises:
                map[crisis.crisis_name] = Crisis(
                            crisis.crisis_name,
                            crisis.crisis_start_date,
                            crisis.crisis_end_date
                )

            self.__crisis_map = map
        return self.__crisis_map

    def get_stress_coeff_ts(self):
        if self.__beta_mult_ts is None:
            from epsilonPhi.core.simulation.SAASimulation import SAASimulation
            SAASimulation.set_beta_multipliers(self)

            dates = self.dates
            crises = self.get_factor_crisis_map()

            arr = np.zeros((len(dates), 1))
            for crisis in crises:
                mask = ((dates > crises[crisis]._start_date) &
                        (dates <= crises[crisis]._end_date))
                arr[mask] = crises[crisis]._stress_coefficient
            self.__beta_mult_ts = CTimeSeries(arr, index=dates, columns=['stress_coeff'])

        return self.__beta_mult_ts


    def get_factor_crises(self):
        return CAppConfig._configUtil.get_factor_crises(
            self.start_date,
            self.end_date,
            False)

    def __load_asset_manager(self):
        if self.__asset_manager is None:
            from epsilonPhi.core.asset.AssetMgr import CAssetMgr
            self.__asset_manager = CAssetMgr(self)

    def get_asset_manager(self):
        if self.__asset_manager is None:
            self.__load_asset_manager()
        return self.__asset_manager

    def get_asset_from_name(self, name):
        from epsilonPhi.core.asset.AssetMgr import CAssetMgr
        return CAssetMgr(self).get_asset_by_name(name)

    def get_extended_schema(self):

        schema = CContext(self.currency,
                 self.frequency,
                 self.dataversion,
                 '27-Feb-1970',
                 self.end_date.strftime("%d-%m-%Y"))

        schema._setup()
        mdl = CAppConfig.get_BaseModel().setup_extended_model(
            schema.frequency,
            schema.end_date
        )
        schema.set_model(mdl)
        return schema




class ContextCreator:

    def __init__(self,
                 currency='USD',
                 frequency=Frequency.BUSINESS_MONTHLY,
                 dataversion=DATAVERSION,
                 start_date=None,
                 end_date=None):

        self._currency = currency
        self._frequency = frequency
        self._dataversion = dataversion
        self._start_date = start_date
        self._end_date = end_date

    def create_context(self):
        self._schema = CContext(self._currency,
                                self._frequency,
                                self._dataversion,
                                self._start_date,
                                self._end_date)
        self._schema._setup()
        self.__load_default_model()
        return self._schema

    def __load_default_model(self):
        CAppConfig.get_BaseModel().setup_default_model(self._frequency,
                                                       self._end_date,
                                                       cache_model=True)
        self._schema.set_model(CAppConfig.get_BaseModel())



if __name__ == "__main__":

    schema = ContextCreator(currency='GBP',
                            start_date='31-Dec-1999',
                            end_date='31-Dec-2022').create_context()

    ex_schema = schema.get_extended_schema()









