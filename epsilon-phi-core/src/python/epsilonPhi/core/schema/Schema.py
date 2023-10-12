from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.utils.DateUtils import DateUtils
from epsilonPhi.core.config.appConfig import CAppConfig
from epsilonPhi.core.env.Env import DATAVERSION, Env
import datetime

__author__ = 'Francis Barker'
__date__ = '01/07/2023'

_Env = Env.get_env()
_START_DATE = None
_END_DATE = None

class CContext(object):

    def __init__(self, currency: str,
                       frequency: Frequency.MONTHLY,
                       data_version: int,
                       start_date: datetime.date,
                       end_date: datetime.date):

        self.__currency = currency
        self.__frequency = frequency
        self.__data_version = data_version
        self.__start_date = start_date
        self.__end_date = end_date
        self.__dates = None

    @property
    def dataversion(self):
        return self.__data_version

    @property
    def start_date(self):
        return self.__start_date

    @property
    def end_date(self):
        return self.__end_date

    @property
    def frequency(self):
        return self.__frequency

    @property
    def annualizing_factor(self):
        return self.frequency.yearfrac()

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
                                                  self.dataverson)
    @property
    def risk_free_rate(self):
        return CAppConfig._configUtil.get_currency_config(self.currency,
                                                          self.frequency,
                                                          self.dataverson).risk_free_rate
    @staticmethod
    def get_risk_free_rate_ticker(currency, frequency, dataversion):
        return CAppConfig._configUtil.get_currency_config(currency,
                                                          frequency,
                                                          dataversion).risk_free_ticker
    @property
    def inflation_rate_ticker(self):
        return CContext.get_inflation_rate_ticker(self.currency,
                                                  self.frequency,
                                                  self.dataverson)
    @property
    def inflation_rate(self):
        return CAppConfig._configUtil.get_currency_config(self.currency,
                                                          self.frequency,
                                                          self.dataverson).inflation_rate

    @staticmethod
    def get_inflation_rate_ticker(currency, frequency, dataversion):
        return CAppConfig._configUtil.get_currency_config(currency,
                                                          frequency,
                                                          dataversion).inflation_ticker

    def get_risk_factor_covariance_matrix(self):
        pass

    def get_factor_panels(self):
        pass

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
                                                          self.dataverson)
    def get_simulation_config(self):
        return CAppConfig._configUtil.get(self.currency,
                                          self.frequency,
                                          self.dataverson)

    def get_estimation_config(self):
        return CAppConfig._configUtil.get_estimation_config()

    def get_factor_config(self):
        return CAppConfig._configUtil.get_factor_config()



class ContextCreator:

    def __init__(self,
                 currency='USD',
                 frequency=Frequency.MONTHLY,
                 dataversion=DATAVERSION,
                 start_date=None,
                 end_date=None):

        self._currency = currency
        self._frequency = frequency
        self._dataversion = dataversion
        self._start_date = start_date
        self._end_date = end_date

    def create_context(self):
        schema = CContext(self._currency,
                                self._frequency,
                                self._dataversion,
                                self._start_date,
                                self._end_date)
        schema._setup()
        return schema

if __name__ == "__main__":

    self = ContextCreator(currency='GBP',
                            start_date='31-Dec-1999',
                            end_date='31-Dec-2022').create_context()
    rfr = self.get_risk_free_rate('USD',Frequency.MONTHLY,1)






