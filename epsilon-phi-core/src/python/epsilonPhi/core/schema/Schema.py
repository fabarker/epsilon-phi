from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.utils.DateUtils import DateUtils
import datetime
import random

__author__ = 'Francis Barker'
__date__ = '01/07/2023'

class CContext(object):

    def __init__(self, currency: str,
                       frequency: Frequency,
                       data_version: int,
                       start_date: datetime.date,
                       end_date: datetime.date):

        self.__currency = currency
        self.__frequency = frequency
        self.__data_version = data_version
        self.__start_date = start_date
        self.__end_date = end_date
        self.__dates = None

        # setup schema
        self._setup()


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
    def dates(self):
        if self.__dates is None:
            self.__load_dates()
        return self.__dates

    def _setup(self):
        self.load_dates()
        self.load_configs()
        self.load_asset_manager()

    def load_configs(self):
        pass

    def load_dates(self):
        self.__dates = DateUtils.get_date_range(self.start_date,
                                                self.end_date,
                                                self.frequency)

    def load_asset_manager(self):
        pass

    #%% Setter Methods
    def set_risk_factor_panels(self):
        pass

    def set_return_factor_panels(self):
        pass

    def set_risk_free_rate(self, currency, asset_name):
        pass

    def get_risk_free_rate(self, currency):
        pass

    def get_risk_free_rate_name(self, currency):
        pass

    def get_asset_by_name(self, asset_name):
        pass




