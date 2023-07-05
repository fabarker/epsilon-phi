import random
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.utils.DateUtils import DateUtils

__author__ = 'Francis Barker'
__date__ = '01/07/2023'

class CContext(object):

    def __init__(self, currency,
                       frequency,
                       start_date,
                       end_date):

        self._currency = currency
        self._frequency = frequency

        self._start_date = start_date
        self._end_date = end_date
        self._dates = None

    def _setup(self):
        self.load_dates()

    def load_dates(self):
        self._dates = DateUtils.get_date_range(self._start_date,
                                               self._end_date,
                                               self._frequency)

    #%% Setter Methods

    def set_risk_model(self):
        pass

    def set_return_model(self):
        pass

    def set_factor_panels(self):
        pass


    #%% Getter Methods

    def get_currency(self):
        return self._currency

    def get_frequency(self):
        return self._frequency

    def get_dates(self):
        if self._dates is None:
            self.load_dates()
        return self._dates

    def set_risk_free_rate(self, currency, asset_name):
        pass

    def get_risk_free_rate(self, currency):
        pass

    def get_risk_free_rate_name(self, currency):
        pass



