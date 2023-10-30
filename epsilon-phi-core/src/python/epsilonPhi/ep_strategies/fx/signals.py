import pandas as pd

from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource

__author__ = 'Francis Barker'
__date__ = 'Mon 30 Oct 16:25'


_datasource = GlobalDataSource()

class Signals(object):

    def __init__(self):
        pass

    @staticmethod
    def get_CAR(currency_pairs, horizon='1m'):

        signal = CTimeSeries()
        for currency_pair in currency_pairs:
            carry = _datasource.get_fx_carry(currency_pair, horizon, price_quotes='mid')
            signal = pd.concat((signal, carry), axis=1)
        return signal.copy()


_G_10_CURRENCIES = ['AUD', 'CAD', 'DKK', 'JPY', 'NZD', 'NOK', 'SEK', 'CHF', 'GBP', 'DEM', 'FRF', 'ITL', 'NLG', 'BEF']