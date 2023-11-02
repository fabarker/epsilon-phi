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
        df_ = -1 * _datasource.get_fx_carry(currency_pairs, horizon, price_quotes='mid')
        df_.columns = df_.columns.get_level_values('bbid')
        return df_.copy().dropna(how='all')

if __name__ == "__main__":


    _G_10_CURRENCIES = ['AUD', 'CAD', 'DKK', 'JPY', 'NZD', 'NOK', 'SEK', 'CHF', 'GBP', 'DEM', 'FRF', 'ITL', 'NLG', 'BEF']

    currency_pairs = [x + '/USD' for x in _G_10_CURRENCIES]
    sig = Signals.get_CAR(currency_pairs)