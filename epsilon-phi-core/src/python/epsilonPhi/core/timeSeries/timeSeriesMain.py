from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
import datetime as datetime
import dateutil
import numpy as np
import pandas as pd

class CTimeSeries(object):

    _cache = {}
    _datasource = GlobalDataSource()

    def __init__(self, df=None, symbol=None):

        self._df = df.copy()
        self._symbol = symbol

    @staticmethod
    def get_time_series(ticker,
                        datafields=None,
                        from_time=None):
        pass


    @property
    def columns(self):
        return self._df.columns

    @property
    def dataframe(self):
        return self._df.copy()

    @property
    def dates(self):
        return pd.to_datetime(self._df.index)

    def get_datafield(self, datafield):
        return self._df.get(datafield)







