from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
import datetime as datetime
import dateutil
import numpy as np
import pandas as pd

class CTimeSeries(pd.DataFrame):

    _datasource = GlobalDataSource()
    @property
    def _constructor(self):
        """This is the key to letting Pandas know how to keep
        derivative `SomeData` the same type as yours.  It should
        be enough to return the name of the Class.  However, in
        some cases, `__finalize__` is not called and `my_attr` is
        not carried over.  We can fix that by constructing a callable
        that makes sure to call `__finlaize__` every time."""
        def _c(*args, **kwargs):
            return CTimeSeries(*args, **kwargs).__finalize__(self)
        return _c

    def __init__(self,
                 dataframe: pd.DataFrame,
                 ts_type: TimeSeriesType = TimeSeriesType.LEVELS):

        super(CTimeSeries, self).__init__(dataframe)
        self._attributes = pd.DataFrame()
        self.__ts_type = ts_type
        self.validate()

    # check we have time series data
    def validate(self):
        assert isinstance(self.index, pd.DatetimeIndex),\
            'Error - index must be pd.DatetimeIndex'

    #%% Properties

    @property
    def type(self):
        return self.__type
    @property
    def dates(self):
        return pd.to_datetime(self.index)
    @property
    def data(self):
        return self.values
    @property
    def data_length(self):
        return self.index.__len__()
    @property
    def number_of_cols(self):
        return self.columns.size
    @property
    def frequency(self):
        if ts.index.index:
            return ts.index.freq
        else:
            return ts.index.inferred_freq

    #%% setter functions




    #%% getter functions

    def get_levels(self):
        if self.type == TimeSeriesType.LEVELS:
            return self.copy()
        else:
            return (1+self).cumprod()

    def get_returns(self, return_type='simple'):
        if self.type == TimeSeriesType.LEVELS:
            return self.pct_change().copy()
        else:
            return self.copy()

    #%% public methods
    def select_subset_dates(self, dates):
        return self.loc[dates].copy()
    def insert_and_select_subset_dates(self, dates):
        return self.reindex[dates].copy()
    def select_subset_columns(self, columns):
        return self.select_subset_labels(self.columns[columns])
    def select_subset_labels(self, labels):
        return self.get(labels).copy()
    def intersect_over_dates(self, df):
        common_dates = np.intersect1d(self.dates, df.index)
        return self.select_subset_dates(common_dates), df.loc[common_dates].copy()
    def intersect_over_date_range(self, df):
        common_dates = np.intersect1d(self.dates, df.index)
        return self[np.min(common_dates):np.max(common_dates)], df[np.min(common_dates):np.max(common_dates)]

    def get_period_levels(self, period):
        pass

    def get_period_returns(self, period):
        pass

    #%% Protected Methods

    def _backfill_levels(self, df):
        pass

    def _backfill_returns(self, df):
        pass

    #%% Private Methods


    #%% Static Methods
    @staticmethod
    def get_timeseries_from_ticker(ticker, fields=None):
        df = GlobalDataSource().get_dataframe_from_ticker(ticker,
                                                          cols=fields,
                                                          index_col='date')
        return CTimeSeries(df)


if __name__ == "__main__":

    self = GlobalDataSource()
    ts = CTimeSeries.get_timeseries_from_ticker('UKPRATE.',fields='IR')






