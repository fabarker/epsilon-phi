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
        if ts.index.freq:
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
    def select_subset_attribute(self, attribute_name, attribute_values):
        pass

    def select_subset_year(self, year):
        pass

    def select_subset_month(self, month):
        pass

    def select_subset_month_year(self, month, year):
        pass

    def insert_dates(self, dates):
        pass

    def intersect_over_dates(self, df):
        common_dates = np.intersect1d(self.dates, df.index)
        return self.select_subset_dates(common_dates), df.loc[common_dates].copy()
    def intersect_over_date_range(self, df):
        common_dates = np.intersect1d(self.dates, df.index)
        return self[np.min(common_dates):np.max(common_dates)], df[np.min(common_dates):np.max(common_dates)]

    def get_periodic_levels(self, period):
        pass

    def get_periodic_returns(self, period):
        pass

    def get_period_ends(self, frequency):
        pass

    def get_week_ends(self):
        return self.get_period_ends('W')

    def get_month_ends(self):
        return self.get_period_ends('M')

    def get_quarter_ends(self):
        return self.get_period_ends('Q')

    def get_year_ends(self):
        return self.get_period_ends('Y')

    def get_weekly_levels(self):
        pass
    def get_weekly_returns(self):
        pass
    def get_monthly_levels(self):
        pass
    def get_monthly_returns(self):
        pass
    def get_quarterly_levels(self):
        pass
    def get_quarterly_returns(self):
        pass
    def get_annual_levels(self):
        pass
    def get_annual_returns(self):
        pass

    #%% Methods assocaited with attributes
    def reset_attributes(self):
        self._attributes = pd.DataFrame()

    def set_attributes(self, attributes: pd.DataFrame):
        self._attributes = attributes

    def set_attribute(self, attribute_name, attribute_vals):
        pass

    def add_attributes(self, attributes):
        pass

    def add_attribute(self, attribute_name, attribute_vals):
        pass

    def append_attributes(self, new_attribute):
        pass

    def get_attribute(self, attribute_name):
        pass

    def sort_by_attribute(self, attribute_name, sort_ascending=False):
        pass

    def merge(self, time_series):
        pass

    def isLevels(self):
        pass

    def isReturns(self):
        pass

    def ind(self, ind_value):
        pass

    #%% Protected Methods

    def _backfill_levels(self, df):
        pass

    def _backfill_returns(self, df):
        pass

    def _frontfill_levels(self, df):
        pass
    def _frontfill_returns(self, df):
        pass
    @staticmethod
    def concatenate(d1, d2, cutoff_date=None):
        pass
    def remove_empty_leading_rows(self):
        pass
    def remove_empty_trailing_rows(self):
        pass
    def remove_empty_leading_trailing_rows(self):
        pass

    def multiply(self, df):
        pass

    def divide(self, df):
        pass

    def subtract(self, df):
        pass

    def add(self, df):
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






