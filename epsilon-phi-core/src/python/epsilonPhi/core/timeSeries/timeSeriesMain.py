from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.utils.FrameUtils import FrameUtils
from epsilonPhi.core.utils.DateUtils import DateUtils
import numpy as np
import datetime as datetime
import dateutil
import pandas as pd

class CTimeSeries(pd.DataFrame):

    _datasource = GlobalDataSource()
    _metadata = ["_added_attributes", "_type"]
    @property
    def _constructor(self):
        """This is the key to letting Pandas know how to keep
        derivative `SomeData` the same type as yours.  It should
        be enough to return the name of the Class.  However, in
        some cases, `__finalize__` is not called and `my_attr` is
        not carried over.  We can fix that by constructing a callable
        that makes sure to call `__finlaize__` every time."""
        def _c(*args, **kwargs):
            return CTimeSeries(*args).__finalize__(self)
        return _c

    def __init__(self,
                 dataframe: pd.DataFrame,
                 ts_type: TimeSeriesType = TimeSeriesType.LEVELS,
                 attributes: pd.DataFrame = None):

        super(CTimeSeries, self).__init__(dataframe)

        # Set attributes in object
        if isinstance(attributes, pd.DataFrame):
            self.set_attributes(attributes)
        else:
            self.reset_attributes()

        self.__setattr__('_type', ts_type)
        self.validate()

    # check we have time series data
    def validate(self):
        assert isinstance(self.index, pd.DatetimeIndex),\
            'Error - index must be pd.DatetimeIndex'

    def _cast_derived_class(self, klass):
        self.__init__(klass, klass._type, klass.attributes)

    #%% Properties
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
        if self.index.freq:
            return self.index.freq
        else:
            return self.index.inferred_freq
    @property
    def attributes(self):
        return self.__getattr__('_added_attributes').get(self.columns)
    @property
    def type(self):
        return self.__getattr__('_type')

    #%% setter functions

    #%% getter functions

    def get_levels(self):
        if self.type == TimeSeriesType.LEVELS:
            return self.copy()
        else:
            levels = np.cumprod((1+self._constructor(self, TimeSeriesType.LEVELS, self.attributes)))

        pd.tseries.offsets.DateOffset(1)


    def get_returns(self, return_type='simple'):
        if self.type == TimeSeriesType.LEVELS:
            return self.pct_change().copy()
        else:
            return self.copy()

    #%% public methods
    def select_subset_dates(self, dates):
        return self.loc[dates].copy()
    def insert_and_select_subset_dates(self, dates, fill_na=False):

        reindexed = self.reindex(self.index.append(dates).unique())
        if fill_na:
           return reindexed.ffill().reindex(dates).copy()
        else:
            return reindexed.reindex(dates).copy()

    def select_subset_columns(self, columns):
        return self.select_subset_labels(self.columns[columns])
    def select_subset_labels(self, labels):
        return self.get(labels).copy()
    def select_subset_attribute(self, attribute_name, attribute_values):
        idx = self.attributes.loc[attribute_name].isin([attribute_values]).values
        return self.iloc[:, idx]

    def select_subset_year(self, year):
        return self.loc[self.index.year == year]

    def select_subset_month(self, month):
        return self.loc[self.index.month == month]

    def select_subset_month_year(self, month, year):
        return self.loc[np.logical_and(self.index.month == month,
                                       self.index.year == year)]

    def insert_dates(self, dates):
        self._cast_derived_class(self.reindex(self.index.append(dates).unique()))

    def intersect_over_dates(self, df):
        common_dates = np.intersect1d(self.dates, df.index)
        return self.select_subset_dates(common_dates), df.loc[common_dates].copy()
    def intersect_over_date_range(self, df):
        common_dates = np.intersect1d(self.dates, df.index)
        return self[np.min(common_dates):np.max(common_dates)], \
            df[np.min(common_dates):np.max(common_dates)]

    def get_periodic_levels(self, period):
        pass

    def get_periodic_returns(self, period):
        pass

    def get_period_ends(self, frequency):
        period_dates = DateUtils.get_date_range(np.min(self.dates), np.max(self.dates), periodicity=frequency)
        return self.select_subset_dates(period_dates)

    def get_week_ends(self):
        return self.get_period_ends(Frequency.WEEKLY)

    def get_month_ends(self):
        return self.get_period_ends(Frequency.MONTHLY)

    def get_quarter_ends(self):
        return self.get_period_ends(Frequency.QUARTERLY)

    def get_year_ends(self):
        return self.get_period_ends(Frequency.YEARLY)

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
        self.set_attributes(pd.DataFrame(columns=self.columns))

    def set_attributes(self, attributes: pd.DataFrame):
        self.__setattr__('_added_attributes',
                         attributes.get(self.columns, pd.DataFrame(columns=self.columns)))

    def set_attribute(self, attribute_name, attribute_vals):

        if not FrameUtils.is_iterable(attribute_vals):
            attribute_vals = [attribute_vals]

        N = len(attribute_vals)
        if N == 1:
            attribute_vals = [attribute_vals] * self.number_of_cols

        if len(attribute_vals) == self.number_of_cols:
            new_att = pd.DataFrame(attribute_vals, columns=self.columns, index=[attribute_name])
            self.append_attributes(new_att)

    def add_attributes(self, attributes: pd.DataFrame):
        self.append_attributes(attributes)

    def add_attribute(self, attribute_name, attribute_vals):

        if not FrameUtils.is_iterable(attribute_vals):
           attribute_vals = [attribute_vals]

        N = len(attribute_vals)
        if N == 1:
            attribute_vals = [attribute_vals] * self.number_of_cols

        if len(attribute_vals) == self.number_of_cols:
            new_att = pd.DataFrame(attribute_vals, columns=self.columns, index=[attribute_name])
            self.append_attributes(new_att)

    def append_attributes(self, new_attributes: pd.DataFrame):
        new_atts = pd.concat((self.attributes, new_attributes), axis=0)
        self.set_attributes(new_atts)

    def get_attributes(self, attribute_name):
        return self.attributes.loc[attribute_name]

    def sort_by_attribute(self, attribute_name, sort_ascending=False):
        pass #self.attributes.loc[attribute_name].sort_values()

    def concat(self, time_series):
        assert self.type == time_series.type, 'ERROR - timeseries must be of the same type to concat'
        new_atts = pd.concat((self.attributes, time_series.attributes), axis=1)
        return CTimeSeries(pd.concat((self, time_series), axis=1), self.type, new_atts)

    def isLevels(self):
        return self.type == TimeSeriesType.LEVELS

    def isReturns(self):
        return self.type == TimeSeriesType.RETURNS or self.type == TimeSeriesType.GROWTH

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
    def remove_empty_leading_rows(self):
        pass
    def remove_empty_trailing_rows(self):
        pass
    def remove_empty_leading_trailing_rows(self):
        pass
    def multiply(self, value):
        return self.copy() * value
    def divide(self, value):
        return self.copy() / value
    def subtract(self, value):
        return self.copy() - value
    def add(self, value):
        return self.copy() + value

    #%% Private Methods


    #%% Static Methods
    @staticmethod
    def get_timeseries_from_ticker(ticker, fields=None):
        df = GlobalDataSource().get_dataframe_from_ticker(ticker,
                                                          cols=fields,
                                                          index_col='date')
        return CTimeSeries(df)


if __name__ == "__main__":

    class test_class(pd.DataFrame):
        _metadata = ["__added_property"]

        @property
        def _constructor(self):
            return test_class


    df = test_class([2, 4])
    df.__added_property = 'test'

    self = CTimeSeries.get_timeseries_from_ticker('UKPRATE.', fields='x')
    self = self / 100






