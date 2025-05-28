from __future__ import annotations
from epsilonPhi.core.timeSeries.timeSeriesInf import slice_metadata, metadata
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.utils.FrameUtils import FrameUtils
from epsilonPhi.core.utils.DateUtils import DateUtils
from typing import Optional, Union
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

class CSlice(pd.Series):

    __pandas_priority__ = 5000
    _metadata = slice_metadata

    @property
    def _constructor(self):
        """This is the key to letting Pandas know how to keep
        derivative `SomeData` the same type as yours.  It should
        be enough to return the name of the Class.  However, in
        some cases, `__finalize__` is not called and `my_attr` is
        not carried over.  We can fix that by constructing a callable
        that makes sure to call `__finlaize__` every time."""
        def _c(*args, **kwargs):
            return CSlice(*args, **kwargs).__finalize__(self)
        return _c

    @property
    def _constructor_expanddim(self):
        def _cs(*args, **kwargs):
            return CTimeSeries(*args, **kwargs).__finalize__(self)
        return _cs

    def __init__(self,
                 data: Optional[Union[pd.Series, CSlice]] = None,
                 ts_type: Optional[TimeSeriesType] = None,
                 returns_type: Optional[ReturnsType] = None,
                 **kwargs):

        super(CSlice, self).__init__(data=data, **kwargs)

        # Set attributes in object
        self.__setattr__('_type', ts_type)
        self.__setattr__('_returns_type', returns_type)



    #############

    def _cast_derived_class(self, klass):
        self.__init__(klass, ts_type=klass.type, returns_type=klass.returns_type)

    def create_new_object(self, *args, **kwargs):
        return self.__class__(*args, **kwargs)

    def deepcopy(self):
        return self.create_new_object(data=self, ts_type=self.type, returns_type=self.returns_type)

    def _create_new_object_same_type(self, data=None, attributes=None, returns_type=None):
        return self.create_new_object(data=data, ts_type=self.type, returns_type=returns_type)

    def _create_new_levels_object(self, data=None, attributes=None, returns_type=None):
        return self.create_new_object(data=data, ts_type=TimeSeriesType.LEVELS, returns_type=returns_type)

    def _create_new_returns_object(self, returns_type, data=None, attributes=None):
        return self.create_new_object(data=data, ts_type=TimeSeriesType.RETURNS, returns_type=returns_type)

    ##############

    @property
    def dates(self):
        return pd.to_datetime(self.index)

    @property
    def data(self):
        return self.values

    @property
    def obs_per_year(self):
        return Frequency(self.frequency).obs_per_year()

    @property
    def data_length(self):
        return self.index.__len__()

    @property
    def number_of_cols(self):
        return None

    @property
    def frequency(self):
        if self.dates.freq:
            return self.dates.freq.name
        elif self.dates.inferred_freq:
            return self.dates.inferred_freq
        else:
            return DateUtils.get_daterange_frequency(self.dates)

    @property
    def type(self):
        return self.__getattr__('_type')

    @property
    def returns_type(self):
        return self.__getattr__('_returns_type')

    @property
    def is_levels(self):
        return (self.type ==
                TimeSeriesType.LEVELS)

    @property
    def is_returns(self):
        return self.type in [TimeSeriesType.RETURNS,
                             TimeSeriesType.GROWTH]

    def get_levels(self):

        if self.is_levels:
            return self.deepcopy()

        idx_nan = self.isna().values
        if self.returns_type in [ReturnsType.SIMPLE,
                                 ReturnsType.SIMPLE.value]:
            newobj = (1 + self).cumprod(axis=0)
            insert_val = 1
        elif self.returns_type in [ReturnsType.LOG,
                                   ReturnsType.LOG.value]:
            newobj = np.exp(self.cumsum(axis=0))
            insert_val = 1
        elif self.returns_type in [ReturnsType.DIFFERENCE,
                                   ReturnsType.DIFFERENCE.value]:
            newobj = self.cumsum(axis=0)
            insert_val = 0
        else:
            raise ValueError('ERROR: {} not supported'.format(self.returns_type))

        # Update the time series type
        newobj.__setattr__('_type', TimeSeriesType.LEVELS)
        newobj.values[idx_nan] = np.nan
        newobj.insert_date_val(DateUtils.shift_date(self.first_valid_index(), newobj.frequency, -1), insert_val)
        return self._create_new_levels_object(newobj, returns_type=newobj.returns_type)

    def get_returns(self, return_type=ReturnsType.SIMPLE):

        if return_type is None:
           return_type = self.returns_type

        if (self.is_returns and
                self.returns_type == return_type):
            return self.deepcopy()


        copyobject = self.remove_empty_leading_trailing_rows().get_levels()
        nan_locs = copyobject.isna().values
        if return_type in [ReturnsType.SIMPLE,
                           ReturnsType.SIMPLE.value]:
            newobj = copyobject.ffill().pct_change().dropna()
        elif return_type in [ReturnsType.LOG,
                             ReturnsType.LOG.value]:
            newobj = np.log(copyobject.ffill()).diff().dropna()
        elif return_type in [ReturnsType.DIFFERENCE,
                             ReturnsType.DIFFERENCE.value]:
            newobj = copyobject.ffill().diff().dropna()
        else:
            raise ValueError('Error - must specify returns type')

        # Update the time series type
        newobj.__setattr__('_returns_type', return_type)
        rtns_locs = np.diff(np.cumsum(nan_locs, axis=0), axis=0) != 0
        newobj.values[rtns_locs] = np.nan
        return self._create_new_returns_object(return_type, data=newobj)

    def insert_dates(self, dates):
        unique_dates = DateUtils.merge([self.dates, dates])
        df_ = self.reindex(unique_dates).sort_index()
        self._cast_derived_class(df_)

    def insert_date(self, date):
        if not DateUtils.is_iterable(date):
            date = pd.DatetimeIndex([date])

        insert_dates = pd.to_datetime(date)
        self.insert_dates(insert_dates)

    def insert_date_val(self, date, val):
        self.insert_date(date)
        self[date] = val

    def select_subset_dates(self, dates):
        return self.loc[dates].deepcopy()

    def insert_and_select_subset_dates(self, dates, fill_na=False):

        unique_dates = DateUtils.merge([self.dates, dates])
        if fill_na:
            return self.reindex(unique_dates).ffill().reindex(dates)
        else:
            return self.reindex(unique_dates).reindex(dates)

    def select_subset_year(self, year):
        return self.loc[self.index.year == year].deepcopy()

    def select_subset_month(self, month):
        return self.loc[self.dates.month == month].deepcopy()

    def select_subset_month_year(self, month, year):
        return self.loc[np.logical_and(self.dates.month == month,
                                       self.dates.year == year)].deepcopy()

    def intersect_over_dates(self, df):
        common_dates = np.intersect1d(self.index, pd.to_datetime(df.index))
        return self.select_subset_dates(common_dates), df.loc[common_dates].deepcopy()

    def intersect_over_date_range(self, df):
        common_dates = np.intersect1d(self.index, pd.to_datetime(df.index))
        return self[np.min(common_dates):np.max(common_dates)].deepcopy(), \
            df[np.min(common_dates):np.max(common_dates)].deepcopy()


    ############
    def get_period_ends(self, frequency):
        period_dates = DateUtils.get_date_range(self.index.min(), self.index.max(), periodicity=frequency)
        return self.insert_and_select_subset_dates(period_dates, fill_na=True)

    def get_periodic_levels(self, periods):
        return self.get_levels().get_period_ends(periods)

    def get_periodic_returns(self, periods, returns_type=None):
        return self.get_levels().get_period_ends(periods).get_returns(returns_type)

    def get_week_ends(self):
        return self.get_period_ends(Frequency.WEEKLY)

    def get_month_ends(self):
        return self.get_period_ends(Frequency.MONTHLY)

    def get_bmonth_ends(self):
        return self.get_period_ends(Frequency.BUSINESS_MONTHLY)

    def get_quarter_ends(self):
        return self.get_period_ends(Frequency.QUARTERLY)

    def get_year_ends(self):
        return self.get_period_ends(Frequency.YEARLY)

    def get_weekly_returns(self, returns_type=None):
        return self.get_periodic_returns(Frequency.WEEKLY, returns_type)

    def get_bmonthly_returns(self, returns_type=None):
        return self.get_periodic_returns(Frequency.BUSINESS_MONTHLY, returns_type)

    def get_quarterly_returns(self, returns_type=None):
        return self.get_periodic_returns(Frequency.QUARTERLY, returns_type)

    def get_annual_returns(self, returns_type=None):
        return self.get_periodic_returns(Frequency.YEARLY, returns_type)

    def get_monthly_returns(self, returns_type=None):
        return self.get_periodic_returns(Frequency.MONTHLY, returns_type)

    def get_weekly_levels(self):
        return self.get_periodic_levels(Frequency.WEEKLY)

    def get_monthly_levels(self):
        return self.get_periodic_levels(Frequency.MONTHLY)

    def get_bmonthly_levels(self):
        return self.get_periodic_levels(Frequency.BUSINESS_MONTHLY)

    def get_quarterly_levels(self):
        return self.get_periodic_levels(Frequency.QUARTERLY)

    def get_annual_levels(self):
        return self.get_periodic_levels(Frequency.YEARLY)

    def concat(self, time_series):
        assert self.type == time_series.type, 'ERROR - timeseries must be of the same type to concat'
        df_ = pd.concat(objs=(self, time_series), axis=1)
        atts = pd.concat(objs=(self.attributes, time_series.attributes), axis=1)
        return df_._create_new_object_same_type(df_, returns_type=self.returns_type)

    def ind(self, ind_value):
        if self.is_levels:
            self._cast_derived_class(
                ind_value * (self / self.loc[self.first_valid_index()]))

    def backfill_returns(self, backfill):
        """Method to backfill two time series objects based on returns.
           Method backfills self with data from B_prime, across matching columns.
       """
        # Make sure both objects are returns objects
        if self.is_levels:
            raise ValueError('Error - backfill_returns only supports returns objects')

        backfill_rtns = backfill.get_returns(self.returns_type).remove_empty_leading_rows()
        A = self.remove_empty_trailing_rows()

        assert self.name == backfill.name, 'Error - series must have the same names'
        backfilled = pd.concat(objs=(backfill_rtns.loc[backfill_rtns.dates < min(A.dates)], A), axis=0).sort_index()
        backfilled.name = self.name

        # preserve the attributes
        return self._create_new_returns_object(self.returns_type, data=backfilled)

    def backfill_levels(self, backfill):

        # Make sure both objects are returns objects
        if self.is_returns:
            raise ValueError('Error - backfill_levels only supports levels objects')

        self_rtns = self.get_returns(self.returns_type)
        backfill_rtns = backfill.get_returns(self.returns_type)

        backfilled_rtns = self_rtns.backfill_returns(backfill_rtns)
        backfilled_lvls = backfilled_rtns.get_levels()

        rescaled = backfilled_lvls * (self.loc[self.first_valid_index()] / backfilled_lvls.loc[self.first_valid_index()])
        rescaled.name = self.name
        return self._create_new_levels_object(rescaled, returns_type=self.returns_type)

    def backfill_series(self, backfill):

        backfill_type = backfill.to_time_series_type(self.type)
        if self.type == TimeSeriesType.LEVELS:
           return self.backfill_levels(backfill_type)
        elif self.type in [TimeSeriesType.GROWTH, TimeSeriesType.RETURNS]:
           return self.backfill_returns(backfill_type)
        else:
           raise ValueError('Error - type {} not supported'.format(self.type))

    def remove_empty_leading_rows(self):
        return self.loc[:self.last_valid_index()].deepcopy()

    def remove_empty_trailing_rows(self):
        return self.loc[self.first_valid_index():].deepcopy()

    def remove_empty_leading_trailing_rows(self):
        return self.remove_empty_trailing_rows().remove_empty_leading_rows()

    def subtract_over_common_dates(self, df):
        A_prime, B_prime = self.intersect_over_dates(df)
        return A_prime - B_prime.values.reshape(A_prime.shape)

    def addition_over_common_dates(self, df):
        A_prime, B_prime = self.intersect_over_dates(df)
        return A_prime + B_prime.values.reshape(A_prime.shape)

    def multiply_over_common_dates(self, df):
        A_prime, B_prime = self.intersect_over_dates(df)
        return A_prime * B_prime.values.reshape(A_prime.shape)

    def division_over_common_dates(self, df):
        A_prime, B_prime = self.intersect_over_dates(df)
        return A_prime / B_prime.values.reshape(A_prime.shape)

    def to_time_series_type(self, ts_type):

        if ts_type == TimeSeriesType.LEVELS:
           return self.get_levels()
        elif ts_type in [TimeSeriesType.RETURNS, TimeSeriesType.GROWTH]:
           return self.get_returns(self.returns_type)
        else:
            raise ValueError('Error - type {} not supported'.format(ts_type))

class CTimeSeries(pd.DataFrame):

    __pandas_priority__ = 5000
    _metadata = metadata


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

    @property
    def _constructor_sliced(self):
        def _cs(*args, **kwargs):
            return CSlice(*args, **kwargs).__finalize__(self)
        return _cs

    def __init__(self,
                 data=None,
                 ts_type: TimeSeriesType = TimeSeriesType.LEVELS,
                 returns_type: ReturnsType = ReturnsType.SIMPLE,
                 attributes: Optional[Union[pd.Series, pd.DataFrame, dict]] = None,
                 **kwargs):

        super(CTimeSeries, self).__init__(data, **kwargs)

        # Set attributes in object
        self.set_attributes(attributes)
        self.__setattr__('_type', ts_type)
        self.__setattr__('_returns_type', returns_type)

    ###################

    def _cast_derived_class(self, klass):
        self.__init__(klass,
                      ts_type=klass.type,
                      attributes=klass.attributes,
                      returns_type=klass.returns_type)

    def create_new_object(self, *args, **kwargs):
        return self.__class__(*args, **kwargs)

    def deepcopy(self):
        return self.create_new_object(data=self,
                                      returns_type=self.returns_type,
                                      ts_type=self.type)

    def _create_new_object_same_type(self, data=None, returns_type=None):
        return self.create_new_object(data=data, ts_type=self.type, returns_type=returns_type)

    def _create_new_levels_object(self, data=None, returns_type=None, attributes=None):
        return self.create_new_object(data=data, ts_type=TimeSeriesType.LEVELS, returns_type=returns_type, attributes=attributes)

    def _create_new_returns_object(self, returns_type=None, data=None):
        return self.create_new_object(data=data, ts_type=TimeSeriesType.RETURNS, returns_type=returns_type)


    ######################

    @property
    def dates(self):
        return pd.to_datetime(self.index)

    @property
    def data(self):
        return self.values

    @property
    def obs_per_year(self):
        return Frequency(self.frequency).obs_per_year()

    @property
    def data_length(self):
        return self.index.__len__()

    @property
    def number_of_cols(self):
        return self.columns.size

    @property
    def frequency(self):
        if self.index.freq:
            return self.index.freq.name
        elif self.index.inferred_freq:
            return self.index.inferred_freq
        else:
            return DateUtils.get_daterange_frequency(self.index)


    @property
    def attributes(self):
        return self.columns.copy()

    @property
    def type(self):
        return self.__getattr__('_type')

    @property
    def returns_type(self):
        return self.__getattr__('_returns_type')

    @property
    def is_levels(self):
        return self.type == TimeSeriesType.LEVELS

    @property
    def is_returns(self):
        return self.type in [TimeSeriesType.RETURNS,
                             TimeSeriesType.GROWTH]

    @property
    def first_valid_date(self):
        return np.min(self.apply(lambda x: x.first_valid_index()))

    @property
    def last_valid_date(self):
        return np.max(self.apply(lambda x: x.last_valid_index()))

    def get_levels(self):
        if self.is_levels:
            return self.deepcopy()

        lvls = self.apply(lambda x: x.get_levels())
        return self._create_new_levels_object(data=lvls, returns_type=self.returns_type)

    def get_returns(self, return_type=ReturnsType.SIMPLE):

        if return_type is None:
           return_type = self.returns_type

        rtns = self.apply(lambda x: x.get_returns(return_type))
        return self._create_new_returns_object(return_type, data=rtns)

    def select_subset_dates(self, dates):
        return self.loc[dates].deepcopy()

    def select_subset_columns(self, columns):
        return self.select_subset_labels(self.columns[columns])

    def select_subset_labels(self, labels):
        return self.get(labels).deepcopy()

    def select_subset_year(self, year):
        return self.loc[self.index.year == year].deepcopy()

    def select_subset_month(self, month):
        return self.loc[self.index.month == month].deepcopy()

    def select_subset_month_year(self, month, year):
        return self.loc[np.logical_and(self.index.month == month,
                                       self.index.year == year)].deepcopy()

    def insert_and_select_subset_dates(self, dates):

        subset = self.deepcopy()
        subset.insert_dates(dates)
        return subset.reindex(dates)

    def insert_date(self, date):
        if not DateUtils.is_iterable(date):
            date = pd.DatetimeIndex([date])

        insert_dates = pd.to_datetime(date)
        self.insert_dates(insert_dates)

    def insert_dates(self, dates):

        unique_dates = DateUtils.merge([self.dates, dates])
        df_ = self.reindex(unique_dates)
        if self.is_levels:
           self._cast_derived_class(df_.ffill())
        else:
           self._cast_derived_class(df_.fillna(0))

    def intersect_over_dates(self, df):
        common_dates = np.intersect1d(self.index, df.index)
        return self.select_subset_dates(common_dates), df.loc[common_dates].copy()

    def intersect_over_date_range(self, df):
        common_dates = np.intersect1d(self.index, df.index)
        return self[np.min(common_dates):np.max(common_dates)], \
            df[np.min(common_dates):np.max(common_dates)].copy()

    def get_period_ends(self, frequency):
        period_dates = DateUtils.get_date_range(np.min(self.index), np.max(self.index), periodicity=frequency)
        return self.insert_and_select_subset_dates(period_dates)

    def get_periodic_returns(self, periods, return_type=None):
        return self.get_levels().get_period_ends(periods).get_returns(return_type)

    def get_periodic_levels(self, periods):
        return self.get_levels().get_period_ends(periods)

    def get_week_ends(self):
        return self.get_period_ends(Frequency.WEEKLY)

    def get_month_ends(self):
        return self.get_period_ends(Frequency.MONTHLY)

    def get_bmonth_ends(self):
        return self.get_period_ends(Frequency.BUSINESS_MONTHLY)

    def get_quarter_ends(self):
        return self.get_period_ends(Frequency.QUARTERLY)

    def get_bquarter_ends(self):
        return self.get_period_ends(Frequency.BUSINESS_QUARTERLY)

    def get_year_ends(self):
        return self.get_period_ends(Frequency.YEARLY)

    def get_weekly_levels(self):
        return self.get_periodic_levels(Frequency.WEEKLY)

    def get_weekly_returns(self, return_type=None):
        return self.get_periodic_returns(Frequency.WEEKLY, return_type)

    def get_monthly_levels(self):
        return self.get_periodic_levels(Frequency.MONTHLY)

    def get_monthly_returns(self, return_type=None):
        return self.get_periodic_returns(Frequency.MONTHLY, return_type)

    def get_bmonthly_levels(self):
        return self.get_periodic_levels(Frequency.BUSINESS_MONTHLY)

    def get_bmonthly_returns(self, return_type=None):
        return self.get_periodic_returns(Frequency.BUSINESS_MONTHLY, return_type)

    def get_quarterly_levels(self):
        return self.get_periodic_levels(Frequency.QUARTERLY)

    def get_quarterly_returns(self, return_type=None):
        return self.get_periodic_returns(Frequency.QUARTERLY, return_type)

    def get_bquarterly_levels(self):
        return self.get_periodic_levels(Frequency.BUSINESS_QUARTERLY)

    def get_bquarterly_returns(self, return_type=None):
        return self.get_periodic_returns(Frequency.BUSINESS_QUARTERLY, return_type)

    def get_annual_levels(self):
        return self.get_periodic_levels(Frequency.YEARLY)

    def get_annual_returns(self, return_type=None):
        return self.get_periodic_returns(Frequency.YEARLY, return_type)

    ############### Methods Associated with Attributes ##################

    def set_attributes(self, attributes: Optional[Union[pd.Series, pd.DataFrame]] = None) -> None:
        pass

    def select_subset_attribute(self, attribute_name, attribute_values):
        return FrameUtils.select_subset_level(self, attribute_name, attribute_values)

    def set_attribute_single(self, attribute_name, attribute_value):
        self._cast_derived_class(
            FrameUtils.set_levels(self, attribute_value, attribute_name))

    def get_attribute(self, attribute_name):
        if attribute_name in self.attributes.names:
           return list(self.attributes.get_level_values(attribute_name))

    def drop_attributes(self, attribute_names):
        for att in attribute_names:
            self.drop_attribute(att)

    def drop_attribute(self, attribute_name):
        if attribute_name in self.columns.names:
            self.columns = self.columns.droplevel(attribute_name)

    def sort_by_attribute(self, attribute_name, inplace=True):
        sorted_frame = FrameUtils.sort_by_level(self, attribute_name)
        if inplace:
            self._cast_derived_class(sorted_frame)
        else:
            return sorted_frame.deepcopy()

    ###################

    def concat(self, time_series, ts_type=None):
        df_concat = pd.concat(objs=(self, time_series), axis=1).sort_index()
        return self.__class__(df_concat, ts_type=ts_type or self.type, returns_type=self.returns_type)

    def combine_left(self, time_series):
        if time_series.size > 0:
            return self.combine_first(time_series) if self.size > 0 else time_series.deepcopy()
        return self.deepcopy()

    def ind(self, ind_value):
        if self.is_levels:
            self._cast_derived_class(
               self.apply(lambda x: ind_value*(x/x.loc[x.first_valid_index()])))

    def backfill_returns(self, backfill):
        """Method to backfill two time series objects based on returns.
           Method backfills self with data from B_prime, across matching columns.
       """
        # Make sure both objects are returns objects
        if self.is_levels:
           raise ValueError('Error - _backfill_returns only supports returns objects')

        if self.size == 0 and backfill.size > 0:
           return backfill.deepcopy()

        backfill_rtns = backfill.get_returns(self.returns_type)
        common_labels = np.intersect1d(self.columns, backfill_rtns.columns)
        not_in_backfill = self.get(np.setdiff1d(self.columns, backfill.columns))

        backfilled = self._create_new_returns_object(self.returns_type)
        for label in common_labels:
            backfilled = backfilled.concat(self.get(label).backfill_returns(backfill_rtns.get(label)))

        self_with_backfill = not_in_backfill.concat(backfilled).select_subset_labels(self.columns)
        self_with_backfill.columns = self.columns
        assert np.all(self_with_backfill.columns == self.columns), 'Error in backfill'
        return self._create_new_returns_object(self.returns_type, data=self_with_backfill)

    def backfill_levels(self, backfill):

        if self.size == 0 and backfill.size > 0:
           return backfill.deepcopy()

        # Make sure both objects are returns objects
        if self.is_returns:
           raise ValueError('Error - _backfill_levels only supports levels objects')

        self_rtns = self.get_returns(self.returns_type)
        backfill_rtns = backfill.get_returns(self.returns_type)

        backfilled_rtns = self_rtns.backfill_returns(backfill_rtns)
        backfilled_lvls = backfilled_rtns.get_levels()

        # Rescaled the backfilled time series
        if self.returns_type in [ReturnsType.DIFFERENCE,
                                   ReturnsType.DIFFERENCE.value]:
            rescaled = backfilled_lvls.apply(lambda x: x + (self.get(x.name).loc[self.get(x.name).first_valid_index()] -
                                                            x.loc[self.get(x.name).first_valid_index()]))
        else:
            rescaled = backfilled_lvls.apply(lambda x: x * (self.get(x.name).loc[self.get(x.name).first_valid_index()] /
                                                            x.loc[self.get(x.name).first_valid_index()]))

        assert np.all(rescaled.columns == self.columns), 'Error in backfill'
        return self._create_new_levels_object(rescaled, returns_type=self.returns_type)

    def backfill_series(self, backfill):

        backfill_type = backfill.to_time_series_type(self.type)
        if self.type == TimeSeriesType.LEVELS:
           return self.backfill_levels(backfill_type)
        elif self.type in [TimeSeriesType.GROWTH, TimeSeriesType.RETURNS]:
           return self.backfill_returns(backfill_type)
        else:
           raise ValueError('Error - type {} not supported'.format(self.type))

    def remove_empty_leading_rows(self, keep_any_nans=True):

        first_valid_indicies = self.apply(lambda x: x.last_valid_index())

        if keep_any_nans:
           return self.loc[:max(first_valid_indicies)].deepcopy()
        else:
           return self.loc[:min(first_valid_indicies):].deepcopy()

    def remove_empty_trailing_rows(self, keep_any_nans=True):

        first_valid_indicies = self.apply(lambda x: x.first_valid_index())

        if keep_any_nans:
            return self.loc[min(first_valid_indicies):]
        else:
            return self.loc[max(first_valid_indicies):]

    def remove_empty_leading_trailing_rows(self, keep_any_nans=True):
        return self.remove_empty_trailing_rows(keep_any_nans).\
                    remove_empty_leading_rows(keep_any_nans)

    def subtract_over_common_dates(self, df):
        A_prime, B_prime = self.intersect_over_dates(df)
        return A_prime - B_prime.values
    def addition_over_common_dates(self, df):
        A_prime, B_prime = self.intersect_over_dates(df)
        return A_prime + B_prime.values

    def multiply_over_common_dates(self, df):
        A_prime, B_prime = self.intersect_over_dates(df)
        return A_prime * B_prime.values

    def division_over_common_dates(self, df):
        A_prime, B_prime = self.intersect_over_dates(df)
        return A_prime / B_prime.values

    def plot_df(self):
        self.plot()
        plt.show()

    def to_time_series_type(self, ts_type):

        if ts_type == TimeSeriesType.LEVELS:
           return self.get_levels()
        elif ts_type in [TimeSeriesType.RETURNS, TimeSeriesType.GROWTH]:
           return self.get_returns(self.returns_type)
        else:
            raise ValueError('Error - type {} not supported'.format(ts_type))

    @staticmethod
    def get_timeseries_from_ticker(ticker, fields=None, ts_type=None):
        from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource

        df = GlobalDataSource().get_dataframe_from_ticker(ticker,
                                                          cols=fields,
                                                          index_col='date')
        if ts_type is None:
            return CTimeSeries(df)
        else:
            return CTimeSeries(df, ts_type=ts_type)


if __name__ == "__main__":


    ds1 = CTimeSeries.get_timeseries_from_ticker('MSSPANL', fields='PI', ts_type=TimeSeriesType.LEVELS)
    ds2 = CTimeSeries.get_timeseries_from_ticker('USESPON', fields='ER', ts_type=TimeSeriesType.LEVELS)
    ds3 = CTimeSeries.get_timeseries_from_ticker('SPANPES', fields='ER', ts_type=TimeSeriesType.LEVELS)

    ds = ds1.concat(ds2.concat(ds3))

    ds.set_attribute_single('location',['Europe'])
    ds1_prime = ds.iloc[:,0]









