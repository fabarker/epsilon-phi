from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.utils.FrameUtils import FrameUtils
from epsilonPhi.core.utils.DateUtils import DateUtils
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class CSlice(pd.Series):

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
            return CSlice(*args, **kwargs).__finalize__(self)
        return _c

    @property
    def _constructor_expanddim(self):
        def _cs(*args, **kwargs):
            return CTimeSeries(*args, **kwargs).__finalize__(self)
        return _cs

    def __init__(self,
                 data=None,
                 attributes: pd.DataFrame = None,
                 ts_type: TimeSeriesType = TimeSeriesType.LEVELS,
                 returns_type: ReturnsType = ReturnsType.SIMPLE,
                 **kwargs):

        super(CSlice, self).__init__(data=data, **kwargs)
        # Set attributes in object
        if isinstance(attributes, pd.DataFrame):
            self.set_attributes(attributes)
        else:
            self.__setattr__('_added_attributes', pd.DataFrame())
        self.__setattr__('_type', ts_type)
        self.__setattr__('_returns_type', returns_type)

    def _cast_derived_class(self, klass):
        self.__init__(klass,
                      ts_type=klass.type,
                      attributes=klass.attributes,
                      returns_type=klass.returns_type)

    def _deepcopy(self):
        return self.__class__(self,
                              ts_type=self._type,
                              attributes=self.attributes,
                              returns_type=self.returns_type)

    def create_new_object(self, data=None, index=None, name=None, attributes=None, ts_type=None, returns_type=None):
        return self.__class__(data=data,
                              index=index,
                              name=name,
                              ts_type=ts_type,
                              attributes=attributes,
                              returns_type=returns_type)

    def _create_new_object_same_type(self, data=None, index=None, name=None, attributes=None):
        return self.create_new_object(data, index, name, attributes, self.type, self.returns_type)

    def _create_new_levels_object(self, data=None, index=None, name=None, attributes=None):
        return self.create_new_object(data, index, name, attributes, TimeSeriesType.LEVELS, self.returns_type)

    def _create_new_returns_object(self, returns_type, data=None, index=None, name=None, attributes=None):
        return self.create_new_object(data, index, name, attributes, TimeSeriesType.RETURNS, returns_type)

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
        return None

    @property
    def frequency(self):
        if self.index.freq:
            return self.index.freq
        else:
            return self.index.inferred_freq

    @property
    def attributes(self):
        self.__update_attributes()
        return self.__getattr__('_added_attributes')

    def __update_attributes(self):
        if self.name:
            atts = self.__getattr__('_added_attributes')
            self.__setattr__('_added_attributes', atts.get([self.name]))

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

    def get_levels(self):

        if self.is_levels:
            return self.copy()

        idx_nan = self.isna().values
        if self.returns_type in [ReturnsType.SIMPLE,
                                 ReturnsType.SIMPLE.value]:
            newobj = (1 + self.fillna(0)).cumprod(axis=0)
        elif self.returns_type in [ReturnsType.LOG,
                                   ReturnsType.LOG.value]:
            newobj = np.exp(self.fillna(0)).cumsum(axis=0)
        elif self.returns_type in [ReturnsType.DIFFERENCE,
                                   ReturnsType.DIFFERENCE.value]:
            newobj = self.fillna(0).cumsum(axis=0)
        else:
            raise ValueError('ERROR: {} not supported'.format(self.returns_type))

        newobj.values[idx_nan] = np.nan
        newobj.insert_date_val(DateUtils.shift_date(newobj.dates[0], newobj.frequency, -1),1)
        return self._create_new_levels_object(newobj, attributes=self.attributes, name=self.name)

    def get_returns(self, return_type=ReturnsType.SIMPLE):

        if self.is_returns:
            return self.copy()

        nan_locs = self.isna().values
        if self.returns_type in [ReturnsType.SIMPLE,
                                 ReturnsType.SIMPLE.value]:
            newobj = self.ffill().pct_change().dropna()
        elif return_type in [ReturnsType.LOG,
                             ReturnsType.LOG.value]:
            newobj = np.log(self.ffill()).diff().dropna()
        elif return_type in [ReturnsType.DIFFERENCE,
                             ReturnsType.DIFFERENCE.value]:
            newobj = self.ffill().diff().dropna()
        else:
            raise ValueError('Error - must specify returns type')

        rtns_locs = np.diff(np.cumsum(nan_locs, axis=0), axis=0) != 0
        newobj.values[rtns_locs] = np.nan
        return self._create_new_returns_object(return_type, data=newobj, name=self.name, attributes=self.attributes)

    def insert_date_val(self, date, val):
        self.insert_date(date)
        self[date] = val

    def select_subset_dates(self, dates):
        return self.loc[dates].copy()

    def insert_and_select_subset_dates(self, dates, fill_na=False):

        reindexed = self.reindex(self.index.append(dates).unique()).sort_index()
        if fill_na:
            return reindexed.ffill().reindex(dates).copy()
        else:
            return reindexed.reindex(dates).copy()

    def select_subset_year(self, year):
        return self.loc[self.index.year == year]

    def select_subset_month(self, month):
        return self.loc[self.index.month == month]

    def select_subset_month_year(self, month, year):
        return self.loc[np.logical_and(self.index.month == month,
                                       self.index.year == year)]

    def insert_date(self, date):
        if not DateUtils.is_iterable(date):
            date = pd.DatetimeIndex([date])

        insert_dates = pd.to_datetime(date)
        self.insert_dates(insert_dates)

    def insert_dates(self, dates):
        self._cast_derived_class(self.reindex(self.index.append(dates).unique()).sort_index())

    def intersect_over_dates(self, df):
        common_dates = np.intersect1d(self.dates, df.index)
        return self.select_subset_dates(common_dates), df.loc[common_dates].copy()

    def intersect_over_date_range(self, df):
        common_dates = np.intersect1d(self.dates, df.index)
        return self[np.min(common_dates):np.max(common_dates)], \
            df[np.min(common_dates):np.max(common_dates)]

    def get_periodic_levels(self, periods):
        return self.get_levels().get_period_ends(periods)

    def get_periodic_returns(self, periods):
        return self.get_levels().get_period_ends(periods).get_returns()

    def get_period_ends(self, frequency):
        period_dates = DateUtils.get_date_range(np.min(self.dates),
                                                np.max(self.dates),
                                                periodicity=frequency)
        return self.insert_and_select_subset_dates(period_dates, fill_na=True)

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

    def get_weekly_levels(self):
        return self.get_periodic_levels(Frequency.WEEKLY)

    def get_weekly_returns(self):
        return self.get_periodic_returns(Frequency.WEEKLY)

    def get_monthly_levels(self):
        return self.get_periodic_levels(Frequency.MONTHLY)

    def get_monthly_returns(self):
        return self.get_periodic_returns(Frequency.MONTHLY)

    def get_bmonthly_levels(self):
        return self.get_periodic_levels(Frequency.BUSINESS_MONTHLY)

    def get_bmonthly_returns(self):
        return self.get_periodic_returns(Frequency.BUSINESS_MONTHLY)

    def get_quarterly_levels(self):
        return self.get_periodic_levels(Frequency.QUARTERLY)

    def get_quarterly_returns(self):
        return self.get_periodic_returns(Frequency.QUARTERLY)

    def get_annual_levels(self):
        return self.get_periodic_levels(Frequency.YEARLY)

    def get_annual_returns(self):
        return self.get_periodic_returns(Frequency.YEARLY)

    def reset_attributes(self):
        self.__setattr__('_added_attributes', pd.DataFrame())

    def set_attributes(self, attributes: pd.DataFrame):
        if isinstance(attributes, pd.DataFrame):
            self.__setattr__('_added_attributes', attributes)

    def add_attributes(self, attributes: pd.DataFrame):
        self.append_attributes(attributes)

    def add_attribute(self, attribute_name, attribute_vals):

        if not FrameUtils.is_iterable(attribute_vals):
            attribute_vals = [attribute_vals]
        new_att = pd.DataFrame(attribute_vals, columns=[self.name], index=[attribute_name])
        self.append_attributes(new_att)

    def append_attributes(self, new_attributes: pd.DataFrame):
        new_atts = pd.concat((self.attributes, new_attributes), axis=0)
        self.set_attributes(new_atts)

    def get_attributes(self, attribute_name):
        return self.attributes.loc[attribute_name]

    def sort_by_attribute(self, attribute_name, sort_ascending=False):
        pass

    def concat(self, time_series):
        assert self.type == time_series.type, 'ERROR - timeseries must be of the same type to concat'
        new_atts = pd.concat((self.attributes, time_series.attributes), axis=1)
        return CTimeSeries(data=pd.concat((self, time_series), axis=1), ts_type=self.type, attributes=new_atts)

    def ind(self, ind_value):
        if self.is_levels:
            self._cast_derived_class(
                ind_value * (self / self.loc[self.first_valid_index()]))

    def _backfill_returns(self, backfill):
        """Method to backfill two time series objects based on returns.
           Method backfills self with data from B_prime, across matching columns.
       """
        # Make sure both objects are returns objects
        if self.is_levels:
            raise ValueError('Error - _backfill_returns only supports returns objects')

        backfill_rtns = backfill.get_returns()
        assert self.name == backfill.name, 'Error - series must have the same names'
        backfilled = pd.concat((backfill.loc[backfill_rtns.dates < min(self.dates)], self), axis=0).sort_index()

        # preserve the attributes
        return self._create_new_returns_object(self.returns_type,
                                               data=backfilled,
                                               name=self.name,
                                               attributes=self.attributes)

    def _backfill_levels(self, backfill):

        # Make sure both objects are returns objects
        if self.is_returns:
            raise ValueError('Error - _backfill_levels only supports levels objects')

        self_rtns = self.get_returns()
        backfill_rtns = backfill.get_returns()

        backfilled_rtns = self_rtns._backfill_returns(backfill_rtns)
        backfilled_lvls = backfilled_rtns.get_levels()

        rescaled = backfilled_lvls * (self.loc[self.first_valid_index()] / backfilled_lvls.loc[self.first_valid_index()])
        return self._create_new_levels_object(rescaled, attributes=self.attributes, name=self.name)

    def remove_empty_leading_rows(self):
        return self.loc[:self.last_valid_index()]

    def remove_empty_trailing_rows(self):
        return self.loc[:self.first_valid_index()]

    def remove_empty_leading_trailing_rows(self):
        return self.remove_empty_trailing_rows().remove_empty_leading_rows()


class CTimeSeries(pd.DataFrame):

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
            return CTimeSeries(*args, **kwargs).__finalize__(self)
        return _c

    @property
    def _constructor_sliced(self):
        def _cs(*args, **kwargs):
            return CSlice(*args, **kwargs).__finalize__(self)
        return _cs

    def __init__(self,
                 data=None,
                 attributes: pd.DataFrame = None,
                 ts_type: TimeSeriesType = TimeSeriesType.LEVELS,
                 returns_type: ReturnsType = ReturnsType.SIMPLE,
                 **kwargs):

        super(CTimeSeries, self).__init__(data, **kwargs)
        # Set attributes in object
        if isinstance(attributes, pd.DataFrame):
            self.set_attributes(attributes)
        else:
            self.__setattr__('_added_attributes', pd.DataFrame())
        self.__setattr__('_type', ts_type)
        self.__setattr__('_returns_type', returns_type)
        #self._validate_index()

    # check we have time series data
    def _validate_index(self):
         if self.index.size > 0:
            if not isinstance(self.index, pd.DatetimeIndex):
                self.index = pd.to_datetime(self.index)

    def _cast_derived_class(self, klass):
        self.__init__(klass,
                      ts_type=klass.type,
                      attributes=klass.attributes,
                      returns_type=klass.returns_type)

    def _deepcopy(self):
        return self.__class__(self,
                              ts_type=self._type,
                              attributes=self.attributes,
                              returns_type=self.returns_type)

    def create_new_object(self, data=None, index=None, columns=None, attributes=None, ts_type=None, returns_type=None):
        return self.__class__(data=data,
                              index=index,
                              columns=columns,
                              ts_type=ts_type,
                              attributes=attributes,
                              returns_type=returns_type)

    def _create_new_object_same_type(self, data=None, index=None, columns=None, attributes=None):
        return self.create_new_object(data, index, columns, attributes, self.type, self.returns_type)

    def _create_new_levels_object(self, data=None, index=None, columns=None, attributes=None):
        return self.create_new_object(data, index, columns, attributes, TimeSeriesType.LEVELS, self.returns_type)

    def _create_new_returns_object(self, returns_type, data=None, index=None, columns=None, attributes=None):
        return self.create_new_object(data, index, columns, attributes, TimeSeriesType.RETURNS, returns_type)

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
        return self.__getattr__('_added_attributes').get(self.columns, pd.DataFrame())

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
            return self.copy()
        lvls = self.apply(lambda x: x.get_levels())
        return self._create_new_levels_object(lvls, attributes=self.attributes)

    def get_returns(self, return_type=ReturnsType.SIMPLE):

        if self.is_returns:
            return self.copy()

        nan_locs = self.isna().values
        if self.returns_type in [ReturnsType.SIMPLE,
                                 ReturnsType.SIMPLE.value]:
            newobj = self.ffill().apply(lambda x: x.pct_change().dropna())
        elif return_type in [ReturnsType.LOG,
                             ReturnsType.LOG.value]:
            newobj = self.ffill().apply(lambda x: np.log(x).diff().dropna())
        elif return_type in [ReturnsType.DIFFERENCE,
                             ReturnsType.DIFFERENCE.value]:
            newobj = self.ffill().apply(lambda x: x.diff().dropna())
        else:
            raise ValueError('Error - must specify returns type')

        rtns_locs = np.diff(np.cumsum(nan_locs, axis=0), axis=0) != 0
        newobj.values[rtns_locs] = np.nan
        return self._create_new_returns_object(return_type, data=newobj, attributes=self.attributes)

    def select_subset_dates(self, dates):
        return self.loc[dates].copy()

    def insert_and_select_subset_dates(self, dates, fill_na=False):

        reindexed = self.reindex(self.index.append(dates).unique()).sort_index()
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

    def insert_date(self, date):
        if not DateUtils.is_iterable(date):
            date = pd.DatetimeIndex([date])

        insert_dates = pd.to_datetime(date)
        self.insert_dates(insert_dates)

    def insert_dates(self, dates):
        self._cast_derived_class(self.reindex(self.index.append(dates).unique()).sort_index())

    def intersect_over_dates(self, df):
        common_dates = np.intersect1d(self.dates, df.index)
        return self.select_subset_dates(common_dates), df.loc[common_dates].copy()

    def intersect_over_date_range(self, df):
        common_dates = np.intersect1d(self.dates, df.index)
        return self[np.min(common_dates):np.max(common_dates)], \
            df[np.min(common_dates):np.max(common_dates)]

    def get_periodic_levels(self, periods):
        return self.get_levels().get_period_ends(periods)

    def get_periodic_returns(self, periods):
        return self.get_levels().get_period_ends(periods).get_returns()

    def get_period_ends(self, frequency):
        period_dates = DateUtils.get_date_range(np.min(self.dates), np.max(self.dates), periodicity=frequency)
        return self.insert_and_select_subset_dates(period_dates, fill_na=True)

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

    def get_weekly_levels(self):
        return self.get_periodic_levels(Frequency.WEEKLY)

    def get_weekly_returns(self):
        return self.get_periodic_returns(Frequency.WEEKLY)

    def get_monthly_levels(self):
        return self.get_periodic_levels(Frequency.MONTHLY)

    def get_monthly_returns(self):
        return self.get_periodic_returns(Frequency.MONTHLY)

    def get_bmonthly_levels(self):
        return self.get_periodic_levels(Frequency.BUSINESS_MONTHLY)

    def get_bmonthly_returns(self):
        return self.get_periodic_returns(Frequency.BUSINESS_MONTHLY)

    def get_quarterly_levels(self):
        return self.get_periodic_levels(Frequency.QUARTERLY)

    def get_quarterly_returns(self):
        return self.get_periodic_returns(Frequency.QUARTERLY)

    def get_annual_levels(self):
        return self.get_periodic_levels(Frequency.YEARLY)

    def get_annual_returns(self):
        return self.get_periodic_returns(Frequency.YEARLY)

    def reset_attributes(self):
        self.__setattr__('_added_attributes', pd.DataFrame())

    def set_attributes(self, attributes: pd.DataFrame):
        self.__setattr__('_added_attributes',
                         attributes.get(self.columns, pd.DataFrame()))

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
        pass

    def concat(self, time_series):
        assert self.type == time_series.type, 'ERROR - timeseries must be of the same type to concat'
        new_atts = pd.concat((self.attributes, time_series.attributes), axis=1)
        df_concat = pd.concat((self, time_series), axis=1)
        return self.__class__(df_concat, ts_type=self.type, attributes=new_atts)

    def combine_left(self, time_series):
        if self.size > 0 and time_series.size > 0:
            return self.combine_first(time_series)
        elif time_series.size > 0:
            return time_series.copy()
        else:
            return self.copy()

    def ind(self, ind_value):
        if self.is_levels:
            self._cast_derived_class(
               self.apply(lambda x: ind_value*(x/x.loc[x.first_valid_index()])))


    def _backfill_returns(self, backfill):
        """Method to backfill two time series objects based on returns.
           Method backfills self with data from B_prime, across matching columns.
       """
        # Make sure both objects are returns objects
        if self.is_levels:
           raise ValueError('Error - _backfill_returns only supports returns objects')

        backfill_rtns = backfill.get_returns()
        common_labels = np.intersect1d(self.columns, backfill_rtns.columns)
        not_in_backfill = self.get(np.setdiff1d(self.columns, backfill.columns))

        backfilled = self._create_new_returns_object(self.returns_type)
        for label in common_labels:
            backfilled = backfilled.concat(self.get(label)._backfill_returns(backfill_rtns.get(label)))
        backfilled.columns = self.columns

        self_with_backfill = not_in_backfill.concat(backfilled).select_subset_labels(self.columns)
        assert np.all(self_with_backfill.columns == self.columns), 'Error in backfill'
        return self_with_backfill.copy()

    def _backfill_levels(self, backfill):

        # Make sure both objects are returns objects
        if self.is_returns:
           raise ValueError('Error - _backfill_levels only supports levels objects')

        self_rtns = self.get_returns()
        backfill_rtns = backfill.get_returns()

        backfilled_rtns = self_rtns._backfill_returns(backfill_rtns)
        backfilled_lvls = backfilled_rtns.get_levels()
        rescaled = backfilled_lvls.apply(lambda x: x * (self.get(x.name).loc[self.get(x.name).first_valid_index()] /
                                                        x.loc[self.get(x.name).first_valid_index()]))
        assert np.all(rescaled.columns == self.columns), 'Error in backfill'
        return self._create_new_levels_object(rescaled, attributes=self.attributes)


    def remove_empty_leading_rows(self, keep_any_nans=True):

        first_valid_indicies = self.apply(lambda x: x.last_valid_index())

        if keep_any_nans:
           return self.loc[:max(first_valid_indicies)]
        else:
           return self.loc[:min(first_valid_indicies):]

    def remove_empty_trailing_rows(self, keep_any_nans=True):

        first_valid_indicies = self.apply(lambda x: x.first_valid_index())

        if keep_any_nans:
            return self.loc[min(first_valid_indicies):]
        else:
            return self.loc[max(first_valid_indicies):]

    def remove_empty_leading_trailing_rows(self, keep_any_nans=True):
        return self.remove_empty_trailing_rows(keep_any_nans).\
                    remove_empty_leading_rows(keep_any_nans)

    def plot_df(self):
        self.plot()
        plt.show()

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


    ds = CTimeSeries.get_timeseries_from_ticker('SPXDELTA20C(O1)', fields='mid', ts_type=TimeSeriesType.LEVELS)
    gsq = CTimeSeries.get_timeseries_from_ticker('SPXDELTA20P(O1)', fields='mid', ts_type=TimeSeriesType.LEVELS)










