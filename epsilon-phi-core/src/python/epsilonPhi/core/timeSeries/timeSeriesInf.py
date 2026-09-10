from abc import ABC, abstractmethod

metadata = ["_type", "_returns_type"]
slice_metadata = metadata + ['_name']

class AbstractTimeSeries(ABC):

    @property
    @abstractmethod
    def _constructor(self):
        pass

    @property
    @abstractmethod
    def _constructor_sliced(self):
        pass

    # check we have time series data
    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def _cast_derived_class(self):
        pass

    @abstractmethod
    def _deepcopy(self):
        pass

    @abstractmethod
    def create_new_object(self):
        pass

    @abstractmethod
    def _create_new_object_same_type(self):
        pass

    @abstractmethod
    def _create_new_levels_object(self):
        pass

    @abstractmethod
    def _create_new_returns_object(self):
        pass

    @property
    @abstractmethod
    def dates(self):
        pass

    @property
    @abstractmethod
    def data(self):
        pass

    @property
    @abstractmethod
    def data_length(self):
        pass

    @property
    @abstractmethod
    def number_of_cols(self):
        pass

    @property
    @abstractmethod
    def frequency(self):
        pass

    @property
    @abstractmethod
    def attributes(self):
        pass

    @property
    @abstractmethod
    def type(self):
        pass

    @property
    @abstractmethod
    def returns_type(self):
        pass

    @property
    @abstractmethod
    def is_levels(self):
        pass

    @property
    @abstractmethod
    def is_returns(self):
        pass

    @abstractmethod
    def get_levels(self):
        pass

    @abstractmethod
    def get_returns(self):
        pass

    @abstractmethod
    def select_subset_dates(self):
        pass

    @abstractmethod
    def insert_and_select_subset_dates(self):
        pass

    @abstractmethod
    def select_subset_columns(self):
        pass

    @abstractmethod
    def select_subset_labels(self):
        pass

    @abstractmethod
    def select_subset_attribute(self):
        pass

    @abstractmethod
    def select_subset_year(self):
        pass

    @abstractmethod
    def select_subset_month(self):
        pass

    @abstractmethod
    def select_subset_month_year(self):
        pass

    @abstractmethod
    def insert_date(self):
        pass

    @abstractmethod
    def insert_dates(self):
        pass

    @abstractmethod
    def intersect_over_dates(self):
        pass

    @abstractmethod
    def intersect_over_date_range(self):
        pass

    @abstractmethod
    def get_periodic_levels(self):
        pass

    @abstractmethod
    def get_periodic_returns(self):
        pass

    @abstractmethod
    def get_period_ends(self):
        pass

    @abstractmethod
    def get_week_ends(self):
        pass

    @abstractmethod
    def get_month_ends(self):
        pass

    @abstractmethod
    def get_bmonth_ends(self):
        pass

    @abstractmethod
    def get_quarter_ends(self):
        pass

    @abstractmethod
    def get_year_ends(self):
        pass

    @abstractmethod
    def get_weekly_levels(self):
        pass

    @abstractmethod
    def get_weekly_returns(self):
        pass

    @abstractmethod
    def get_monthly_levels(self):
        pass

    @abstractmethod
    def get_monthly_returns(self):
        pass

    @abstractmethod
    def get_bmonthly_levels(self):
        pass

    @abstractmethod
    def get_bmonthly_returns(self):
        pass

    @abstractmethod
    def get_quarterly_levels(self):
        pass

    @abstractmethod
    def get_quarterly_returns(self):
        pass

    @abstractmethod
    def get_annual_levels(self):
        pass

    @abstractmethod
    def get_annual_returns(self):
        pass

    @abstractmethod
    def reset_attributes(self):
        pass

    @abstractmethod
    def set_attributes(self):
        pass

    @abstractmethod
    def add_attributes(self):
        pass

    @abstractmethod
    def add_attribute(self):
        pass

    @abstractmethod
    def append_attributes(self):
        pass

    @abstractmethod
    def get_attributes(self):
        pass

    @abstractmethod
    def sort_by_attribute(self):
        pass

    @abstractmethod
    def concat(self):
        pass

    @abstractmethod
    def combine_left(self):
        pass

    @abstractmethod
    def ind(self):
        pass

    @abstractmethod
    def _backfill_returns(self):
        pass

    @abstractmethod
    def _backfill_levels(self):
        pass

    @abstractmethod
    def remove_empty_leading_rows(self):
        pass

    @abstractmethod
    def remove_empty_trailing_rows(self):
        pass

    @abstractmethod
    def remove_empty_leading_trailing_rows(self):
        pass