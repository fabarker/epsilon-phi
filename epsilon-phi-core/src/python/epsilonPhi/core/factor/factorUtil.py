from epsilonPhi.core.utils.PickleUtils import PickleUtils

__author__ = 'Francis Barker'
__date__ = '03/08/2023'

class CFactorUtil:

    @staticmethod
    def is_pickled(factor_name, end_date, frequency):
        pickle_name = CFactorUtil.get_factor_pickle_name(factor_name, end_date, frequency)
        return PickleUtils.is_pickled(pickle_name)

    @staticmethod
    def get_factor_pickle_name(factor_name, end_date, frequency):
        return factor_name + end_date + frequency



    @staticmethod
    def combine_factors(df_, betas):
        pass

    @staticmethod
    def to_excess_return_time_series(asset):
        pass

    @staticmethod
    def create_factor_from_dataframe(self, df, is_ER=False, currency=None):
        pass

    @staticmethod
    def create_factor_from_asset(self, asset):
        pass

    @staticmethod
    def add_factor_to_risk_model(factor):
        pass

    @staticmethod
    def add_factor_to_return_model(factor):
        pass


