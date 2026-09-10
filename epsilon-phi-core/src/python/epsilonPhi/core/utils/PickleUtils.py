import pickle
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
import pandas as pd
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency


sessionMgr = SessionMgr()

class PickleUtils(object):

    _pickle_type_short_name = {'Factor':'F', 'FactorPanel':'FP', 'Bootstrap':'B', 'Asset':'A', 'Other':'O'}
    def __init__(self):
        pass

    @staticmethod
    def pickle_it(pickle_object, pickle_name):
        pickled_object = pickle.dumps(pickle_object)
        sessionMgr.save_pickle_to_database(pickled_object, pickle_name)

    @staticmethod
    def is_pickled(pickle_name):
        return sessionMgr.is_pickled(pickle_name)

    @staticmethod
    def load_pickle(pickle_name):
        return sessionMgr.load_pickle_from_database(pickle_name)

    @staticmethod
    def is_factor_pickled(factor_name, end_date, frequency):

        end_date = pd.to_datetime(end_date).strftime('%Y%m%d')
        if isinstance(frequency, Frequency):
           frequency = frequency.value
        pickle_name = factor_name + end_date + frequency
        return PickleUtils.is_pickled(pickle_name)

    @staticmethod
    def load_factor_from_pickles(factor_name, end_date, frequency):

        end_date = pd.to_datetime(end_date).strftime('%Y%m%d')
        if isinstance(frequency, Frequency):
            frequency = frequency.value
        pickle_name = factor_name + end_date + frequency

        factor = PickleUtils.load_pickle(pickle_name)
        factor.index = pd.to_datetime(factor.index)
        return factor

    @staticmethod
    def pickle_factor(factor_object, factor_name, end_date, frequency):
        end_date = pd.to_datetime(end_date).strftime('%Y%m%d')
        if isinstance(frequency, Frequency):
            frequency = frequency.value
        pickle_name = factor_name + end_date + frequency
        PickleUtils.pickle_it(factor_object, pickle_name)

    @staticmethod
    def delete_pickle(pickle_id):
        if PickleUtils.is_pickled(pickle_id):
            sessionMgr.delete_pickle_from_database(pickle_id)






