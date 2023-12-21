from abc import ABC, abstractmethod
class CSignalInf(object):

    @abstractmethod
    def set_vol_function(asset):
        pass

    @abstractmethod
    def get_signal(asset):
        pass
