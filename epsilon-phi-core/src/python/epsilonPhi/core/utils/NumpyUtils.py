import numpy as np

class NumpyUtils(object):
    pass

    @staticmethod
    def flat_meshgrid(*args):
        res = np.meshgrid(*args)
        return [ res[x].ravel() for x in range(len(res)) ]