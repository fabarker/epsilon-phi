from epsilonPhi.ep_strategies.estimators.risk.RiskFunctions import vol_functions
import pandas as pd


# Risk Estimator

# Equal weighted
# Exponentially weighted
# Custom weighted
# Arch
# Garch
# yang zhang
# Principle components
# Robust / MonCovDet
# shrunk
# Ledoit-Wolf covariance
# Oracle Approximating Shrinkage


class Estimator(object):

    class mixed_estimator(object):
        def __init__(self):
            pass

    class rolling_close_equal_weighted(object):
        def __init__(self, window):
            self._window = window
            self._func = getattr(vol_functions, 'close_close_equal_weighted_moving_std')

        def compute(self, series):
            return self._func(series, self._window)

    class rolling_close_exponential_weighted(object):
        def __init__(self, span):
            self._span = span
            self._func = getattr(vol_functions, 'close_close_exponential_moving_std')

        def compute(self, series):
            return self._func(series, self._span)

    class arch(object):
        def __init__(self, p, q):
            self._p = p
            self._q = q
            self._func = getattr(vol_functions, 'arch')

        def compute(self, series):
            sig = pd.concat([self._func(series.get(x).dropna(), self._p, self._q)
                             for x in series.columns], axis=1)
            sig.columns = series.columns
            return sig.copy()

    class garch(object):
        def __init__(self, p, q):
            self._p = p
            self._q = q
            self._func = getattr(vol_functions, 'garch')

        def compute(self, series):

            sig = pd.concat([ self._func(series.get(x).dropna(), self._p, self._q)
                              for x in series.columns ], axis=1)
            sig.columns = series.columns
            return sig.copy()

    class yang_zhang(object):
        def __init__(self):
            self._func = getattr(vol_functions, 'yang_zhang')

        def compute(self, series):
            return self._func(series)

    class roger_satchell(object):
        def __init__(self):
            self._func = getattr(vol_functions, 'roger_satchell')

        def compute(self, series):
            return self._func(series)

    class garman_klass(object):
        def __init__(self):
            self._func = getattr(vol_functions, 'garman_klass')

        def compute(self, series):
            return self._func(series)


class Volatility(object):

    def __init__(self, estimator_class, *args, **kwargs):
        self.set_estimator(estimator_class, *args, **kwargs)

    def set_estimator(self, estimator_class, *args, **kwargs):
        self._estimator = estimator_class(*args, **kwargs)

    def estimate(self, data):
        return self._estimator.compute(data)

    @staticmethod
    def estimate_volatility(prices, estimator_class, *args, **kwargs):
        vol = Volatility(estimator_class, *args, **kwargs)
        return vol.estimate(prices)




if __name__ == "__main__":


    self = Volatility(Estimator.rolling_close_equal_weighted, window=21)


