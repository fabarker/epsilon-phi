from epsilonPhi.ep_strategies.estimators.risk.RiskFunctions import covariance_functions
import pandas as pd

class Estimator(object):

    class mixed_estimator(object):
        def __init__(self):
            pass

    class diagonal(object):

        def __init__(self, window):
            self._window = window
            self._func = getattr(covariance_functions, 'close_close_exponential_moving_std')

        def compute(self, series):
            return self._func(series, self._window)


    class rolling_close_equal_weighted(object):
        def __init__(self, window):
            self._window = window
            self._func = getattr(covariance_functions, 'close_close_equal_weighted_moving_covariance')

        def compute(self, series):
            return self._func(series, self._window)

    class rolling_close_exponential_weighted(object):
        def __init__(self, span):
            self._span = span
            self._func = getattr(covariance_functions, 'close_close_exponential_moving_covariance')

        def compute(self, series):
            return self._func(series, self._span)

    class arch(object):
        def __init__(self, p, q):
            self._p = p
            self._q = q
            self._func = getattr(covariance_functions, 'arch')

        def compute(self, series):
            sig = pd.concat([self._func(series.get(x).dropna(), self._p, self._q)
                             for x in series.columns], axis=1)
            sig.columns = series.columns
            return sig.copy()

    class garch(object):
        def __init__(self, p, q):
            self._p = p
            self._q = q
            self._func = getattr(covariance_functions, 'garch_covariance_matrix')

        def compute(self, series):

            return self._func(series)

    class yang_zhang(object):
        def __init__(self):
            self._func = getattr(covariance_functions, 'yang_zhang')

        def compute(self, series):
            return self._func(series)

    class roger_satchell(object):
        def __init__(self):
            self._func = getattr(covariance_functions, 'roger_satchell')

        def compute(self, series):
            return self._func(series)

    class garman_klass(object):
        def __init__(self):
            self._func = getattr(covariance_functions, 'garman_klass')

        def compute(self, series):
            return self._func(series)



class Covariance(object):

    def __init__(self, estimator_class, *args, **kwargs):

        self._shrink = False
        self.set_estimator(estimator_class, *args, **kwargs)


    @property
    def shrink(self):
        return self._shrink

    @shrink.setter
    def shrink(self, value):
        self._shrink = value

    def set_estimator(self, estimator_class, *args, **kwargs):
        self._estimator = estimator_class(*args, **kwargs)

    def estimate(self, data):
        return self._estimator.compute(data)

    @staticmethod
    def estimate_covariance(prices, estimator_class, *args, **kwargs):
        covar = Covariance(estimator_class, *args, **kwargs)
        return covar.estimate(prices)


if __name__ == "__main__":

    from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource

    gds = GlobalDataSource()
    prices = gds.get_dataframe_from_tickers(['CAFCS00','CTTCS00','CCFCS00','CYMCS00','NAYCS00','NCJCS00','NPSCS00','NVICS00'], cols='PS')

    self = Covariance(Estimator.rolling_close_exponential_weighted, span=252)
    Covariance.shrink = True
    self.estimate(prices)

