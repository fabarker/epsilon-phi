from enum import Enum
from epsilonPhi.core.factor.factorPanel import CFactorPanels
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
import datetime

class CModelFactory(object):

    class REGRESSION_TYPE(object):
        class OLS(object):
            def __init__(self):
                pass

        class RIDGE(object):
            def __init__(self, lmbda):
                self._lambda = lmbda

        class LASSO(object):
            def __init__(self, lmbda):
                self._lambda = lmbda

        class ELASTIC_NET(object):
            def __init__(self, lmbda):
                self._lambda = lmbda

        class ROBUST(object):
            def __init__(self, lmbda):
                self._lambda = lmbda

        class BAYESIAN(object):
            def __init__(self, lmbda):
                self._lambda = lmbda

    class WINDOW_TYPE(object):
        class STATIC(object):
            def __init__(self):
                pass

        class ROLLING(object):

            def __init__(self, sample_size_in_years):
                self._sample_size_in_years = sample_size_in_years

            def __repr__(self):
                return f"ModelFactory.WINDOW_TYPE.ROLLING(sample_size_in_years={self._sample_size_in_years})"

        class EXPANDING(object):
            def __init__(self, starting_window_size_years):
                self._starting_window_size_years = starting_window_size_years

    class ORTHOGONALIZE(object):
        def __init__(self, orthogonalize_list):
            self._orthogonalize_list = orthogonalize_list

    class WEIGHTING_SCHEME(object):
        class EQUAL(object):
            def __init__(self):
                pass

        class EXPONENTIAL(object):
            def __init__(self, decay):
                self._decay = decay

    class USE_STATISTICAL_MODEL(object):
        def __init__(self, boolean=False):
            self._boolean = boolean

    def __init__(self):
        self._regression_type = None
        self._window_type = None
        self._orthogonalize = None
        self._weighting_scheme = None
        self._set_factor_Sharpe_cap = None

    def set_regression_type(self, regression_type: REGRESSION_TYPE):
        self._regression_type = regression_type

    def set_window_type(self, window_type: WINDOW_TYPE):
        self._window_type = window_type

    def set_orthogonalize(self, orthogonalize: ORTHOGONALIZE):
        self._orthogonalize = orthogonalize

    def set_weighting_scheme(self, weighting_scheme: WEIGHTING_SCHEME):
        self._weighting_scheme = weighting_scheme

    def use_statistical_model(self, use_statistical_model: WEIGHTING_SCHEME):
        self._use_statistical_model = use_statistical_model

    def set_risk_factor_list(self, risk_factor_list):
        pass

    def set_return_factor_list(self, return_factor_list):
        pass

    def set_factor_Sharpe_cap(self, factor, Sharpe_cap):
        self._factor_Sharpe_cap[factor] = Sharpe_cap

    def __set_factor_panels(self):

    def create_model(self, end_date, frequency):
        CModelFactory
        pass


if __name__ == "__main__":

    model_factory = CModelFactory()
    model = model_factory.create_model(frequency=Frequency.BUSINESS_MONTHLY,
                                       end_date=datetime.date(year=2022, month=12, day=31))