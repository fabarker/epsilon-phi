import pandas as pd
from typing import Optional
from epsilonPhi.core.factor.factorPanel import CFactorPanels
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.dataModel.enums.Factor import FACTOR
from epsilonPhi.core.dataModel.enums.Model import REGRESSION_TYPE, SAMPLING_TYPE, WEIGHTING_SCHEME
from epsilonPhi.core.timeSeries.regression import Regression
from epsilonPhi.core.config.appConfig import CAppConfig
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import datetime as dt
import numpy as np

class BaseModel(object):

    __DEFAULT_RISK_FACTORS = FACTOR.get_default_risk_factor_list()
    __DEFAULT_RETURN_FACTORS = FACTOR.get_default_return_factor_list()
    __DEFAULT_END_DATE = dt.date(year=2022, month=12, day=31)
    __DEFAULT_SHARPE_FACTOR_END_DATE = dt.date(year=2018, month=12, day=31)
    __DEFAULT_FREQUENCY = Frequency.BUSINESS_MONTHLY

    def __init__(self,
                 frequency=None,
                 end_date=None,
                 ):

        if frequency is None:
            self.__frequency = BaseModel.__DEFAULT_FREQUENCY
        else:
            self.__frequency = frequency

        if end_date is None:
           self.__end_date = BaseModel.__DEFAULT_END_DATE
        else:
           self.__end_date = end_date

        # List of return factors
        self.__RISK_FACTORS_LIST = []
        self.__RETURN_FACTORS_LIST = []

        # Associated with factor model spec
        self.__regression_type = None
        self.__sampling_type = None
        self.__orthogonalize_list = None
        self.__weighting_scheme = None
        self.__use_statistical_model = False
        self.__factor_Sharpe_cap = {}
        self._factorPanels = None
        self.__regression = None

    def create_model(self):
        self.__factorPanels = CFactorPanels(self.factor_list,
                                            self.frequency,
                                            self.end_date)
        self.__factorPanels.load_factors()

    @staticmethod
    def setup_default_model(frequency=None, end_date=None, cache_model=False):

        mdl = BaseModel(frequency, end_date)
        mdl.set_risk_factor_list(mdl.__DEFAULT_RISK_FACTORS)
        mdl.set_return_factor_list(mdl.__DEFAULT_RETURN_FACTORS)
        mdl.set_orthogonalize([FACTOR.EQUITY_EMERGING_ISG.name, FACTOR.FUNDING_US_ISG.name])
        mdl.set_regression_type(Regression.REGRESSION_TYPES.OLS, Regression.SAMPLING_TYPE.ROLLING(60), Regression.WEIGHTING_SCHEME.EQUAL)
        mdl.set_factor_Sharpe_cap(FACTOR.EQUITY_EMERGING_ISG.name, 0.2)
        mdl.create_model()

        if cache_model:
           CAppConfig._BaseModel = mdl
        else:
           return mdl

    # Public Properties
    @property
    def factor_panels(self):
        return self.__factorPanels
    @property
    def end_date(self):
        return self.__end_date
    @property
    def frequency(self):
        return self.__frequency
    @property
    def risk_factor_df(self):
        return None
    @property
    def return_factor_df(self):
        return None
    @property
    def return_factor_list(self):
        return self.__return_factor_list

    @property
    def risk_factor_list(self):
        return self.__risk_factor_list
    @property
    def factor_list(self):
        return list(np.unique(self.__return_factor_list + self.__risk_factor_list))
    @property
    def regression_type(self):
        return self.__regression_type
    @property
    def sampling_type(self):
        return self.__sampling_type
    @property
    def is_statistical_model(self):
        return self.__use_statistical_model
    @property
    def factor_Sharpe_caps(self):
        return self.__factor_Sharpe_cap
    @property
    def regression(self):
        return self.__regression
    @property
    def orthogonal_list(self):
        return self.__orthogonalize_list
    def is_orthogonalized(self, factor_name):
        return factor_name in self.__orthogonalize_list

    ############ setter methods ###############

    def set_regression_type(self,
                            regression_type: REGRESSION_TYPE,
                            sampling:  Optional[SAMPLING_TYPE] = None,
                            weights: Optional[WEIGHTING_SCHEME] = None,
                            **kwargs):

        """
        Sets the regression type and optionally specifies sampling and weighting schemes.

        :param regression_type: The type of regression model to be used.
        :param sampling: Optional sampling type to be used.
        :param weights: Optional weighting scheme to be applied.
        :param kwargs: Additional keyword arguments for the regression model.
        """

        self.__regression = Regression(regression_type,
                                       weights,
                                       sampling,
                                       **kwargs)

    def set_orthogonalize(self, orthogonalize_list):
        self.__orthogonalize_list = orthogonalize_list

    def use_statistical_model(self, use_statistical_model: bool):
        self.__use_statistical_model = use_statistical_model

    def set_risk_factor_list(self, risk_factor_list):

        assert isinstance(risk_factor_list, list), 'ERROR - Must be list of return factor enums'
        boolean = [isinstance(x, FACTOR) for x in risk_factor_list]
        assert np.all(boolean), 'Error - factor list must be a list of FACTOR enumerators'
        self.__risk_factor_list = [x.name for x in risk_factor_list]

    def set_return_factor_list(self, return_factor_list):

        assert isinstance(return_factor_list, list), 'ERROR - Must be list of return factor enums'
        boolean = [isinstance(x, FACTOR) for x in return_factor_list]
        assert np.all(boolean), 'Error - factor list must be a list of FACTOR enumerators'
        self.__return_factor_list = [x.name for x in return_factor_list]

    def set_factor_Sharpe_cap(self, factor, Sharpe_cap):
        self.__factor_Sharpe_cap[factor] = Sharpe_cap

    ###################### PUBLIC METHODS ######################
    def get_return_factor_df(self, orthogonalize=False):

        if orthogonalize:
            df_ = self.factor_panels.get_factors_df(self.return_factor_list)
            return self.regression.orthogonalize_columns(df_, self.__orthogonalize_list)
        return self.factor_panels.get_factors_df(self.return_factor_list)

    def get_risk_factor_df(self, orthogonalize=False):

        if orthogonalize:
            df_ = self.factor_panels.get_factors_df(self.risk_factor_list)
            return self.regression.orthogonalize_columns(df_, self.__orthogonalize_list)
        return self.factor_panels.get_factors_df(self.risk_factor_list)

    def get_factor(self, factor_name, orthogonalized=False):

        if orthogonalized:
            df_ = self.factor_panels.get_factors_df(self.return_factor_list)

            X = df_.get(np.setdiff1d(self.return_factor_list, self.orthogonal_list))
            y = self.factor_panels.get_factor(factor_name).select_subset_dates(X.index)
            return self.regression.orthorgonalize(X, y)
        return self.factor_panels.get_factor(factor_name)

    def get_return_factor_Sharpe_ratios(self):
        return pd.DataFrame([self.get_return_factor_Sharpe_ratio(x) for x in self.return_factor_list],
                             index=self.return_factor_list)

    def get_return_factor_Sharpe_ratio(self, factor_name):

        if self.is_orthogonalized(factor_name):
            sr = self.get_factor(factor_name, True).get_historical_Sharpe()
        else:
            sr = self.get_factor(factor_name).get_historical_Sharpe()
        return np.minimum(sr, self.factor_Sharpe_caps.get(factor_name, np.inf))

    def get_return_factor_Sharpe_ratios_uncapped(self):
        return pd.DataFrame([self.get_return_factor_Sharpe_ratio_uncapped(x) for x in self.return_factor_list],
                             index=self.return_factor_list)

    def get_return_factor_Sharpe_ratio_uncapped(self, factor_name):

        if self.is_orthogonalized(factor_name):
            return self.get_factor(factor_name, True).get_historical_Sharpe()
        else:
            return self.get_factor(factor_name).get_historical_Sharpe()

    def get_risk_factor_covariance(self, dates, orthogonalize=True):
        panel = self.get_risk_factor_df(orthogonalize).loc[dates]
        return panel.cov()

    def get_risk_factor_correlation(self, dates):
        panel = self.get_risk_factor_df().loc[dates]
        return panel.corr()

    def get_factor_panels(self):
        return self.__factorPanels

    def get_principle_component_factors(self):
        df_ = self.get_risk_factor_df()
        cov_mat = np.cov(df_, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Project the data onto the principal components
        pca_result = np.dot(df_, eigenvectors)

        return pd.DataFrame(pca_result, index=df_.index)


        # Methods associated with factors


if __name__ == "__main__":

    return_factors = FACTOR.get_default_return_factor_list()

    model = BaseModel.setup_default_model()
    factors = model.get_principle_component_factors()

    model.set_return_factor_list(FACTOR.get_default_return_factor_list())
    model.set_risk_factor_list(FACTOR.get_default_risk_factor_list())
    model.create_model()

