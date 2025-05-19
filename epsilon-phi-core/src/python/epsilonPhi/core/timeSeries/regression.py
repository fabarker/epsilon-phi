from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from epsilonPhi.core.dataModel.enums.Model import REGRESSION_TYPE, WEIGHTING_SCHEME, SAMPLING_TYPE
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import statsmodels.api as sm
from sklearn.svm import SVR
import xgboost as xgb
import pandas as pd
import numpy as np
import copy

class Regression(object):

    REGRESSION_TYPES = REGRESSION_TYPE

    class WEIGHTING_SCHEME(object):

        class EQUAL(object):
            def weights(self, n):
                return None

        class EXPONENTIAL(object):
            def __init__(self, half_life):
                self._half_life = half_life
                self._decay = np.power(0.5, 1/self._half_life)

            def weights(self, n):
                return np.power(self._decay, np.arange(n)) / np.sum(np.power(self._decay, np.arange(n)))

    class SAMPLING_TYPE(object):
        class STATIC(object):
            def __init__(self):
                pass

            def samples(self, n):
                return [np.arange(0, n)]
        class ROLLING(object):
            def __init__(self, window):
                self._param = window

            def samples(self, n):
                return [np.arange(i, i + self._param) for i in range(n - self._param + 1)]

        class EXPANDING(object):
            def __init__(self, initial_window_size):
                self._param = initial_window_size

            def samples(self, n):
                return [np.arange(0, i + self._param) for i in range(n - self._param + 1)]


    def __init__(self,
                 regression_type=None,
                 weighting_scheme=None,
                 sampling_type=None,
                 **kwargs):

        self._type = regression_type
        if isinstance(weighting_scheme, type):
            self._weighting_scheme = weighting_scheme()
        else:
            self._weighting_scheme = weighting_scheme

        if isinstance(sampling_type, type):
            self._sampling_type = sampling_type()
        else:
            self._sampling_type = sampling_type

        self._kwargs = kwargs
        self._setup()

    def _setup(self):
        if self._type == REGRESSION_TYPE.OLS:
            self._reg = LinearRegression(**self._kwargs)
        elif self._type == REGRESSION_TYPE.RIDGE:
            self._reg = Ridge(**self._kwargs)
        elif self._type == REGRESSION_TYPE.LASSO:
            self._reg = Lasso(**self._kwargs)
        elif self._type == REGRESSION_TYPE.ELASTIC_NET:
            self._reg = ElasticNet(**self._kwargs)
        elif self._type == REGRESSION_TYPE.SVR:
            self._reg = SVR(**self._kwargs)
        elif self._type == REGRESSION_TYPE.DECISION_TREE_REGERSSOR:
            self._reg = DecisionTreeRegressor(**self._kwargs)
        elif self._type == REGRESSION_TYPE.XGBOOST:
            self._reg = xgb.XGBRegressor(**self._kwargs)
        elif self._type == REGRESSION_TYPE.RANDOM_FOREST_REGRESSION:
            self._reg = RandomForestRegressor(**self._kwargs)
        else:
            raise ValueError('Error - regression type {} not supported'.format(self._type))


    def regress(self, X, y, orthogonalize_columns=None, normalize=False):
        T, N = X.shape
        assert T == y.shape[0], 'Error - X {} and y {} lengths are inconsistent'.format(X.shape[0], y.shape[0])

        _samples = self._sampling_type.samples(T)
        pars = np.full((len(_samples), N + 1), np.nan)

        for ctr, s in enumerate(_samples):

            # Get the weighting scheme for the regression problem
            wts = self._weighting_scheme.weights(len(s))

            # sample predictors
            x_prime = X.iloc[s]

            # Orthogonalize and/or normalize columns where necessary
            if orthogonalize_columns is not None:
                x_prime = self.orthogonalize_columns(x_prime, orthogonalize_columns)
            if normalize:
                x_prime = self.normalize_columns(x_prime)

            # Fitting the model
            self._fit(x_prime, y.iloc[s].values.flatten(), wts)
            pars[ctr, :] = self.params
        return pars

    def _fit(self, X, y, sample_weight=None):

        """
        Fit the regression model to the provided data.
        For traditional models, this typically involves finding coefficients that minimize a loss function.

        :param X: Independent variables (features)
        :param y: Dependent variable (target)
        """
        self._reg.fit(X, y, sample_weight=sample_weight)

    def predict(self, X):

        """
        Predict the target variable using the regression model.

        :param X: Independent variables (features) for which predictions are to be made
        :return: Predicted values of the dependent variable
        """

        return self._reg.predict(X)

    def score(self, X, y):

        """
        For most models, this method returns the coefficient of determination (R^2) of the prediction.

        :param X: Independent variables (features)
        :param y: Dependent variable (target)
        :return: Score indicating the performance of the model
        """

        return self._reg.score(X, y)

    def orthorgonalize(self, X, y):

        """
        Calculate and return the residuals of the model. Residuals are the differences
        between the observed values (y) and the values predicted by the model.

        :param X: Independent variables (features) used for prediction
        :param y: Actual observed values of the dependent variable
        :return: Residuals (difference between observed and predicted values)
        """

        self._reg.fit(X, y)
        residuals = y - X @ self._reg.coef_
        return residuals


    def residuals(self, X, y):

        """
        Calculate and return the residuals of the model. Residuals are the differences
        between the observed values (y) and the values predicted by the model.

        :param X: Independent variables (features) used for prediction
        :param y: Actual observed values of the dependent variable
        :return: Residuals (difference between observed and predicted values)
        """

        self._reg.fit(X, y)
        predictions = self.predict(X)
        residuals = y - predictions
        return residuals

    @property
    def params(self):
        """
        Retrieve the regression parameters of the model.

        :return: Coefficients of the model. For some models like Decision Trees,
                         this might not be applicable.
        """
        return np.hstack((self._reg.intercept_, self._reg.coef_))

    @property
    def betas(self):
        """
        Retrieve the regression coefficients (betas) of the model.

        :return: Coefficients of the model. For some models like Decision Trees,
                 this might not be applicable.
        """

        if hasattr(self._reg, 'coef_'):
            return self._reg.coef_
        else:
            raise NotImplementedError("This model type does not support extraction of coefficients.")

    @property
    def alpha(self):
        """
        Retrieve the intercept (alpha) of the model, if applicable.

        :return: Intercept of the model. For models without an intercept or
                 where it's not applicable (like Decision Trees), this might return None.
        """

        if hasattr(self._reg, 'intercept_'):
            return self._reg.intercept_
        else:
            # For models without an intercept or where it's not applicable
            return None

    def normalize_columns(self, df, normalize=True):

        if normalize is False:
            return df

        # Calculate the standard deviation of each column
        sig = np.std(df, ddof=1, axis=0)

        # Check if any standard deviation is zero
        if np.any(sig == 0):
            # If any standard deviation is zero, return the original DataFrame
            return df
        else:
            # Normalize the DataFrame columns
            return df / sig

    def orthogonalize_columns(self, df, columns):

        if columns is None:
           return df

        copy_df = df.copy()
        cols = np.intersect1d(columns, copy_df.columns)
        for col in cols:
            copy_df.loc[:, col] = self.residuals(copy_df.loc[:, np.setdiff1d(copy_df.columns, col)], copy_df.loc[:, col])
        return copy_df

    @staticmethod
    def regress_against_factors(y, X):

        reg = Regression(Regression.REGRESSION_TYPES.OLS,
                         Regression.WEIGHTING_SCHEME.EQUAL,
                         Regression.SAMPLING_TYPE.STATIC)

        regstats = reg.regress(X, y)
        return np.hstack((reg._reg.intercept_, reg._reg.coef_))

    @staticmethod
    def simple_regression_OLS_with_array(
            X: np.ndarray,
            y: np.array
    ):

        if not len(X) == len(y):
            raise Exception('X and y must have same length')

        X = np.append(X, np.ones((X.shape[0], 1), dtype=int), axis=1)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

        alpha = beta[-1]
        betas = beta[:X.shape[1]-1]
        return alpha, betas






if __name__ == "__main__":

    from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
    gds = GlobalDataSource()

    y = gds.get_time_series_data_from_ticker('HFRIEMN', 'RI')
    X = gds.get_time_series_data_from_ticker('MSWRLDL', 'RI').loc[y.index]

    y_prime = y.get_returns()
    X_prime = X.get_returns()


    # Regression
    reg = Regression(Regression.REGRESSION_TYPES.OLS,
                     Regression.WEIGHTING_SCHEME.EQUAL,
                     Regression.SAMPLING_TYPE.ROLLING(60))


    alpha, betas = reg.regress(X_prime, y_prime)

    # Getting alpha and beta
    beta = reg.betas()
    alpha = reg.alpha()
    res = reg.residuals(X, y)


    print(f"Alpha (intercept): {alpha}")
    print(f"Beta (coefficient of S&P 500): {beta}")

