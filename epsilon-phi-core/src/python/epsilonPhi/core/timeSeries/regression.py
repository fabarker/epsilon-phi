from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from epsilonPhi.core.dataModel.enums.Model import REGRESSION_TYPE
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import statsmodels.api as sm
from sklearn.svm import SVR
import xgboost as xgb
import pandas as pd
import numpy as np
import copy


class Regression(object):
    def __init__(self, regression_type, **kwargs):

        self._regression_type = regression_type
        self._kwargs = kwargs
        self._setup()

    def _setup(self):
        if self._regression_type == REGRESSION_TYPE.OLS:
            self._reg = LinearRegression(**self._kwargs)
        elif self._regression_type == REGRESSION_TYPE.RIDGE:
            self._reg = Ridge(**self._kwargs)
        elif self._regression_type == REGRESSION_TYPE.LASSO:
            self._reg = Lasso(**self._kwargs)
        elif self._regression_type == REGRESSION_TYPE.ELASTIC_NET:
            self._reg = ElasticNet(**self._kwargs)
        elif self._regression_type == REGRESSION_TYPE.SVR:
            self._reg = SVR(**self._kwargs)
        elif self._regression_type == REGRESSION_TYPE.DECISION_TREE_REGERSSOR:
            self._reg = DecisionTreeRegressor(**self._kwargs)
        elif self._regression_type == REGRESSION_TYPE.XGBOOST:
            self._reg = xgb.XGBRegressor(**self._kwargs)
        elif self._regression_type == REGRESSION_TYPE.RANDOM_FOREST_REGRESSION:
            self._reg = RandomForestRegressor(**self._kwargs)
        else:
            raise ValueError('Error - regression type {} not supported'.format(self._regression_type))

    def fit(self, X, y):

        """
        Fit the regression model to the provided data.
        For traditional models, this typically involves finding coefficients that minimize a loss function.

        :param X: Independent variables (features)
        :param y: Dependent variable (target)
        """

        self._reg.fit(X, y)

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

    def orthogonalize_columns(self, df, columns):

        cols = np.intersect1d(columns, df.columns)
        for col in cols:
            self.fit(df.get(np.setdiff1d(df.columns, cols)), df.get(col))
            df[col] = self.residuals(X, y)
        return df.copy


    @staticmethod
    def simple_regression_fast_OLS(X: np.array, y: np.array, intercept=True) -> np.array:

        """
        Perform a fast OLS regression.
        :param X: Independent variable(s) as a 2D array.
        :param y: Dependent variable as a 1D array.
        :param intercept: If True, an intercept will be added.
        :return: Coefficients as an array.
        """

        if intercept:
            # Add a column of ones to X for the intercept term
            ones = np.ones((X.shape[0], 1))
            X = np.hstack((ones, X))

        return np.linalg.solve(X.T @ X, X.T @ y)

    @staticmethod
    def get_residuals_from_OLS_fit(X: np.array, y: np.array, intercept=True) -> np.array:
        betas = Regression.simple_regression_fast_OLS(X, y, intercept)
        return y - X @ betas

    @staticmethod
    def zero_intercept_regression(X: np.array, y: np.array) -> np.array:
        """
        Perform a fast OLS regression without an intercept.
        :param X: Independent variable(s) as a 2D array.
        :param y: Dependent variable as a 1D array.
        :return: Coefficients as an array.
        """
        return Regression.simple_regression_fast_OLS(X, y)

    @staticmethod
    def simple_weighted_regression(X: np.array, y: np.array, w: np.array) -> np.array:

        """
        Perform a weighted least squares regression.

        :param X: Independent variable(s) as a 2D array.
        :param y: Dependent variable as a 1D array.
        :param w: Weights as a 1D array.
        :return: Coefficients as an array.
        """

        try:
            y = y[y.size-X.size:]
            w = w[w.size-w.size:]
            X = np.stack((np.ones(X.size, dtype=int), X), axis=-1)
            result = sm.WLS(y, X, w).fit()
            if result is not None:
                return result.params
        except Exception as e:
            print(e)
            return None

    @staticmethod
    def regress_against_factors(y: pd.Series, X: pd.DataFrame) -> np.array:
        """
        Perform a fast OLS regression using pandas data structures.
        :param X: Independent variable(s) as a DataFrame.
        :param y: Dependent variable as a Series.
        :return: Coefficients as an array.
        """
        assert np.all(X.index == y.index), 'Error - date misalignment'
        return Regression.simple_regression_fast_OLS(X.values,
                                                     y.values,
                                                     True)

    @staticmethod
    def simple_regression_OLS_with_array(X: np.array, y: np.array) -> np.array:
        """
        Perform an OLS regression.

        :param X: Independent variable(s) as a 2D array.
        :param y: Dependent variable as a 1D array.
        :return: Coefficients as an array.
        """
        if len(X) != len(y):
            raise Exception('Error - length X and y dont match')

        rows, _ = X.shape
        augmented_X = np.empty((rows, X.shape[1] + 1), dtype=X.dtype)
        augmented_X[:, :-1] = X
        augmented_X[:, -1] = 1
        reg = np.linalg.lstsq(augmented_X, y, rcond=None)
        return reg[0]

    @staticmethod
    def rolling_regression(X: np.array,
                           y: np.array,
                           orthog_cols: np.array,
                           sample_size: int,
                           normalize=True):

        idx = np.array(range(sample_size))
        while max(idx) < y.size:
            reg = Regression.simple_regression_OLS_orthogonalized(X[idx],
                                                                  y[idx],
                                                                  orthog_cols,
                                                                  normalize)
            idx += 1
        return reg

    @staticmethod
    def simple_regression_OLS_orthogonalized(X: np.array,
                                             y: np.array,
                                             orthog_cols: np.array,
                                             normalize=True):

           Xs = copy.deepcopy(X)
           for col in orthog_cols:
               res = Regression.simple_regression_fast_OLS(X[:,col],
                                                           X[:,col])

               Xs[:,col] = Xs[:,col] - Xs[:, ~col].dot(res.params)

           if normalize:
              Xs = Xs / np.std(Xs, axis=0, ddof=1)

           return Regression.simple_regression_fast_OLS(Xs, y)

    @staticmethod
    def rolling_regression_generic(X: np.array,
                                   y: np.array,
                                   orthog_cols: np.array,
                                   sample_size: int):
        return Regression.rolling_regression(X,
                                             y,
                                             orthog_cols,
                                             sample_size,
                                             normalize=False)





if __name__ == "__main__":

    np.random.seed(0)  # For reproducibility

    # Assume 250 trading days
    dates = pd.date_range(start='2020-01-01', periods=250, freq='B')
    sp500_returns = np.random.normal(loc=0.0008, scale=0.01, size=len(dates))  # Daily returns for S&P 500
    stock_returns = np.random.normal(loc=0.0009, scale=0.02, size=len(dates))  # Daily returns for our stock

    # Create a DataFrame
    data = pd.DataFrame({
        'S&P500': sp500_returns,
        'Stock': stock_returns
    }, index=dates)

    # Regression
    reg = Regression(REGRESSION_TYPE.OLS)
    X = data[['S&P500']]  # Independent variable (S&P 500 returns)
    y = data['Stock']  # Dependent variable (Stock returns)

    reg.fit(X, y)

    # Getting alpha and beta
    beta = reg.betas()
    alpha = reg.alpha()
    res = reg.residuals(X, y)


    print(f"Alpha (intercept): {alpha}")
    print(f"Beta (coefficient of S&P 500): {beta}")

