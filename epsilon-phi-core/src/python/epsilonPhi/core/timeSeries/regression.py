import copy
import statsmodels.api as sm
import numpy as np
import pandas as pd
import numba

class Regression(object):

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







