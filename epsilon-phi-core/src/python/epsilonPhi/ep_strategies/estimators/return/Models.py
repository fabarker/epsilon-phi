from epsilonPhi.core.dataModel.enums.Model import PREDICTOR_TYPE
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Lars, RANSACRegressor, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb


class EqualWeighted(object):
    def predict(self, Xs):
        return Xs.mean()

    def fit(self, signals, y):
        self._fitted = signals.copy()


class PySystemTrade(object):
    _average_abs_signal = 10

    def __init__(self, **kwargs):
        self._fitted = None

    def predict(self, Xs):
        return np.mean((Xs.flatten() * self._fitted).clip(-20, 20))

    def fit(self, signals, y):
        self._fitted = PySystemTrade._average_abs_signal / signals.abs().mean(axis=0)


# This function takes a set of signals and returns and builds a model
# describing the returns from the signals
class Predictor(object):
    def __init__(self, model_type, **kwargs):

        self._model_type = model_type
        self._kwargs = kwargs

        self._model = None
        self._setup = False

        self._y = None
        self._X = None

    def _setup_model(self):



        if self._model_type == PREDICTOR_TYPE.EQUAL_WEIGHTED:
            self._model = EqualWeighted(**self._kwargs)
        elif self._model_type == PREDICTOR_TYPE.OLS_REGRESSION:
            self._model = LinearRegression(**self._kwargs)
        elif self._model_type == PREDICTOR_TYPE.RIDGE_REGRESSION:
            self._model = Ridge(**self._kwargs)
        elif self._model_type == PREDICTOR_TYPE.LASSO_REGRESSION:
            self._model = Lasso(**self._kwargs)
        elif self._model_type == PREDICTOR_TYPE.ELASTIC_NET_REGRESSION:
             self._model = ElasticNet(**self._kwargs)
        elif self._model_type == PREDICTOR_TYPE.SUPPORT_VECTOR_REGRESSION:
             self._model = SVR(**self._kwargs)
        elif self._model_type == PREDICTOR_TYPE.DECISION_TREE_REGERSSIOM:
            self._model = DecisionTreeRegressor(**self._kwargs)
        elif self._model_type == PREDICTOR_TYPE.XGBOOST_REGRESSION:
            self._model = xgb.XGBRegressor(**self._kwargs)
        elif self._model_type == PREDICTOR_TYPE.RANDOM_FOREST_REGRESSION:
            self._model = RandomForestRegressor(**self._kwargs)
        elif self._model_type == PREDICTOR_TYPE.LEAST_ANGLE_REGERSSION:
            self._model = Lars(**self._kwargs)
        elif self._model_type == PREDICTOR_TYPE.ROBUST_REGRESSION:
            self._model = RANSACRegressor(**self._kwargs)
        elif self._model_type == PREDICTOR_TYPE.LOGISTIC_REGRESSION:
            self._model = LogisticRegression(**self._kwargs)
        elif self._model_type == PREDICTOR_TYPE.SUPPORT_VECTOR_MACHINES:
            self._model = SVC(**self._kwargs)
        elif self._model_type == PREDICTOR_TYPE.K_NEAREST_NEIGHBOUR:
            self._model = KNeighborsClassifier(**self._kwargs)
        elif self._model_type == PREDICTOR_TYPE.DECISION_TREE_CLASSIFIER:
            self._model = DecisionTreeClassifier(**self._kwargs)
        elif self._model_type == PREDICTOR_TYPE.ARTIFICIAL_NEURAL_NETWORK_CLASSIFIER:
            self._model = MLPClassifier(**self._kwargs)
        elif self._model_type == PREDICTOR_TYPE.BAGGING_REGRESSOR:
            self._model = BaggingRegressor(**self._kwargs)
        elif self._model_type == PREDICTOR_TYPE.BAGGING_CLASSIFICATION:
            self._model = BaggingClassifier(**self._kwargs)
        elif self._model_type == PREDICTOR_TYPE.NAIVE_BAYES_CLASSIFIER:
            self._model =  GaussianNB(**self._kwargs)
        elif self._model_type == PREDICTOR_TYPE.XGBOOST_CLASSIFIER:
            self._model = xgb.XGBClassifier(**self._kwargs)
        else:
            raise ValueError('Error - regression type {} not supported'.format(self._model_type))

        self._setup = True

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    def prepare_variables(self, X, y):
        return X.intersect_over_dates(y)

    def fit_and_predict(self, X, y, Xs):
        self.fit(X, y)
        return self.predict(Xs.reshape(1, -1))

    def fit(self, X, y):

        # remove nans
        nan_locs_X = X.isna().any(axis=1)  # Find rows in X with any NaN values
        nan_locs_y = y.isna().any(axis=1)  # Find NaN values in y

        # Combine the two to get locations where either X or y has NaN values
        nan_locs = nan_locs_X | nan_locs_y

        X_prime = X[~nan_locs]
        y_prime = y[~nan_locs]

        if not self._setup:
            self._setup_model()

        self._model.fit(X_prime, y_prime)
        self._X = X_prime.copy()
        self._y = y_prime.copy()

    def predict(self, X=None):
        if not self._setup:
            self._setup_model()

        if X is not None:
            return self._model.predict(X)
        else:
            # Assuming self._X is defined elsewhere in your class to hold default prediction data
            return self._model.predict(self._X)

    def score(self, X=None, y=None):
        if X and y:
          self.fit(X, y)
        return self._model.score(self._X, self._y)

    def residuals(self, X=None, y=None):
        if X and y:
          self.fit(X, y)
        return (self._y -
                predictor.predict(self._X))




if __name__ == "__main__":

    import numpy as np
    from epsilonPhi.core.dataModel.dataSources.futures.Futures import Futures
    from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
    from epsilonPhi.ep_strategies.futures.signals.BaseSignals import SignalFactory

    gds = GlobalDataSource()

    future = Futures()
    prices = future.get_futures_continuous_series_settlement_price('BEJ')
    price_rx = prices.get_returns()

    emac_4_16 = SignalFactory.univariate_signal.exponential_moving_average_crossover(prices, fast=4, slow=16)
    emac_16_32 = SignalFactory.univariate_signal.exponential_moving_average_crossover(prices, fast=16, slow=32)
    emac_32_250 = SignalFactory.univariate_signal.exponential_moving_average_crossover(prices, fast=32, slow=250)
    signals = SignalFactory.multivariate_signal(emac_4_16,
                                                emac_16_32,
                                                emac_32_250)

    emac_4_16.get_normalized_signal()


    common_dates = np.intersect1d(price_rx.dates, factor.index)
    X = factor.loc[common_dates]
    y = price_rx.loc[common_dates]

    predictor = Predictor(PREDICTOR_TYPE.OLS_REGRESSION)

    # Regress next period asset return from current period factor returns,
    # that is, we use todays factor return to predict tomorrow asset return
    predictor.fit(X.shift(1), y)

    # Make a prediction
    predicted_returns = predictor.predict(X)

