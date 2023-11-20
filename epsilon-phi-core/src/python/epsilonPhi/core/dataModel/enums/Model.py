from enum import Enum

class REGRESSION_TYPE(Enum):

    OLS = 1
    RIDGE = 2
    LASSO = 3
    ELASTIC_NET = 4
    ROBUST = 5
    BAYESIAN = 6
    SVR = 7
    DECISION_TREE_REGERSSOR = 8
    XGBOOST = 9
    RANDOM_FOREST_REGRESSION = 10
    KERAS_RESNET = 11


class SAMPLING_TYPE(Enum):

    ROLLING = 1
    EXPANDING = 2
    STATIC = 3

class WEIGHTING_SCHEME(Enum):

    EQUAL = 1
    EXPONENTIAL = 2