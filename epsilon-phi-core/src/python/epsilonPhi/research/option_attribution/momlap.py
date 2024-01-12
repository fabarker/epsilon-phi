import numpy as np

def momlap(b, X):
    nobs, nvars = X.shape
    y = X[:, nvars - 1]
    X2 = X[:, 0:nvars - 1]
    e = X2 * ((y - X2 @ b)[:, np.newaxis] * np.ones((1, nvars - 1)))
    return e
