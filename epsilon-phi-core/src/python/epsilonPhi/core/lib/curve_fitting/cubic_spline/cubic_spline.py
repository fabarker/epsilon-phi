import numpy as np
import numba
from numba import float64

@numba.njit(cache=True, fastmath=True)
def calc_spline_params(x, y):

    n = x.size - 1
    a = y.copy()
    h = x[1:] - x[:-1]
    alpha = 3 * ((a[2:] - a[1:-1]) / h[1:] - (a[1:-1] - a[:-2]) / h[:-1])
    c = np.zeros(n + 1)
    ell, mu, z = np.ones(n + 1), np.zeros(n), np.zeros(n + 1)
    for i in range(1, n):
        ell[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / ell[i]
        z[i] = (alpha[i - 1] - h[i - 1] * z[i - 1]) / ell[i]
    for i in range(n - 1, -1, -1):
        c[i] = z[i] - mu[i] * c[i + 1]
    b = (a[1:] - a[:-1]) / h + (c[:-1] + 2 * c[1:]) * h / 3
    d = np.diff(c) / (3 * h)

    return a[1:], b, c[1:], d, x

@numba.njit(cache=True, fastmath=True,)
def func_spline(x, ix, x0, a, b, c, d):
    dx = x - x0[1:][ix]
    return a[ix] + (b[ix] + (c[ix] + d[ix] * dx) * dx) * dx


@numba.njit(cache=True, fastmath=True)
def searchsorted_merge(a, b, sort_b):
    idx = np.zeros((len(b),), dtype=np.int64)
    if sort_b:
        ib = np.argsort(b)
    pa, pb = 0, 0
    while pb < len(b):
        if pa < len(a) and a[pa] < (b[ib[pb]] if sort_b else b[pb]):
            pa += 1
        else:
            idx[pb] = pa
            pb += 1
    return idx


@numba.njit(cache=True, fastmath=True)
def piece_wise_spline(x, x0, a, b, c, d):
    x = np.asarray(x)
    xsh = x.shape
    x = x.ravel()
    ix = searchsorted_merge(x0[1: -1], x, True)
    y = func_spline(x, ix, x0, a, b, c, d)
    y = np.ascontiguousarray(y).reshape(xsh)
    return y


@numba.njit(fastmath=True, cache=True)
def cubic_spline(x0, y0, x):
    a, b, c, d, e = calc_spline_params(x0, y0)
    r = piece_wise_spline(x, x0, a, b, c, d)
    return r

def fit_spline(x, y):

    x_ = x[~np.isnan(y)]
    y_ = y[~np.isnan(y)]

    _loc = np.argsort(x_)
    x_sorted = x_[_loc]
    y_sorted = y_[_loc]
    p = calc_spline_params(x_sorted, y_sorted)
    return p


@numba.njit(cache=True)
def fit_cubic_spline(x, y):

    # 1. Remove Nans...
    x_ = x[~np.isnan(y)]
    y_ = y[~np.isnan(y)]

    # 2. Keep it Semi-Arbitrage Free
    if ((np.any(np.abs(x_) == 0.5)) and
                (np.all(y_[np.abs(x_) == 0.5] > y_[np.abs(x_) != 0.5]))):
        y_ = y_[np.abs(x_) != 0.5]
        x_ = x_[np.abs(x_) != 0.5]

    if len(x_) < 4:
        p = (np.array([np.nan]), np.array([np.nan]),
             np.array([np.nan]), np.array([np.nan]), np.array([np.nan]))
        return p

    _loc = np.argsort(x_)
    x_sorted = x_[_loc]
    y_sorted = y_[_loc]
    p = calc_spline_params(x_sorted, y_sorted)
    return p

@numba.njit(fastmath=True, cache=True)
def eval_cubic_spline(x, params):
    val = piece_wise_spline(x,
                            params[4],
                            params[0],
                            params[1],
                            params[2],
                            params[3])
    return val

