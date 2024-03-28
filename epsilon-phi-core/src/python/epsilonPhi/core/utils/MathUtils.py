from numba import njit, float64, vectorize
from numba.np.arraymath import binary_search_with_guess
import warnings
import numpy as np
from numba.core.errors import NumbaPendingDeprecationWarning
from epsilonPhi.core.lib.curve_fitting.cubic_spline.cubic_spline import cubic_spline as cubicspline

# Suppress NumbaPendingDeprecationWarning
warnings.filterwarnings('ignore', category=NumbaPendingDeprecationWarning)

PI = 3.14159265358979323846
inv_root_two_pi = 0.3989422804014327

ONE_MILLION = 1000000
TEN_MILLION = 10000000
ONE_BILLION = 1000000000


##@njit(float64[:, :](float64[:], float64[:, :], float64[:]), fastmath=True, cache=True)
def flat_forward_interp(x_fix, y_fix_, x_var):

    T = y_fix_.shape[0]

    _nan_locs = ~np.isnan(y_fix_)
    y_prime = np.full((T, len(x_var)), np.nan)
    for t in range(T):

        # Throw away any tenors that have nan's
        y = y_fix_[t, _nan_locs[t]]
        x = x_fix[_nan_locs[t]]

        if len(y) == 0:
           continue

        keep_cols = (x_var <= np.max(x)) & (x_var >= np.min(x))
        _x_var = x_var[keep_cols]

        # If we have sufficient observations, interpolate
        if len(_x_var) > 0:

            # Interpolate all the target maturities
            x_repeat = np.repeat(_x_var[:, None], len(x)).reshape((len(_x_var), len(x)))
            distances = np.abs(x_repeat - x)
            x_indices = np.searchsorted(x, _x_var)

            weights = np.zeros_like(distances)
            idx = np.sum(distances == 0, axis=1) == 0.0
            weights[distances == 0] = 1.0

            weights[idx, x_indices[idx] - 1] = x_fix[x_indices[idx] - 1] / _x_var[idx]
            weights[idx, x_indices[idx]] = (1.0 - x_fix[x_indices[idx] - 1] / _x_var[idx])
            weights /= np.sum(weights, axis=1)[:, None]
            y_prime[t, np.isin(x_var, _x_var)] = np.sqrt((weights @ np.power(y.T, 2.0)).T)
    return y_prime

def flat_forward_interpolation(x_fix, y_fix, x_var):

    x_fix = np.array(x_fix)
    if np.isscalar(x_var):
        x_var = np.array([x_var])

    x_repeat = np.tile(x_var[:, None], (len(x_fix),))
    distances = np.abs(x_repeat - x_fix)
    x_indices = np.searchsorted(x_fix, x_var)

    weights = np.zeros_like(distances)
    idx = np.arange(len(x_indices))
    weights[idx, x_indices - 1] = x_fix[x_indices-1] / x_var
    weights[idx, x_indices] = (1-x_fix[x_indices-1] / x_var)
    weights /= np.sum(weights, axis=1)[:, None]

    weights[np.any(distances == 0, axis=1), :] = 0
    weights[distances == 0] = 1
    return np.sqrt((weights @ np.power(y_fix.T, 2)).T)


def linear_interpolate(x_fix, y_fix, x_var):

    x_fix = np.array(x_fix)
    if np.isscalar(x_var):
       x_var = np.array([x_var])

    x_repeat = np.tile(x_var[:, None], (len(x_fix),))
    distances = np.abs(x_repeat - x_fix)
    x_indices = np.searchsorted(x_fix, x_var)

    weights = np.zeros_like(distances)
    idx = np.arange(len(x_indices))
    weights[idx, x_indices] = distances[idx, x_indices - 1]
    weights[idx, x_indices - 1] = distances[idx, x_indices]
    weights /= np.sum(weights, axis=1)[:, None]
    return (weights @ y_fix.T).T

@njit(fastmath=True, cache=True)
def linear_interpolate_1d_rowise(x, X0, y0):

    T = np.reshape(x, (-1, 1)).repeat(X0.shape[1], axis=1)

    LB_locs = ((X0 <= T) & (~np.isnan(y0)))
    UB_locs = ((X0 > T)  & (~np.isnan(y0)))

    LB = np.nanmax(X0 * np.where(LB_locs, LB_locs, np.nan), axis=1)
    UB = np.nanmin(X0 * np.where(UB_locs, UB_locs, np.nan), axis=1)

    y_LB = np.log(y0[np.where(np.isnan(LB), 0, LB).reshape(-1, 1) == X0])
    y_UB = np.log(y0[np.where(np.isnan(UB), 0, UB).reshape(-1, 1) == X0])

    W = (x - UB) / (LB - UB)
    fwd_mat = np.exp(W * y_LB + (1 - W) * y_UB)
    fwd_mat[LB == x] = np.exp(y_LB[LB == x])
    return fwd_mat


###############################################################################

@njit(fastmath=True, cache=True)
def nprime(x: float):
    """Calculate the first derivative of the Cumulative Normal CDF which is
    simply the PDF of the Normal Distribution """

    inv_root_two_pi = 0.3989422804014327
    return np.exp(-x * x / 2.0) * inv_root_two_pi


###############################################################################


@njit(fastmath=True, cache=True)
def normpdf(x: float):
    """ Calculate the probability density function for a Gaussian (Normal)
    function at value x"""
    return np.exp(-x * x / 2.0) * inv_root_two_pi

###############################################################################


@njit(float64(float64), fastmath=True, cache=True)
def N(x):
    """ Fast Normal CDF function based on Hull OFAODS  4th Edition Page 252.
    This function is accurate to 6 decimal places. """

    #phi = np.asarray(np.nan).astype(np.float64)

    a1 = 0.319381530
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    g = 0.2316419

    k = 1.0 / (1.0 + g * np.abs(x))
    k2 = k * k
    k3 = k2 * k
    k4 = k3 * k
    k5 = k4 * k

    if x >= 0.0:
        c = (a1 * k + a2 * k2 + a3 * k3 + a4 * k4 + a5 * k5)
        phi = 1.0 - c * np.exp(-x*x/2.0) * inv_root_two_pi
    else:
        phi = 1.0 - N(-x)
    return phi

###############################################################################

@vectorize([float64(float64)], fastmath=True, cache=True)
def n_vect(x):
    return N(x)

###############################################################################

@vectorize([float64(float64)], fastmath=True, cache=True)
def n_prime_vect(x):
    return nprime(x)

###############################################################################

#@jit(fastmath=True, cache=True, nopython=True)
@vectorize([float64(float64)], fastmath=True, cache=True)
def norminvcdf(p):

    """  This algorithm computes the inverse Normal CDF and is based on the
    algorithm found at (http:#home.online.no/~pjacklam/notes/invnorm/)
    which is by John Herrero (3-Jan-03) """

    # Define coefficients in rational approximations
    a1 = -39.6968302866538
    a2 = 220.946098424521
    a3 = -275.928510446969
    a4 = 138.357751867269
    a5 = -30.6647980661472
    a6 = 2.50662827745924

    b1 = -54.4760987982241
    b2 = 161.585836858041
    b3 = -155.698979859887
    b4 = 66.8013118877197
    b5 = -13.2806815528857

    c1 = -7.78489400243029E-03
    c2 = -0.322396458041136
    c3 = -2.40075827716184
    c4 = -2.54973253934373
    c5 = 4.37466414146497
    c6 = 2.93816398269878

    d1 = 7.78469570904146E-03
    d2 = 0.32246712907004
    d3 = 2.445134137143
    d4 = 3.75440866190742

    inverse_cdf = 0.0

    # Define break-points
    p_low = 0.02425
    p_high = 1.0 - p_low

    # If argument out of bounds, raise error
    if p < 0.0 or p > 1.0:
        raise ValueError("p must be between 0.0 and 1.0")

    if p == 0.0:
        p = 1e-10

    if p == 1.0:
        p = 1.0 - 1e-10

    if p < p_low:
        # Rational approximation for lower region
        q = np.sqrt(-2.0 * np.log(p))
        inverse_cdf = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5)
                       * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q
                                    + 1.0)
    elif p <= p_high:
        # Rational approximation for lower region
        q = p - 0.5
        r = q * q
        inverse_cdf = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * \
                      q / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    elif p < 1.0:
        # Rational approximation for upper region
        q = np.sqrt(-2.0 * np.log(1 - p))
        inverse_cdf = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5)
                        * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    return inverse_cdf

@njit(cache=True)
def interp_N(x, xp, fp, _type):

    x_ = np.asarray(x, dtype=np.float64)
    _res = np.empty((len(fp), x_.shape[0]), dtype=_type)

    for n in range(len(fp)):
        _res[n, :] = np_interp_1d(x_, xp, fp[n, :], _type)
    return _res

@njit(cache=True)
def interp_N_vect(x, xp, fp, _type):

    x_ = np.asarray(x, dtype=np.float64)
    _res = np.empty((len(fp), x_.shape[1]), dtype=_type)

    for n in range(len(fp)):
        _res[n, :] = np_interp_1d(x_[n, :], xp, fp[n, :], _type)
    return _res

@njit(cache=True)
def np_interp_1d(x, xp, fp, dtype):
    # NOTE: Do not refactor... see note in np_interp function impl below
    # this is a facsimile of arr_interp post 1.16:
    # https://github.com/numpy/numpy/blob/maintenance/1.16.x/numpy/core/src/multiarray/compiled_base.c    # noqa: E501
    # Permanent reference:
    # https://github.com/numpy/numpy/blob/971e2e89d08deeae0139d3011d15646fdac13c92/numpy/core/src/multiarray/compiled_base.c#L473     # noqa: E501

    x_ = xp[~np.isnan(fp)]
    y_ = fp[~np.isnan(fp)]

    dz = np.asarray(x, dtype=np.float64)
    dx = np.asarray(x_, dtype=np.float64)
    dy = np.asarray(y_, dtype=np.float64)

    if len(dx) == 0:
        return np.full(dz.shape, dtype=dtype, fill_value=np.nan)

    if len(dx) != len(dy):
        raise ValueError('fp and xp are not of the same size.')

    if dx.size == 1:
        dres = np.full(dz.shape, fill_value=dy[0], dtype=dtype)
        return dres

    dres = np.empty(dz.shape, dtype=dtype)

    lenx = dz.size
    lenxp = len(dx)
    lval = dy[0]
    rval = dy[lenxp - 1]

    if lenxp == 1:
        xp_val = dx[0]
        fp_val = dy[0]

        for i in range(lenx):
            x_val = dz.flat[i]
            if x_val < xp_val:
                dres.flat[i] = lval
            elif x_val > xp_val:
                dres.flat[i] = rval
            else:
                dres.flat[i] = fp_val

    else:
        j = 0

        # only pre-calculate slopes if there are relatively few of them.
        if lenxp <= lenx:
            slopes = (dy[1:] - dy[:-1]) / (dx[1:] - dx[:-1])
        else:
            slopes = np.empty(0, dtype=dtype)

        for i in range(lenx):
            x_val = dz.flat[i]

            if np.isnan(x_val):
                dres.flat[i] = x_val
                continue

            j = binary_search_with_guess(x_val, dx, lenxp, j)

            if j == -1:
                dres.flat[i] = lval
            elif j == lenxp:
                dres.flat[i] = rval
            elif j == lenxp - 1:
                dres.flat[i] = dy[j]
            elif dx[j] == x_val:
                # Avoid potential non-finite interpolation
                dres.flat[i] = dy[j]
            else:
                if slopes.size:
                    slope = slopes[j]
                else:
                    slope = (dy[j + 1] - dy[j]) / (dx[j + 1] - dx[j])

                dres.flat[i] = slope * (x_val - dx[j]) + dy[j]

                # NOTE: this is in np1.17
                # https://github.com/numpy/numpy/blob/maintenance/1.17.x/numpy/core/src/multiarray/compiled_base.c    # noqa: E501
                # Permanent reference:
                # https://github.com/numpy/numpy/blob/91fbe4dde246559fa5b085ebf4bc268e2b89eea8/numpy/core/src/multiarray/compiled_base.c#L610-L616    # noqa: E501
                #
                # If we get nan in one direction, try the other
                if np.isnan(dres.flat[i]):
                    dres.flat[i] = slope * (x_val - dx[j + 1]) + dy[j + 1]  # noqa: E501
                    if np.isnan(dres.flat[i]) and dy[j] == dy[j + 1]:
                        dres.flat[i] = dy[j]

    return dres

@njit(cache=True)
def np_fwd_flat_interp_1d(x, xp, fp, dtype):
    # NOTE: Do not refactor... see note in np_interp function impl below
    # this is a facsimile of arr_interp post 1.16:
    # https://github.com/numpy/numpy/blob/maintenance/1.16.x/numpy/core/src/multiarray/compiled_base.c    # noqa: E501
    # Permanent reference:
    # https://github.com/numpy/numpy/blob/971e2e89d08deeae0139d3011d15646fdac13c92/numpy/core/src/multiarray/compiled_base.c#L473     # noqa: E501

    x_ = xp[~np.isnan(fp)]
    y_ = fp[~np.isnan(fp)]

    dz = np.asarray(x, dtype=np.float64)
    dx = np.asarray(x_, dtype=np.float64)
    dy = np.asarray(y_, dtype=np.float64)

    if len(dx) == 0:
        return np.full(dz.shape, dtype=dtype, fill_value=np.nan)

    if len(dx) != len(dy):
        raise ValueError('fp and xp are not of the same size.')

    if dx.size == 1:
        dres = np.full(dz.shape, fill_value=dy[0], dtype=dtype)
        return dres

    dres = np.empty(dz.shape, dtype=dtype)

    lenx = dz.size
    lenxp = len(dx)
    lval = dy[0]
    rval = np.nan

    if lenxp == 1:
        xp_val = dx[0]
        fp_val = dy[0]

        for i in range(lenx):
            x_val = dz.flat[i]

            if x_val < 1/365.25:
                dres.flat[i] = np.nan
                continue

            if x_val < xp_val:
                dres.flat[i] = lval
            elif x_val > xp_val:
                dres.flat[i] = rval
            else:
                dres.flat[i] = fp_val

    else:
        j = 0

        for i in range(lenx):
            x_val = dz.flat[i]

            if np.isnan(x_val):
                dres.flat[i] = x_val
                continue

            if x_val < 1/365.25:
                dres.flat[i] = np.nan
                continue

            j = binary_search_with_guess(x_val, dx, lenxp, j)

            if j == -1:
                dres.flat[i] = lval
            elif j == lenxp:
                dres.flat[i] = rval
            elif j == lenxp - 1:
                dres.flat[i] = dy[j]
            elif dx[j] == x_val:
                # Avoid potential non-finite interpolation
                dres.flat[i] = dy[j]
            else:
                #slope = (dy[j + 1] - dy[j]) / (dx[j + 1] - dx[j])
                w = (dx[j] * (dx[j + 1] - x_val)) / (x_val * (dx[j + 1] - dx[j]))

                #dres.flat[i] = slope * (x_val - dx[j]) + dy[j]
                dres.flat[i] = w * dy[j] + (1-w) * dy[j+1]

    return dres

@njit(cache=True)
def forward_flat_interpolation_N(x, xp, fp, dtype):

    x_ = np.asarray(x, dtype=np.float64)
    _res = np.empty((len(fp), x_.size), dtype=dtype)
    for n in range(len(fp)):
        _res[n, :] = np_fwd_flat_interp_1d(x_, xp, fp[n, :], dtype)
    return _res

def cubic_spline(x, xq, y):

    x_ = np.asarray(xq, dtype=np.float64)
    _res = np.empty((len(y), x_.size), dtype=np.float64)
    for t in range(len(y)):
        _res[t, :] = cubicspline(x, y[t, :], xq)
    return _res

@njit(float64(float64[:], float64[:], float64), fastmath=True, cache=True)
def quad_poly_regression(x0, y0, x):
    X = np.column_stack((np.ones(x0.shape[0]), x0, x0 ** 2))
    b = (np.linalg.inv(X.T @ X) @ X.T @ y0)
    res = np.array([1, x, x ** 2], dtype=np.float64) @ b
    return res


if __name__ == "__main__":

    import numpy as np
    import numba as nb

    from epsilonPhi.core.lib.cpp.fastfind.find_1st import *
    import numpy as np

    x = np.arange(100).reshape(10, 10)
    y = np.arange(100).reshape(10, 10)
    z = np.arange(10)

    import pandas as pd
    path = '/Users/francisbarker/Desktop/Numba/Numba.xlsx'
    df = pd.read_excel(path, sheet_name='Sheet9', index_col=0, header=[0, 1])
    df = df.droplevel(level='date', axis=1)
    df.index = np.array([x.toordinal() for x in pd.to_datetime(df.index)])


    @njit()
    def to_ordinal(x):
        o = x - (-719162)
        return o

    _stacked = df.stack()
    hours = np.array(_stacked.index.get_level_values('t') * (365.25 * 24), dtype=np.int32)
    input = np.asarray((_stacked.index.get_level_values(0).astype(str) + hours.astype(str)).astype(int))
    limit = input.repeat(10)

    input = input.astype(int).reshape(-1, 1)
    input = np.sort(input, axis=0)
    limit = limit.astype(int).reshape(-1, 1)

    idx = find_1st(input, limit)

    N = df.shape[0]

    x_fix = np.array(df.columns)[None, :].repeat(N, axis=0)
    x_var = np.array([1/12])[None, :].repeat(N, axis=0)
    y_fix_ = df.values

    @nb.guvectorize([(nb.int64[:,:], nb.int64[:,:], nb.int64[:], nb.int64[:])], '(n,m),(n,m),(n)->(n)', nopython=True)
    def g(x, y, z, res):
        for i in range(x.shape[0]):

            y_ = y[~np.isnan(y[i])]
            x_ = x[~np.isnan(x[i])]

            if len(x_) > 1:
                res[i] = cubicspline(x_, y_, z[i])
            else:
                res[i] = np.float64(np.nan)

    res = g(x_fix, y_fix_, x_var)







