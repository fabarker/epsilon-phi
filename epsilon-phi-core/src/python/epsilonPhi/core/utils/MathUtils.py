from numba import njit, boolean, int64, float64, vectorize
import numpy as np
from numba import jit
from epsilonPhi.core.utils.Error import Error as FinError

PI = 3.14159265358979323846
inv_root_two_pi = 0.3989422804014327

ONE_MILLION = 1000000
TEN_MILLION = 10000000
ONE_BILLION = 1000000000

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

@njit(float64[:, :](float64[:, :]), cache=True, fastmath=True)
def cholesky(rho):
    """ Numba-compliant wrapper around Numpy cholesky function. """
    chol = np.linalg.cholesky(rho)
    return chol

###############################################################################
# Copyright (C) 2018, 2019, 2020 Dominic O'Kane
###############################################################################


# from math import exp, sqrt, fabs, log
from numba import njit, boolean, int64, float64, vectorize
import numpy as np

PI = 3.14159265358979323846
inv_root_two_pi = 0.3989422804014327

ONE_MILLION = 1000000
TEN_MILLION = 10000000
ONE_BILLION = 1000000000


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

@jit('float64[:](float64[:])', nopython=True, cache=True)
def fast_norminvcdf_(p):
    """  This algorithm computes the inverse Normal CDF and is based on the
    algorithm found at (http:#home.online.no/~pjacklam/notes/invnorm/)
    which is by John Herrero (3-Jan-03) """

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

    inverse_cdf = np.full(len(p), 0.0)

    # Define break-points
    p_low = 0.02425
    p_high = 1.0 - p_low

    # If argument out of bounds, raise error
    p[p < 0.0] = np.nan
    p[p > 1.0] = np.nan
    p[p == 0.0] = 1e-10
    p[p == 1.0] = 1.0 - 1e-10

    # Rational approximation for lower region
    q_low = np.sqrt(-2.0 * np.log(p[p < p_low]))
    inverse_cdf[p < p_low] = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5)
                       * q_low + c6) / ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)

    # Rational approximation for upper region
    q_h = p[np.logical_and(p <= p_high, p >= p_low)] - 0.5
    r = q_h * q_h
    inverse_cdf[np.logical_and(p <= p_high, p >= p_low)] = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * \
            q_h / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # Rational approximation for upper region
    qur = np.sqrt(-2.0 * np.log(1 - p[np.logical_and(p < 1.0, p > p_high)]))
    inverse_cdf[np.logical_and(p < 1.0, p > p_high)] = -(((((c1 * qur + c2) * qur + c3) * qur + c4) * qur + c5)
                        * qur + c6) / ((((d1 * qur + d2) * qur + d3) * qur + d4) * qur + 1.0)
    return inverse_cdf

@jit(nopython=True, cache=True)
def fast_norminvcdf(p):
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
        raise FinError("p must be between 0.0 and 1.0")

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


@jit(nopython=True, cache=True)
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
        raise FinError("p must be between 0.0 and 1.0")

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

###############################################################################

