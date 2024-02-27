import numpy as np
from numba import float64, int64, vectorize, njit, jit

#@jit(nopython=True, cache=True)
def norminv_vector(p_):
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

    if np.isscalar(p_):
        p = np.full(1, p_)
    else:
        p = p_

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

@jit([float64(float64), float64[:](float64[:])], nopython=True, cache=True)
def norminv_scalar(p):
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

#@jit([float64(float64), float64[:](float64[:])], nopython=True, cache=True)
#def norminv(p):
#    if np.isscalar(p):
#        print('Executing norminv_scalar')
#        norm_ = norminv_scalar(p)
#    else:
#        print('Executing norminv_vector')
#        norm_ = norminv_vector(p)
#    return norm_


p = np.random.uniform(low=1e-10, high=1.0-1e-10, size=1000000)
res_2 = norminv_vector(p[0])
