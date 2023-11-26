from numba import njit, boolean, int64, float64, vectorize
import numpy as np

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