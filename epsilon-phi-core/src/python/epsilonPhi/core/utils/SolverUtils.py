from collections import namedtuple
from numba import njit, jit, float64
import numpy as np
import operator
from epsilonPhi.core.utils.Error import Error as FinError

_ECONVERGED = 0
_ECONVERR = -1

_iter = 100
_xtol = 2e-12
_rtol = 4*np.finfo(float).eps

results = namedtuple('results', 'root function_calls iterations converged')

###############################################################################

@njit(cache=True, fastmath=True)
def _results(r):
    r"""Select from a tuple of(root, funccalls, iterations, flag)"""
    x, fun_calls, iterations, flag = r
    return x  # results(x, fun_calls, iterations, flag == 0)

###############################################################################
# DO NOT TOUCH THIS FUNCTION AS IT IS USED IN FX VOL CALIBRATION !!!!!!!!!
# IT NEEDS TO PASS IN ARGS AS A TUPLE AS ONE OF THE ARGS IS AN NDARRAY
###############################################################################
# UNABLE TO NJIT THIS DUE TO ERROR
# FIXED ERROR BY MAKING CACHE=FALSE!!!????


@njit(fastmath=True, cache=False)
def newton_secant(func, x0, args=(), tol=1.48e-8, maxiter=50, disp=True):
    """
    Find a zero from the secant method using the jitted version of
    Scipy's secant method.

    Note that `func` must be jitted via Numba.

    Parameters
    ----------
    func : callable and jitted
        The function whose zero is wanted. It must be a function of a
        single variable of the form f(x,a,b,c...), where a,b,c... are extra
        arguments that can be passed in the `args` parameter.
    x0 : float
        An initial estimate of the zero that should be somewhere near the
        actual zero.
    args : tuple, optional(default=())
        Extra arguments to be used in the function call.
    tol : float, optional(default=1.48e-8)
        The allowable error of the zero value.
    maxiter : int, optional(default=50)
        Maximum number of iterations.
    disp : bool, optional(default=True)
        If True, raise a RuntimeError if the algorithm didn't converge.

    Returns
    -------
    results : namedtuple
        A namedtuple containing the following items:
        ::

            root - Estimated location where function is zero.
            function_calls - Number of times the function was called.
            iterations - Number of iterations needed to find the root.
            converged - True if the routine converged
    """

    if tol <= 0.0:
        raise FinError("Tolerance should be positive.")

    if maxiter < 1:
        raise FinError("maxiter must be greater than 0")

    # Convert to float (don't use float(x0); this works also for complex x0)
    eps = 1e-4
    p0 = 1.0 * x0
    fun_calls = 0
    status = _ECONVERR

    p1 = x0 * (1.0 + eps)

    if p1 > 0.0:
        p1 = p1 + eps
    else:
        p1 = p1 - eps

    q0 = func(p0, *args)
    fun_calls += 1
    q1 = func(p1, *args)
    fun_calls += 1

    if np.abs(q1) < np.abs(q0):
        p0, p1, q0, q1 = p1, p0, q1, q0

    for _ in range(maxiter):

        if q1 == q0:
            if p1 != p0:
                raise FinError("Tolerance reached")

            p = (p1 + p0) / 2.0
            status = _ECONVERGED
            break
        else:
            if np.abs(q1) > np.abs(q0):
                p = (-q0 / q1 * p1 + p0) / (1.0 - q0 / q1)
            else:
                p = (-q1 / q0 * p0 + p1) / (1.0 - q1 / q0)

        if np.abs(p - p1) < tol:
            status = _ECONVERGED
            return p

        p0, q0 = p1, q1
        p1 = p
        q1 = func(p1, *args)
        fun_calls += 1

    if disp and status == _ECONVERR:
        msg = "Failed to converge"
        raise FinError(msg)

    return p
