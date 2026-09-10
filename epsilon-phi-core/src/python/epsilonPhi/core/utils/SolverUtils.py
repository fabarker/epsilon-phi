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

_eps = 1.4902e-100
_golden = 0.381966011250105097

def brent(f, a, b, args=(), rtol=_eps, atol=_eps, maxiter=500):
    """Seeks a minimum of a function via Brent's method.

    Given a function ``f`` with a minimum in the interval ``a <= b``, seeks a local
    minimum using a combination of golden section search and successive parabolic
    interpolation.

    Let ``tol = rtol * abs(x0) + atol``, where ``x0`` is the best guess found so far.
    It converges if evaluating a next guess would imply evaluating ``f`` at a point that
    is closer than ``tol`` to a previously evaluated one or if the number of iterations
    reaches ``maxiter``.

    Parameters
    ----------
    f : object
        Objective function to be minimized.
    a : float, optional
        Interval's lower limit. Defaults to ``-inf``.
    b : float, optional
        Interval's upper limit. Defaults to ``+inf``.
    x0 : float, optional
        Initial guess. Defaults to ``None``, which implies that::

            x0 = a + 0.382 * (b - a)
            f0 = f(x0)

    f0 : float, optional
        Function evaluation at ``x0``.
    rtol : float
        Relative tolerance. Defaults to ``1.4902e-08``.
    atol : float
        Absolute tolerance. Defaults to ``1.4902e-08``.
    maxiter : int
        Maximum number of iterations.


    Returns
    -------
    float
        Best guess ``x`` for the minimum of ``f``.
    float
        Value ``f(x)``.
    int
        Number of iterations performed.

    References
    ----------
    - http://people.sc.fsu.edu/~jburkardt/c_src/brent/brent.c
    - Numerical Recipes 3rd Edition: The Art of Scientific Computing
    - https://en.wikipedia.org/wiki/Brent%27s_method
    """
    # a, b: interval within the minimum should lie
    #       no function evaluation will be requested
    #       outside that range.
    # x0: least function value found so far (or the most recent one in
    #                                            case of a tie)
    # x1: second least function value
    # x2: previous value of x1
    # (x0, x1, x2): Memory triple, updated at the end of each interation.
    # u : point at which the function was evaluated most recently.
    # m : midpoint between the current interval (a, b).
    # d : step size and direction.
    # e : memorizes the step size (and direction) taken two iterations ago
    #      and it is used to (definitively) fall-back to golden-section steps
    #      when its value is too small (indicating that the polynomial fitting
    #      is not helping to speedup the convergence.)
    #
    #
    # References: Numerical Recipes: The Art of Scientific Computing
    # http://people.sc.fsu.edu/~jburkardt/c_src/brent/brent.c

    if a > b:
        raise ValueError("'a' must be equal or smaller than 'b'")

    x0 = a + _golden * (b - a)
    f0 = f(x0, *args)

    x1 = x0
    x2 = x1
    niters = -1
    d = 0.0
    e = 0.0
    f1 = f0
    f2 = f1

    for niters in range(maxiter):

        m = 0.5 * (a + b)
        tol = rtol * abs(x0) + atol
        tol2 = 2.0 * tol

        # Check the stopping criterion.
        if abs(x0 - m) <= tol2 - 0.5 * (b - a):
            break

        r = 0.0
        q = r
        p = q

        # "To be acceptable, the parabolic step must (i) fall within the
        # bounding interval (a, b), and (ii) imply a movement from the best
        # current value x0 that is less than half the movement of the step
        # before last."
        #   - Numerical Recipes 3rd Edition: The Art of Scientific Computing.

        if tol < abs(e):
            # Compute the polynomial of the least degree (Lagrange polynomial)
            # that goes through (x0, f0), (x1, f1), (x2, f2).
            r = (x0 - x1) * (f0 - f2)
            q = (x0 - x2) * (f0 - f1)
            p = (x0 - x2) * q - (x0 - x1) * r
            q = 2.0 * (q - r)
            if 0.0 < q:
                p = -p
            q = abs(q)
            r = e
            e = d

        if abs(p) < abs(0.5 * q * r) and q * (a - x0) < p and p < q * (b - x0):
            # Take the polynomial interpolation step.
            d = p / q
            u = x0 + d

            # Function must not be evaluated too close to a or b.
            if (u - a) < tol2 or (b - u) < tol2:
                if x0 < m:
                    d = tol
                else:
                    d = -tol
        else:
            # Take the golden-section step.
            if x0 < m:
                e = b - x0
            else:
                e = a - x0
            d = _golden * e

        # Function must not be evaluated too close to x0.
        if tol <= abs(d):
            u = x0 + d
        elif 0.0 < d:
            u = x0 + tol
        else:
            u = x0 - tol

        # Notice that we have u \in [a+tol, x0-tol] or
        #                     u \in [x0+tol, b-tol],
        # (if one ignores rounding errors.)
        fu = f(u, *args)

        # Housekeeping.

        # Is the most recently evaluated point better (or equal) than the
        # best so far?
        if fu <= f0:

            # Decrease interval size.
            if u < x0:
                if b != x0:
                    b = x0
            else:
                if a != x0:
                    a = x0

            # Shift: drop the previous third best point out and
            # include the newest point (found to be the best so far).
            x2 = x1
            f2 = f1
            x1 = x0
            f1 = f0
            x0 = u
            f0 = fu

        else:
            # Decrease interval size.
            if u < x0:
                if a != u:
                    a = u
            else:
                if b != u:
                    b = u

            # Is the most recently evaluated point at better (or equal)
            # than the second best one?
            if fu <= f1 or x1 == x0:
                # Insert u between (rank-wise) x0 and x1 in the triple
                # (x0, x1, x2).
                x2 = x1
                f2 = f1
                x1 = u
                f1 = fu
            elif fu <= f2 or x2 == x0 or x2 == x1:
                # Insert u in the last position of the triple (x0, x1, x2).
                x2 = u
                f2 = fu

    return x0, f0, niters + 1

@njit(cache=True, fastmath=True)
def brentsmethod(f, a, b, args=()):

    tol = 0
    root_tol = 2e-12
    maxitr = 100

    fa = f(a, *args)
    fb = f(b, *args)

    ctr = 0
    if abs(fa) < abs(fb):
        d = 0
        c = b
        b = a
        a = c
        t = fa
        fa = fb
        fb = t

    if fa * fb > 0:
        raise ValueError('Error root is not bracketed')

    # If endpoint of the interval is at most tol then output
    if abs(fb) <= tol:
        root = np.nan
        return root

    c = a
    mflag = 1
    while abs(b-a) > root_tol and ctr < maxitr:

        fc = f(c, *args)
        if fa != fc and fb != fc:
            s = a * fb * fc / ((fa - fb) * (fa - fc)) + b * fa * fc / ((fb - fa) * (fb - fc)) + c * fa * fb / ((fc - fa) * (fc - fb))
        else:
            s = b - fb * (b - a) / (fb - fa)

        if (s < min((3 * a + b) / 4, b) or s > max((3 * a + b) / 4, b)) or \
                (mflag == 1 and abs(s - b) >= abs(b - c) / 2) or \
                (mflag == 0 and abs(s - b) >= abs(c - d) / 2) or \
                (mflag == 1 and abs(b - c) < root_tol) or \
                (mflag == 0 and abs(c - d) < root_tol):
            s = (a + b) / 2
            mflag = 1
        else:
            mflag = 0

        ctr += 1
        fs = f(s, *args)
        if abs(fs) <= tol:
           root = s
           return root

        if fa * fs < 0:
            d = c  #if c was b_{k - 1}, d would be b_{k - 2}
            c = b
            b = s
            fb = fs
        else:
            a = s
            fa = fs

        if abs(fa) < abs(fb):
            d = c
            c = b
            b = a
            a = c
            fc = fb
            fb = fa
            fa = fc

    if ctr > maxitr:
       root = s
       return root

    root = s
    return root


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
