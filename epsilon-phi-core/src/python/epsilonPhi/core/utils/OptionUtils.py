import numpy as np
from numba import float64, int64, vectorize, njit
from epsilonPhi.core.utils.Error import Error as FinError
from epsilonPhi.core.utils.DateUtils import DateUtils
from epsilonPhi.core.utils.MathUtils import *
from epsilonPhi.core.dataModel.enums.ImpliedVolatility import *
from epsilonPhi.core.utils.SolverUtils import *
from enum import Enum

gDaysInYear = DateUtils.days_per_year
g_small = 1e-12
gNotebookMode = False

class OptionTypes(Enum):
    EUROPEAN_CALL = 1
    EUROPEAN_PUT = 2
    AMERICAN_CALL = 3
    AMERICAN_PUT = 4
    DIGITAL_CALL = 5
    DIGITAL_PUT = 6
    ASIAN_CALL = 7
    ASIAN_PUT = 8
    COMPOUND_CALL = 9
    COMPOUND_PUT = 10

@njit(fastmath=True, cache=True)
def g(K, *args):
    """ This is the objective function used in the determination of the FX
    option implied strike which is computed in the class below. """

    s = args[0]
    t = args[1]
    r_d = args[2]
    r_f = args[3]
    volatility = args[4]
    delta_method_value = args[5]
    option_type_value = args[6]
    delta_target = args[7]
    return delta_target - fast_delta(s, t, K, r_d, r_f,
                           volatility,
                           delta_method_value,
                           option_type_value)



@njit(float64(float64, float64, float64, float64, int64, float64,
              int64, float64), fastmath=True)
def solve_for_strike(spot_fx_rate,
                     tdel, rd, rf,
                     option_type_value,
                     delta_target,
                     delta_method_value,
                     volatility):
    """ This function determines the implied strike of an FX option
    given a delta and the other option details. It uses a one-dimensional
    Newton root search algorithm to determine the strike that matches an
    input volatility. """

    # =========================================================================
    # IMPORTANT NOTE:
    # =========================================================================
    # For some delta quotation conventions I can solve for K explicitly.
    # Note that as I am using the function norm_inv_delta to calculate the
    # inverse value of delta, this may not, on a round trip using N(x), give
    # back the value x as it is calculated to a different number of decimal
    # places. It should however agree to 6-7 decimal places. Which is OK.
    # =========================================================================

    if delta_method_value == DeltaType.SPOT_DELTA.value:

        dom_df = np.exp(-rd*tdel)
        for_df = np.exp(-rf*tdel)

        if option_type_value == OptionTypes.EUROPEAN_CALL.value:
            phi = +1.0
        else:
            phi = -1.0

        F0T = spot_fx_rate * for_df / dom_df
        vsqrtt = volatility * np.sqrt(tdel)
        arg = delta_target*phi/for_df  # CHECK THIS !!!
        norm_inv_delta = norminvcdf(arg)
        K = F0T * np.exp(-vsqrtt * (phi*norm_inv_delta - vsqrtt/2.0))
        return K

    elif delta_method_value == DeltaType.FORWARD_DELTA.value:

        dom_df = np.exp(-rd*tdel)
        for_df = np.exp(-rf*tdel)

        if option_type_value == OptionTypes.EUROPEAN_CALL.value:
            phi = +1.0
        else:
            phi = -1.0

        F0T = spot_fx_rate * for_df / dom_df
        vsqrtt = volatility * np.sqrt(tdel)
        arg = delta_target*phi   # CHECK THIS!!!!!!!!
        norm_inv_delta = norminvcdf(arg)
        K = F0T * np.exp(-vsqrtt * (phi*norm_inv_delta - vsqrtt/2.0))
        return K

    elif delta_method_value == DeltaType.SPOT_DELTA_PREM_ADJ.value:
        argtuple = (spot_fx_rate, tdel, rd, rf, volatility,
                    delta_method_value, option_type_value, delta_target)
        k = 0.0
        return k
    elif delta_method_value == DeltaType.FORWARD_DELTA_PREM_ADJ.value:
        k = 0.0
        return k
    else:

        raise FinError("Unknown FinFXDeltaMethod")


@vectorize([float64(float64, float64, float64, float64,
                    float64, float64)], fastmath=True, cache=True)
def d_plus(s, t, k, r, q, v):

    t = np.maximum(t, g_small)

    v_sqrt_t = np.maximum(v, g_small) * np.sqrt(t)
    ss = s * np.exp(-q * t)
    kk = np.maximum(k, g_small) * np.exp(-r * t)
    d1 = np.log(ss/kk) / v_sqrt_t + v_sqrt_t / 2.0
    return d1

@vectorize([float64(float64, float64, float64, float64,
                    float64, float64)], fastmath=True, cache=True)
def d_minus(s, t, k, r, q, v):
    d2 = d_plus(s, t, k, r, q, v) - np.maximum(v, g_small) * np.sqrt(t)
    return d2

@vectorize([float64(float64, float64, float64, float64, float64, float64,
                    int64)], fastmath=True, cache=True)
def bs_value(s, t, k, r, q, v, option_type_value):
    """Price a derivative using Black-Scholes model."""

    if option_type_value == OptionTypes.EUROPEAN_CALL.value:
        phi = 1.0
    elif option_type_value == OptionTypes.EUROPEAN_PUT.value:
        phi = -1.0
    else:
        raise ValueError("Unknown option type value")

    k = np.maximum(k, g_small)
    t = np.maximum(t, g_small)
    v = np.maximum(v, g_small)

    v_sqrt_t = v * np.sqrt(t)
    ss = s * np.exp(-q*t)
    kk = k * np.exp(-r*t)
    d1 = np.log(ss/kk) / v_sqrt_t + v_sqrt_t / 2.0
    d2 = d1 - v_sqrt_t

    value = phi * ss * N(phi * d1) - phi * kk * N(phi * d2)
    return value

###############################################################################

@njit(fastmath=True, cache=True)
def fast_delta(s, t, k, rd, rf, vol, deltaTypeValue, option_type_value):
    """ Calculation of the option delta. Used in the determination of
    the volatility surface. """

    spot_delta = bs_delta(s, t, k, rd, rf, vol, option_type_value)

    if deltaTypeValue == DeltaType.SPOT_DELTA.value:
        delta = spot_delta
    elif deltaTypeValue == DeltaType.FORWARD_DELTA.value:
        delta = spot_delta * np.exp(rf*t)
    elif deltaTypeValue == DeltaType.SPOT_DELTA_PREM_ADJ.value:
        vpctf = bs_value(s, t, k, rd, rf, vol, option_type_value) / s
        delta = spot_delta - vpctf
    elif deltaTypeValue == DeltaType.FORWARD_DELTA_PREM_ADJ.value:
        vpctf = bs_value(s, t, k, rd, rf, vol, option_type_value) / s
        delta = np.exp(rf*t) * (spot_delta - vpctf)
    else:
        raise FinError("Unknown DeltaMethod")
    return delta


###############################################################################


@vectorize([float64(float64, float64, float64, float64,
                    float64, float64, int64)], fastmath=True, cache=True)
def bs_delta(s, t, k, r, q, v, option_type_value):
    """Price a derivative using Black-Scholes model."""
    if option_type_value == OptionTypes.EUROPEAN_CALL.value:
        phi = +1.0
    elif option_type_value == OptionTypes.EUROPEAN_PUT.value:
        phi = -1.0
    else:
        raise FinError("Unknown option type value")
        return 0.0

    k = np.maximum(k, g_small)
    t = np.maximum(t, g_small)
    v = np.maximum(v, g_small)

    v_sqrt_t = v * np.sqrt(t)
    ss = s * np.exp(-q*t)
    kk = k * np.exp(-r*t)
    d1 = np.log(ss/kk) / v_sqrt_t + v_sqrt_t / 2.0
    delta = phi * np.exp(-q*t) * n_vect(phi * d1)
    return delta

###############################################################################


@vectorize([float64(float64, float64, float64, float64,
                    float64, float64)], fastmath=True, cache=True)
def bs_gamma(s, t, k, r, q, v):
    """Price a derivative using Black-Scholes model."""

    k = np.maximum(k, g_small)
    t = np.maximum(t, g_small)
    v = np.maximum(v, g_small)

    v_sqrt_t = v * np.sqrt(t)
    ss = s * np.exp(-q*t)
    kk = k * np.exp(-r*t)
    d1 = np.log(ss/kk) / v_sqrt_t + v_sqrt_t / 2.0
    gamma = np.exp(-q*t) * n_prime_vect(d1) / s / v_sqrt_t
    return gamma

###############################################################################


@vectorize([float64(float64, float64, float64, float64,
                    float64, float64)], fastmath=True, cache=True)
def bs_vega(s, t, k, r, q, v):
    """Price a derivative using Black-Scholes model."""
    k = np.maximum(k, g_small)
    t = np.maximum(t, g_small)
    v = np.maximum(v, g_small)

    sqrt_t = np.sqrt(t)
    v_sqrt_t = v * sqrt_t
    ss = s * np.exp(-q*t)
    kk = k * np.exp(-r*t)
    d1 = np.log(ss/kk) / v_sqrt_t + v_sqrt_t / 2.0
    vega = ss * sqrt_t * n_prime_vect(d1)
    return vega

###############################################################################


@vectorize([float64(float64, float64, float64, float64,
                    float64, float64, int64)], fastmath=True, cache=True)
def bs_theta(s, t, k, r, q, v, option_type_value):
    """Price a derivative using Black-Scholes model."""

    if option_type_value == OptionTypes.EUROPEAN_CALL.value:
        phi = 1.0
    elif option_type_value == OptionTypes.EUROPEAN_PUT.value:
        phi = -1.0

    k = np.maximum(k, g_small)
    t = np.maximum(t, g_small)
    v = np.maximum(v, g_small)

    sqrt_t = np.sqrt(t)
    v_sqrt_t = v * sqrt_t
    ss = s * np.exp(-q*t)
    kk = k * np.exp(-r*t)
    d1 = np.log(ss/kk) / v_sqrt_t + v_sqrt_t / 2.0
    d2 = d1 - v_sqrt_t
    theta = - ss * n_prime_vect(d1) * v / 2.0 / sqrt_t
    theta = theta - phi * r * k * np.exp(-r*t) * n_vect(phi * d2)
    theta = theta + phi * q * ss * n_vect(phi * d1)
    return theta

###############################################################################


@vectorize([float64(float64, float64, float64, float64,
                    float64, float64, int64)], fastmath=True, cache=True)
def bs_rho(s, t, k, r, q, v, option_type_value):
    """Price a derivative using Black-Scholes model."""

    if option_type_value == OptionTypes.EUROPEAN_CALL.value:
        phi = 1.0
    elif option_type_value == OptionTypes.EUROPEAN_PUT.value:
        phi = -1.0

    k = np.maximum(k, g_small)
    t = np.maximum(t, g_small)
    v = np.maximum(v, g_small)

    sqrt_t = np.sqrt(t)
    v_sqrt_t = v * sqrt_t
    ss = s * np.exp(-q*t)
    kk = k * np.exp(-r*t)
    d1 = np.log(ss/kk) / v_sqrt_t + v_sqrt_t / 2.0
    d2 = d1 - v_sqrt_t
    rho = phi * k * t * np.exp(-r*t) * n_vect(phi * d2)
    return rho

###############################################################################

@vectorize([float64(float64, float64, float64, float64,
                    float64, float64)], fastmath=True, cache=True)
def bs_vanna(s, t, k, r, q, v):
    """Price a derivative using Black-Scholes model."""

    k = np.maximum(k, g_small)
    t = np.maximum(t, g_small)
    v = np.maximum(v, g_small)

    sqrt_t = np.sqrt(t)
    v_sqrt_t = v * sqrt_t
    ss = s * np.exp(-q*t)
    kk = k * np.exp(-r*t)
    d1 = np.log(ss/kk) / v_sqrt_t + v_sqrt_t / 2.0
    d2 = d1 - v_sqrt_t
    vanna = np.exp(-q*t) * sqrt_t * n_prime_vect(d1) * (d2/v)
    return vanna
