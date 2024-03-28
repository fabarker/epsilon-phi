from scipy.stats import norm
from numba import float64, int64, vectorize, njit, guvectorize
from epsilonPhi.core.utils.DateUtils import DateUtils
from epsilonPhi.core.utils.MathUtils import *
from epsilonPhi.core.dataModel.enums.ImpliedVolatility import *
from epsilonPhi.core.utils.SolverUtils import *
import pandas as pd
from enum import Enum

gDaysInYear = DateUtils.days_per_year
g_small = 1e-12
gNotebookMode = False

class OptionTypes(Enum):
    EUROPEAN_CALL = 1
    EUROPEAN_PUT = -1

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

    if delta_method_value in [DeltaType.SPOT_DELTA.value,
                              DeltaType.SPOT_DELTA]:

        dom_df = np.asarray(np.exp(-rd * tdel), dtype=np.float64)
        for_df = np.asarray(np.exp(-rf * tdel), dtype=np.float64)

        phi = np.sign(option_type_value)

        F0T = np.array(spot_fx_rate) * np.array(for_df / dom_df)
        vsqrtt = volatility * np.sqrt(tdel)
        arg = delta_target*phi/for_df  # CHECK THIS !!!
        norm_inv_delta = norm.ppf(arg, 0.0, 1.0)
        K = F0T * np.exp(-vsqrtt * (phi*norm_inv_delta - vsqrtt/2.0))
        return K

    elif delta_method_value in [DeltaType.FORWARD_DELTA.value,
                                DeltaType.FORWARD_DELTA]:

        dom_df = np.asarray(np.exp(-rd*tdel), dtype=np.float64)
        for_df = np.asarray(np.exp(-rf*tdel), dtype=np.float64)

        phi = np.sign(option_type_value)

        F0T = np.array(spot_fx_rate) * for_df / dom_df
        vsqrtt = volatility * np.sqrt(tdel)
        arg = delta_target*phi   # CHECK THIS!!!!!!!!
        norm_inv_delta = norm.ppf(arg, 0.0, 1.0)
        K = F0T * np.exp(-vsqrtt * (phi*norm_inv_delta - vsqrtt/2.0))
        return K

    elif delta_method_value == [DeltaType.SPOT_DELTA_PREM_ADJ.value,
                                DeltaType.SPOT_DELTA_PREM_ADJ]:
        argtuple = (spot_fx_rate, tdel, rd, rf, volatility,
                    delta_method_value, option_type_value, delta_target)
        raise FinError('Error - not impletments')
    elif delta_method_value == [DeltaType.FORWARD_DELTA_PREM_ADJ.value,
                                DeltaType.FORWARD_DELTA_PREM_ADJ]:
        raise FinError('Error - not impletments')
    else:
        raise FinError("Unknown DeltaMethod")


def d1(F, k, v, t):
    d1 = (np.log(F / k) + 0.5 * (v ** 2) * t) / (v * np.sqrt(t))
    return d1

def d2(F, k, v, t):
    return  d1(F, k, v, t) - v * np.sqrt(t)

def d_plus(s, t, k, r, q, v):

    t = np.maximum(t, g_small)

    v_sqrt_t = np.maximum(v, g_small) * np.sqrt(t)
    ss = s * np.exp(-q * t)
    kk = np.maximum(k, g_small) * np.exp(-r * t)
    d1 = np.log(ss/kk) / v_sqrt_t + v_sqrt_t / 2.0
    return d1

def d_minus(s, t, k, r, q, v):
    d2 = d_plus(s, t, k, r, q, v) - np.maximum(v, g_small) * np.sqrt(t)
    return d2

def bs_value(s, t, k, r, q, v, option_type_value):
    """Price a derivative using Black-Scholes model."""

    phi = np.sign(option_type_value)
    k = np.maximum(k, g_small)
    t = np.maximum(t, g_small)
    v = np.maximum(v, g_small)

    v_sqrt_t = v * np.sqrt(t)
    ss = s * np.exp(-q*t)
    kk = k * np.exp(-r*t)
    d1 = np.log(ss/kk) / v_sqrt_t + v_sqrt_t / 2.0
    d2 = d1 - v_sqrt_t

    value = phi * ss * norm.cdf(phi * d1, 0.0, 1.0) - phi * kk * norm.cdf(phi * d2, 0.0, 1.0)
    return value


def blsprice(f, t, k, rf, v, option_type_value):

    """Price a derivative using Black-Scholes model."""

    phi = np.sign(option_type_value)
    k = np.maximum(k, g_small)
    t = np.maximum(t, g_small)
    v = np.maximum(v, g_small)

    v_sqrt_t = v * np.sqrt(t)
    d1 = np.log(f/k) / v_sqrt_t + v_sqrt_t / 2.0
    d2 = d1 - v_sqrt_t

    value = phi * np.exp(-rf * t) * (f * norm.cdf(phi * d1, 0.0, 1.0) - k * norm.cdf(phi * d2, 0.0, 1.0))
    return value


###############################################################################

def atm_delta_neutral_strike(s, t, rd, rf, vol, deltaTypeValue):

    """ Calculation of the option delta. Used in the determination of
       the volatility surface. """

    dom_df = np.exp(-rd * t)
    for_df = np.exp(-rf * t)
    f = s * for_df / dom_df

    if deltaTypeValue in [DeltaType.SPOT_DELTA, DeltaType.SPOT_DELTA.value]:
        return f * np.exp(0.5 * vol * vol * t)
    elif deltaTypeValue in [DeltaType.FORWARD_DELTA, DeltaType.FORWARD_DELTA.value]:
        return f * np.exp(0.5 * vol * vol * t)
    elif deltaTypeValue == [DeltaType.SPOT_DELTA_PREM_ADJ, DeltaType.SPOT_DELTA_PREM_ADJ.value]:
        return f * np.exp(-0.5 * vol * vol * t)
    elif deltaTypeValue == [DeltaType.FORWARD_DELTA_PREM_ADJ, DeltaType.FORWARD_DELTA_PREM_ADJ.value]:
        return f * np.exp(-0.5 * vol * vol * t)
    else:
        raise FinError("Unknown DeltaMethod")

def delta_from_delta_neutral_straddle_quote(t, rf, vol, deltaTypeValue, option_type_value):

    """ Calculation of the option delta. Used in the determination of
       the volatility surface. """

    if deltaTypeValue in [DeltaType.SPOT_DELTA, DeltaType.SPOT_DELTA.value]:
        return 0.5 * option_type_value * np.exp(-1 * rf * t)
    elif deltaTypeValue == [DeltaType.FORWARD_DELTA, DeltaType.FORWARD_DELTA.value]:
        return 0.5 * option_type_value
    elif deltaTypeValue == [DeltaType.SPOT_DELTA_PREM_ADJ, DeltaType.SPOT_DELTA_PREM_ADJ.value]:
        return 0.5 * option_type_value * np.exp(-1 * rf * t) * np.exp(-0.5 * vol * vol * t)
    elif deltaTypeValue == [DeltaType.FORWARD_DELTA_PREM_ADJ, DeltaType.FORWARD_DELTA_PREM_ADJ.value]:
        return 0.5 * option_type_value * np.exp(-0.5 * vol * vol * t)
    else:
        raise FinError("Unknown DeltaMethod")


def fast_delta(s, t, k, rd, rf, vol, deltaTypeValue, option_type_value):
    """ Calculation of the option delta. Used in the determination of
    the volatility surface. """

    assert vol.shape == k.shape, 'Error - K and V dim mis-match'
    if k.ndim > 1 and not isinstance(s, pd.DataFrame):
        s = s.reshape(-1, 1)
        t = t.reshape(-1, 1)
        rd = rd.reshape(-1, 1)
        rf = rf.reshape(-1, 1)

    spot_delta = bs_delta(s, t, k, rd, rf, vol, option_type_value)

    if deltaTypeValue in [DeltaType.SPOT_DELTA, DeltaType.SPOT_DELTA.value]:
        delta = spot_delta
    elif deltaTypeValue in [DeltaType.FORWARD_DELTA, DeltaType.FORWARD_DELTA.value]:
        delta = spot_delta * np.exp(rf*t)
    elif deltaTypeValue in [DeltaType.SPOT_DELTA_PREM_ADJ, DeltaType.SPOT_DELTA_PREM_ADJ.value]:
        vpctf = bs_value(s, t, k, rd, rf, vol, option_type_value) / s
        delta = spot_delta - vpctf
    elif deltaTypeValue in [DeltaType.FORWARD_DELTA_PREM_ADJ, DeltaType.FORWARD_DELTA_PREM_ADJ.value]:
        vpctf = bs_value(s, t, k, rd, rf, vol, option_type_value) / s
        delta = np.exp(rf*t) * (spot_delta - vpctf)
    else:
        raise FinError("Unknown DeltaMethod")
    return delta


###############################################################################

def bs_delta(s, t, k, r, q, v, option_type_value):
    """Price a derivative using Black-Scholes model."""

    phi = np.sign(option_type_value)
    k = np.maximum(k, g_small)
    t = np.maximum(t, g_small)
    v = np.maximum(v, g_small)

    v_sqrt_t = np.array(v * np.sqrt(t))
    ss = s * np.exp(-q*t)
    kk = k * np.exp(-r*t)
    d1 = np.log(ss/kk) / v_sqrt_t + v_sqrt_t / 2.0
    delta = phi * np.exp(-q*t) * norm.cdf(phi * d1, 0.0, 1.0)
    return delta

def black_delta(f, t, k, q, v, delta_type_value, option_type):

    phi = np.sign(option_type)
    k = np.maximum(k, g_small)
    t = np.maximum(t, g_small)
    v = np.maximum(v, g_small)

    d1_ = (np.log(f/k) + v * v * t / 2.0) / (v * np.sqrt(t))

    if delta_type_value == 1:
        return phi * np.exp(-q * t) * n_vect(phi * d1_)
    else:
        return phi * np.exp(-q * t) * n_vect(phi * d1_) * np.exp(q*t)

###############################################################################

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

def black_gamma(f, t, k, r, v):
    """Return gamma of a derivative using Black model. """
    d1 = (np.log(f/k) + v * v * t / 2.0) / (v * np.sqrt(t))
    return np.exp(-r*t) * n_prime_vect(d1) / (f * v * np.sqrt(t))

###############################################################################

def black_vega(f, t, k, r, v):
    """Return vega of a derivative using Black model. """
    d1 = (np.log(f/k) + v * v * t / 2.0) / (v * np.sqrt(t))
    return np.exp(-r*t) * f * np.sqrt(t) * n_prime_vect(d1)

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
    theta_1 = - ss * n_prime_vect(d1) * v / 2.0 / sqrt_t
    theta_2 = - phi * r * k * np.exp(-r*t) * n_prime_vect(phi * d2)
    theta_3 = phi * q * ss * n_prime_vect(phi * d1)
    return theta_1 + theta_2 + theta_3

def black_theta(f, t, k, r, q, v, option_type):
    """Return theta of a derivative using Black model. """

    if option_type == OptionTypes.EUROPEAN_CALL.value:
        phi = 1.0
    elif option_type == OptionTypes.EUROPEAN_PUT.value:
        phi = -1.0

    d1 = (np.log(f/k) + v * v * t / 2.0) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)
    sqrt_t =  np.sqrt(t)

    theta_1 = - f * np.exp(-r*t) * n_prime_vect(d1) * v / 2.0 / sqrt_t
    theta_2 = - phi * r * k * np.exp(-r * t) * n_prime_vect(phi * d2)
    theta_3 = phi * q * f * np.exp(-r*t) * n_prime_vect(phi * d1)
    return theta_1 + theta_2 + theta_3

###############################################################################

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
    vanna = - np.exp(-q*t) * n_prime_vect(d1) * (d2/v)
    return vanna

def black_vanna(f, t, k, q, vol):
    d1 = (np.log(f/k) + vol * vol * t / 2.0) / (vol * np.sqrt(t))
    d2 = d1 - vol * np.sqrt(t)
    vanna_ = -1 * np.exp(-q*t) * n_prime_vect(d1) * d2 / vol
    return vanna_

def vanna_wu(f, t, k, q, vol):
    gamma = black_gamma(f, t, k, q, vol)
    z_p = np.log(k/f) + 0.5 * np.power(vol, 2) * t
    vanna = z_p * gamma * f / vol
    return vanna


###############################################################################

def bs_volga(s, t, k, r, q, v):
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
    volga = np.exp(-q*t) * sqrt_t * n_prime_vect(d1) * ((d1*d2)/v)
    return volga

def black_volga(f, t, k, r, vol):
    d1 = (np.log(f / k) + vol * vol * t / 2.0) / (vol * np.sqrt(t))
    d2 = d1 - vol * np.sqrt(t)
    return f * np.exp(-r*t) * np.sqrt(t) * n_prime_vect(d1) * d1 * d2 * (1/vol)


###############################################################################


@vectorize([float64(float64, float64, float64, float64, int64, float64,
              int64, float64)], fastmath=True, nopython=True)
def nb_strike(s, t, rd, rf, option_type_value, delta, delta_method_value, v):


    phi = np.sign(option_type_value)
    if delta_method_value == DeltaType.SPOT_DELTA.value:

        dom_df = np.exp(-rd*t)
        for_df = np.exp(-rf*t)

        F0T = s * for_df / dom_df
        vsqrtt = v * np.sqrt(t)
        arg = delta*phi/for_df  # CHECK THIS !!!
        norm_inv_delta = norminvcdf(arg)
        K = F0T * np.exp(-vsqrtt * (phi*norm_inv_delta - vsqrtt/2.0))
        return K

    elif delta_method_value == DeltaType.FORWARD_DELTA.value:

        dom_df = np.exp(-rd*t)
        for_df = np.exp(-rf*t)

        F0T = s * for_df / dom_df
        vsqrtt = v * np.sqrt(t)
        arg = delta*phi   # CHECK THIS!!!!!!!!
        norm_inv_delta = norminvcdf(arg)
        K = F0T * np.exp(-vsqrtt * (phi*norm_inv_delta - vsqrtt/2.0))
        return K
    else:
        raise FinError("Unknown FinFXDeltaMethod")

@vectorize([float64(float64, float64, float64, float64, float64, float64,
                    int64)], fastmath=True, cache=True, nopython=True)
def blsvalue(s, t, k, r, q, v, option_type_value):
    """Price a derivative using Black-Scholes model."""

    phi = np.sign(option_type_value)

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

@vectorize([float64(float64, float64, float64, float64,
                    float64, float64, int64)], fastmath=True, cache=True, nopython=True)
def blsdelta(s, t, k, r, q, v, option_type_value):
    """Price a derivative using Black-Scholes model."""

    phi = np.sign(option_type_value)
    k = np.maximum(k, g_small)
    t = np.maximum(t, g_small)
    v = np.maximum(v, g_small)

    v_sqrt_t = v * np.sqrt(t)
    ss = s * np.exp(-q*t)
    kk = k * np.exp(-r*t)

    d1 = np.log(ss/kk) / v_sqrt_t + v_sqrt_t / 2.0

    delta = phi * np.exp(-q*t) * n_vect(phi * d1)
    return delta



#@njit(float64[:,:](float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:],
#                    int64, int64), fastmath=True, cache=True)
@vectorize([float64(float64, float64, float64, float64, float64, float64,
                    int64, int64)], fastmath=True, cache=True, nopython=True)
def nb_delta(s, t, k, rd, rf, vol, deltaTypeValue, option_type_value):
    """ Calculation of the FX Option delta. Used in the determination of
    the volatility surface. Avoids discount curve interpolation so it
    should be slightly faster than the full calculation of delta. """

    pips_spot_delta = blsdelta(s, t, k, rd, rf, vol, option_type_value)

    if deltaTypeValue == DeltaType.SPOT_DELTA.value:
        return pips_spot_delta
    elif deltaTypeValue == DeltaType.FORWARD_DELTA.value:
        pips_fwd_delta = pips_spot_delta * np.exp(rf*t)
        return pips_fwd_delta
    elif deltaTypeValue == DeltaType.SPOT_DELTA_PREM_ADJ.value:
        vpctf = blsvalue(s, t, k, rd, rf, vol, option_type_value) / s
        pct_spot_delta_prem_adj = pips_spot_delta - vpctf
        return pct_spot_delta_prem_adj
    elif deltaTypeValue == DeltaType.SPOT_DELTA_PREM_ADJ.value:
        vpctf = blsvalue(s, t, k, rd, rf, vol, option_type_value) / s
        pct_fwd_delta_prem_adj = np.exp(rf*t) * (pips_spot_delta - vpctf)
        return pct_fwd_delta_prem_adj
    else:
        raise FinError("Unknown FinFXDeltaMethod")


def black_volga(f, t, k, r, vol):
    d1 = (np.log(f / k) + vol * vol * t / 2.0) / (vol * np.sqrt(t))
    d2 = d1 - vol * np.sqrt(t)
    return f * np.exp(-r*t) * np.sqrt(t) * n_prime_vect(d1) * d1 * d2 * (1/vol)


def delta_bump(f, t, k, rd, v, option_type_value):

    bump = 0.0001 * f

    v_ = blsprice(f, t, k, rd, v, option_type_value)
    v_bumped = blsprice(f + bump, t, k, rd, v, option_type_value)
    delta = (v_bumped - v_) / bump
    return delta


def gamma_bump(f, t, k, rd, v, delta_type_value, option_type_value):

    bump = 0.0001 * f

    v_ = black_delta(f, t, k, rd, v, delta_type_value, option_type_value)
    v_bumped_dn = black_delta(f-bump, t, k, rd, v, delta_type_value, option_type_value)
    v_bumped_up = black_delta(f+bump, t, k, rd, v, delta_type_value, option_type_value)

    gd = (v_bumped_dn - v_) / bump
    gu = (v_bumped_up - v_) / bump

    gamma = (gu - gd) / 2.0
    return gamma

def vega_bump(f, t, k, rd, v, option_type_value):

    bump = 0.01

    v_ = blsprice(f, t, k, rd, v, option_type_value)
    v_bumped = blsprice(f, t, k, rd, v+bump, option_type_value)
    vega = (v_bumped - v_) / bump
    return vega

def theta_bump(f, t, k, rd, v, option_type_value):

    bump = 1/DateUtils.days_per_year

    v_ = blsprice(f, t, k, rd, v, option_type_value)
    v_bumped = blsprice(f, t-bump, k, rd, v, option_type_value)
    theta = (v_bumped - v_) / bump
    return theta

def vanna_bump(f, t, k, rd, v):

    bump = 0.0001 * f

    v_ = black_vega(f, t, k, rd, v)
    v_bumped = black_vega(f+bump, t, k, rd, v)
    vanna  = (v_bumped - v_) / bump
    return vanna


def volga_bump(f, t, k, rd, v):

    bump = 0.01

    v_ = black_vega(f, t, k, rd, v)
    v_bumped = black_vega(f, t, k, rd, v+bump)
    volga = (v_bumped - v_) / bump
    return volga






if __name__ == "__main__":

    from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource

    import pandas as pd
    import numpy as np

    path = '/Users/francisbarker/Desktop/ivols.csv'
    df = pd.read_csv(path, index_col=0)
    _df = df.pivot(columns=['x','t'])
    N = _df.shape[0]

    r = _df.get('rf').values
    q = _df.get('rd').values
    s =  _df.get('s').values
    k =  _df.get('k').values
    v = _df.get('sig').values

    t = np.array(_df.get('sig').columns.get_level_values('t')).reshape(1, -1).repeat(N, axis=0)

    _s = s[:,:].astype(np.float64)
    _r = r[:,:].astype(np.float64)
    _q = q[:,:].astype(np.float64)
    _k = k[:,:].astype(np.float64)
    _v = v[:,:].astype(np.float64)
    _t = t[:,:].astype(np.float64)

    _d = nb_delta(_s, _t, _k, _r, _q, _v, DeltaType.FORWARD_DELTA.value, -1)
    __D = blsdelta(_s, _t, _k, _r, _q, _v, -1)



    ks = nb_strike(_s, _t, _r, _q, -1, _d, 1, _v)

    delta = blsdelta(_s, _t, _k, _r, _q, _v, -1)



