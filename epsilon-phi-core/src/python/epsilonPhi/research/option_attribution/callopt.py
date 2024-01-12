import numpy as np
from scipy.stats import norm


def callopt(F, K, r, t, st, v):
    """
    Call option pricing formula. Calculates both value and greeks (delta, vega)
    in forward space.

    Parameters:
    F : Forward price of the underlying asset
    K : Strike price
    r : Risk-free interest rate
    t : Time to expiration
    st : Time steps
    v : Volatility of the underlying asset

    Returns:
    P : Call option price
    fdelta : Delta of the call option
    fvega : Vega of the call option
    """
    sv = v * st
    d1 = (np.log(F / K) + 0.5 * sv ** 2) / sv
    d2 = d1 - sv
    B = np.exp(-r * t)
    Nd1 = norm.cdf(d1)
    fdelta = B * Nd1
    P = B * (F * Nd1 - K * norm.cdf(d2))
    fvega = F * st * norm.pdf(d1)

    return P, fdelta, fvega