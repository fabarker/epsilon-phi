import numpy as np
import pandas as pd
from scipy.stats import norm


def put_price(F, K, r, t, st, v):
    """
    Put option pricing formula. Calculates both value and greeks (delta, vega)
    in forward space.

    Parameters:
    F : Forward price of the underlying asset
    K : Strike price
    r : Risk-free interest rate
    t : Time to expiration
    st : Time steps
    v : Volatility of the underlying asset

    Returns:
    P : Put option price
    fdelta : Delta of the put option
    fvega : Vega of the put option
    """

    if isinstance(F, pd.Series):
        F = F.to_frame()
        F = pd.concat([F] * K.shape[1], axis=1)
        F.columns = K.columns

    sv = v * st
    d1 = (np.log(F / K) + 0.5 * sv ** 2) / sv
    d2 = d1 - sv
    B = np.exp(-r * t)
    Nd1 = pd.DataFrame(norm.cdf(-d1), columns=d1.columns, index=d1.index)
    Nd2 = pd.DataFrame(norm.cdf(-d2), columns=d1.columns, index=d1.index)
    fdelta = B * (Nd1 - 1)
    P = B * (K * Nd2 - F * Nd1)
    fvega = F * st * norm.pdf(d1)

    return P, fdelta, fvega


def call_price(F, K, r, t, st, v):
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

    if isinstance(F, pd.Series):
       F = F.to_frame()
       F = pd.concat([F] * K.shape[1], axis=1)
       F.columns = K.columns

    sv = v * st
    d1 = (np.log(F / K) + 0.5 * sv ** 2) / sv
    d2 = d1 - sv
    B = np.exp(-r * t)
    Nd1 = pd.DataFrame(norm.cdf(d1), columns=d1.columns, index=d1.index)
    fdelta = B * Nd1
    P = B * (F * Nd1 - K * norm.cdf(d2))
    fvega = F * st * norm.pdf(d1)

    return P, fdelta, fvega