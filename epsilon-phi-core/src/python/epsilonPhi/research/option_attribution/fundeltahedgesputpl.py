import numpy as np
from scipy.stats import norm


def fundeltahedgesputpl(v, F, K, tv, stv, rv, BR):
    P, pdelta = putopt(F, K, rv, tv, stv, v)
    PL = (-P[-1] + P[0] * BR[0]) + np.dot(BR[:-1], pdelta[:-1] * np.diff(F))
    return PL


def putopt(F, K, r, t, st, v):
    nt = len(t)
    P = np.zeros(nt)
    fdelta = np.zeros(nt)
    indt = t <= 0

    if np.sum(indt) > 0:
        P[indt] = np.maximum(0, K - F[indt])
        fdelta[indt] = np.sign(P[indt])

        if np.sum(~indt) > 0:
            t = t[~indt]
            st = st[~indt]
            F = F[~indt]
            sv = v * st
            d1 = (np.log(F / K) + 0.5 * sv ** 2) / sv
            d2 = d1 - sv
            B = np.exp(-r * t)
            Nd1 = norm.cdf(d1)
            fdelta[~indt] = -B * (1 - Nd1)
            P[~indt] = B * (F * Nd1 - K * norm.cdf(d2) - F + K)
    else:
        sv = v * st
        d1 = (np.log(F / K) + 0.5 * sv ** 2) / sv
        d2 = d1 - sv
        B = np.exp(-r * t)
        Nd1 = norm.cdf(d1)
        fdelta = -B * (1 - Nd1)
        P = B * (F * Nd1 - K * norm.cdf(d2) - F + K)

    return P, fdelta
