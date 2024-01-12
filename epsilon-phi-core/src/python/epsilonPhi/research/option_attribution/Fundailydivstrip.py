import numpy as np
from scipy.stats import norm

def Fundailydivstrip(t, ud, dd, IVS, lmh, xv, nm, nx, pos):

    # Initialize arrays to store results
    Z = np.full((1, nx, nm), np.nan)
    LIt, DIt, LCt, DCt, LPt, DPt = Z.copy(), Z.copy(), Z.copy(), Z.copy(), Z.copy(), Z.copy()

    # Find indices for the current and next trading days
    indt = dd == ud[t]
    indn = dd == ud[t + 1]
    IVt = IVS[indt, :]
    IVn = IVS[indn, :]

    # Calculate the log return and median implied volatility for the next trading day
    Rvt = np.log(np.nanmedian(IVn[:, pos['S']]) / np.nanmedian(IVt[:, pos['S']]))
    Svt = np.nanmedian(IVt[:, pos['S']])

    # Find common options between the current and next trading days
    C, IA, IB = np.intersect1d(IVt[:, pos['id']], IVn[:, pos['id']], return_indices=True)
    nobs = len(C)

    # Check if there are enough observations to proceed
    if nobs > 10:
        IVt = IVt[IA, :]
        IVn = IVn[IB, :]
        dy = np.log(IVn[:, pos['iv']] / IVt[:, pos['iv']])

        y = IVt[:, pos['iv']]
        p = IVt[:, pos['fm']]

        # price cahnge of
        dp = IVn[:, pos['fm']] - IVt[:, pos['fm']]

        days = IVt[:, pos['days']]
        tau = days / 365
        lm = np.log(tau)

        # log moneyness
        k = np.log(IVt[:, pos['K']] / IVt[:, pos['F']])

        # z-positive
        zp = (k + 0.5 * y ** 2 * tau) / (y * np.sqrt(tau))
        # z-negative
        zm = (k - 0.5 * y ** 2 * tau) / (y * np.sqrt(tau))

        wpc = norm.cdf(zm)
        pc = IVt[:, pos['CP']]
        wpc[pc == 2] = 1 - wpc[pc == 2]
        wpc[wpc < 0.2] = 0
        callonly = np.zeros(nobs)
        callonly[pc == 1] = 1
        putonly = np.zeros(nobs)
        putonly[pc == 2] = 1

        sigmaz = np.std(zp)
        hz = 1 * (4 / 3) ** (1 / 5) * sigmaz / nobs ** (1 / 5)
        sigmam = np.std(lm)
        hm = 2 * (4 / 3) ** (1 / 5) * sigmam / nobs ** (1 / 5)

        for m in range(nm):
            xm = np.abs(lm - lmh[m]) / hm
            wm = wpc * np.exp(-xm ** 2 / 2)
            wmc = callonly * np.exp(-xm ** 2 / 2)
            wmp = putonly * np.exp(-xm ** 2 / 2)

            for j in range(nx):

                # Equation 16 for Gaussian Smoothing, hz is the bandwidth
                xk = np.abs(zp - xv[j]) / hz
                wk = wm * np.exp(-xk ** 2 / 2)
                wkc = wmc * np.exp(-xk ** 2 / 2)
                wkp = wmp * np.exp(-xk ** 2 / 2)

                # Calculate various weighted statistics for options
                DIt[0, j, m] = np.sum(wk * dy) / np.sum(wk)
                LIt[0, j, m] = np.sum(wk * y) / np.sum(wk)

                LCt[0, j, m] = np.sum(wkc * p) / np.sum(wkc)
                DCt[0, j, m] = np.sum(wkc * dp) / np.sum(wkc)

                LPt[0, j, m] = np.sum(wkp * p) / np.sum(wkp)
                DPt[0, j, m] = np.sum(wkp * dp) / np.sum(wkp)

    return Rvt, Svt, LIt, DIt, LCt, DCt, LPt, DPt
