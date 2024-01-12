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

    # Check if there are enough observations to proceed, must have more than 10 options
    if nobs > 10:
        IVt = IVt[IA, :]
        IVn = IVn[IB, :]

        # get the LOG change in IV for the options that we can observe on consecutive dates
        # R(i, t+1)
        dy = np.log(IVn[:, pos['iv']] / IVt[:, pos['iv']])

        # Y is the Implied Vol on the Current Date t
        y = IVt[:, pos['iv']]
        # p is the forward option price on the currency date
        p = IVt[:, pos['fm']]

        # price CHANGE OF FORWARD OPTION PRICE
        dp = IVn[:, pos['fm']] - IVt[:, pos['fm']]

        # DAYS TO EXPIRY
        days = IVt[:, pos['days']]
        # TIME TO MATURITY IN YEARS
        tau = days / 365
        # LOG TIME TO MATURITY
        lm = np.log(tau)

        # log moneyness
        k = np.log(IVt[:, pos['K']] / IVt[:, pos['F']])

        # z-positive
        zp = (k + 0.5 * y ** 2 * tau) / (y * np.sqrt(tau))
        # z-negative
        zm = (k - 0.5 * y ** 2 * tau) / (y * np.sqrt(tau))

        # norm cdf of z-negative N(d2), which is the forward delta of the option
        wpc = norm.cdf(zm)

        # Get the call and put identifiers
        pc = IVt[:, pos['CP']]

        # We put more weight on the out-of-the-money option contract, which
        # tends to be more actively traded and hence tends to have a more reliable
        # quote.

        # We use one minus the absolute value of the option’s forward delta as
        # the weight, and further truncate the weight to zero when the absolute delta
        # is greater than 80%. The truncation sets the weights on deep in-the-money
        # options to zero when the absolute delta is over 80%, where the quotes tend to
        # become unreliable.

        wpc[pc == 2] = 1 - wpc[pc == 2]
        # set deep in the money weights to zero
        # wpc is the weighted calls and puts togetherr
        wpc[wpc < 0.2] = 0

        callonly = np.zeros(nobs)
        callonly[pc == 1] = 1

        putonly = np.zeros(nobs)
        putonly[pc == 2] = 1

        # calucalute bandwidth 1, h(x)
        sigmaz = np.std(zp)
        hz = 1 * (4 / 3) ** (1 / 5) * sigmaz / nobs ** (1 / 5)

        # calculate bandwidth 2, h(tau)
        sigmam = np.std(lm)
        hm = 2 * (4 / 3) ** (1 / 5) * sigmam / nobs ** (1 / 5)

        # loop through all maturities
        for m in range(nm):

            # get the distance of the option from the target log time to maturity
            xm = np.abs(lm - lmh[m]) / hm

            # weights call and put comb
            wm = wpc * np.exp(-xm ** 2 / 2)

            # weights for calls only
            wmc = callonly * np.exp(-xm ** 2 / 2)

            # weights for puts only
            wmp = putonly * np.exp(-xm ** 2 / 2)

            # Now that we have the weights across maturities
            # dimension, interpolate across moneyness dimension
            for j in range(nx):

                # Equation 16 for Gaussian Smoothing, hz is the bandwidth
                # xv is the target moneynness of the vols, hz is bandwidth, zp z-score of options
                xk = np.abs(zp - xv[j]) / hz

                # weights in k space calls and puts
                wk = wm * np.exp(-xk ** 2 / 2)
                # weights in k space calls only
                wkc = wmc * np.exp(-xk ** 2 / 2)
                # weights in k space puts only
                wkp = wmp * np.exp(-xk ** 2 / 2)

                # Calculate various weighted statistics for options

                # weighted sum of change in log IV from t, t+1
                DIt[0, j, m] = np.sum(wk * dy) / np.sum(wk)
                # weighted sum of level of IV on date t
                LIt[0, j, m] = np.sum(wk * y) / np.sum(wk)

                # weighted sum of mid price of call options (forward adjusted)
                LCt[0, j, m] = np.sum(wkc * p) / np.sum(wkc)
                # weighted sum of change in price of call options (forward adjusted)
                DCt[0, j, m] = np.sum(wkc * dp) / np.sum(wkc)
                # same as above but for puts
                LPt[0, j, m] = np.sum(wkp * p) / np.sum(wkp)
                DPt[0, j, m] = np.sum(wkp * dp) / np.sum(wkp)

    return Rvt, Svt, LIt, DIt, LCt, DCt, LPt, DPt
    # Output is
    # Rvt = log return of spot t to t+1
    # Svt = spot price on date t
    # LIt = IV Level on date t interpolated calls and puts
    # DIt = Change in log IV from t to t+1 interpolated calls and puts