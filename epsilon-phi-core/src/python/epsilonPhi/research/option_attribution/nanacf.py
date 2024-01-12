import numpy as np

def nanacf(data, nrho):
    nobs, nvars = data.shape
    rho = np.zeros((nrho, nvars))
    A = data - np.nanmean(data, axis=0)

    for i in range(nvars):
        x = A[:, i]
        gamma0 = np.nanmean(x * x)
        for k in range(1, nrho + 1):
            num = np.nanmean(x[:-k] * x[k:])
            rho[k - 1, i] = num / gamma0

    return rho
