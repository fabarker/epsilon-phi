import numpy as np
from statsmodels.tsa.stattools import acovf

def newey2(b, X, L, moments):
    e = moments(b, X)
    n, m = e.shape

    # Modification: demean
    e -= np.mean(e, axis=0)

    # Calculate autocovariance function
    R = acovf(e, unbiased=False, fft=True, nlag=L+1)

    W = np.zeros((m, m))

    # Construct initial covariance matrix
    for i in range(m):
        for j in range(m):
            W[i, j] = R[i + (j - 1) * m, 0]

    # Triangle window function
    w = np.arange(1, 2*L+1)
    w = w / np.max(w)

    # Adjust covariance matrix using Newey-West method
    for ii in range(2, L + 2):
        W1 = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                W1[i, j] = R[i + (j - 1) * m, ii - 1]
        W += w[L + ii - 2] * (W1 + W1.T)

    # Invert the matrix
    W = np.linalg.inv(W)

    return W
