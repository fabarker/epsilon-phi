import numpy as np

def jac(bb, X, moments):
    eps = 1e-12
    nn, np = bb.shape
    e = np.mean(moments(bb, X), axis=0)
    neq, nn = e.shape
    np = len(bb)
    j = np.zeros((neq, np))

    x1 = np.mean(moments(bb, X), axis=0)
    for i in range(np):
        stepsizei = (eps ** 0.2) * bb[i]
        if stepsizei < np.finfo(float).tiny:
            stepsizei = eps
        tempj = bb[i]
        bb[i] += stepsizei
        stepsizei = bb[i] - tempj
        x2 = np.mean(moments(bb, X), axis=0)
        j[:, i] = (x2 - x1) / stepsizei
        bb[i] = tempj

    return j.T
