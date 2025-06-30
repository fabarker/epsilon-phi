import numpy as np

def interpolate_single_date(mat, unique_m, x, unique_x, mids):

    N = len(mids)

    # get the std of the reference on each date
    sig_x = x.std(ddof=1)
    h_x = 1 * np.power(4 / 3, 1 / 5) * sig_x / np.power(N, 1 / 5)

    # do the same in the maturity direction
    sig_lm = np.log(mat).std(ddof=1)
    h_m = 2 * np.power(4 / 3, 1 / 5) * sig_lm / np.power(N, 1 / 5) * 0.1

    lnmat_ = np.log(mat).reshape(-1, 1)

    m_diff = np.abs(lnmat_ - np.log(unique_m.repeat(N, 0))) / h_m
    x_diff = np.abs(x.reshape(-1, 1) - unique_x.repeat(N, 0)) / h_x

    x_wts = np.exp(-1 * np.power(x_diff, 2) / 2)
    x_wts = np.repeat(x_wts[:, :, np.newaxis], unique_m.size, axis=2)

    m_wts = np.exp(-1 * np.power(m_diff, 2) / 2).reshape(m_diff.shape[0], 1, m_diff.shape[1])
    m_wts = m_wts.repeat(x_wts.shape[1], 1)

    wts = (x_wts * m_wts) / np.sum(x_wts * m_wts, axis=0)
    ivol = np.repeat(mids.reshape(-1, 1).repeat(wts.shape[1], 1)[:, :, np.newaxis], unique_m.size, axis=2)
    ivols = np.sum(wts * ivol, axis=0).flatten()
    return ivols

