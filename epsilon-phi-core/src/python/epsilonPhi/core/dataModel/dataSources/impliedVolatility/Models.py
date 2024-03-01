import numpy as np

from epsilonPhi.core.utils.OptionUtils import *
from epsilonPhi.core.dataModel.alchemist.SessionManager import *

def polynomial_regression_2d(x, y, x_prime, y_prime, z):

    reg = np.concatenate((np.vstack(np.ones(z.size)),
                          np.vstack(x),
                          np.vstack(x * x)),
                          np.vstack(y),
                          np.vstack(y * x),
                          axis=1)
    b = np.linalg.lstsq(reg, z, rcond=-1)[0]
    return b[0] + x_prime * b[1] + (x_prime ** 2) * b[2] + y_prime * b[3] + (x_prime * y_prime) * b[4]

def polynomial_regression_1d(x, y, x_prime):

    reg = np.concatenate((np.vstack(np.ones(y.size)),
                          np.vstack(x),
                          np.vstack(x * x)),
                          axis=1)
    b = np.linalg.lstsq(reg, y, rcond=-1)[0]
    return (x_prime**2)*b[2] + x_prime*b[1] + b[0]

def vanna_volga_2d(F, k_prime, t, kput, katm, kcall, sigput, sigatm, sigcal):

    k_d, m_d = np.meshgrid(k_prime, t)

    kput = np.array(kput).reshape(-1, 1).repeat(m_d.shape[1], axis=1)
    katm = np.array(katm).reshape(-1, 1).repeat(m_d.shape[1], axis=1)
    kcall = np.array(kcall).reshape(-1, 1).repeat(m_d.shape[1], axis=1)
    sigput = np.array(sigput).reshape(-1, 1).repeat(k_d.shape[1], axis=1)
    sigatm = np.array(sigatm).reshape(-1, 1).repeat(k_d.shape[1], axis=1)
    sigcal = np.array(sigcal).reshape(-1, 1).repeat(k_d.shape[1], axis=1)

    # First Interpolate the Cross-Section
    w_put = (np.log(katm / k_d) * np.log(kcall / k_d)) / (np.log(katm / kput) * np.log(kcall / kput))
    w_atm = (np.log(k_d / kput) * np.log(kcall / k_d)) / (np.log(katm / kput) * np.log(kcall / katm))
    w_cal = (np.log(k_d / kput) * np.log(k_d / katm)) / (np.log(kcall / kput) * np.log(kcall / katm))

    # First Interpolate the Cross-Section
    f = F.reshape(-1, 1).repeat(m_d.shape[1], axis=1)
    T = t.reshape(-1, 1).repeat(f.shape[1], 1)
    d1d2 = d1(f, k_d, sigatm, T) * d2(f, k_d, sigatm, T)

    # First Interpolate the Cross-Section
    vv_fo = (w_put * sigput + w_atm * sigatm + w_cal * sigcal) - sigatm
    vv_so = (w_put * d1(f, kput, sigput, T) * d2(f, kput, sigput, T) * np.power(sigput - sigatm, 2) +
             w_atm * d1(f, katm, sigatm, T) * d2(f, katm, sigatm, T) * np.power(sigatm - sigatm, 2) +
             w_cal * d1(f, kcall, sigcal, T) * d2(f, kcall, sigcal, T) * np.power(sigcal - sigatm, 2))

    # First Interpolate the Cross-Section
    vol = sigatm + (-sigatm + np.sqrt(sigatm ** 2 + d1d2 * (2 * sigatm * vv_fo + vv_so))) / d1d2
    return vol

def vanna_volga_1d(F, k_prime, t_prime, t, kput, katm, kcall, sigput, sigatm, sigcal):

    kput = np.array(kput)
    katm = np.array(katm)
    kcall = np.array(kcall)
    sigput = np.array(sigput)
    sigatm = np.array(sigatm)
    sigcal = np.array(sigcal)

    # First Interpolate the Cross-Section
    w_put = (np.log(katm / k_prime) * np.log(kcall / k_prime)) / (np.log(katm / kput) * np.log(kcall / kput))
    w_atm = (np.log(k_prime / kput) * np.log(kcall / k_prime)) / (np.log(katm / kput) * np.log(kcall / katm))
    w_cal = (np.log(k_prime / kput) * np.log(k_prime / katm)) /  (np.log(kcall / kput) * np.log(kcall / katm))

    # First Interpolate the Cross-Section
    d1d2 = d1(F, k_prime, sigatm, t) * d2(F, k_prime, sigatm, t)

    # First Interpolate the Cross-Section
    vv_fo = (w_put * sigput + w_atm * sigatm + w_cal * sigcal) - sigatm
    vv_so = (w_put * d1(F, kput, sigput, t) * d2(F, kput, sigput, t) * np.power(sigput - sigatm, 2) +
             w_atm *  d1(F, katm, sigatm, t) * d2(F, katm, sigatm, t)  * np.power(sigatm - sigatm, 2) +
             w_cal * d1(F, kcall, sigcal, t) * d2(F, kcall, sigcal, t) * np.power(sigcal - sigatm, 2))

    # First Interpolate the Cross-Section
    vol = sigatm + (-sigatm + np.sqrt(sigatm ** 2 + d1d2 * (2 * sigatm * vv_fo + vv_so))) / d1d2
    return vol

def gaussian_kernel_smoother_1d(x, y, x_prime, h=None):

    x = x.reshape(-1, 1)
    x_prime = x_prime.reshape(1, -1)
    x_d = np.abs(x - x_prime) / h

    _wts = np.exp(-1.0 * np.power(x_d, 2.0)/2)
    _wts = _wts / np.sum(_wts, axis=0)
    return y @ _wts

# @jit(nopython=True, fastmath=True, cache=True)
def gaussian_kernel_smoother_2d(z, z_prime, x, x_prime, y, h_x, h_z):

    N = len(y)
    if h_x is None:
        h_x = (1.0 * np.power(4.0 / 3.0, 1.0 / 5.0) * np.std(x) /
               np.power(N, 1.0 / 5.0))

    if h_z is None:
        h_z = (2.0 * np.power(4.0 / 3.0, 1.0 / 5.0) * np.std(z) /
               np.power(N, 1.0 / 5.0) * 0.1)

    lnmat_ = z.reshape((-1, 1))
    m_diff = np.abs(lnmat_ - np.log(z_prime.repeat(N).reshape((z_prime.size, N)).T)) / h_z
    x_diff = np.abs(x.reshape((-1, 1)) - x_prime.repeat(N).reshape((x_prime.size, N)).T) / h_x

    x_wts = np.exp(-1.0 * np.power(x_diff, 2.0) / 2.0)
    x_wts = np.expand_dims(x_wts, -1).repeat(z_prime.size).reshape((x_wts.shape[0], x_wts.shape[1], z_prime.size))

    m_wts = np.exp(-1.0 * np.power(m_diff, 2.0) / 2.0).reshape((m_diff.shape[0], 1, m_diff.shape[1]))
    m_wts_ = np.full(x_wts.shape, np.nan) * np.nan
    for dim in range(x_wts.shape[1]):
        m_wts_[:, dim, :] = m_wts[:, 0, :]

    wts = (x_wts * m_wts) / np.sum(x_wts * m_wts_, axis=0)
    y_prime = y.repeat(wts.shape[1] * z_prime.size).reshape(m_wts_.shape)
    y_primes = np.sum(wts * y_prime, axis=0).flatten()
    return y_primes

if __name__ == "__main__":

    sessionMgr = SessionMgr()

    underlier = 'EURUSD'
    df = sessionMgr.get_ivols(underlier, 'NYC')

    # single date
    df_t = df.set_index('date').loc['2009-01-20']
    df_t = df_t[~df_t.relative_strike.isin([1, 0.45, 0.55])]
    df_t['t'] =  DateUtils.Rdate_to_mat(df_t.get('tenor'))
    df_t.loc[df_t.relative_strike < 0, 'relative_strike'] = df_t.loc[df_t.relative_strike < 0, 'relative_strike'] + 1
    df_t.loc[df_t.relative_strike == -998, 'relative_strike'] = 0.5

    _s = df_t.set_index(['t','relative_strike']).get('mid').unstack(level=0)
    _s = _s.sort_index(axis=1)

    # 1d polynomial regression
    y = _s.get(1).values
    x = np.array(_s.index)

    # Fit the Polynomial Regression
    fit = polynomial_regression_1d(x, y, np.linspace(0.1, 0.9, 1000))
    fit_ = gaussian_kernel_smoother_1d(x, y, np.linspace(0.1, 0.9, 1000), 0.2)

    # Fit the 1-d Gaussian Kernel

    from matplotlib import pyplot as plt # Change - use least squares
    plt.scatter(x, y)
    u = np.linspace(0.1, 0.9, 1000)
    plt.plot(u, fit)
    plt.show()

    plt.plot(u, fit_)
    plt.show()
