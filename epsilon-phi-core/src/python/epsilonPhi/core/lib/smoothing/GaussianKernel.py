import numpy as np
from numba import njit

@njit(cache=True)
def smooth_2d(x, y, xp, yp, z):

        # get the std of the reference on each date
        sig_x = np.nanstd(x)
        h_x = 2.0 * 1.059 * sig_x * len(z) ** (-1/5.)


        # do the same in the maturity direction
        sig_y = np.nanstd(x)
        h_m = 2.0 * 1.059 * sig_y * len(z) ** (-1/5.)

        m_diff = (y - yp) / h_m
        x_diff = (x - xp) / h_x

        x_wts = np.exp(-1.0 * np.power(x_diff, 2.0) / 2.0)
        m_wts = np.exp(-1.0 * np.power(m_diff, 2.0) / 2.0)

        wts = (x_wts * m_wts) / np.nansum(x_wts * m_wts)
        return np.nansum(wts * z)

class GaussianKernel(object):
    def __init__(self):
        pass

    @staticmethod
    def smooth_1d(x, xp, z):
        pass

    @staticmethod
    @njit(cache=True)
    def smooth_2d(x, y, xp, yp, z):

        # get the std of the reference on each date
        sig_x = np.std(x)
        h_x = 1.0 * np.power(4.0 / 3.0, 1.0 / 5.0) * sig_x / np.power(len(z), 1.0 / 5.0)

        # do the same in the maturity direction
        sig_y = np.std(y)
        h_m = 2.0 * np.power(4.0 / 3.0, 1.0 / 5.0) * sig_y / np.power(len(z), 1.0 / 5.0) * 0.1

        m_diff = (y - yp) / h_m
        x_diff = (x - xp) / h_x

        x_wts = np.exp(-1.0 * np.power(x_diff, 2.0) / 2.0)
        m_wts = np.exp(-1.0 * np.power(m_diff, 2.0) / 2.0)

        wts = (x_wts * m_wts) / np.sum(x_wts * m_wts, axis=0)
        return wts @ z
