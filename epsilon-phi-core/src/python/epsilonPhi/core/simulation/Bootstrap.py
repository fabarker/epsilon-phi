import numpy as np
import pandas as pd
import typing as tp


class AbstractBootstrapper(object):
    _state = 5489
    _R = np.random.RandomState(5489)

    def __init__(self):
        pass

    @staticmethod
    def optimal_block_length(data: np.ndarray):

        def lam(kk: np.ndarray) -> np.ndarray:
            """
            Helper function, calculates the flattop kernel weights.

            Adapted for Python August 12, 2018 by Michael C. Nowotny
            """
            return (np.abs(kk) >= 0) * (np.abs(kk) < 0.5) \
                + 2 * (1.0 - np.abs(kk)) * (np.abs(kk) >= 0.5) * (np.abs(kk) <= 1)

        def mlag(x: np.ndarray,
                 n: tp.Optional[int] = 1,
                 init: tp.Optional[float] = 0.0) -> np.ndarray:

            nobs, nvar = x.shape

            xlag = np.ones((nobs, nvar * n), dtype=x.dtype) * init
            icnt = 0
            for i in range(nvar):
                for j in range(n):
                    xlag[j + 1:, icnt + j] = x[0:-j - 1, i]
                icnt += n

            return xlag

        """
        This is a function to select the optimal (in the sense of minimising the MSE
        of the estimator of the long-run variance) block length for the stationary
        bootstrap or circular bootstrap.
        The code follows Politis and White, 2001,
        "Automatic Block-Length Selection for the Dependent Bootstrap".


        NOTE: The optimal average block length for the stationary bootstrap,
        and it does not need to be an integer.
        The optimal block length for the circular bootstrap should be an
        integer. Politis and White suggest rounding the output UP to the
        nearest integer.

        Args:
          data, an nxk matrix

          Returns: a 2xk NumPy array of optimal bootstrap block lengths,
                      [[b_star_sb], [b_star_cb]], where
                      b_star_sb: optimal block length for stationary bootstrap
                      b_star_cb: optimal block length for circular bootstrap
        """

        if data.ndim == 1:
            data = data.reshape((-1, 1))
        elif data.ndim > 2:
            raise ValueError(
                'data must be a two dimensional NumPy array'
                '(number of observations x number of variables)')
        n, k = data.shape

        # these are optional in the original Matlab implementation
        # opt_block_length_full.m, but fixed at default values here
        kn = int(max(5, np.sqrt(np.log10(n))))

        # adding kn extra lags to employ Politis' (2002) suggestion
        # for finding largest significant m
        m_max = int(np.ceil(np.sqrt(n)) + kn)

        # maximum value of b_star_sb to consider.
        # dec07: new idea for rule-of-thumb to put an upper bound on estimated
        # optimal block length
        b_max = np.ceil(min(3 * np.sqrt(n), n / 3))

        c = 2
        original_data = data
        # b_star_final = np.zeros((2, k), dtype=np.float64)
        b_star_final = []

        for i in range(k):
            data = original_data[:, i].reshape((-1, 1))

            # FIRST STEP: finding m_hat-> the largest lag for which the
            # auto-correlation is still significant.
            temp = mlag(data, m_max)

            # dropping the first m_max rows, as they are filled with zeros
            temp = temp[m_max:, :]
            temp = np.corrcoef(np.hstack((data[m_max:], temp)), rowvar=False)
            temp = temp[1:, 0].reshape((-1, 1))

            # We follow the empirical rule suggested in
            # Politis, 2002, "Adaptive Bandwidth Choice".
            # as suggested in Remark 2.3, setting c=2, kn=5

            # looking at vectors of auto-correlations,
            # from lag m_hat to lag m_hat+kn
            temp2 = np.hstack((np.transpose(mlag(temp, kn)), temp[-kn:]))

            # dropping the first kn-1, as the vectors have empty cells
            temp2 = temp2[:, kn:]

            # checking which are less than the critical value
            temp2 = np.abs(temp2) < (c * np.sqrt(np.log10(n) / n)
                                     * np.ones((kn, m_max - kn + 1)))

            # this counts the number of insignificant autocorrelations
            temp2 = np.sum(temp2, axis=0).reshape((1, -1))
            temp3 = np.hstack((np.arange(1, temp2.shape[1] + 1).reshape((-1, 1)),
                               temp2.transpose()))

            # selecting all rows where ALL kn auto-correlations are not significant
            temp3 = temp3[np.squeeze(temp2 == kn), :]

            if temp3.size == 0:
                # this means that NO collection of kn auto-correlations were all
                # insignificant, so pick largest significant lag
                m_hat = max(
                    np.flatnonzero(np.abs(temp) > (c * np.sqrt(np.log10(n) / n))))
            else:
                # if more than one collection is possible, choose the smallest m
                m_hat = temp3[0, 0]

            if 2 * m_hat > m_max:
                m = m_max
            else:
                m = 2 * m_hat

            del temp, temp2, temp3

            # SECOND STEP: computing the inputs to the function for b_star_sb
            kk = np.arange(-m, m + 1).reshape((-1, 1))
            if m > 0:
                temp = mlag(data, m)

                # dropping the first m_max rows, as they're filled with zeros
                temp = temp[m:, :]
                temp = np.cov(np.hstack((data[m:], temp)).transpose())

                # auto-covariances
                acv = temp[:, 0].reshape((-1, 1))
                acv2 = np.hstack(
                    (-np.arange(1, m + 1).reshape((-1, 1)), acv[1:, :]))
                if acv2.shape[0] > 1:
                    acv2 = acv2[acv2[:, 0].argsort(),]

                # auto-covariances from -m to m
                acv = np.vstack((acv2[:, 1].reshape((-1, 1)), acv))
                del acv2

                g_hat = np.sum(lam(kk / m) * np.abs(kk) * acv)
                dcb_hat = (4.0 / 3.0) * np.sum(lam(kk / m) * acv) ** 2

                # first part of dsb_hat (note cos(0)=1)
                dsb_hat = 2 * (np.sum(lam(kk / m) * acv) ** 2)

                # FINAL STEP: constructing the optimal block length estimator

                # optimal block lenght for stationary bootstrap
                b_star_sb = ((2 * (g_hat ** 2) / dsb_hat) ** (1.0 / 3.0)) \
                            * (n ** (1.0 / 3.0))
                if b_star_sb > b_max:
                    b_star_sb = b_max

                # optimal block length for circular bootstrap
                b_star_cb = ((2 * (g_hat ** 2) / dcb_hat) ** (1.0 / 3.0)) \
                            * (n ** (1.0 / 3.0))
                if b_star_cb > b_max:
                    b_star_cb = b_max

                # b_star = (b_star_sb, b_star_cb)
                return b_star_sb,b_star_cb
            else:
                return 1, 1

    @staticmethod
    def stationary_block_bootstrap(T, N, q):

        nRands = T * N
        R = np.random.RandomState(AbstractBootstrapper._state)
        blockLocations = R.choice(range(0, T), size=(nRands, 1), replace=True)
        blockLengths = np.minimum(R.geometric(q, (nRands, 1)), T)

        block_locs_list = list()
        L = 0
        b = 0

        while L < nRands:
            block_size = int(blockLengths[b])
            block_loc = int(blockLocations[b])

            locs = np.mod(block_loc + np.array(range(0, block_size, 1)), T)
            block_locs_list.extend(locs.tolist())

            L = L + block_size
            if np.mod(L, T) < block_size:
                L = L - np.mod(L, T)
            b = b + 1

        del blockLengths
        del blockLocations

        return np.reshape(np.array(block_locs_list)[:nRands], (N, T)).T

    @staticmethod
    def random_generator(output_row, output_col, output_range, use_replacement=False):

        low = 0
        high = 0
        if isinstance(output_range, range):
            low = output_range[0]
            high = len(output_range) + 1
        elif isinstance(output_range, int):
            high = output_range

        return AbstractBootstrapper._R.choice(range(low, high), size=(output_row, output_col), replace=use_replacement)

    @staticmethod
    def block_bootstrap(T, N, q):

        theta = np.zeros((T, N), dtype=int)
        for strap in range(N):
            t = 0
            theta[t, strap] = AbstractBootstrapper.random_generator(1, 1, range(0, T-1))
            while t < T-1:
                t = t + 1
                U = AbstractBootstrapper._R.random()
                if U < q:
                    theta[t, strap] = AbstractBootstrapper.random_generator(1, 1, range(0, T-1))
                else:
                    if theta[t-1, strap] + 1 > T-1:
                        theta[t, strap] = 0
                    else:
                        theta[t, strap] = theta[t-1, strap] + 1
        return theta


    def circular_bootstrap(self):
        pass

    def bootstrap(self):
        pass

class TimeSeriesBootstrapper(AbstractBootstrapper):
    def __init__(self, df, N, q):
        super(TimeSeriesBootstrapper, self).__init__()

        self._df = df.copy()
        self._bootstrap_df()

        self._T = df.shape[0]
        self._N = N
        self._q = q

    def _bootstrap_df(self):
        pass

    def _bootstrap_indicies(self, N, q):
        pass


class _Bootstrapper(AbstractBootstrapper):
    def __init__(self, function_apply, sample):
        super(_Bootstrapper, self).__init__()

        self._function_apply = function_apply
        self._sample = sample



class SAABootstrapper(AbstractBootstrapper):
    def __init__(self):
        super(SAABootstrapper, self).__init__()

    @staticmethod
    def extract_shocks(df):

        df_prime = df.dropna()
        Y = df_prime.values[1:]
        X = df_prime.values[:-1]

        T = len(Y)
        N = len(X.columns)

        X_prime = np.append(X, np.ones([T, 1]), axis=1)
        stats = np.linalg.lstsq(X_prime, Y, recond=None)
        A = stats[-1]
        beta = stats[:N]

        shocks = df.values[1:] - A - beta * df.values[:,-1]
        shocks = np.insert(shocks, 0, 0)
        return A, beta, shocks

    def bootstrap_indicies(self, N, T, q):
        return self.stationary_block_bootstrap(T, N, q)



