import numpy as np
import pandas as pd


class AbstractBootstrapper(object):
    _state = 5489
    _R = np.random.RandomState(5489)

    def __init__(self):
        pass

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



