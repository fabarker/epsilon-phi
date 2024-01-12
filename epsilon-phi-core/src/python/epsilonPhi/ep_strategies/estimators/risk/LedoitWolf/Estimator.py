import numpy as np
import pandas as pd
from sklearn import covariance

class Covariance(object):
    pass

    @staticmethod
    def diagonal_sample_variance_shrinkage(series, delta=None):
        from epsilonPhi.ep_strategies.estimators.risk.RiskModel import RiskFunctions

        """
        
        A well-conditioned estimator for large-dimensional covariance matrices

        Shrink a sample covariance matrix to the identity matrix (scaled by the
        sample variances)

        :param delta: shrinkage parameter, if not supplied solve for minimum estimation error
        :type delta: float, optional
        :return: shrunk sample covariance matrix
        :rtype: np.ndarray
        
        covDiag equivalent
        """

        matrix = RiskFunctions.sample_covariance(series)
        F = np.diag(np.diag(matrix))

        if delta is None:

            N = series.shape[0] - 1

            series_ = series - series.mean(axis=0)
            sq_ = np.power(series_, 2)
            sample_sq_ = (1/N) * sq_.T @ sq_

            pi = sample_sq_ - np.power(matrix, 2)
            pi_ = sum(np.sum(pi))

            g_ = np.linalg.norm(matrix - F, ord='fro') ** 2
            rho = np.trace(pi)

            kappa_ = (pi_ - rho) / g_
            delta = max(0, min(1, kappa_ / N))

        # Shrinkage
        shrunk_cov = delta * F + (1 - delta) * matrix
        return RiskFunctions._prepare_matrix(shrunk_cov, series)

    @staticmethod
    def mean_variance_diagonal_shrinkage(series, delta=None):
        from epsilonPhi.ep_strategies.estimators.risk.RiskModel import RiskFunctions

        """
        Shrink a sample covariance matrix to the identity matrix (scaled by the average
        sample variance). This method does not estimate an optimal shrinkage parameter,
        it requires manual input.

        :param delta: shrinkage parameter, defaults to 0.2.
        :type delta: float, optional
        :return: shrunk sample covariance matrix
        :rtype: np.ndarray
        
        equivalent to cov1Para
    
        """

        if delta is None:
            return covariance.ledoit_wolf(series)
        else:
            matrix = RiskFunctions.sample_covariance(series)
            N = series.shape[1]

            # Shrinkage target
            mu = np.trace(matrix) / N
            F = np.identity(N) * mu

            # Shrinkage
            shrunk_cov = delta * F + (1 - delta) * matrix
            return RiskFunctions._prepare_matrix(shrunk_cov, series), delta

    @staticmethod
    def single_factor_shrinkage(series, delta=None):
        from epsilonPhi.ep_strategies.estimators.risk.RiskModel import RiskFunctions

        """
        Helper method to calculate the Ledoit-Wolf shrinkage estimate
        with the Sharpe single-factor matrix as the shrinkage target.
        See Ledoit and Wolf (2001).

        :return: shrunk sample covariance matrix, shrinkage constant
        :rtype: np.ndarray, float
        
        covMarket equivalent
        """

        X = pd.DataFrame(series).values

        # De-mean returns
        t, n = np.shape(X)
        Xm = X - X.mean(axis=0)
        xmkt = Xm.mean(axis=1).reshape(t, 1)

        # compute sample covariance matrix
        sample = np.cov(np.append(Xm, xmkt, axis=1), rowvar=False) * (t - 1) / t
        betas = sample[0:n, n].reshape(n, 1)
        varmkt = sample[n, n]
        sample = sample[:n, :n]
        F = np.dot(betas, betas.T) / varmkt
        F[np.eye(n) == 1] = np.diag(sample)

        if delta is None:

            # compute shrinkage parameters
            c = np.linalg.norm(sample - F, "fro") ** 2
            y = Xm ** 2
            p = 1 / t * np.sum(np.dot(y.T, y)) - np.sum(sample ** 2)

            # r is divided into diagonal
            # and off-diagonal terms, and the off-diagonal term
            # is itself divided into smaller terms
            rdiag = 1 / t * np.sum(y ** 2) - sum(np.diag(sample) ** 2)
            z = Xm * np.tile(xmkt, (n,))
            v1 = 1 / t * np.dot(y.T, z) - np.tile(betas, (n,)) * sample
            roff1 = (
                    np.sum(v1 * np.tile(betas, (n,)).T) / varmkt
                    - np.sum(np.diag(v1) * betas.T) / varmkt
            )
            v3 = 1 / t * np.dot(z.T, z) - varmkt * sample
            roff3 = (
                    np.sum(v3 * np.dot(betas, betas.T)) / varmkt ** 2
                    - np.sum(np.diag(v3).reshape(-1, 1) * betas ** 2) / varmkt ** 2
            )
            roff = 2 * roff1 - roff3
            r = rdiag + roff

            # compute shrinkage constant
            k = (p - r) / c
            delta = max(0, min(1, k / t))

        # compute the estimator
        shrunk_cov = delta * F + (1 - delta) * sample
        return RiskFunctions._prepare_matrix(shrunk_cov, series), delta

    @staticmethod
    def constant_correlation_shrinkage(series, delta=None):
        from epsilonPhi.ep_strategies.estimators.risk.RiskModel import RiskFunctions

        """
        Helper method to calculate the Ledoit-Wolf shrinkage estimate
        with the constant correlation matrix as the shrinkage target.
        See Ledoit and Wolf (2003)

        :return: shrunk sample covariance matrix, shrinkage constant
        :rtype: np.ndarray, float
        covCor equivalent
        
        The constant correlation model would not be appropriate if the assets came from
        different asset classes, such as stocks and bonds. But in such cases more general models
        for the shrinkage target are available

        """

        X = pd.DataFrame(series).dropna(how="all")
        X = np.nan_to_num(X.values)
        t, n = np.shape(X)

        S = RiskFunctions.sample_covariance(X)

        # Constant correlation target
        var = np.diag(S).reshape(-1, 1)
        std = np.sqrt(var)

        _var = np.tile(var, (n,))
        _std = np.tile(std, (n,))

        r_bar = (np.sum(S / (_std * _std.T)) - n) / (n * (n - 1))
        F = r_bar * (_std * _std.T)
        F[np.eye(n) == 1] = var.reshape(-1)

        if delta is None:
            Xm = X - X.mean(axis=0)
            y = Xm ** 2
            pi_mat = np.dot(y.T, y) / t - 2 * np.dot(Xm.T, Xm) * S / t + S ** 2
            pi_hat = np.sum(pi_mat)

            # Theta matrix, expanded term by term
            term1 = np.dot((Xm ** 3).T, Xm) / t
            help_ = np.dot(Xm.T, Xm) / t
            help_diag = np.diag(help_)
            term2 = np.tile(help_diag, (n, 1)).T * S
            term3 = help_ * _var
            term4 = _var * S
            theta_mat = term1 - term2 - term3 + term4
            theta_mat[np.eye(n) == 1] = np.zeros(n)
            rho_hat = sum(np.diag(pi_mat)) + r_bar * np.sum(
                np.dot((1 / std), std.T) * theta_mat
            )

            # Estimate gamma
            gamma_hat = np.linalg.norm(S - F, "fro") ** 2

            # Compute shrinkage constant
            kappa_hat = (pi_hat - rho_hat) / gamma_hat
            delta = max(0.0, min(1.0, kappa_hat / t))

        # Compute shrunk covariance matrix
        shrunk_cov = delta * F + (1 - delta) * S
        return RiskFunctions._prepare_matrix(shrunk_cov, series), delta

    @staticmethod
    def mean_varcov_shrinkage(series, delta=None):
        from epsilonPhi.ep_strategies.estimators.risk.RiskModel import RiskFunctions

        """

        A well-conditioned estimator for large-dimensional covariance matrices

        Shrink a sample covariance matrix to the identity matrix (scaled by the
        sample variances). This method does not estimate an optimal shrinkage parameter,
        it requires manual input.

        :param delta: shrinkage parameter, if not supplied solve for minimum estimation error
        :type delta: float, optional
        :return: shrunk sample covariance matrix
        :rtype: np.ndarray

        cov2para equivalent
        """

        matrix = RiskFunctions.sample_covariance(series)
        k = matrix.shape[0]

        mu_var = np.trace(matrix) / k
        mu_cov = np.nanmean(np.diag([np.nan] * k) + matrix)
        F = mu_var * np.eye(k) + mu_cov * (1 - np.eye(k))

        if delta is None:

            N = series.shape[0] - 1

            series_ = series - series.mean(axis=0)
            sq_ = np.power(series_, 2)
            sample_sq_ = (1 / N) * sq_.T @ sq_
            pi = sample_sq_ - np.power(matrix, 2)
            pi_ = sum(np.sum(pi))

            g_ = np.linalg.norm(matrix - F, ord='fro') ** 2
            rho = (1/k) * (sum(np.sum(sample_sq_)) - np.trace(matrix) ** 2)

            s1 = series_.sum(axis=1)
            s2 = sq_.sum(axis=1)
            tmp = np.power(s1, 2) - s2

            rho_ = np.sum(np.power(tmp, 2)) / (k * N)
            rho__ = (sum(np.sum(matrix)) - np.trace(matrix)) ** 2 / k
            rho_off = (rho_ - rho__) / (k - 1)

            # compute shrinkage intensity
            kappa_ = (pi_ - (rho + rho_off)) / g_
            delta = max(0, min(1, kappa_ / N))

        # Shrinkage
        shrunk_cov = delta * F + (1 - delta) * matrix
        return RiskFunctions._prepare_matrix(shrunk_cov, series)

    @staticmethod
    def geometric_inverse_shrinkage(series):

        """ Implements the geometric-inverse shrinkage (QIS) estimator.

            This is a nonlinear shrinkage estimator based on the Symmetrized
            Kullback-Leibler loss; it can be viewed as geometrically averaging
            linear-inverse shrinkage (LIS) with quadratic-inverse shrinkage (QIS)

        """

        # Set df dimensions
        N, p = series.shape

        # vars
        n = N - 1  # adjust effective sample size
        c = p / n  # concentration ratio

        # Cov df: sample covariance matrix
        smpl = series.cov()
        smpl = 0.5 * ( smpl + smpl.T )

        # Spectral decomposition
        L, V = np.linalg.eig(smpl)
        L = L.real.clip(min=0)
        V = V.real

        # Sort the eigenvalues and eigenvectors from smallest to largest
        V_ = V[:, L.argsort()]
        L_ = L[L.argsort()].reshape(-1, 1)

        # Compute Quadratic-Inverse Shrinkage estimator of the covariance matrix
        h = (min(c ** 2, 1 / c ** 2) ** 0.35) / p ** 0.35  # smoothing parameter

        inv_L = 1 / L_[max(1, p - n + 1) - 1:p]
        Lj = inv_L.repeat(min(p, n), axis=1)
        Lj_i = Lj - Lj.T

        # Smoothed Stein Shrinker
        theta_ = np.mean((Lj * Lj_i) / ((Lj_i * Lj_i) + (Lj * Lj * h ** 2)), axis=0)

        # Conjugate of Smoothed Stein Shrinkage
        theta__ = np.mean((Lj * (Lj * h)) / ((Lj_i * Lj_i) + (Lj * Lj * h ** 2)), axis=0)

        # Squared Amplitude
        _theta_ = theta_ ** 2 + theta__ ** 2

        if p <= n:  # case where sample covariance matrix is not singular
            delta_ = (1 - c) * inv_L + 2 * c * inv_L * theta_.reshape(-1,1)  # shrunk inverse eigenvalues (LIS)

            _delta = 1 / ((1 - c) ** 2 * inv_L + 2 * c * (1 - c) * inv_L * theta_.reshape(-1,1)\
                         + c ** 2 * inv_L * _theta_.reshape(-1,1))  # optimally shrunk eigenvalues
        else:  # case where sample covariance matrix is singular
            print('p must be <= n for the Symmetrized Kullback-Leibler divergence')
            return -1


        delta_[delta_ < min(inv_L)] = min(inv_L)
        tmp_ = np.diag((_delta / delta_).flatten() ** 0.5)
        tmp__ = V_.T.conjugate()

        # reconstruct covariance matrix
        return np.matmul(np.matmul(V_, tmp_), tmp__)

    @staticmethod
    def linear_inverse_shrinkage(series):

        # Set df dimensions
        N, p = series.shape

        # vars
        n = N - 1  # adjust effective sample size
        c = p / n  # concentration ratio

        # Cov df: sample covariance matrix
        smpl = series.cov()
        smpl = 0.5 * (smpl + smpl.T)

        # Spectral decomposition
        L, V = np.linalg.eig(smpl)
        L = L.real.clip(min=0)
        V = V.real

        # Sort the eigenvalues and eigenvectors from smallest to largest
        V_ = V[:, L.argsort()]
        L_ = L[L.argsort()].reshape(-1, 1)

        # Compute Quadratic-Inverse Shrinkage estimator of the covariance matrix
        h = (min(c ** 2, 1 / c ** 2) ** 0.35) / p ** 0.35  # smoothing parameter

        inv_L = 1 / L_[max(1, p - n + 1) - 1:p]
        Lj = inv_L.repeat(min(p, n), axis=1)
        Lj_i = Lj - Lj.T

        # Smoothed Stein Shrinker
        theta_ = np.mean((Lj * Lj_i) / ((Lj_i * Lj_i) + (Lj * Lj * h ** 2)), axis=0)

        if p <= n:  # case where sample covariance matrix is not singular
            delta_ = (1 - c) * inv_L + 2 * c * inv_L * theta_.reshape(-1,1)  # shrunk inverse eigenvalues (LIS)
        else:  # case where sample covariance matrix is singular
            print("p must be <= n for Stein's loss")
            return -1

        delta_[delta_ < min(inv_L)] = min(inv_L)
        tmp_ = np.diag(1 / delta_.flatten())
        tmp__ = V_.T.conjugate()

        return np.matmul(np.matmul(V_, tmp_), tmp__)

    @staticmethod
    def quadratic_inverse_shrinkage(series):

        # Set df dimensions
        N, p = series.shape

        # vars
        n = N - 1  # adjust effective sample size
        c = p / n  # concentration ratio

        # Cov df: sample covariance matrix
        smpl = series.cov()
        smpl = 0.5 * (smpl + smpl.T)

        # Spectral decomposition
        L, V = np.linalg.eig(smpl)
        L = L.real.clip(min=0)
        V = V.real

        # Sort the eigenvalues and eigenvectors from smallest to largest
        V_ = V[:, L.argsort()]
        L_ = L[L.argsort()].reshape(-1, 1)

        # Compute Quadratic-Inverse Shrinkage estimator of the covariance matrix
        h = (min(c ** 2, 1 / c ** 2) ** 0.35) / p ** 0.35  # smoothing parameter

        inv_L = 1 / L_[max(1, p - n + 1) - 1:p]
        Lj = inv_L.repeat(min(p, n), axis=1)
        Lj_i = Lj - Lj.T

        # Smoothed Stein Shrinker
        theta_ = np.mean((Lj * Lj_i) / ((Lj_i * Lj_i) + (Lj * Lj * h ** 2)), axis=0)

        # Conjugate of Smoothed Stein Shrinkage
        theta__ = np.mean((Lj * (Lj * h)) / ((Lj_i * Lj_i) + (Lj * Lj * h ** 2)), axis=0)

        # Squared Amplitude
        _theta_ = theta_ ** 2 + theta__ ** 2

        if p <= n:  # case where sample covariance matrix is not singular
            _delta = 1 / ((1 - c) ** 2 * inv_L + 2 * c * (1 - c) * inv_L * theta_.reshape(-1, 1)\
                          + c ** 2 * inv_L * _theta_.reshape(-1, 1))  # optimally shrunk eigenvalues
        else:  # case where sample covariance matrix is singular
            d_0 = 1 / ((c - 1) * np.mean(inv_L))  # shrinkage of null eigenvalues
            _delta = np.repeat(d_0, p - n)
            _delta = np.concatenate((_delta, 1 / (inv_L * _theta_.reshape(-1, 1))), axis=None)

        d_qis = _delta * (sum(L_) / sum(_delta))  # preserve trace
        tmp = np.diag(d_qis.flatten())
        tmp_ = V_.T.conjugate()

        # reconstruct covariance matrix
        return np.matmul(np.matmul(V_, tmp), tmp_)


if __name__ == "__main__":

    from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource

    gds = GlobalDataSource()
    prices = gds.get_dataframe_from_tickers(['CAFCS00','CTTCS00','CCFCS00','CYMCS00','NAYCS00','NCJCS00','NPSCS00','NVICS00'], cols=['PS'])

    s = prices.pct_change().dropna(how='any')
    cov = Covariance.quadratic_inverse_shrinkage(s)


