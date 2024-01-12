import numpy as np
import pandas as pd
import warnings
from arch import arch_model
from sklearn import covariance
from epsilonPhi.core.simulation.Bootstrap import TimeSeriesBootstrapper
from epsilonPhi.core.utils.FrameUtils import FrameUtils
from typing import Tuple
import math

class RiskFunctions:

    @staticmethod
    def eig(matrix):
        return np.linalg.eigh(matrix)

    @staticmethod
    def eigenvalues(matrix):
        w, _ = RiskFunctions.eig(matrix)
        return w

    @staticmethod
    def eigenvectors(matrix):
        _, v = RiskFunctions.eig(matrix)

    @staticmethod
    def _preprocess_data(series):
        return pd.DataFrame(series).dropna(how='any', axis=1)

    @staticmethod
    def _prepare_matrix(series, _cls):

        if isinstance(_cls, pd.DataFrame):
           return pd.DataFrame(series, index=_cls.columns, columns=_cls.columns)
        elif isinstance(_cls, np.ndarray):
           return np.asarray(series)
        else:
            raise ValueError('Error - class type {} not supported'.format(_cls))

    @staticmethod
    def is_positive_semidefinite(matrix):

        """
        Helper function to check if a given matrix is positive semidefinite.
        Any method that requires inverting the covariance matrix will struggle
        with a non-positive semidefinite matrix

        :param matrix: (covariance) matrix to test
        :type matrix: np.ndarray, pd.DataFrame
        :return: whether matrix is positive semidefinite
        :rtype: bool
        """

        try:
            # Significantly more efficient than checking eigenvalues (stackoverflow.com/questions/16266720)
            np.linalg.cholesky(matrix + 1e-16 * np.eye(len(matrix)))
            return True
        except np.linalg.LinAlgError:
            return False

    @staticmethod
    def fix_nonpositive_semidefinite(matrix, fix_method="spectral"):

        """
        Check if a covariance matrix is positive semidefinite, and if not, fix it
        with the chosen method.

        The ``spectral`` method sets negative eigenvalues to zero then rebuilds the matrix,
        while the ``diag`` method adds a small positive value to the diagonal.

        :param matrix: raw covariance matrix (may not be PSD)
        :type matrix: pd.DataFrame
        :param fix_method: {"spectral", "diag"}, defaults to "spectral"
        :type fix_method: str, optional
        :raises NotImplementedError: if a method is passed that isn't implemented
        :return: positive semidefinite covariance matrix
        :rtype: pd.DataFrame
        """

        if RiskFunctions.is_positive_semidefinite(matrix):
            return matrix

        warnings.warn(
            "The covariance matrix is non positive semidefinite. Amending eigenvalues."
        )

        # Eigendecomposition
        q, V = np.linalg.eigh(matrix)

        if fix_method == "spectral":
            # Remove negative eigenvalues - set to zero
            q = np.where(q > 0, q, 0)
            # Reconstruct matrix
            fixed_matrix = V @ np.diag(q) @ V.T
        elif fix_method == "diag":
            min_eig = np.min(q)
            fixed_matrix = matrix - 1.1 * min_eig * np.eye(len(matrix))
        else:
            raise NotImplementedError("Method {} not implemented".format(fix_method))

        if not RiskFunctions.is_positive_semidefinite(matrix):  # pragma: no cover
            warnings.warn(
                "Could not fix matrix. Please try a different risk model.", UserWarning
            )

        # Rebuild labels if provided
        if isinstance(matrix, pd.DataFrame):
            tickers = matrix.index
            return pd.DataFrame(fixed_matrix, index=tickers, columns=tickers)
        else:
            return fixed_matrix

    @staticmethod
    def cov_to_corr(cov_matrix):

        """
        Convert a covariance matrix to a correlation matrix.

        :param cov_matrix: covariance matrix
        :type cov_matrix: pd.DataFrame
        :return: correlation matrix
        :rtype: pd.DataFrame
        """

        if not isinstance(cov_matrix, pd.DataFrame):
            warnings.warn("cov_matrix is not a dataframe", RuntimeWarning)
            cov_matrix = pd.DataFrame(cov_matrix)

        Dinv = np.diag(1 / np.sqrt(np.diag(cov_matrix)))
        corr = np.dot(Dinv, np.dot(cov_matrix, Dinv))
        return pd.DataFrame(corr, index=cov_matrix.index, columns=cov_matrix.index)

    @staticmethod
    def corr_to_cov(corr_matrix, stdevs):
        """
        Convert a correlation matrix to a covariance matrix

        :param corr_matrix: correlation matrix
        :type corr_matrix: pd.DataFrame
        :param stdevs: vector of standard deviations
        :type stdevs: array-like
        :return: covariance matrix
        :rtype: pd.DataFrame
        """
        if not isinstance(corr_matrix, pd.DataFrame):
            warnings.warn("corr_matrix is not a dataframe", RuntimeWarning)
            corr_matrix = pd.DataFrame(corr_matrix)

        return corr_matrix * np.outer(stdevs, stdevs)

    @staticmethod
    def autoencoder_denoised(series,
                             layer_dims=[64, 32, 4],
                             learn_rate=0.001,
                             epochs=2,
                             batch_size=10,
                             shuffle_batches=True,
                             ):

        idxs = TimeSeriesBootstrapper.stationary_block_bootstrap(252, 20000, 1/126)
        series_pd = pd.DataFrame(series).rolling(window=252).apply(lambda x: RiskFunctions.sample_covariance(x))
        _training_covs = RiskFunctions.sample_covariance(series)


    @staticmethod
    def sample_covariance(series):
        return RiskFunctions.empirical_covariance(series)

    @staticmethod
    def empirical_covariance(series,
                             _fix_negative_eigenvalues=False):

        cov = pd.DataFrame(series).cov()
        if _fix_negative_eigenvalues:
            cov = RiskFunctions.fix_nonpositive_semidefinite(cov)
        return RiskFunctions._prepare_matrix(cov, series)

    @staticmethod
    def exponentially_weighted_covariance(series, alpha=0.94):

        weights = (1-alpha) ** np.arange(len(series))[::-1]
        normalized = (series-series.mean()).fillna(0).to_numpy()
        cov = (weights * normalized.T) @ normalized / weights.sum()
        return RiskFunctions._prepare_matrix(cov, series)

    @staticmethod
    def empirical_yang_zhang(series):

        cov = RiskFunctions.empirical_covariance(series)
        sig = RiskFunctions.yang_zhang(series)
        cov[np.diag_indices_from(cov)] = sig
        return RiskFunctions._prepare_matrix(cov, series)

    @staticmethod
    def exponentially_weighted_yang_zhang(prices: pd.DataFrame, alpha=0.94, close_col='ps'):


        returns = prices.iloc[:, prices.columns.get_level_values('field') == 'PS'].pct_change()

        returns = prices.iloc[:, prices.columns.get_level_values('field') == 'PS'].pct_change()
        cov = RiskFunctions.exponentially_weighted_covariance(returns, alpha)

        # Update the diagonals to reflect those from the yang zhang estimator
        cov.values[np.diag_indices_from(cov)] = np.power(RiskFunctions.yang_zhang(prices), 2).flatten()
        return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)

    @staticmethod
    def garch_ccc(series, p, q):

        """
        Estimate volatility using ARCH (Autoregressive Conditional Heteroskedasticity) model.
        By default, this function uses the GARCH(1,1) model, but it can be adjusted for different ARCH type models.

        :param series: DataFrame with at least one column containing returns
        :param returns_column: The name of the column containing returns
        :param vol_model: Type of volatility model to use ('ARCH', 'GARCH', etc.)
        :param p: Order of the autoregressive component
        :param q: Order of the moving average component
        :return: Fitted ARCH model
        """

        def _univariate_garch(series, p, q):
            model = arch_model(series, mean='Constant', vol='GARCH', p=p, o=0, q=q).fit(update_freq=0, disp='off')
            return model.conditional_volatility[-1], model.resid / model.conditional_volatility

        N = np.shape(series)
        for col in series.columns:
            sig, std_res = _univariate_garch(series.get(col), p=p, q=q)

        # calculate the constant conditional correlation matrix (CCC) R:
        R = std_res.transpose().dot(std_res).div(len(std_res))
        return RiskFunctions.corr_to_cov(R, sigs)

    @staticmethod
    def bayesian_covariance(series):
        pass

    @staticmethod
    def kalman_covariance(series):
        pass

    @staticmethod
    def particle_filter(series):
        pass

    @staticmethod
    def robust_covariance(series, random_state=None):
        cov = RiskFunctions.min_covariance_determinant_matrix(series, random_state)
        return RiskFunctions._prepare_matrix(cov, series)

    @staticmethod
    def principle_components(series: pd.DataFrame) -> pd.DataFrame():

        """
        Calculate the covariance matrix after dropping
        eigenvectors outside of Marchecko-Pasteur limits

        :return:  covariance matrix
        :rtype: pd.DataFrame
        """

        matrix = RiskFunctions.sample_covariance(series)
        q, V = np.linalg.eigh(matrix)

        # Drop eigenvalues outside Markchecko-Pasteur
        n, p = series.shape
        q = p / n
        lambda_plus = (1 + np.sqrt(q)) ** 2
        lambda_minus = (1 - np.sqrt(q)) ** 2

        locs = (q > lambda_minus) & (q < lambda_plus)
        cov = V[:, locs] @ np.diag(q[locs]) @ V[:, locs].T
        return RiskFunctions._prepare_matrix(cov, series)

    @staticmethod
    def oracle_approximating_shrinkage(series):

        """
        Calculate the Oracle Approximating Shrinkage estimate

        :return: shrunk sample covariance matrix
        :rtype: np.ndarray
        """

        cov = covariance.oas(series)
        return RiskFunctions._prepare_matrix(cov, series)

    @staticmethod
    def min_covariance_determinant_matrix(series: pd.DataFrame,
                                          random_state=None):



        X = series.dropna()
        raw_cov_array = covariance.fast_mcd(X.values, random_state=random_state)[1]
        cov = pd.DataFrame(raw_cov_array, index=X.columns, columns=X.columns)
        #return fix_nonpositive_semidefinite(cov, kwargs.get("fix_method", "spectral"))
        return RiskFunctions._prepare_matrix(cov, series)

    @staticmethod
    def shrink_covariance_identity(series, delta=0.5):

        """
        Shrink a sample covariance matrix to the identity matrix (scaled by the average
        sample variance). This method does not estimate an optimal shrinkage parameter,
        it requires manual input.

        :param delta: shrinkage parameter, defaults to 0.2.
        :type delta: float, optional
        :return: shrunk sample covariance matrix
        :rtype: np.ndarray
         """

        matrix = RiskFunctions.sample_covariance(series)
        N = matrix.shape[1]

        # Shrinkage target
        mu = np.trace(matrix) / N
        F = np.identity(N) * mu

        # Shrinkage
        shrunk_cov = delta * F + (1 - delta) * matrix
        return RiskFunctions._prepare_matrix(shrunk_cov, series)
        #return self._format_and_annualize(shrunk_cov)

    @staticmethod
    def shrink_covariance_ledoit_wolf_zero_correlation_equal_variance(series):

        """
        Ledoit - Wolf is a particular form of shrinkage, where the shrinkage coefficient is computed using O.Ledoit and M.Wolf’s
        formula as described in “A Well - Conditioned Estimator for Large - Dimensional Covariance Matrices”,
        Ledoit and Wolf, Journal of Multivariate Analysis, Volume 88, Issue 2, February 2004, pages 365-411.

        This estimator from sklearn use's shrinkage to zero correlation equal variance target and
        implies no systematic risk and equal total risk of all stocks.

        """

        return covariance.ledoit_wolf(series)

    @staticmethod
    def shrink_covariance_ledoit_wolf_single_factor_model(series):

        """
        Helper method to calculate the Ledoit-Wolf shrinkage estimate
        with the Sharpe single-factor matrix as the shrinkage target.
        See Ledoit and Wolf (2001).

        :return: shrunk sample covariance matrix, shrinkage constant
        :rtype: np.ndarray, float
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
    def shrink_covariance_ledoit_wolf_constant_correlation(series):

        """
        Helper method to calculate the Ledoit-Wolf shrinkage estimate
        with the constant correlation matrix as the shrinkage target.
        See Ledoit and Wolf (2003)

        :return: shrunk sample covariance matrix, shrinkage constant
        :rtype: np.ndarray, float
        """

        X = pd.DataFrame(series).dropna(how="all")
        X = np.nan_to_num(X.values)
        t, n = np.shape(X)

        S = RiskFunctions.sample_covariance(series)

        # Constant correlation target
        var = np.diag(S).reshape(-1, 1)
        std = np.sqrt(var)

        _var = np.tile(var, (n,))
        _std = np.tile(std, (n,))

        r_bar = (np.sum(S / (_std * _std.T)) - n) / (n * (n - 1))
        F = r_bar * (_std * _std.T)
        F[np.eye(n) == 1] = var.reshape(-1)

        # Estimate pi
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
    def yang_zhang(prices):

        """
        Estimate the Yang-Zhang volatility of a price series.
        The formula for Yang-Zhang volatility is:
        sigma_YZ^2 = k * (sigma_O^2 + sigma_C^2 + sigma_RS^2)
        where k = 0.34 / (1.34 + (n+1)/(n-1)),
        sigma_O is the overnight volatility,
        sigma_C is the close-to-close volatility,
        and sigma_RS is the Rogers-Satchell volatility.

        :param price_data: DataFrame with columns ['Open', 'High', 'Low', 'Close']
        :return: Yang-Zhang volatility
        """

        assert isinstance(prices.columns, pd.MultiIndex), \
            ('Error - dataframe must be multiindexed columns with level_0 asset name'
             'and level_1 price quote type')

        assert np.all(prices.fillna(0) >= 0), \
            'Error - cannot estimate volatility with negative price series '

        n = len(prices)
        assets = list(set(prices.columns.get_level_values(0)))

        PH = FrameUtils.select_subset_level(prices, level_name='field', values='PH', drop_nans=False).get(assets)
        PS = FrameUtils.select_subset_level(prices, level_name='field', values='PS', drop_nans=False).get(assets)
        PO = FrameUtils.select_subset_level(prices, level_name='field', values='PO', drop_nans=False).get(assets)
        PL = FrameUtils.select_subset_level(prices, level_name='field', values='PL', drop_nans=False).get(assets)

        log_hl = np.log(PH.values / PL).values
        log_co = np.log(PS.values / PO).values
        log_oc = np.log(PO.values / PS.shift(1)).values
        log_oc = log_oc[1:]  # Remove NaN

        # Calculate components of the volatility
        sigma_o = np.nanstd(log_oc, ddof=1, axis=0)
        sigma_c = np.nanstd(log_co, ddof=1, axis=0)
        sigma_rs = np.sqrt((1 / n) * np.nansum((log_hl * (log_co - 0.5 * log_hl)) ** 2))

        # Calculate k
        k = 0.34 / (1.34 + (n + 1) / (n - 1))

        # Calculate Yang-Zhang volatility
        return pd.DataFrame(np.sqrt(k * (sigma_o ** 2 + sigma_c ** 2 + sigma_rs ** 2)), index=assets)


if __name__ == "__main__":

    from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource

    gds = GlobalDataSource()
    prices = gds.get_dataframe_from_tickers(['CAFCS00','CTTCS00','CCFCS00','CYMCS00','NAYCS00','NCJCS00','NPSCS00','NVICS00'], cols=['PH','PS','PL','PO'])

    sig = RiskFunctions.exponentially_weighted_yang_zhang(prices)
    series = prices.pct_change().dropna(how='any')







