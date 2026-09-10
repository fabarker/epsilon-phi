import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd

class MGARCH_DCC(tf.keras.Model):
    _learning_rate = 1e-2

    """

    Tensorflow/Keras implementation of multivariate GARCH under dynamic conditional correlation (DCC) specification.

    Further reading:
        - Engle, Robert. "Dynamic conditional correlation: A simple class of multivariate generalized autoregressive conditional heteroskedasticity models."
        - Bollerslev, Tim. "Modeling the Coherence in Short-Run Nominal Exchange Rates: A Multi-variate Generalized ARCH Model."
        - Lütkepohl, Helmut. "New introduction to multiple time series analysis."

    """

    def __init__(self, y):
        super().__init__()
        self._y = y.copy()
        self._setup()

    @property
    def n_dims(self):
        return self._n_dims
    @property
    def Y(self):
        return np.float32(self._y)
    @property
    def mu(self):
        return self._mu
    @property
    def sigma_0(self):
        return self._sigma_0
    @property
    def alpha_0(self):
        return self._alpha_0
    @property
    def beta(self):
        return self._beta
    @property
    def alpha(self):
        return self._alpha
    @property
    def L_0(self):
        return self._L_0
    @property
    def A(self):
        return self._A
    @property
    def B(self):
        return self._B

    @staticmethod
    def cov_to_corr(S):

        """

        Transforms covariance matrix to a correlation matrix via matrix operations
        Args:
        S: Symmetric, positive semidefinite covariance matrix (tf.Tensor)

        """

        D = tf.linalg.LinearOperatorDiag(1 / (tf.linalg.diag_part(S) ** 0.5))
        return D @ S @ D

    def _setup(self):

        np.random.seed(123)
        tf.random.set_seed(123)

        # Create TensorFlow variables ofr model parameters
        self._n_dims = self.Y.shape[1]
        self._mu = tf.Variable(np.mean(self.Y, 0))
        self._sigma_0 = tf.Variable(np.std(self.Y, 0))
        self._alpha_0 = tf.Variable(np.std(self.Y, 0))
        self._A = tf.Variable(tf.zeros(shape=(1,)) + 0.9)
        self._B = tf.Variable(tf.zeros(shape=(1,)) + 0.05)
        self._alpha = tf.Variable(tf.zeros(shape=(self.n_dims,)) + 0.25)
        self._beta = tf.Variable(tf.zeros(shape=(self.n_dims,)) + 0.25)
        self._L_0 = tf.Variable(np.float32(np.linalg.cholesky(np.corrcoef(self.Y.T))))

        # set optimizer
        self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=MGARCH_DCC._learning_rate))


    def fit(self, **kwargs):
        super(MGARCH_DCC, self).fit(self.Y,
                                    self.Y,
                                    batch_size=len(self.Y),
                                    shuffle=False,
                                    epochs=300,
                                    verbose=False)

    def get_conditional_correlations(self):
        pass

    def get_conditional_volatilities(self):
        pass

    def get_conditional_covariances(self):
        pass

    def call(self, inputs, training=None, mask=None):
        return self.get_conditional_distributions(inputs)

    def get_log_probabilities(self, y):

        """
        Calculate log probabilities for a given matrix of time-series observations
        Args:
        y: NxM numpy.array of N observations of M correlated time-series
        """

        return self.get_conditional_distributions(y).log_prob(y)

    @tf.function
    def get_conditional_distributions(self, y):

        """
        Calculate conditional distributions for given observations
        Args:
        y: NxM numpy.array of N observations of M correlated time-series
        """

        T= tf.shape(y)[0]

        # create containers for looping
        mus = tf.TensorArray(tf.float32, size=T)  # observation mean container
        Sigmas = tf.TensorArray(tf.float32, size=T)  # observation covariance container
        sigmas = tf.TensorArray(tf.float32, size=T + 1)
        us = tf.TensorArray(tf.float32, size=T + 1)
        Qs = tf.TensorArray(tf.float32, size=T + 1)

        # initialize respective values for t=0
        sigmas = sigmas.write(0, self.sigma_0)
        A_0 = tf.transpose(self.L_0) @ self.L_0
        Qs = Qs.write(0, A_0)  # set initial unnormalized correlation equal to mean matrix
        us = us.write(0, tf.zeros(shape=(self.n_dims,)))  # initial observations equal to zero

        # convenience
        alpha_0_ = self.alpha_0 ** 2  # ensure positivity
        for t in tf.range(T):
            # tm1 = 't minus 1'
            # suppress conditioning on past in notation

            # 1) calculate conditional standard deviations
            u_tm1 = us.read(t)
            sigma_tm1 = sigmas.read(t)

            sigma_t = (alpha_0_ + self.alpha * sigma_tm1 ** 2 + self.beta * u_tm1 ** 2) ** 0.5

            # 2) calculate conditional correlations
            u_tm1_standardized = u_tm1 / sigma_tm1
            Psi_tilde_tm1 = tf.reshape(u_tm1_standardized, (self.n_dims, 1)) @ tf.reshape(u_tm1_standardized,
                                                                                          (1, self.n_dims))

            Q_tm1 = Qs.read(t)
            Q_t = A_0 + self.A * (Q_tm1 - A_0) + self.B * (Psi_tilde_tm1 - A_0)
            R_t = self.cov_to_corr(Q_t)

            # 3) calculate conditional covariance
            D_t = tf.linalg.LinearOperatorDiag(sigma_t)
            Sigma_t = D_t @ R_t @ D_t

            # 4) store values for next iteration
            sigmas = sigmas.write(t + 1, sigma_t)
            us = us.write(t + 1, y[t, :] - self.mu)  # we want to model the zero-mean disturbances
            Qs = Qs.write(t + 1, Q_t)

            mus = mus.write(t, self.mu)
            Sigmas = Sigmas.write(t, Sigma_t)

        scale_tril = tf.linalg.cholesky(Sigmas.stack())
        return tfp.distributions.MultivariateNormalTriL(loc=mus.stack(), scale_tril=scale_tril)

    def train_step(self, data):

        """

        Custom training step to handle keras model.fit given that there is no input-output structure in our model
        Args:
        S: Symmetric, positive semidefinite covariance matrix (tf.Tensor)

        """

        x, y = data
        with tf.GradientTape() as tape:
            loss = -tf.math.reduce_mean(self.get_log_probabilities(y))

        trainable_vars = self.trainable_weights
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {"Current loss": loss}

if __name__ == "__main__":

    import yfinance as yf
    import numpy as np
    import matplotlib.pyplot as plt

    data = yf.download("^GDAXI ^GSPC", start="2009-09-10", end="2022-09-10", interval="1d")

    close = data["Close"]
    returns = np.log(close).diff().dropna()

    model = MGARCH_DCC(returns)
    model.fit()

    conditional_dists = model(np.float32(returns.values))
    cond_covs = conditional_dists.covariance()
    cond_pmat = [np.array(model.cov_to_corr(x.numpy())) for x in cond_covs]
    cond_corr = [x[1][0] for x in cond_pmat]
    df = pd.DataFrame(cond_corr, index=returns.index)
    df.to_clipboard()

