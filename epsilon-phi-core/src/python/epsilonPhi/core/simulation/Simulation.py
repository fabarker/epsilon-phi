import numpy as np


class BrownianBridge:


    @staticmethod
    def simulate(S0, ST, M, T, sigma=1.0):
        """
        Simulate M Brownian Bridge paths from S0 to ST over T time steps.

        Parameters:
        - S0: starting value
        - ST: ending value
        - M: number of simulation paths
        - T: number of time steps
        - sigma: volatility (default 1.0 for standard Brownian Bridge)

        Returns:
        - ndarray of shape (T + 1, M): paths from t=0 to t=T
        """
        dt = 1.0 / T
        time_grid = np.linspace(0, 1, T + 1)

        # Standard Brownian motion
        W = np.cumsum(np.random.normal(0, np.sqrt(dt), size=(M, T)), axis=1)
        W = np.hstack((np.zeros((M, 1)), W))  # Ensure W(0) = 0

        # Construct the Brownian Bridge
        bridge = S0 + (ST - S0) * time_grid + \
                 sigma * (W - np.outer((time_grid), W[:, -1]).T)

        return bridge.T  # Shape: (T+1, M)

class MixtureGBM(object):

    @staticmethod
    def simulate(r, sigma, p, M, T, S0):

        d = r.shape[0]
        z = np.random.choice(d, size=M, replace=True, p=p)

        S = []
        for idxD in range(d):
            mask = z == idxD

            mu = r[idxD]
            gbm = GBM.simulate(
                mu, sigma, M, T, S0
            )

            if len(S) == 0:
                S = gbm[mask, :]
            else:
                S = np.vstack((S, gbm[mask, :]))
        return S.T


class GBM(object):

    @staticmethod
    def simulate(mu, sigma, M, T, S0):

        """
        Simulate M paths of Geometric Brownian Motion (GBM) over T time steps.

        Parameters:
        - mu: expected return (drift)
        - sigma: volatility
        - M: number of simulation paths
        - T: number of time steps
        - S0: initial stock price

        Returns:
        - S: ndarray of shape (M, T) containing simulated paths
        """

        dt = 1/T
        S = np.full((M, T), np.nan)
        S[:, 0] = S0

        eps = np.random.normal(0, 1, (M, T))
        for i in range(1, T):
            drift = (mu - 0.5 * np.power(sigma, 2)) * dt
            shock = sigma * np.sqrt(dt) * eps[:, i]
            S[:, i] = S[:, i-1] * np.exp(drift + shock)

        return S.T