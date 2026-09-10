import numpy as np
import pandas as pd
from typing import Optional, Union

class Geltner:

    def __init__(self, returns: Optional[Union[pd.Series, pd.DataFrame]]):
        self._returns = returns.dropna()

    @staticmethod
    def unsmooth_returns(
            returns,
            target_rho: float = 0.2):

        # to numpy array
        returns = np.array(returns)

        # clean nans
        returns = pd.Series(returns[~np.isnan(returns)])

        # current auto-correlation
        current_rho = returns.autocorr(lag=1)

        if not (-1 < current_rho < 1):
            raise ValueError(f"Invalid observed autocorrelation: {current_rho:.4f}")
        if not (-1 < target_rho < 1):
            raise ValueError("Target autocorrelation must be between -1 and 1 (exclusive)")

        # Adjustment factor
        alpha = (current_rho - target_rho) / (1 - target_rho)

        adjusted = (returns - alpha * returns.shift(1)) / (1 - alpha)
        return adjusted.dropna()

    def unsmooth(self, target_rho: float = 0.2) -> pd.Series:

        """
        Unsmooth a return series to reach a specified target autocorrelation.

        Parameters:
        - returns: pd.Series of smoothed returns
        - target_rho: desired autocorrelation (e.g. 0 for full unsmoothing, or any valid rho in (-1, 1))

        Returns:
        - pd.Series of adjusted (less smoothed) returns with autocorr ≈ target_rho
        """

        returns = self._returns
        current_rho = returns.autocorr(lag=1)

        if not (-1 < current_rho < 1):
            raise ValueError(f"Invalid observed autocorrelation: {current_rho:.4f}")
        if not (-1 < target_rho < 1):
            raise ValueError("Target autocorrelation must be between -1 and 1 (exclusive)")

        # Adjustment factor
        alpha = (current_rho - target_rho) / (1 - target_rho)

        adjusted = (returns - alpha * returns.shift(1)) / (1 - alpha)
        return adjusted.dropna()