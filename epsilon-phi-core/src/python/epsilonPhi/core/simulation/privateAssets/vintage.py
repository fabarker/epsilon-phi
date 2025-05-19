from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.Asset import PrivateAsset
import pandas as pd
import numpy as np


class vintage(object):

    def __init__(self,
                 strategy_type,
                 commitment_size,
                 fund_age,
                 realized_NAV=None,
                 cumltv_realized_contributions=None,
                 cumltv_realized_distributions=None,
                 cash_flow_frequency=1):

        self._commitments_size = commitment_size
        assert fund_age >= 0, 'Error - commitment year must be neutral-positive'

        self._commitment_year = fund_age
        self._strategy_type = strategy_type
        self._cash_flow_frequency = cash_flow_frequency

        self._REALIZED_NAV = realized_NAV
        self._REALIZED_CONTRIBUTIONS = cumltv_realized_contributions
        self._REALIZED_DISTRIBUTIONS = cumltv_realized_distributions

        self.set_default_properties()
        self._estimate_cash_flows()

    @property
    def T(self):
        return self.distribution_assumptions.shape[0]

    def _load_distribution_assumptions(self):
        distributions = GlobalDataSource().get_private_asset_distribution_assumptions(self._strategy_type)
        idxs = np.arange(1/self._cash_flow_frequency, distributions.index.max() + 1/self._cash_flow_frequency, 1/self._cash_flow_frequency)
        self.distribution_assumptions = distributions.reindex(idxs).interpolate().fillna(0)

    def _load_capital_call_assumptions(self):
        capital_calls = GlobalDataSource().get_private_asset_capital_call_assumptions(self._strategy_type)
        idxs = np.arange(1/self._cash_flow_frequency, capital_calls.index.max() + 1/self._cash_flow_frequency, 1/self._cash_flow_frequency)
        self.capital_call_assumptions = capital_calls.reindex(idxs).bfill() / self._cash_flow_frequency

    def _load_strategy_returns(self):
        rtns = [-1 + np.power((1 + 0.11480), 1 / self._cash_flow_frequency)] * self.T
        self.return_df = pd.DataFrame(rtns, index=self.capital_call_assumptions.index, columns=[self._strategy_type])

    def set_default_properties(self):
        self._load_distribution_assumptions()
        self._load_capital_call_assumptions()
        self._load_strategy_returns()

    @property
    def capital_calls(self):
        return self._df.get('CALLS')

    @property
    def distributions(self):
        return self._df.get('DISTR')

    @property
    def net_flows(self):
        return self.distributions - self.capital_calls

    @property
    def cumulative_distributions(self):
        return self._df.get('DISTR').cumsum()

    @property
    def cumulative_capital_calls(self):
        return self._df.get('CALLS').cumsum()

    @property
    def cumulative_net_flows(self):
        return np.cumsum(self.net_flows)

    def estimate_cash_flows_PME(self):
        pass

    def _estimate_cash_flows(self):

        VALS = np.zeros((self.T+1, 4))
        for t in range(1, self.T):

            VALS[t, 0] = VALS[t-1, -1]
            EOY_NAV_t = VALS[t, 0] * (1 + self.return_df.values[t])
            VALS[t, 1] = self.capital_call_assumptions.values[t] * self._commitments_size
            VALS[t, 2] = EOY_NAV_t * self.distribution_assumptions.values[t]
            VALS[t, 3] = EOY_NAV_t * (1-self.distribution_assumptions.values[t]) + self.capital_call_assumptions.values[t] * self._commitments_size
        self._df = pd.DataFrame(VALS, columns=['BOY_NAV', 'CALLS', 'DISTR', 'EOY_NAV'], index=range(0, self.T+1))

    def _calc_irr(self, start_year=2000):
        """
        Calculate the IRR for a private equity fund vintage from a dataframe.

        Parameters:
        - df: A pandas DataFrame with columns ['BOY_NAV', 'CALLS', 'DISTR', 'EOY_NAV']
        - start_year: The base year corresponding to index 0

        Returns:
        - Annualized IRR as a float
        """
        from datetime import datetime, timedelta

        df = self._df.copy()

        start_date = datetime(start_year, 1, 1)
        dates = [start_date + timedelta(days=365 * i) for i in range(len(df))]

        df = df.copy()
        df["cash_flow"] = df["DISTR"] - df["CALLS"]
        df.loc[len(df) - 1, "cash_flow"] += df.loc[len(df) - 1, "EOY_NAV"]

        year_fractions = [(d - dates[0]).days / 365.25 for d in dates]
        cash_flows = df["cash_flow"].tolist()

        def xirr(cash_flows, times, guess=0.1, tol=1e-6, max_iter=100):
            r = guess
            for _ in range(max_iter):
                f = sum([cf / (1 + r) ** t for cf, t in zip(cash_flows, times)])
                df = sum([-t * cf / (1 + r) ** (t + 1) for cf, t in zip(cash_flows, times)])
                r_new = r - f / df
                if abs(r - r_new) < tol:
                    return r_new
                r = r_new
            raise RuntimeError("XIRR calculation did not converge")

        return xirr(cash_flows, year_fractions)

if __name__ == "__main__":

    vy = vintage(strategy_type=PrivateAsset.BUYOUT,
                 commitment_size=100,
                 fund_age=0,
                 cash_flow_frequency=1)
    vy._calc_irr()




