from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.Asset import PrivateAsset
import pandas as pd
import numpy as np

from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency


# VintageYear(commitment: float, initial_value, start_age, calls, distributions, returns, shocks)

class Vintage(object):

    _returns = {}

    def __init__(self,
                 strategy_type,
                 schema,
                 commitment_size,
                 fund_age=0,
                 realized_nav=0.0,
                 cumltv_realized_contributions=None,
                 cumltv_realized_distributions=None,
                 cash_flow_frequency=1):

        self._schema = schema
        self._commitments_size = commitment_size
        assert fund_age >= 0, 'Error - commitment year must be neutral-positive'

        self._commitment_year = fund_age
        self._strategy_type = strategy_type
        self._cash_flow_frequency = cash_flow_frequency
        self._fund_age = fund_age

        self._realized_nav = realized_nav
        self._realized_contributions = cumltv_realized_contributions
        self._realized_distributions = cumltv_realized_distributions

        self._BOY_NAV = np.full((21,), np.nan)
        self._EOY_NAV = np.full((21,), np.nan)
        self._BOY_NAV[0] = self._realized_nav

        self.set_default_properties()
        self._estimate_cash_flows()


    @property
    def T(self):
        return self.distribution_assumptions.shape[0]

    def reset_vintage(self):
        self._commitment_year = 0
        self._commitments_size = 0
        self._realized_nav = 0
        self._BOY_NAV = np.full((21,), np.nan)
        self._EOY_NAV = np.full((21,), np.nan)
        self._BOY_NAV[0] = self._realized_nav

    def _load_distribution_assumptions(self):
        distributions = GlobalDataSource().get_private_asset_distribution_assumptions(self._strategy_type)
        idxs = np.arange(1/self._cash_flow_frequency, distributions.index.max() + 1/self._cash_flow_frequency, 1/self._cash_flow_frequency)
        self.distribution_assumptions = distributions.reindex(idxs).interpolate().fillna(0)

    def _load_capital_call_assumptions(self):
        capital_calls = GlobalDataSource().get_private_asset_capital_call_assumptions(self._strategy_type)
        idxs = np.arange(1/self._cash_flow_frequency, capital_calls.index.max() + 1/self._cash_flow_frequency, 1/self._cash_flow_frequency)
        self.capital_call_assumptions = capital_calls.reindex(idxs).bfill() / self._cash_flow_frequency

    def _load_strategy_returns(self):

        if self.type not in Vintage._returns:
            rtns, _ = self.simulate_returns()
            #rtns = [-1 + np.power((1 + 0.11480), 1 / self._cash_flow_frequency)] * self.T
            Vintage._returns[self.type] = pd.DataFrame(np.mean(rtns, axis=1), index=self.capital_call_assumptions.index, columns=[self.type])
        self.return_df = Vintage._returns[self.type]


    def simulate_returns(self):
        return self._schema.get_asset_from_name(
            self._strategy_type.asset_name
        ).simulate(frequency=Frequency.YEARLY)

    def set_default_properties(self):
        self._load_distribution_assumptions()
        self._load_capital_call_assumptions()
        self._load_strategy_returns()

    @property
    def type(self):
        return self._strategy_type

    @property
    def capital_calls(self):
        return self.capital_call_assumptions.values * self._commitments_size

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

    def get_return(self, year):
        return self.return_df.values[year-1].item()

    def get_MV_after_growth(self, year):
        return self.get_BOY_NAV(year) * (1 + self.get_return(year))

    def get_BOY_NAV(self, year):

        if year <= 0:
            return self._realized_nav
        elif np.isnan(self._BOY_NAV[year]):
            self._BOY_NAV[year] = self.get_EOY_NAV(year-1) + self.get_capital_call(year-1)
            return self._BOY_NAV[year]
        else:
            return self._BOY_NAV[year]

    def get_EOY_NAV(self, year):
        return self.get_MV_after_growth(year) - self.get_distribution(year)

    # This is the distribution flow at end of year
    def get_distribution(self, year):
        if year + self._fund_age - 1 < 0 or year + self._fund_age - 1 >= self.T:
            return 0
        else:
            return self.distribution_assumptions.values[year + self._fund_age - 1] * self.get_MV_after_growth(year)

    def get_capital_call(self, year):
        if year + self._fund_age >= len(self.capital_calls):
            return 0
        else:
            return self.capital_calls[year + self._fund_age]

    def _estimate_cash_flows(self):

        VALS = np.zeros((self.T+1, 6))
        for t in range(0, self.T+1):

            VALS[t, 0] = self.get_BOY_NAV(t)
            VALS[t, 1] = self.get_MV_after_growth(t)
            VALS[t, 2] = self.get_capital_call(t)
            VALS[t, 3] = self.get_distribution(t)
            VALS[t, 4] = self.get_EOY_NAV(t)
            VALS[t, 5] = self.get_return(t)
        self._df = pd.DataFrame(VALS, columns=['BOY_NAV', "MV_AFTER_GROWTH", 'CALLS', 'DISTR', 'EOY_NAV', "RETURN"], index=range(0, self.T+1))

    def  _calc_irr(self, start_year=2000):
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


    from epsilonPhi.core.schema.Schema import ContextCreator
    from epsilonPhi.core.portfolio.SAAPortfolio import SAAPortfolio

    schema = ContextCreator(currency='USD',
                            start_date='30-Nov-1983',
                            end_date='31-Dec-2022').create_context()


    vy = Vintage(strategy_type=PrivateAsset.BUYOUT,
                 schema=schema,
                 commitment_size=100,
                 fund_age=0,
                 realized_nav=0,
                 cash_flow_frequency=1)

    self= vy




