import os, sys
import numpy as np

CC = np.array([0.18, 0.18, 0.08, 0.06, 0.05, 0.04, 0.4])
CC = CC / CC.sum()

D = np.array([])
class vintage(object):

    _distributions = None
    _capital_calls = None
    def __init__(self, strategy_type, commitment_size, commitment_year):

        _commitments_size = commitment_size

        assert commitment_year > 0, 'Error - commitment year must be positive'
        _commitment_year = commitment_year
        _strategy_type = strategy_type
        self.set_default_properties()
        self._estimate_cash_flows()

    def set_default_properties(self):
        self._annualized_return = 0.09

    @property
    def capital_calls(self):
        return self._capital_calls
    @property
    def distributions(self):
        return self._distributions

    @property
    def net_flows(self):
        return self.distributions - self.capital_calls
    @property
    def cumulative_distributions(self):
        return np.cumsum(self.distributions)
    @property
    def cumulative_capital_calls(self):
        return np.cumsum(self.capital_calls)
    @property
    def cumulative_net_flows(self):
        return np.cumsum(self.net_flows)

    def estimate_cash_flows_PME(self):
        pass

    def _estimate_cash_flows(self):

        # Project the flows forward until the NAV goes to 0
        ctr = 0
        NAV_t = np.inf
        NAV_T = np.inf
        while NAV_t > 0:
            pass



