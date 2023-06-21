import os, sys
import numpy as np
import pandas as pd

class CashFlowAssumptions:

    @staticmethod
    def get_strategy_capital_call_assumptions(strategy):
        if strategy.upper() == 'BUYOUT':
            return [0.18126888, 0.18126888, 0.18126888, 0.10372608, 0.08056395, 0.06243706, 0.04833837, 0.03726083, 0.02920443, 0.02316214, 0.01812689, 0.01409869, 0.01107755, 0.00805639, 0.0060423, 0.00503525, 0.0040282, 0.00302115, 0.0020141, 0]
        elif strategy.upper() == 'SECONDARIES':
            return [0.25, 0.25, 0.112, 0.087, 0.067, 0.052, 0.041, 0.032, 0.025, 0.019]
        elif strategy.upper() == 'GROWTH':
            return [0.22, 0.22, 0.125, 0.097, 0.075, 0.059, 0.046, 0.035, 0.027, 0.021]
        elif strategy.upper() == 'VENTURE':
            return [0.18, 0.18, 0.117, 0.096, 0.078, 0.064, 0.052, 0.043, 0.035, 0.028]
        elif strategy.upper() == 'PRIVATE_CREDIT':
            return [0.35, 0.35, 0.10, 0.067, 0, 0, 0, 0, 0, 0]
        elif strategy.upper() == 'REAL_ESTATE':
            return [0.25, 0.25, 0.167, 0.111, 0.074, 0, 0, 0, 0, 0]
        elif strategy.upper() == 'INFRASTRUCTURE':
            return [0.4, 0.24, 0.144, 0.086, 0.052, 0.031, 0.019, 1.1, 0.07, 0.04]
        else:
            raise ValueError('Strategy {} not supported')

    @staticmethod
    def get_strategy_distribution_assumptions(strategy):
        if strategy.upper() == 'BUYOUT':
            return [0, 0, 0.002, 0.012, 0.036, 0.081, 0.151, 0.248, 0.368, 0.503, 0.637, 0.752, 0.820, 0.826, 0.851, 0.874, 0.894, 0.910, 0.924, 1]
        elif strategy.upper() == 'SECONDARIES':
            return []
        elif strategy.upper() == 'GROWTH':
            return []
        elif strategy.upper() == 'VENTURE':
            return []
        elif strategy.upper() == 'PRIVATE_CREDIT':
            return []
        elif strategy.upper() == 'REAL_ESTATE':
            return []
        elif strategy.upper() == 'INFRASTRUCTURE':
            return []
        else:
            raise ValueError('Strategy {} not supported')


class vintage(object):


    def __init__(self,
                 strategy_type,
                 commitment_size,
                 commitment_year,
                 realized_NAV=None,
                 cumltv_realized_contributions=None,
                 cumltv_realized_distributions=None,
                 cash_flow_frequency=1):

        self._commitments_size = commitment_size
        assert commitment_year >= 0, 'Error - commitment year must be neutral-positive'

        self._commitment_year = commitment_year
        self._strategy_type = strategy_type
        self._cash_flow_frequency = cash_flow_frequency

        self._REALIZED_NAV = realized_NAV
        self._REALIZED_CONTRIBUTIONS = cumltv_realized_contributions
        self._REALIZED_DISTRIBUTIONS = cumltv_realized_distributions

        self.set_default_properties()
        self._estimate_cash_flows()

    def set_default_properties(self):
        self.CAGR = 0.1148
        self._capital_calls = CashFlowAssumptions.get_strategy_capital_call_assumptions(self._strategy_type)
        self._distributions = CashFlowAssumptions.get_strategy_distribution_assumptions(self._strategy_type)

    @property
    def CAGR(self):
        return self.__CAGR

    @CAGR.setter
    def CAGR(self, value):
        self.__CAGR = value

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

        T = self._cash_flow_frequency * len(self._capital_calls) - self._commitment_year

        VALS = np.ones((T, 4)) * np.nan
        VALS[0, 0] = 0
        VALS[0, 1] = self._commitments_size * self._capital_calls[0]
        VALS[0, 2] = 0
        VALS[0, 3] = self._commitments_size * self._capital_calls[0]
        for t in range(1, T):

            VALS[t, 0] = VALS[t-1, -1]
            EOY_NAV_t = VALS[t, 0] * (1 + self.CAGR)
            VALS[t, 1] = self._capital_calls[t] * self._commitments_size
            VALS[t, 2] = EOY_NAV_t * self._distributions[t]
            VALS[t, 3] = EOY_NAV_t * (1-self._distributions[t]) + self._capital_calls[t] * self._commitments_size
        self._df = pd.DataFrame(VALS, columns=['BOY_NAV', 'CALLS', 'DISTR', 'EOY_NAV'], index=range(1, T+1))

if __name__ == "__main__":

    vy = vintage(strategy_type='Buyout',
                 commitment_size=100,
                 commitment_year=2,
                 cash_flow_frequency=1)




