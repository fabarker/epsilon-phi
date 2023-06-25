from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.dataModel.enums.Asset import PrivateAsset
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.alchemist.DataModel import *
import numpy as np
import pandas as pd
import os, sys


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

    def _load_capital_call_assumptions(self):
        yearly_capital_calls = GlobalDataSource().get_private_asset_capital_call_assumptions(self._strategy_type)
        self._capital_call_assumptions = GlobalDataSource().get_private_asset_capital_call_assumptions(self._strategy_type)

    def _load_distribution_assumptions(self):
        distributions = GlobalDataSource().get_private_asset_capital_call_assumptions(self._strategy_type)
        idxs = np.arange(1/self._cash_flow_frequency, distributions.index.max(), 1/self._cash_flow_frequency)
        self._distribution_assumptions = distributions.reindex(idxs).ffill() / self._cash_flow_frequency

    def set_default_properties(self):
        self._load_distribution_assumptions()
        self._load_capital_call_assumptions()
        self.CAGR = 0.1148

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

    vy = vintage(strategy_type=PrivateAsset.BUYOUT,
                 commitment_size=100,
                 commitment_year=2,
                 cash_flow_frequency=4)




