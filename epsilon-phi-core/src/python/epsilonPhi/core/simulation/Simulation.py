import numpy as np
import pandas as pd

class HullWhiteDynamicMean(object):

    def __init__(self):
        pass

    def simulate(self):
        pass

    def get_forward_curve(self):
        pass

    def extract_shocks(self):
        pass

    def fit_params(self):
        pass



class YieldCurve(object):

    def __init__(self, currency):
        pass

    def get_shocks_single_maturity(self, maturity):
        pass

    def extract_shocks(self, df):
        alpha, beta = self._bootstap.extract_shocks(df)


    def simulate_yield_single_maturity(self):

        shocks, beta = self.bootstrap_shocks_single_maturity(maturity)
        periodicForwardRates = self.get_forward_curve_for_maturity(maturity)

        T = periodicForwardRates.shape[0]
        N = shocks.shape[1]

        rates = np.ones((T, N)) * np.nan
        mean_reversion_speed = self._mean_reversion_strength
        for t in range(0, T):
            if t == 0:
                rates[t, :] = periodicForwardRates[t]
            else:
                dR = mean_reversion_speed * (periodicForwardRates[t] - rates[t-1]) + shocks
                rates[t, :] = rates[t, :] + dR
        return pd.DataFrame(rates)

    def simulate_bond_returns_single_maturity(self, maturity):

        rates = self.simulate_yield_single_maturity(maturity).values
        T, N = rates.shape

        tau = 1/12

        total_return = np.ones(rates.shape) * np.nan
        income = np.ones(rates.shape) * np.nan
        price = np.ones(rates.shape) * np.nan

        for t in range(1,T):
            income[t,:] = -1 + np.power(1+rates[t-1,:], 1/tau)
            price[t,:] = -1 * maturity * (rates[t, :] - rates[t-1,:])
        total_return = income + price
        return pd.DataFrame(total_return).dropna().copy()




