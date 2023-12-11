import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
import cvxpy as cp

class ConstraintsParser(object):
    def __init__(self):
        pass

    @staticmethod
    def parse_constraints_string(string: str):

        spltstr = sorted([x for x in string.split(';') if x])
        N = len(spltstr)
        LB = np.ones((N, )) * 0
        UB = np.ones((N, ))

        for s in spltstr:

            # Remove % sign if its there
            if s[-1] == '%':
               s = s[0:-1]

            LT_idx = s.find('<')
            GT_idx = s.find('>')
            ET_idx = s.find('=')

            if LT_idx > -1:
                pass
               # Do something
            elif GT_idx > -1:
                pass
               # Do something
            elif ET_idx > -1:
                pass
               # Do something


class CVXOptimizer():
    def __init__(self):
        self._ws = None

    # Create Inequality Constraints Aeq * x = beq
    def deconstruct_constraints(self):
        pass

    def build_constaints_from_string(self, assetList, constraints):

        N = len(assetList)
        T = len(constraints.lowerbounds)

        b_low = np.array(constraints.lowerbounds)
        b_high = np.array(constraints.upperbounds)

        A = np.zeros((N ,T))
        for i in range(T-1):
            for j in range(N):
                s = assetList[j]
                A[i, j] = constraints._constraints[i][s]

        return b_low, b_high, A


    def optimizeGeneric(self):
        pass

    def optimizeRobust(self, mu, sigma, kappa, T, target_volatility):

        assert len(np.diag(sigma)) == len(mu)
        assert len(np.diag(sigma)) == len(T)

        # number of assets
        n = len(mu)

        # Sigma is the covariance matrix
        sig = sqrtm(sigma)

        # T is the uncertainty vector (standard errors of the means
        T12 = np.diag(T)

        ub = []
        lb = []
        A = []
        b = []
        Aeq = []
        beq = []
        b_low = []
        b_high = []

        w = cp.Variable(n)
        objfun = cp.Maximize(mu @ w - 0.5 * kappa * cp.norm2(T12) @ w)
        constraints = [w >= lb,
                       w <= ub,
                       b_low <= A @ w,
                       A @ w <= b_high,
                       sum(w) == 1,
                       cp.norm2(sig @ w) <= target_volatility]

        # cp.norm2(sqrtm(cov * 12) @ weights).value
        # is equivalent to
        # np.sqrt(cp.quad_form(weights, cov * 12).value)

        problem = cp.Problem(objfun, constraints)
        problem.solve()
        return problem.w


    # Mean Variance is robust with a kappa of 0
    def optimizeMeanVariance(self, assetList, mu, sigma, T, target_volatility):
        return self.optimizeRobust(assetList, mu, sigma, 0, T, target_volatility)

    def optimizeMinVol(self):
        pass

    def leastRegretOptimize(self):
        pass

    def optimizeTrackingError(self):
        pass


if __name__ == "__main__":

    import pandas as pd

    project_path = '/Users/francisbarker/Library/Mobile Documents/com~apple~CloudDocs/Data/Misc/Global Factor Premiums.xlsx'
    df = pd.read_excel(project_path, 'Data', header=[0,1], index_col=0)
    df = df.dropna()
    df.index = pd.to_datetime(df.index) + pd.tseries.offsets.MonthEnd(0)

    n_assets = df.shape[1] # Number of assets
    n_constraints = 2  # Number of inequality constraints

    mu = df.mean().values * 12  # Expected returns
    cov = df.cov().values * 12 # Covariance matrix
    sig = sqrtm(cov)
    serror = np.sqrt(df.std() / df.shape[0])
    T12 = np.diag(serror.values)

    # Define variables
    weights = cp.Variable(n_assets)  # Portfolio weights

    # Define problem constraints
    constraints = [
        cp.sum(weights) == 1,  # Sum of weights must be 1
        weights >= 0,
        cp.norm2(sig @ weights) <= 0.07]

    # Define problem
    risk_aversion = 6
    kappa = 50
    objective = cp.Maximize(mu @ weights - risk_aversion * cp.quad_form(weights, cov))
    objfun = cp.Maximize(mu @ weights - 0.5 * kappa * cp.norm2(T12 @ weights))

    problem = cp.Problem(objfun, constraints)

    # Solve problem
    problem.solve()

    # Retrieve optimal solution
    optimal_weights = weights.value

    # Print optimal weights
    print("Optimal Weights:")
    for i in range(n_assets):
        print(f"Asset {i + 1}: {optimal_weights[i]}")


