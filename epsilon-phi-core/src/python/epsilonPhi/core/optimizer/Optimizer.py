from epsilonPhi.core.optimizer.constraints.Parser import ConstraintsParser, Constraints
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
import cvxpy as cp
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizerPars:
    def __init__(self,
                 mu: np.array,
                 sigma: np.array,
                 T12: np.array,
                 asset_list: list,
                 ):
        self._mu = mu
        self._sigma = sigma
        self._T12 = T12
        self._asset_list = asset_list

    @property
    def mu(self):
        return self._mu

    @property
    def sigma(self):
        return self._sigma

    @property
    def T12(self):
        return self._T12

    @property
    def asset_list(self):
        return self._asset_list


class CVXOptimizer(object):
    def __init__(self):
        self._ws = None

    def find_kappa_old(self, constraints, optim_pars, target_vol, skew_kappa_l, skew_kappa_r):

        f_kappa = None
        error = 0

        prec = 0.0001
        kappa_init = 10
        kappa_lim = 10000

        kappa = kappa_init
        kappa_l = prec

        kappa_r = math.ceil(skew_kappa_r / 100) * 100
        if kappa < kappa_lim:
            loop = 0

            while abs(kappa_l - kappa_r) > 0.01:
                loop += 1

                kappa = (kappa_l + kappa_r) / 2
                if kappa < skew_kappa_l:
                    kappa_l = kappa
                elif skew_kappa_r < kappa:
                    kappa_r = kappa
                else:

                    wts, error = self.optimizeRobust(
                        constraints,
                        optim_pars,
                        target_vol,
                        kappa
                    )

                    if error != 0:
                        return 0, error

                    sig = 100 * math.sqrt(
                        np.matmul(
                            np.matmul(
                                weights.T, optim_pars.sigma
                            ), weights
                        )
                    )

                    if sig >= target_vol - prec:
                        kappa_l = kappa
                    else:
                        kappa_r = kappa

                f_kappa = max(kappa - 0.01, 0)
        else:
            f_kappa = kappa_lim

        logger.info('find_kappa_old() done with kappa {}'.format(f_kappa))
        return f_kappa, error

    def find_kappa(self,
                   constraints,
                   optim_pars,
                   target_vol):

        optimize_loop = False

        p = 0.0001
        kappa_init = 20
        kappa_limit = 10000

        kappa = kappa_init
        kappa_l = p
        kappa_r = kappa_limit * 1.1

        # Binary search to fin largest kappa for which we reach the target vol
        logger.info("starting find_kappa")
        loop = 0
        while abs(kappa_l - kappa_r) > 0.01 and kappa < kappa_limit:
            loop += 1

            wts, error_code = self.optimizeRobust(
                constraints,
                optim_pars,
                target_vol,
                kappa
            )

            if error_code != 0:
                if error_code == -1:
                    if optimize_loop:
                        break
                else:
                    fkappa = None
                    return fkappa, error_code

            sig = 100 * math.sqrt(
                np.matmul(
                    np.matmul(
                        weights.T, optim_pars.sigma
                    ), weights
                )
            )

            if sig is not None:
                if sig >= (target_vol - p):
                    kappa_l = kappa
                else:
                    kappa_r = kappa
            else:
                raise Exception('find_kappa, sigma is none')

            skew = max(1, np.log10((kappa_r - kappa_l)))
            kappa = (skew * kappa_l + kappa_r) / (skew + 1)
            logger.info('kappa = [{}] and skew = [{}]'.format(kappa, skew))
        logger.info('find_kappa done with kappa [{}]'.format(kappa))
        fkappa, error_code = self.find_kappa_old(constraints, optim_pars, target_vol, kappa_l, kappa_r)
        return kappa, error_code

    def generate_constraints(self,
                             constraint_str: str,
                             asset_list: list,
                             LB=None,
                             UB=None,
                             not_modeled=0):

        # Instantiate parser
        parser = ConstraintsParser(constraint_str)

        if constraint_str and constraint_str != '':
            if LB is None:
                res, const, LB_, UB_, error = parser.generate_constraints_UB(constraint_str, asset_list)
            else:
                res, const, LB_, UB_, error = parser.generate_constraints_UB_LB(constraint_str, asset_list)

            if error != 0:
                raise Exception('Error - constraint error in constraint {}'.format(error))

            res.UB = UB_
            res.LB = LB_
            res.constraints = const
            res.UB_box = UB if UB is not None else [1] * len(asset_list)
            res.LB_box = LB if LB is not None else [0] * len(asset_list)
        else:
            res = Constraints()
            res.UB_box = UB if UB is not None else [1] * len(asset_list)
            res.LB_box = LB if LB is not None else [0] * len(asset_list)
        return res

    @staticmethod
    def get_optim_pars(portfolio):
        return OptimizerPars(
            portfolio.get_assets_total_return(),
            portfolio.get_sigma(),
            portfolio.get_uncertainty_matrix(),
            portfolio.get_asset_names(),
        )

    def optimize_robust(self, portfolio, target_vol, constraints, LB=None, UB=None):

        # Get optimization parameters
        optim_pars = self.get_optim_pars(portfolio)

        # Generate Constraints
        const_struct = self.generate_constraints(
            constraints, optim_pars.asset_list, LB, UB)

        # Kappa
        kappa = self.find_max_kappa(
            target_vol,
            optim_pars,
            const_struct
        )

        # Run the CVX Optimizer
        wts, _ = self.run_cvx_robust(
            kappa,
            target_vol,
            optim_pars,
            const_struct)

        return wts, kappa

    def find_max_kappa(
            self,
            target_vol,
            optim_pars,
            const_struct,
            kappa_low=0.0,
            kappa_high=1000.0,
            tol=1e-3,
            max_iter=10000,
    ):
        """
        Finds the kappa that results in volatility closest to the target.

        Returns:
            float: Best kappa such that realized volatility ≈ target_vol
        """
        best_kappa = kappa_low
        best_vol = 0

        for _ in range(max_iter):
            mid_kappa = (kappa_low + kappa_high) / 2
            try:
                w, realized_vol = self.run_cvx_robust(mid_kappa,
                                                      target_vol,
                                                      optim_pars,
                                                      const_struct)
            except Exception:
                realized_vol = np.inf

            # Track the kappa that gets closest to target_vol
            if np.abs(best_vol - target_vol) > np.abs(realized_vol - target_vol):
                best_vol = realized_vol
                best_kappa = mid_kappa

            # Early stopping if very close
            if np.abs(best_vol - target_vol) < 0.0001:
                break

            # Update bisection bounds
            if realized_vol < target_vol:
                # Too conservative → try lower kappa
                kappa_high = mid_kappa
            else:
                # Acceptable or too risky → try higher kappa
                kappa_low = mid_kappa

            if abs(kappa_high - kappa_low) < tol:
                break

        if np.abs(best_vol - target_vol) > tol:
            raise ValueError('Error - target vol not obtainable')

        return best_kappa

    def run_cvx_robust(
            self,
            kappa,
            target_vol,
            optim_pars,
            constraints,
    ):

        sigma = optim_pars.sigma
        T = optim_pars.T12
        mu = optim_pars.mu

        assert len(np.diag(sigma)) == len(mu)
        assert len(np.diag(sigma)) == len(T)

        # number of assets
        n = len(mu)

        # Sigma is the covariance matrix
        sig = sqrtm(sigma)

        # Variable of insterest
        w = cp.Variable(n)

        # Objective Function
        objfun = cp.Maximize(mu @ w - 0.5 * kappa * cp.norm2(T @ w))

        # === Core constraints ===
        constraints_list = []

        # Volatility constraint
        constraints_list.append(cp.norm2(sig @ w) <= target_vol)

        # Full investment
        constraints_list.append(cp.sum(w) == 1)

        # Box constraints
        constraints_list.append(w >= constraints.LB_box)
        constraints_list.append(w <= constraints.UB_box)  # UB_box is incorrect

        # Equality/Inequality Constraints
        if constraints.LB is not None:
            constraints_list.append(constraints.LB <= constraints.mat @ w)
        if constraints.UB is not None:
            constraints_list.append(constraints.mat @ w <=  )

        problem = cp.Problem(objfun, constraints_list)
        problem.solve()
        return w.value, cp.norm2(sig @ w).value


if __name__ == "__main__":
    from epsilonPhi.core.schema.Schema import ContextCreator
    from epsilonPhi.core.portfolio.SAAPortfolio import SAAPortfolio

    schema = ContextCreator(
        currency='USD',
        start_date='30-Nov-1983',
        end_date='31-Dec-2022'
    ).create_context()

    ptf = SAAPortfolio('portfolio', schema)
    ptf.add_asset_by_name('MSUSAML', 0.5, 0)
    ptf.add_asset_by_name('LHAGGBD', 0.5, 0)
    ptf.setup()

    self = CVXOptimizer()
    wts = self.optimize_robust(
        ptf,
        target_vol=0.07,
        constraints=None,
    )

    ptf.optimize(target_vol=0.07)
