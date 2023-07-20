from epsilonPhi.core.factor.factorPanelInf import CFactorPanelInf

class CFactorPanels(CFactorPanelInf):
    def __init__(self):
        super(CFactorPanels, self).__init__()

        self._return_factors = None
        self._risk_factors = None

        self._return_factors_df = None
        self._risk_factor_df = None

    def set_risk_factors(self):
        pass

    def set_return_factors(self):
        pass

    def orthogonalize_factors(self):
        pass

    def get_return_factor_Sharpe_ratios(self):
        pass

    def  set_return_factor_Sharpe_ratio(self, factor_name, Sharpe_ratio):
        pass

    def get_historical_factor_Sharpe_ratio(self, factor_name, orthogonalize=False):
        pass


    def get_historical_factor_volatility(self, factor_name, orthogonalize=False):
        pass
