from epsilonPhi.core.asset.CAssetInf import CAssetInf
from epsilonPhi.core.timeSeries.timeSeriesMain import CSlice
from epsilonPhi.core.schema.Schema import CContext
from datetime import datetime
import pandas as pd
import numpy as np
import math

class CAsset(CSlice, CAssetInf):
    def __init__(self,
                 ticker: str,
                 currency: str,
                 hedging_ratio: float,
                 schema: CContext,
                 dataframe=None):

        super(CAsset, self).__init__(dataframe)
        self._currency = currency
        self._ticker = ticker
        self._hedging_ratio = hedging_ratio
        self._schema = schema

    @property
    def asset_ticker(self):
        return self._ticker
    @property
    def currency(self):
        return self._currency
    @property
    def name(self):
        return self._ticker

    def get_alpha(self):
        pass

    def set_alpha(self):
        pass

    def get_historical_sharpe_ratio(self):
        pass

    def get_Sharpe_ratio(self):
        pass

    def get_data_length(self):
        pass

    def get_return_betas(self, normalized=True):
        betas = self._schema.getEstimationMgr().getReturnBetas(self, normalized=True)
        return betas

    def get_return_betas_not_normalized(self):
        return self.get_return_betas(False)

    def get_total_return(self):
        return self.get_risk_premia() + self._schema.get_risk_free_rate() + self.get_alpha()

    def get_risk_betas(self):
        return self._schema.getEstimationMgr().get_risk_betas(self, self.hedging_ratio)

    def get_idiosyncratic_variance(self):
        return self._schema.getEstimationMgr().get_idiosyncratic_variance(self, self.hedging_ratio)

    def get_volatility(self):

        betas = self.get_risk_betas()
        idio = self.get_idiosyncratic_variance()

        factor_covariance = self._schema.get_factor_panels().get_risk_factor_covariance_matrix()
        systematic_var = betas @ factor_covariance @ betas
        return np.sqrt(systematic_var + idio)

    def get_risk_premia(self):
        return self._schema.getEstimationMgr().get_risk_premium(self)

    def get_historical_risk_premium(self):
        pass

    def get_excess_return_df(self):
        pass

    def set_hedging_ratio(self, hedging_ratio):
        self._hedging_ratio = hedging_ratio

    def get_hedging_option(self, hedging_option):
        self._hedging_option = hedging_option












