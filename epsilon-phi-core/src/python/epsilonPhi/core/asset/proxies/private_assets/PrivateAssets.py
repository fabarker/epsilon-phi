from epsilonPhi.core.utils.DateUtils import DateUtils
from epsilonPhi.core.asset.Asset import CAsset
from epsilonPhi.core.schema.Schema import CContext
from typing import Optional
import numpy as np
import datetime as dt
import math

__all__ = ['CPrivateAsset']

class CPrivateAsset(CAsset):
    _metadata = (CAsset._metadata +
                 ['_pme', '_pme_name', '_risk_premias'])

    def __init__(
            self,
            schema: Optional[CContext] = None,
            pme_name: Optional[str] = None,
            asset_name: Optional[str] = None,
    ) -> None:

        self._schema = schema
        self._pme_name = pme_name
        self._pme = None
        self._risk_premias = None

        pme = self.get_public_market_equivalent()
        pme.name = (asset_name, 'RI')
        super(CPrivateAsset, self).__init__(
            pme,
            schema,
            **pme.get_asset_params())

    @property
    def liq_factor_name(self):
        return 'LIQUIDITY_US_PASTOR_STAMBAUGH'

    @property
    def premium_liq_to_total(self):
        return 1/3

    @property
    def liq_adjustment(self):
        return 1 / (1 - self.premium_liq_to_total) -1

    @property
    def beta_priv_to_pub(self):
        return 0.8
        #return 1.08

    def load_public_market_equivalent(self):
        if self._pme is None:
            self._pme = self._schema.get_asset_from_name(self._pme_name)
            self._pme.set_currency_hedge_ratio(0)

    def get_public_market_equivalent(self):
        if self._pme is None:
           self.load_public_market_equivalent()
        return self._pme.deepcopy()

    def get_pars(self):
        if self._pme is None:
            self.load_public_market_equivalent()
        return {'currency': self._pme.currency,
                'frequency': self._pme.frequency}

    def get_return_betas(self, hedging_ratio=None, normalized=True):
        rp = self.get_risk_premias()
        hist_sharpe =  np.array(self._schema.get_return_factors_sharpe_ratios()).flatten()
        return rp / hist_sharpe / math.sqrt(self._schema.frequency.obs_per_year())

    def get_excess_return_df(self, from_date=None, to_date=None):
        raise Exception("Not supported for private assets")

    def get_liquidity_factor_index(self):
        return self._schema.get_return_factors_panel().columns.get_loc(self.liq_factor_name)

    def get_risk_premias_pme(self):
        return np.copy(self.get_public_market_equivalent().get_risk_premias())

    def get_risk_premias_new(self):

        liq_idx = self.get_liquidity_factor_index()
        rp_pme = self.get_risk_premias_pme()
        beta_adj_rp = rp_pme * self.beta_priv_to_pub
        return_fac_panel = self._schema.get_orthog_return_factors_panel()

        # factor vols in schema dates
        fact_sig = np.std(return_fac_panel, axis=1, ddof=1)
        fact_sharpes = self._schema.get_return_factors_sharpe_ratios()
        beta_adj_rp[liq_idx] = 0.8 * fact_sharpes[liq_idx] * fact_sig[liq_idx] * np.sqrt(12)
        return beta_adj_rp


    def get_risk_premias(self):

        if self._risk_premias is None:
            rp_pme = self.get_risk_premias_pme()  # shape: [n_factors]
            beta_adjusted_rp = self.beta_priv_to_pub * rp_pme  # shape: [n_factors]

            liq_idx = self.get_liquidity_factor_index()
            total_rp = beta_adjusted_rp.sum()
            liq_contrib = self.liq_adjustment * total_rp

            # Create adjusted vector in one go
            liq_adj = np.zeros_like(beta_adjusted_rp)
            liq_adj[0, liq_idx] = liq_contrib

            self._risk_premias = beta_adjusted_rp + liq_adj
        return self._risk_premias

    def get_historical_sharpe(self):
        return self.get_public_market_equivalent().get_historical_sharpe()

    def get_risk_betas(self):

        # liquidity index
        liq_idx = self.get_liquidity_factor_index()

        # Get the unhedged beta risk - no hedging of private assets
        betas, _ = self.get_public_market_equivalent().get_beta_and_idio_risk(0)

        # risk premia of public market equivalent
        rp_pme = self.get_public_market_equivalent().get_risk_premias()

        # get the liquidity factor (normalized or no?)
        liq_factor = self._schema.get_factor_panels().get_factor(self.liq_factor_name)
        liq_prem = np.mean(liq_factor) * self._schema.frequency.obs_per_year()

        beta_adj_betas = np.copy(betas)
        beta_adj_betas[liq_idx] = beta_adj_betas[liq_idx] + self.liq_adjustment * np.sum(rp_pme) / liq_prem
        beta_adj_betas = beta_adj_betas * self.beta_priv_to_pub

        # vol of PME
        std_pme = self.get_public_market_equivalent().get_volatility()

        # factor covariance - stop at factor sharpe end date
        fac_cov = self._schema.get_risk_factor_covariance()

        # adjust the variance
        adj_var = (beta_adj_betas @ fac_cov) @ beta_adj_betas
        if adj_var > np.power(std_pme, 2):
           idio = 0
        else:
           idio = np.power(std_pme, 2) - adj_var

        return beta_adj_betas, idio

    def get_data_length(self):

        if self._schema.currency in ['CHF', 'USD']:
            SD = dt.datetime(1999, 6, 30)
        else:
            SD = dt.datetime(1998, 6, 30)

        return DateUtils.get_date_delta(SD, self._schema.end_date, False).item() / 252


if __name__ == "__main__":

    from epsilonPhi.core.asset.AssetMgr import CAssetMgr
    from epsilonPhi.core.schema.Schema import ContextCreator
    schema = ContextCreator(
        currency='USD',
        start_date='30-Nov-1983',
        end_date='31-Dec-2022'
    ).create_context()

    tt = CPrivateAsset(schema, 'MSWRLD$', 'PE_BUYOUT')
    rp_pme = tt.get_return_betas()








