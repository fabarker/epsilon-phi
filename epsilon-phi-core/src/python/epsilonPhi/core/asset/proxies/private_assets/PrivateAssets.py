from epsilonPhi.core.utils.DateUtils import DateUtils
from epsilonPhi.core.asset.Asset import CAsset
from epsilonPhi.core.schema.Schema import CContext
from epsilonPhi.core.timeSeries.timeSeriesMain import CSlice, CTimeSeries
from typing import Optional, Union
from typing import Optional
import numpy as np
import datetime as dt
import math

__all__ = ['CPrivateAsset']

beta_pri_to_pub = {
    ("PE_BUYOUT", "RI"): 1.0,
    ("PE_DISTRESSED", "RI"): 1.0,
    ("PE_VENTURE", "RI"): 1.0,
    ("PE_GROWTH", "RI"): 1.0,
    ("PA_INFRA", "RI"): 0.9,
}

class CPrivateAsset(CAsset):
    _metadata = (CAsset._metadata +
                 ['_pme_name', '_risk_premias'])

    @property
    def _constructor(self):
        def _c(*args, **kwargs):
            return CPrivateAsset(*args, **kwargs).__finalize__(self)

        return _c

    def __init__(
            self,
            pme: Union[CTimeSeries, CSlice],
            schema: Optional[CContext] = None,
            asset_name: Optional[str] = None,
            **kwargs
    ) -> None:

        self._risk_premias = None

        # Only attempt to extract name if it's a timeseries-like object
        if hasattr(pme, 'name'):
            self._pme_name = pme.name[0] if isinstance(pme.name, tuple) else pme.name
        else:
            self._pme_name = None  # fallback if being constructed via pandas internals

        # Optional: set name if asset_name is explicitly provided
        if asset_name and hasattr(pme, 'name'):
            pme.name = (asset_name, "RI")

        super(CPrivateAsset, self).__init__(pme, schema, **kwargs)
        self.set_currency_hedge_ratio(0)

    def create_new_object(self, *args, **kwargs):
        kwargs["pme"] = kwargs.pop("data")
        return self.__class__(*args, **kwargs)

    @property
    def schema(self):
        return self._schema

    @property
    def returns_type(self):
        if not self._returns_type:
            self._returns_type = self.get_public_market_equivalent().returns_type
        return self._returns_type

    @property
    def type(self):
        if not self._type:
            self._type = self.get_public_market_equivalent().type
        return self._type

    @property
    def denominated_currency(self):
        if not self._denominated_currency:
            self._denominated_currency = self.get_public_market_equivalent().denominated_currency
        return self._denominated_currency

    @property
    def exposure_currency(self):
        if not self._exposure_currency:
            self._exposure_currency = self.get_public_market_equivalent().exposure_currency
        return self._exposure_currency

    @property
    def liq_factor_name(self):
        return 'LIQUIDITY_US_PASTOR_STAMBAUGH'

    @property
    def premium_liq_to_total(self):
        return 1 / 3

    @property
    def liq_adjustment(self):
        return 1 / (1 - self.premium_liq_to_total) - 1

    @property
    def beta_priv_to_pub(self):
        return beta_pri_to_pub.get(self.name, 0.8)

    def get_liquidity_beta(self):

        # 1. Get PME beta
        b, _ = self.get_public_market_equivalent().get_beta_and_idio_risk(0)

        # get liquidity beta index
        idx = self.get_liquidity_factor_index()

        # get liquidity factor premium
        liq_premium = self.get_liquidity_factor_premium()
        return b[idx] + self.liq_adjustment * np.sum(self.get_risk_premias_pme()) / liq_premium

    def get_liquidity_factor_premium(self):
        return self.schema.get_factor_panels().get_factor(
            self.liq_factor_name).mean() * 12

    def load_public_market_equivalent(self):
        if not hasattr(self, '_pme'):
            self._pme = self.schema.get_asset_from_name(self._pme_name)
            self._pme.set_currency_hedge_ratio(0)

    def get_public_market_equivalent(self):
        if not hasattr(self, '_pme'):
            self.load_public_market_equivalent()
        return self._pme.deepcopy()

    def get_pars(self):
        if self._pme is None:
            self.load_public_market_equivalent()
        return {'currency': self._pme.currency,
                'frequency': self._pme.frequency}

    def get_return_betas(self, hedging_ratio=None, normalized=True):
        rp = self.get_risk_premias()
        hist_sharpe = np.array(self._schema.get_return_factors_sharpe_ratios()).flatten()
        return rp / hist_sharpe / math.sqrt(self._schema.frequency.obs_per_year())

    def get_excess_return_df(self, from_date=None, to_date=None):
        raise Exception("Not supported for private assets")

    def get_volatility(self, hedging_ratio=None):
        betas, idio = self.get_beta_and_idio_risk(hedging_ratio)
        return np.sqrt((betas @ self.schema.get_risk_factor_covariance()) @ betas + idio)

    def get_liquidity_factor_index(self):
        return self._schema.get_return_factors_panel().columns.get_loc(self.liq_factor_name)

    def get_risk_premias_pme(self):
        return np.copy(self.get_public_market_equivalent().get_risk_premias())

    def get_beta_and_idio_risk(self, hedging_ratio=0):
        return self.get_risk_betas(), self.get_idiosyncratic_variance(hedging_ratio)

    def get_risk_premias(self):
        beta_adj_rp = self.get_risk_premias_pme() * self.beta_priv_to_pub
        liq_idx = self.get_liquidity_factor_index()
        beta_adj_rp[0, liq_idx] = self.get_liquidity_beta() * self.get_liquidity_factor_premium()
        return beta_adj_rp

    def get_risk_premias_orig(self):

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

    def get_idiosyncratic_variance(self, hedging_ratio=0):

        risk_betas = self.get_risk_betas()
        risk_betas_target = risk_betas*(self.beta_priv_to_pub / 0.8)

        fac_cov = self._schema.get_risk_factor_covariance()
        pme_idio = self.get_public_market_equivalent().get_idiosyncratic_variance(hedging_ratio)

        curr_var = (risk_betas @ fac_cov) @ risk_betas
        targ_var = (risk_betas_target @ fac_cov) @ risk_betas_target

        #std_pme = self.get_public_market_equivalent().get_volatility(hedging_ratio)

        # factor covariance - stop at factor sharpe end date
        #fac_cov = self._schema.get_risk_factor_covariance()

        #adj_var = (risk_betas @ fac_cov) @ risk_betas
        #if adj_var > np.power(std_pme, 2):
        #    idio = 0
        #else:
        #    idio = np.power(std_pme, 2) - adj_var
        #return idio

        return targ_var - curr_var + pme_idio


    def get_risk_betas_new(self):

        # liquidity index
        liq_idx = self.get_liquidity_factor_index()

        # Get the unhedged beta risk - no hedging of private assets
        betas, _ = self.get_public_market_equivalent().get_beta_and_idio_risk(0)

        # adjust the liquidity beta
        beta_adj_betas = np.copy(betas)
        beta_adj_betas[liq_idx] = self.get_liquidity_beta()
        beta_adj_betas = beta_adj_betas * self.beta_priv_to_pub
        return beta_adj_betas

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
        #beta_adj_betas = beta_adj_betas * self.beta_priv_to_pub
        return beta_adj_betas * 0.8

    def get_data_length(self):

        if self._schema.currency in ['CHF', 'USD']:
            SD = dt.datetime(1999, 6, 30)
        else:
            SD = dt.datetime(1998, 6, 30)

        return DateUtils.get_date_delta(SD, self._schema.end_date, False).item() / 252

    def get_uncertainty(self):
        return super(CPrivateAsset, self).get_uncertainty() * 2


if __name__ == "__main__":
    from epsilonPhi.core.schema.Schema import ContextCreator

    schema = ContextCreator(
        currency='USD',
        start_date='30-Nov-1983',
        end_date='31-Dec-2022'
    ).create_context()

    pme = schema.get_asset_from_name('MSWRLD$')

    tt = CPrivateAsset(pme=pme, schema=schema, asset_name='PE_BUYOUT')
    betas = tt.get_idiosyncratic_variance()
    beta_orig = tt.get_risk_premias()
