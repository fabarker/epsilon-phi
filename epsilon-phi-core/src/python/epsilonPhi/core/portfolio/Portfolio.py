import math, sys
from collections import OrderedDict
from datetime import datetime as dt

import numpy as np

from epsilonPhi.core.schema.Schema import CContext
from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.asset.Asset import CAsset
from epsilonPhi.logging import *
from typing import Union

class CPortfolio(object):
    _sqrt_epsilon = math.sqrt(sys.float_info.epsilon)

    def __init__(self, name: str, context: CContext):

        self._created = dt.now()

        self._context = context
        self._name = name
        self._portfolio_mgr = None

        self._weights = []
        self._asset_names = []
        self._assets = OrderedDict()
        self._hedging_ratios = []
        self._current_value = 100

        self._returns_ts = None
        self._is_setup = False

    def reset_properties(self):
        self._returns_ts = None

    @property
    def create_date(self):
        return self._created

    @property
    def num_assets(self):
        return len(self._assets)

    @property
    def portfolio_mgr(self):
        return self._portfolio_mgr if self._portfolio_mgr else None

    @property
    def dates(self):
        return self._context.dates

    @property
    def name(self):
        return self._name

    @property
    def current_value(self):
        return self._current_value

    @property
    def context(self):
        return self._context

    @property
    def schema(self):
        return self._context

    @property
    def assets(self):
        return self._assets

    @property
    def asset_names(self):
        return list(self._assets.keys())

    @property
    def reporting_currency(self):
        return self._context.currency

    @property
    def frequency(self):
        return self._context.frequency

    @property
    def created(self):
        return self._created

    ########## Setter Methods ############

    def setup(self):
        from epsilonPhi.core.portfolio.PortfolioMgr import CPortfolioMgr
        if not self._is_setup and self._portfolio_mgr is None:
            self._portfolio_mgr = CPortfolioMgr(self._context)
            self._portfolio_mgr.set_portfolio(self)
            self._portfolio_mgr.set_tax_info(self)
            self._is_setup = True

    def set_current_value(self, value):
        self._current_value = value

    def set_hedging_ratios(self, ratios) -> None:

        # Hedging ratios are as the asset level
        for i, asset_name in enumerate(self.get_asset_names()):
            self.set_single_asset_hedge_ratio(asset_name, ratios[i])
        self._hedging_ratios = ratios

    def set_single_asset_hedge_ratio(self, asset_name, hedge_ratio) -> None:
        self.portfolio_mgr.set_single_asset_hedge_ratio(asset_name, hedge_ratio)

    def set_weights(self, weights):
        self._portfolio_mgr.set_weights(weights)

    def is_in_portfolio(self, asset: Union[CAsset, str]) -> bool:

        if isinstance(asset, CAsset):
           return asset.name in self._assets
        elif isinstance(asset, str):
           return asset in self._assets
        else:
            raise ValueError('Invalid asset type')


    ######### Getter Methods ##########

    def get_portfolio_mgr(self):
        if self._portfolio_mgr is None:
            from epsilonPhi.core.portfolio.PortfolioMgr import CPortfolioMgr
            self._portfolio_mgr = CPortfolioMgr(self._context)
            self._portfolio_mgr.set_portfolio(self)
        return self._portfolio_mgr

    def get_weights(self):
        wts = []
        for i, asset_name in enumerate(self.get_asset_names()):
            wts.append(self.get_asset(asset_name).weight)
        return np.asanyarray(wts).reshape(-1, 1)

    def get_asset_names(self):
        return self.asset_names

    def get_asset(self, asset_name):
        if asset_name not in self._assets:
            raise KeyError("Asset name '{}' not found in portfolio".format(asset_name))
        return self._assets[asset_name]

    def get_assets(self):
        return self._assets.values()

    def get_historical_risk_premia(self):
        return self.get_portfolio_mgr().get_historical_risk_premia()

    def get_tax_rates(self):
        return self.get_portfolio_mgr().get_tax_rates()

    def get_asset_alphas(self):
        return self.get_portfolio_mgr().get_asset_alphas()

    def get_hedging_ratios(self):
        return np.asarray([x.hedging_ratio for x in self.get_assets()])

    def get_risk_free_asset(self):
        return self.portfolio_mgr.assetMgr.get_risk_free_asset(self.reporting_currency)

    def get_risk_free_rate(self):
        return self.schema.risk_free_rate

    def get_flattened_weights(self):
        return self.get_weights().flatten()

    ############ Public Methods ##############

    def add_asset(self,
                  asset: CAsset,
                  weight: float,
                  hedging_ratio: float
                  ) -> None:

        # cache asset in portfolio
        if not self.is_in_portfolio(asset):
            self._assets[asset.name] = asset
            self._assets[asset.name].set_weight(weight)
            self._assets[asset.name].set_currency_hedge_ratio(hedging_ratio)
        else:
            raise ValueError("Asset '{}' already added to portfolio".format(asset.name))

        # Invalidate Cache
        self.reset_properties()

    def add_asset_by_name(self, asset_name, weight, hedging_ratio=0.0, asset_time_series=None) -> None:
        self.portfolio_mgr.add_asset_by_name(asset_name, weight, hedging_ratio, asset_time_series)

    def remove_asset_by_name(self, asset_name, rebalance=False) -> None:

        if asset_name not in self.asset_names:
            return

        self._assets.pop(asset_name)
        self.reset_properties()

        if rebalance:
           wts_pre_rebalance = self.get_weights()
           wts_post_rebalance = wts_pre_rebalance / np.sum(wts_pre_rebalance)
           self.set_weights(wts_post_rebalance)

    def check_weights(self, weights=None):
        weights = self.get_weights() if weights is None else weights
        if abs(sum(weights) - 1.0) > 1e-5:
            error = 'Error: sum of weights should be 1 – got {} for weights {}'.format(sum(weights), weights)
            logger.error(error)
            raise Exception(error)

    def validate_weights(self, weights=None):
        self.check_weights(weights)

    def has_single_stock(self):
        return self.get_portfolio_mgr().has_single_stock()

    ################### Historical Related #######################

    def get_historical_stress_tests(self):
        return self._portfolio_mgr.get_historical_stress_tests()

    def get_realized_asset_return_panel(self):
        return self._portfolio_mgr.get_realized_asset_return_panel()

    def get_historical_return_time_series(self):
        return self._portfolio_mgr.get_historical_return_time_series()

    def get_historical_cuml_return_series(self):
        return self.portfolio_mgr.get_historical_cuml_return_series()

    def get_historical_real_cuml_return_series(self):
        return self.portfolio_mgr.get_historical_real_cuml_return_series()

    def get_historical_worst_peak_to_trough_loss(self):
        pass

    def get_historical_max_drawdown(self):
        pass

    def get_get_worst_periodic_return(self):
        pass

    def get_worst_periodic_real_return(self):
        pass

    def get_historical_excess_return(self):
        pass

    def get_historical_beta(self):
        pass






if __name__ == "__main__":

    from epsilonPhi.core.asset.AssetMgr import CAssetMgr
    from epsilonPhi.core.schema.Schema import ContextCreator
    schema = ContextCreator(currency='USD',
                            start_date='30-Nov-1983',
                            end_date='31-Dec-2022').create_context()

    assetMgr = CAssetMgr(schema)
    asset = assetMgr.get_asset_by_name('MSUSAML')

    self = CPortfolio('portfolio', schema)
    self.add_asset_by_name('MSUSAML', 1, 0)
    self.get_portfolio_mgr()

    import inspect
    all_methods = [name for name, member in inspect.getmembers(self) if inspect.isroutine(member)]
    print("All methods:", all_methods)









