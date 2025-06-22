from epsilonPhi.core.asset.Asset import CAsset
from epsilonPhi.core.schema.Schema import CContext
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType
from epsilonPhi.core.timeSeries.timeSeriesMain import CSlice, CTimeSeries
from epsilonPhi.core.estimator.estimationMgr import EstimationMgr
from epsilonPhi.core.config.appConfig import CAppConfig
from typing import Optional
from typing import Optional, Union
import numpy as np
import math

__all__ = ['CSingleStock']


class CSingleStock(CAsset):
    _category = 'Concentrated Equity Position'

    def __init__(
            self,
            schema,
            single_stock_time_series: Union[CTimeSeries, CSlice],
            denominated_currency: str,
            exposure_currency: str,
            ts_hedge_ratio: Union[float, int],
            market_equivalent: str,
    ) -> None:
        super(CSingleStock, self).__init__(
            single_stock_time_series,
            schema,
            denominated_currency,
            exposure_currency,
            ts_hedge_ratio,
            single_stock_time_series.returns_type,
            single_stock_time_series.type
        )

        # single stocks are unhedged
        super(CSingleStock, self).set_currency_hedge_ratio(0)

        # set the market equivalent
        self._mkt = schema.get_asset_from_name(market_equivalent)
        self._mkt.set_currency_hedge_ratio(0)

        # idio var is the total vol
        self._idio_var = None
        self._cache_asset()

    def _cache_asset(self):
        CAssetMgr(self.schema).add_asset_to_cache(self)

    def set_currency_hedge_ratio(self, hedge_ratio: Optional[Union[float, int]]):
        raise ValueError("Method not supported for single stocks")

    def set_idio_vol(self, idio_vol):
        self._idio_var = np.power(idio_vol, 2)

    def set_idio_var(self, idio_var):
        self._idio_var = idio_var

    def get_volatility(self, hedging_ratio=None):
        if self._idio_var is None:
            self._idio_var = self.get_historical_volatility() ** 2
        return np.sqrt(self._idio_var)

    def get_beta_and_idio_risk(self, hedging_ratio=None):
        # Get the betas from the index
        betas, _ = self._mkt.get_beta_and_idio_risk(hedging_ratio)
        idio = max(0, np.power(self.get_volatility(hedging_ratio), 2) - self.get_systematic_variance(hedging_ratio))
        return betas, idio

    def get_data_length(self):
        return self._ss.get_data_length()

    def get_risk_premia(self):
        return self._mkt.get_risk_premia()

    def get_risk_premias(self):
        return self._mkt.get_risk_premias()

    def get_stressed_factor_based_risk_premium(self):
        raise ValueError('Error - not supported for single stocks')

    def get_stressed_factor_based_returns(self):
        raise ValueError('Error - not supported for single stocks')

    def get_factor_backfilled_returns(self):
        raise ValueError('Error - not supported for single stocks')

    def get_risk_premias_in_current_environment(self):
        raise ValueError('Error - not supported for single stocks')

    def get_risk_premia_in_current_environment(self):
        raise ValueError('Error - not supported for single stocks')

    def get_return_betas(self, hedging_ratio=None, normalized=True):
        return self._mkt.get_return_betas(hedging_ratio, normalized)

    def get_rolling_return_betas(self, hedging_ratio=None, normalized=True):
        return self._mkt.get_rolling_return_betas(hedging_ratio, normalized)

    def get_risk_betas(self):
        return self._mkt.get_risk_betas()

    def get_rsq(self):
        return self.get_systematic_variance() / (self.get_systematic_variance() + self.get_idiosyncratic_variance())

    def get_single_stock_return_betas(self):
        return self.get_single_stock_rolling_return_betas().mean(axis=0)

    def get_single_stock_rolling_return_betas(self):
        return EstimationMgr.get_return_betas(self, self.hedging_ratio, True)

    def get_single_stock_risk_premias(self):
        hist_sharpe = CAppConfig.get_BaseModel().get_return_factor_Sharpe_ratios()
        betas = self.get_single_stock_return_betas().values
        return betas * hist_sharpe.values.T * math.sqrt(self.schema.obs_per_year)

    def get_single_stock_risk_premia(self):
        return self.get_single_stock_risk_premias().sum()

    def simulate(self):
        raise ValueError('Method not supported for single stocks')


if __name__ == "__main__":
    from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource

    gds = GlobalDataSource()

    ts = gds.get_time_series_data_from_ticker('U:JNJ', cols='RI', ts_type=TimeSeriesType.LEVELS)
    ts = ts.get_returns()

    from epsilonPhi.core.asset.AssetMgr import CAssetMgr
    from epsilonPhi.core.schema.Schema import ContextCreator

    schema = ContextCreator(
        currency='USD',
        start_date='30-Nov-1983',
        end_date='31-Dec-2022'
    ).create_context()

    mkt_equi = "MSEXUKL"

    tt = CSingleStock(
        schema,
        ts,
        'USD',
        'USD',
        0,
        mkt_equi
    )

    risk = tt.get_single_stock_risk_premia()
