from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType
from epsilonPhi.core.asset.Asset import CAsset
from epsilonPhi.core.schema.Schema import CContext
from typing import Optional, Union

gds = GlobalDataSource()

__all__ = ['CBankLoans']

class CBankLoans(CAsset):
    _asset_name = 'CSLEVLOANS'
    _RP_TARGET  = 0.011

    def __init__(
            self,
            schema: Optional[CContext] = None,
            **kwargs
    ) -> None:

        data = self.get_time_series()
        super().__init__(data, schema, **kwargs)
        self._set_asset_alpha()

    @staticmethod
    def get_time_series():
        return gds.get_total_return_series_from_ticker(
            CBankLoans._ASSET_NAME,
            TimeSeriesType.RETURNS
        )

    @staticmethod
    def get_time_series_params():
        return {
            'denominated_currency':'USD',
            'exposure_currency':'USD',
            'ts_hedge_ratio': 0.0,
            'returns_type': ReturnsType.SIMPLE
        }

    def _set_asset_alpha(self):
        self.set_alpha(CBankLoans._RP_TARGET - self.get_risk_premia().sum())
