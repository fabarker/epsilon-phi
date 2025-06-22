import pandas as pd

from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType
from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries
from epsilonPhi.core.asset.Asset import CAsset
from epsilonPhi.core.schema.Schema import CContext
from epsilonPhi.core.lib.smoothing.Unsmoothing import Geltner
from typing import Optional


gds = GlobalDataSource()

__all__ = ['CUSHighYield']
class CUSHighYield(CAsset):
    _asset_name = 'LHYIELD_GE20'
    _underlier = 'LHYIELD'

    _reporting_name = 'US High Yield'
    _category = 'Other Fixed Income'

    def __init__(
            self,
            schema: Optional[CContext] = None,
    ) -> None:

        data = self.get_time_series(schema)
        pars = CUSHighYield.get_time_series_params()
        super(CUSHighYield, self).__init__(
            data,
            schema,
            **pars)

    @staticmethod
    def get_time_series(schema):
        asset = CUSHighYield.get_asset_from_name(
             CUSHighYield._underlier,
             schema
        )

        res = Geltner.unsmooth_returns(asset, 0.0)
        res.index = asset.index[1:]
        res.name = (CUSHighYield._asset_name, 'IN')
        return CTimeSeries(res.to_frame(res.name))


    @staticmethod
    def get_asset_from_name(name, schema):
        from epsilonPhi.core.asset.AssetMgr import CAssetMgr
        return CAssetMgr(schema).get_asset_by_name(name)

    @staticmethod
    def get_time_series_params():
        return {
            'denominated_currency':'USD',
            'exposure_currency':'USD',
            'ts_hedge_ratio': 0.0,
            'returns_type': ReturnsType.SIMPLE,
            'ts_type': TimeSeriesType.RETURNS,
        }


if __name__ == "__main__":

    from epsilonPhi.core.asset.AssetMgr import CAssetMgr
    from epsilonPhi.core.schema.Schema import ContextCreator
    schema = ContextCreator(
        currency='USD',
        start_date='30-Nov-1983',
        end_date='31-Dec-2022'
    ).create_context()

    tt = CUSHighYield(schema)

