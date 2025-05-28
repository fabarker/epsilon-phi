from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType
from epsilonPhi.core.asset.Asset import CAsset
from epsilonPhi.core.schema.Schema import CContext
from typing import Optional


gds = GlobalDataSource()

__all__ = ['CTacticalTrading']

class CTacticalTrading(CAsset):
    _ASSET_NAME = 'CSFBMTT'
    _MF = 'CSTMNFH'
    _GM = 'CSTGLMH'

    def __init__(
            self,
            schema: Optional[CContext] = None,
    ) -> None:

        data = self.get_time_series(schema)
        pars = CTacticalTrading.get_time_series_params()
        super(CTacticalTrading, self).__init__(
            data,
            schema,
            **pars)

    @staticmethod
    def get_time_series(schema):
        mf = CTacticalTrading.get_managed_futures_time_series(schema)
        gm = CTacticalTrading.get_global_macro_time_series(schema)
        res = 0.5 * mf.addition_over_common_dates(gm)
        res.name = (CTacticalTrading._ASSET_NAME, 'RI')
        return res

    @staticmethod
    def get_asset_from_name(name, schema):
        from epsilonPhi.core.asset.AssetMgr import CAssetMgr
        return CAssetMgr(schema).get_asset_by_name(name)

    @staticmethod
    def get_managed_futures_time_series(schema):
        return CTacticalTrading.get_asset_from_name(
            CTacticalTrading._MF,
            schema
        )

    @staticmethod
    def get_global_macro_time_series(schema):
        return CTacticalTrading.get_asset_from_name(
            CTacticalTrading._GM,
            schema
        )

    @staticmethod
    def get_time_series_params():
        return {
            'denominated_currency':'USD',
            'exposure_currency':'USD',
            'ts_hedge_ratio': 0.0,
            'returns_type': ReturnsType.SIMPLE
        }


if __name__ == "__main__":

    from epsilonPhi.core.asset.AssetMgr import CAssetMgr
    from epsilonPhi.core.schema.Schema import ContextCreator
    schema = ContextCreator(
        currency='USD',
        start_date='30-Nov-1983',
        end_date='31-Dec-2022'
    ).create_context()

    tt = CTacticalTrading(schema)

