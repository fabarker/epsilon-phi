from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType
from epsilonPhi.core.timeSeries.timeSeriesMain import CSlice, CTimeSeries
from epsilonPhi.core.asset.Asset import CAsset
from epsilonPhi.core.estimator.estimationMgr import EstimationMgr
from epsilonPhi.core.config.appConfig import CAppConfig
from epsilonPhi.core.asset.AssetMgr import CAssetMgr
from epsilonPhi.core.schema.Schema import CContext
from typing import Optional, Union

gds = GlobalDataSource()

__all__ = ['CInfra']


class CInfra(CAsset):
    _asset_name = 'INFRA_EQUITY'
    _PROXIES = [('ACEMLPIT', .50), ('MSWDIF', .50)]

    _infra = 'MSWDIFL'
    _mlp = 'ACEMLPT'

    def __init__(
            self,
            schema: Optional[CContext] = None,
            **kwargs
    ) -> None:

        data = self.get_time_series(schema)
        pars = CInfra.get_time_series_params()
        super(CInfra, self).__init__(
            data,
            schema,
            **pars)

    @staticmethod
    def get_time_series(schema):
        mf = CInfra.get_MLP_time_series(schema)
        gm = CInfra.get_infra_time_series(schema)
        res = 0.5 * mf.addition_over_common_dates(gm)
        res.name = (CInfra._asset_name, 'RI')
        return res

    @staticmethod
    def get_asset_from_name(name, schema):
        from epsilonPhi.core.asset.AssetMgr import CAssetMgr
        return CAssetMgr(schema).get_asset_by_name(name)

    @staticmethod
    def get_MLP_time_series(schema):
        return CInfra.get_asset_from_name(
            CInfra._mlp,
            schema
        )

    @staticmethod
    def get_infra_time_series(schema):
        return CInfra.get_asset_from_name(
            CInfra._infra,
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

    tt = CInfra(schema)