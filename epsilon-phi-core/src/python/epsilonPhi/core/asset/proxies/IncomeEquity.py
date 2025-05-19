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

class CIncomeEquity(CAsset):
    _ASSET_NAME = 'INCOME_EQUITY'
    _PROXIES = [('ACEMLPIT', .50), ('MSWDIF', .50)]

    def __init__(
            self,
            schema: Optional[CContext] = None,
            **kwargs
    ) -> None:

        data = self.get_time_series()
        super(CIncomeEquity, self).__init__(data, schema, **kwargs)
        self._set_asset_alpha()

    @staticmethod
    def get_time_series():
        pass

    @staticmethod
    def get_time_series_params():
        return {
            'denominated_currency': 'USD',
            'exposure_currency': 'USD',
            'ts_hedge_ratio': 0.0,
            'returns_type': ReturnsType.SIMPLE
        }