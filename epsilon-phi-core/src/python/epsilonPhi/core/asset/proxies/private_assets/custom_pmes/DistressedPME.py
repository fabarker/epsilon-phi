from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType
from epsilonPhi.core.asset.Asset import CAsset
from epsilonPhi.core.schema.Schema import CContext
from typing import Optional

gds = GlobalDataSource()

__all__ = ['DistressedPME']

class DistressedPME(CAsset):
    _asset_name = 'DISTRESSED_PME'
    _hf = 'CSTEVDH' # Event Driven
    _eq = 'MSWRLD$' # MSCI World

    _reporting_name = 'Distressed PME'
    _category = 'Public Equity'

    def __init__(
            self,
            schema: Optional[CContext] = None,
    ) -> None:

        self._schema = schema
        data = self.get_time_series()
        super(DistressedPME, self).__init__(
            data,
            schema,
            **DistressedPME.get_time_series_params())

    def get_time_series(self):
        hf = self.get_asset_from_name(self._hf)
        eq = self.get_asset_from_name(self._eq)
        res = 0.5 * hf.addition_over_common_dates(eq)
        res.name = (DistressedPME._asset_name, 'RI')
        return res

    def get_asset_from_name(self, name):
        return self._schema.get_asset_from_name(name)

    @staticmethod
    def get_time_series_params():
        return {
            'denominated_currency':'USD',
            'exposure_currency':'USD',
            'ts_hedge_ratio': 0.0,
            'returns_type': ReturnsType.SIMPLE,
            'ts_type': TimeSeriesType.RETURNS
        }


if __name__ == "__main__":

    from epsilonPhi.core.asset.AssetMgr import CAssetMgr
    from epsilonPhi.core.schema.Schema import ContextCreator
    schema = ContextCreator(
        currency='USD',
        start_date='30-Nov-1983',
        end_date='31-Dec-2022'
    ).create_context()

    tt = DistressedPME(schema)

