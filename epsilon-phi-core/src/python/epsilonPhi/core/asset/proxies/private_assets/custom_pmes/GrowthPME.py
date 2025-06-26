from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType
from epsilonPhi.core.asset.Asset import CAsset
from epsilonPhi.core.schema.Schema import CContext
from typing import Optional

gds = GlobalDataSource()

__all__ = ['GrowthPME']


class GrowthPME(CAsset):
    _asset_name = 'GROWTH_PME'
    _US_SCG = 'FRUS2GR'
    _GL_ACG = 'MSGWLD$'

    WT = 0.4

    _reporting_name = 'Growth PME'
    _category = 'Public Equity'

    def __init__(
            self,
            schema: Optional[CContext] = None,
    ) -> None:
        self._schema = schema
        data = self.get_time_series()
        super(GrowthPME, self).__init__(
            data,
            schema,
            **GrowthPME.get_time_series_params())

    def get_time_series(self):
        US_G = self.WT * self.get_asset_from_name(self._US_SCG)
        WL_G = (1-self.WT) * self.get_asset_from_name(self._GL_ACG)
        res = US_G.addition_over_common_dates(WL_G)
        res.name = (GrowthPME._asset_name, 'RI')
        return res

    def get_asset_from_name(self, name):
        return self._schema.get_asset_from_name(name)

    @staticmethod
    def get_time_series_params():
        return {
            'denominated_currency': 'USD',
            'exposure_currency': 'USD',
            'ts_hedge_ratio': 0.0,
            'returns_type': ReturnsType.SIMPLE,
            'ts_type': TimeSeriesType.RETURNS
        }


if __name__ == "__main__":

    from epsilonPhi.core.schema.Schema import ContextCreator

    schema = ContextCreator(
        currency='USD',
        start_date='30-Nov-1983',
        end_date='31-Dec-2022'
    ).create_context()

    tt = GrowthPME(schema)
