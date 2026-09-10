from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType
from epsilonPhi.core.asset.Asset import CAsset
from epsilonPhi.core.schema.Schema import CContext
from typing import Optional
import pandas as pd

gds = GlobalDataSource()

__all__ = ['VenturePME']


class VenturePME(CAsset):
    _asset_name = 'VENTURE_PME'
    _underlier = 'NASCOMP'  # Event Driven

    _reporting_name = 'Venture PME'
    _category = 'Public Equity'

    def __init__(
            self,
            schema: Optional[CContext] = None,
    ) -> None:
        self._schema = schema
        data = self.get_time_series()
        super(VenturePME, self).__init__(
            data,
            schema,
            **VenturePME.get_time_series_params())

    def get_time_series(self):
        ts = gds.get_time_series_data_from_ticker(self._underlier)

        ri = ts.get((self._underlier, 'RI')).dropna()
        pi = ts.get((self._underlier, 'PI')).dropna()
        pi.name = (self._underlier, 'RI')
        return ri.backfill_series(pi).reindex(self._schema.dates).get_returns()

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
    from epsilonPhi.core.asset.AssetMgr import CAssetMgr
    from epsilonPhi.core.schema.Schema import ContextCreator

    schema = ContextCreator(
        currency='USD',
        start_date='30-Nov-1983',
        end_date='31-Dec-2022'
    ).create_context()

    tt = VenturePME(schema)
