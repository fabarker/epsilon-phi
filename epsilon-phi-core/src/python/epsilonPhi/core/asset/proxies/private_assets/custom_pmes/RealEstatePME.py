from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType
from epsilonPhi.core.asset.Asset import CAsset
from epsilonPhi.core.schema.Schema import CContext
from typing import Optional
import pandas as pd

gds = GlobalDataSource()

__all__ = ['RealEstatePME']

class RealEstatePME(CAsset):
    _asset_name = 'REAL_ESTATE_PME'

    _reporting_name = 'Private Real Estate PME'
    _category = 'Public Equity'

    def __init__(
            self,
            schema: Optional[CContext] = None,
    ) -> None:

        self._schema = schema
        data = self.get_time_series()
        super(RealEstatePME, self).__init__(
            data,
            schema,
            **RealEstatePME.get_time_series_params())

    def get_time_series(self):
        glob = self.get_global_series()
        us = self.get_us_series()

        # Ensure consistent columns for backfilling
        us.columns = glob.columns

        # Backfill and resample to desired frequency, then compute returns
        filled = glob.backfill_series(us)
        resampled = filled.resample(self._schema.frequency.value).asfreq()
        returns = resampled.get_returns()

        # Reindex to match schema dates and validate
        reindexed = returns.reindex(self._schema.dates)

        if reindexed.isna().any().item():
            raise ValueError(f"NaNs found in asset '{self._asset_name}' after reindexing. Check source data integrity.")

        return reindexed

    def get_global_series(self):
        return gds.get_time_series_data_from_ticker('SBBRGLL', cols='RI')

    def get_us_series(self):
        return gds.get_time_series_data_from_ticker('NAREQR$', cols='RI')

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

    tt = RealEstatePME(schema)

