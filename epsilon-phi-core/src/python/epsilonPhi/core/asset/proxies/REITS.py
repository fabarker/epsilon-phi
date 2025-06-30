from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType
from epsilonPhi.core.timeSeries.timeSeriesMain import CSlice, CTimeSeries
from epsilonPhi.core.asset.Asset import CAsset
from epsilonPhi.core.estimator.estimationMgr import EstimationMgr
from epsilonPhi.core.config.appConfig import CAppConfig
from epsilonPhi.core.asset.AssetMgr import CAssetMgr
from epsilonPhi.core.schema.Schema import CContext
from typing import Optional, Union
import numpy as np

gds = GlobalDataSource()

__all__ = ['CREITs']


class CREITs(CAsset):
    _asset_name = 'GLOBAL_REITS'
    _underlier = 'SBBRGLL'

    _reporting_name = 'Global REITs'
    _category = 'Public Equity'

    def __init__(
            self,
            schema: Optional[CContext] = None,
    ) -> None:

        data = self.get_time_series(schema)
        pars = CREITs.get_time_series_params()
        super(CREITs, self).__init__(
            data,
            schema,
            **pars)


    @staticmethod
    def get_time_series(schema):
        asset = CREITs.get_asset_from_name(
            CREITs._underlier,
            schema
        )

        mkt = schema.get_return_factors_panel().get("EQUITY_GLOBAL_ISG")
        er = asset.get_excess_return_df()
        rfr = asset - er
        y, X = er.intersect_over_dates(mkt)

        # Add intercept column to X
        X = np.column_stack((np.ones_like(X), X))  # shape: (n_samples, 2)
        _, beta = np.linalg.inv(X.T @ X) @ (X.T @ y)  # Final coefficients

        asset = ((y - beta * 0.22 * X[:, 1]) + rfr).dropna()
        asset.name = (CREITs._asset_name, 'RI')
        return CTimeSeries(asset.to_frame(asset.name))

    @staticmethod
    def get_asset_from_name(name, schema):
        from epsilonPhi.core.asset.AssetMgr import CAssetMgr
        return CAssetMgr(schema).get_asset_by_name(name)

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

    tt = CREITs(schema)
    tt.get_rolling_return_betas()