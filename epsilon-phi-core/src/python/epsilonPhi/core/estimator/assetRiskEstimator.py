from epsilonPhi.core.estimator.assetEstimatorInf import CAssetRiskEstimatorInf
from epsilonPhi.core.config.configUtil import CAppConfig
import numpy as np
import math

class AssetRiskEstimator(CAssetRiskEstimatorInf):
    _cache = {}

    def __init__(self):
        super(AssetRiskEstimator, self).__init__()


    @staticmethod
    def calc_betas_and_idio_variance(asset, hedging_ratio):

        # Get the asset in schema currency with correct hedging proportions
        schema_currency = asset.schema.currency
        asset_in_schema_currency = asset.convert_asset_to_currency(
            schema_currency,
            hedging_ratio
        )

        # convert asset to factor / excess return
        rx = asset_in_schema_currency.get_excess_return_df()

        # get the unorthogonalized risk factor panel
        factor_df = asset.schema.get_risk_factors_panel()

        # Intersect over dates before running orthogonalization
        y, X = rx.intersect_over_date_range(factor_df)
        assert np.all(y.index == X.index), 'Error - date mismatch in regression'

        # Orthogonalize the factor panel
        model = CAppConfig.get_BaseModel()
        orth_factor_df = model.regression.orthogonalize_columns(
            X,
            model.orthogonal_list
        )

        # Run simple OLS regression with array (Linear, one-shot with intercept)
        alpha, betas = model.regression.simple_regression_OLS_with_array(
            orth_factor_df, y
        )


        residuals = y - orth_factor_df @ betas
        return betas, np.var(residuals, ddof=1) * asset.schema.obs_per_year, residuals

    @staticmethod
    def get_beta_and_idio_variance(asset, hedging_ratio):
        if (asset.name, hedging_ratio) not in AssetRiskEstimator._cache:
            AssetRiskEstimator._cache[(asset.name, hedging_ratio)] = AssetRiskEstimator.calc_betas_and_idio_variance(asset, hedging_ratio)
        betas, idio, _ = AssetRiskEstimator._cache[(asset.name, hedging_ratio)]
        return betas, idio

    @staticmethod
    def get_residuals(asset, hedging_ratio):
        if (asset.name, hedging_ratio) not in AssetRiskEstimator._cache:
            AssetRiskEstimator._cache[(asset.name, hedging_ratio)] = AssetRiskEstimator.calc_betas_and_idio_variance(asset, hedging_ratio)
        _, _, residuals = AssetRiskEstimator._cache[(asset.name, hedging_ratio)]
        return residuals

    @staticmethod
    def get_risk_factor_stdev(asset, hedging_ratio):

        betas, idio = AssetRiskEstimator.get_beta_and_idio_variance(asset, hedging_ratio)

        factor_covariance = asset.schema.get_risk_factor_covariance()
        systematic_var = betas @ factor_covariance @ betas
        return np.sqrt(systematic_var + idio)


    @staticmethod
    def get_betas_and_idio_risk(asset, hedging_ratio):
        return AssetRiskEstimator.get_beta_and_idio_variance(asset, hedging_ratio)

    @staticmethod
    def get_risk_betas(asset, hedging_ratio):
        betas, _ = AssetRiskEstimator.get_beta_and_idio_variance(asset, hedging_ratio)
        return betas

    @staticmethod
    def get_idiosyncratic_variance(asset, hedging_ratio):
        _, idio = AssetRiskEstimator.get_beta_and_idio_variance(asset, hedging_ratio)
        return idio

    @staticmethod
    def get_systematic_variance(asset, hedging_ratio):
        betas, idio = AssetRiskEstimator.get_beta_and_idio_variance(asset, hedging_ratio)

        factor_covariance = asset.schema.get_risk_factor_covariance()
        systematic_var = betas @ factor_covariance @ betas
        return systematic_var



if __name__ == "__main__":

    from epsilonPhi.core.schema.Schema import ContextCreator
    from epsilonPhi.core.asset.AssetMgr import CAssetMgr

    schema = ContextCreator(currency='GBP',
                            start_date='31-Dec-1999',
                            end_date='31-Dec-2022').create_context()

    assetMgr = CAssetMgr(schema)

    asset = assetMgr.get_asset_by_name('S&PCOMP')
    ER = AssetRiskEstimator.get_historical_volatility(asset)






