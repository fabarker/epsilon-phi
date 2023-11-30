from epsilonPhi.core.estimator.assetEstimatorInf import CAssetRiskEstimatorInf
from epsilonPhi.core.config.configUtil import CAppConfig
import numpy as np
import math

class AssetRiskEstimator(CAssetRiskEstimatorInf):
    def __init__(self):
        super(AssetRiskEstimator, self).__init__()

    @staticmethod
    def get_beta_and_idio_variance(asset, hedging_ratio):

        schema = asset.schema
        schema_currency = schema.currency
        asset_in_schema_currency = asset.convert_asset_to_currency(schema_currency, hedging_ratio)

        model = CAppConfig.get_BaseModel()
        factor_df = asset.schema.get_risk_factors_panel()

        rx = asset_in_schema_currency.get_excess_return_df()
        y, X = rx.intersect_over_dates(factor_df)
        regstats = model.regression.regress(X,
                                            y,
                                            orthogonalize_columns=model.orthogonal_list,
                                            normalize=False)

        betas = regstats[:, 1:]
        residuals = y - model.regression.orthogonalize_columns(X, model.orthogonal_list) @ np.mean(betas, axis=0)
        return np.mean(betas, axis=0), np.var(residuals, ddof=1) * schema.obs_per_year

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



if __name__ == "__main__":

    from epsilonPhi.core.schema.Schema import ContextCreator
    from epsilonPhi.core.asset.AssetMgr import CAssetMgr

    schema = ContextCreator(currency='GBP',
                            start_date='31-Dec-1999',
                            end_date='31-Dec-2022').create_context()

    assetMgr = CAssetMgr(schema)

    asset = assetMgr.get_asset_by_name('S&PCOMP')
    ER = AssetRiskEstimator.get_historical_volatility(asset)






