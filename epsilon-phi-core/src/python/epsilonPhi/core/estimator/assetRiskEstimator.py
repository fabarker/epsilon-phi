import numpy as np
import math

from epsilonPhi.core.estimator.assetEstimatorInf import CAssetRiskEstimatorInf

class AssetRiskEstimator(CAssetRiskEstimatorInf):
    def __init__(self):
        super(AssetRiskEstimator, self).__init__()

    @staticmethod
    def get_beta_and_idio_variance(asset, hedging_ratio):

        schema = asset.schema
        schema_currency = schema.currency

        hedged_name, unhedged_name = asset.get_risk_mapping()
        hedged_asset = schema.get_asset_mgr().get_asset_by_name(hedged_name)
        unhedged_asset = schema.get_asset_mgr().get_asset_by_name(unhedged_name)

        hedged_betas, hedged_idio = hedged_asset.get_risk_betas_no_hedge()
        unhdgd_betas, unhdgd_idio = unhedged_asset.get_risk_betas_no_hedge()

        asset.set_hedging_ratio(hedging_ratio)
        betas, idio_var = asset.get_risk_betas_no_hedge()
        return betas, idio_var

    @staticmethod
    def get_risk_factor_stdev(asset):

        factor_covar_mat = asset.schema.get_risk_factor_covar_matrix()
        betas, idio_var = asset.get_beta_and_idio_var(asset.get_hedging_ratio)
        return math.sqrt(np.matmul(np.matmul(betas.cong().T, factor_covar_mat.vallues), betas) + idio_var)

    @staticmethod
    def get_risk_factor_betas_no_hedge(asset):
        return asset.schema.get_factor_panels().get_risk_factor_betas_no_hedge(asset)

    @staticmethod
    def get_risk_betas(asset, hedging_ratio):
        betas, _ = AssetRiskEstimator.get_beta_and_idio_variance(asset, hedging_ratio)
        return betas

    @staticmethod
    def get_idiosyncratic_variance(asset, hedging_ratio):
        _, idio = AssetRiskEstimator.get_beta_and_idio_variance(asset, hedging_ratio)
        return idio

    @staticmethod
    def get_historical_volatility(asset):
        return asset.std() * math.sqrt(asset.schema.annualizing_factor)

    @staticmethod
    def get_historical_max_drawdown(asset):
        pass

    @staticmethod
    def get_historical_value_at_risk(asset):
        pass

    @staticmethod
    def get_historical_crisis_period_performance(asset):
        pass

    @staticmethod
    def get_historical_beta_to_equity_market(asset):
        pass






