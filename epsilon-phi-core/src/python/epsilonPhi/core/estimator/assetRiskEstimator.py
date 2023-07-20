from epsilonPhi.core.estimator.assetEstimatorInf import CAssetRiskEstimatorInf

class AssetRiskEstimator(CAssetRiskEstimatorInf):
    def __init__(self):
        super(AssetRiskEstimator, self).__init__()

    @staticmethod
    def get_risk_betas(asset, hedging_ratio):
        pass

    @staticmethod
    def get_idiosyncratic_variance(asset, hedging_ratio):
        pass
