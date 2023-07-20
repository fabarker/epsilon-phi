from epsilonPhi.core.estimator.assetEstimatorInf import CAssetReturnEstimatorInf


class AssetReturnEstimator(CAssetReturnEstimatorInf):
    def __init__(self):
        super(AssetReturnEstimator, self).__init__()

    @staticmethod
    def get_risk_premium(asset):
        pass

    @staticmethod
    def get_excess_return_timeseries(asset):
        pass

    @staticmethod
    def get_return_betas(asset, normalized=True):
        pass
