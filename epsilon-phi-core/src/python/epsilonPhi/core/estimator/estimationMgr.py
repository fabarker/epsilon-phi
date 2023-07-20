from epsilonPhi.core.estimator.estimationMgrInf import CEStimationMgrInf

class EstimationMgr(CEStimationMgrInf):

    @staticmethod
    def get_risk_premium(asset):
        from epsilonPhi.core.estimator.assetReturnEstimator import AssetReturnEstimator
        pass

    @staticmethod
    def get_excess_return_timeseries(asset):
        from epsilonPhi.core.estimator.assetReturnEstimator import AssetReturnEstimator
        pass

    @staticmethod
    def get_return_betas(asset, normalized=True):
        from epsilonPhi.core.estimator.assetReturnEstimator import AssetReturnEstimator
        pass

    @staticmethod
    def get_idiosyncratic_variance(asset, hedging_ratio):
        from epsilonPhi.core.estimator.assetRiskEstimator import AssetRiskEstimator
        pass

    @staticmethod
    def get_risk_betas(asset, hedging_ratio):
        from epsilonPhi.core.estimator.assetRiskEstimator import AssetRiskEstimator
        pass