from epsilonPhi.core.estimator.estimationMgrInf import CEStimationMgrInf

class EstimationMgr(CEStimationMgrInf):

    @staticmethod
    def get_risk_premium(asset):
        from epsilonPhi.core.estimator.assetReturnEstimator import AssetReturnEstimator
        return AssetReturnEstimator.get_risk_premium(asset)

    @staticmethod
    def get_excess_return_timeseries(asset):
        from epsilonPhi.core.estimator.assetReturnEstimator import AssetReturnEstimator
        return AssetReturnEstimator.get_excess_return_timeseries(asset)

    @staticmethod
    def get_return_betas(asset, normalized=True):
        from epsilonPhi.core.estimator.assetReturnEstimator import AssetReturnEstimator
        return AssetReturnEstimator.get_return_betas(asset, normalized)

    @staticmethod
    def get_idiosyncratic_variance(asset, hedging_ratio):
        from epsilonPhi.core.estimator.assetRiskEstimator import AssetRiskEstimator
        return AssetRiskEstimator.get_idiosyncratic_variance(asset, hedging_ratio)

    @staticmethod
    def get_risk_betas(asset, hedging_ratio):
        from epsilonPhi.core.estimator.assetRiskEstimator import AssetRiskEstimator
        return AssetRiskEstimator.get_risk_betas(asset, hedging_ratio)

    @staticmethod
    def set_factor_panels(factor_panels):
        EstimationMgr._factor_panel = factor_panels


