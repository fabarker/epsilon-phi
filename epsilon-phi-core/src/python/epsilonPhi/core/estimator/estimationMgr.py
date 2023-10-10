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
    def get_beta_and_idio_variance(asset, hedging_ratio):
        from epsilonPhi.core.estimator.assetRiskEstimator import AssetRiskEstimator
        return AssetRiskEstimator.get_risk_betas(asset, hedging_ratio)

    @staticmethod
    def get_risk_factor_stdev(asset, hedging_ratio):
        from epsilonPhi.core.estimator.assetRiskEstimator import AssetRiskEstimator
        return AssetRiskEstimator.get_risk_betas(asset, hedging_ratio)

    @staticmethod
    def get_historical_max_drawdown(asset, hedging_ratio):
        from epsilonPhi.core.estimator.assetRiskEstimator import AssetRiskEstimator
        return AssetRiskEstimator.get_risk_betas(asset, hedging_ratio)

    @staticmethod
    def get_historical_value_at_risk(asset, hedging_ratio):
        from epsilonPhi.core.estimator.assetRiskEstimator import AssetRiskEstimator
        return AssetRiskEstimator.get_risk_betas(asset, hedging_ratio)

    @staticmethod
    def get_historical_crisis_period_performance(asset, hedging_ratio):
        from epsilonPhi.core.estimator.assetRiskEstimator import AssetRiskEstimator
        return AssetRiskEstimator.get_risk_betas(asset, hedging_ratio)

    @staticmethod
    def get_historical_beta_to_equity(asset, hedging_ratio):
        from epsilonPhi.core.estimator.assetRiskEstimator import AssetRiskEstimator
        return AssetRiskEstimator.get_risk_betas(asset, hedging_ratio)

    @staticmethod
    def get_historical_Sharpe_ratio(asset):
        from epsilonPhi.core.estimator.assetReturnEstimator import AssetReturnEstimator
        return AssetReturnEstimator.get_historical_Sharpe_ratio(asset)

    @staticmethod
    def get_historical_risk_premia(asset):
        from epsilonPhi.core.estimator.assetReturnEstimator import AssetReturnEstimator
        return AssetReturnEstimator.get_historical_risk_premia(asset)

    @staticmethod
    def get_historical_volatility(asset):
        from epsilonPhi.core.estimator.assetRiskEstimator import AssetRiskEstimator
        return AssetRiskEstimator.get_historical_volatility(asset)

    @staticmethod
    def get_risk_premia_with_hedging(asset, hedge_ratio):
        from epsilonPhi.core.estimator.assetReturnEstimator import AssetReturnEstimator
        return AssetReturnEstimator.get_risk_premia_with_hedging(asset, hedge_ratio)

    @staticmethod
    def set_factor_panels(factor_panels):
        EstimationMgr._factor_panel = factor_panels




