from epsilonPhi.core.estimator.estimationMgrInf import CEStimationMgrInf
import numpy as np

class EstimationMgr(CEStimationMgrInf):

    @staticmethod
    def get_risk_premium(asset):
        from epsilonPhi.core.estimator.assetReturnEstimator import AssetReturnEstimator
        return AssetReturnEstimator.get_risk_premium(asset)

    @staticmethod
    def get_estimation_length(asset):
        from epsilonPhi.core.estimator.assetReturnEstimator import AssetReturnEstimator
        return AssetReturnEstimator.get_estimation_length(asset)

    @staticmethod
    def get_excess_return_timeseries(asset):
        from epsilonPhi.core.estimator.assetReturnEstimator import AssetReturnEstimator
        return AssetReturnEstimator.get_excess_return_timeseries(asset)

    @staticmethod
    def get_return_betas(asset, hedging_ratio, normalized=True):
        from epsilonPhi.core.estimator.assetReturnEstimator import AssetReturnEstimator
        return AssetReturnEstimator.get_return_betas(asset, hedging_ratio, normalized)

    ######################

    @staticmethod
    def get_idiosyncratic_variance(asset, hedging_ratio):
        from epsilonPhi.core.estimator.assetRiskEstimator import AssetRiskEstimator
        return AssetRiskEstimator.get_idiosyncratic_variance(asset, hedging_ratio)

    @staticmethod
    def get_systematic_variance(asset, hedging_ratio):
        from epsilonPhi.core.estimator.assetRiskEstimator import AssetRiskEstimator
        return AssetRiskEstimator.get_systematic_variance(asset, hedging_ratio)

    @staticmethod
    def get_residuals(asset, hedging_ratio):
        from epsilonPhi.core.estimator.assetRiskEstimator import AssetRiskEstimator
        return AssetRiskEstimator.get_residuals(asset, hedging_ratio)

    @staticmethod
    def get_risk_betas(asset, hedging_ratio):
        from epsilonPhi.core.estimator.assetRiskEstimator import AssetRiskEstimator
        return AssetRiskEstimator.get_risk_betas(asset, hedging_ratio)

    @staticmethod
    def get_beta_and_idio_variance(asset, hedging_ratio):
        from epsilonPhi.core.estimator.assetRiskEstimator import AssetRiskEstimator
        return AssetRiskEstimator.get_betas_and_idio_risk(asset, hedging_ratio)

    @staticmethod
    def get_risk_factor_stdev(asset, hedging_ratio):
        from epsilonPhi.core.estimator.assetRiskEstimator import AssetRiskEstimator
        return AssetRiskEstimator.get_risk_factor_stdev(asset, hedging_ratio)



    ##################

    @staticmethod
    def get_historical_total_return(asset):
        from epsilonPhi.core.estimator.tseriesEstimator import CTimeSeriesEstimator
        return CTimeSeriesEstimator(asset, asset.returns_type, asset.type).item()

    @staticmethod
    def get_historical_excess_return_df(asset):
        from epsilonPhi.core.estimator.tseriesEstimator import CTimeSeriesEstimator
        return CTimeSeriesEstimator(asset, asset.returns_type, asset.type).excess_return_series(asset.denominated_currency)

    @staticmethod
    def get_historical_risk_premium(asset):
        from epsilonPhi.core.estimator.tseriesEstimator import CTimeSeriesEstimator
        return CTimeSeriesEstimator(asset, asset.returns_type, asset.type).risk_premium(asset.denominated_currency).item()

    @staticmethod
    def get_historical_volatility(asset):
        from epsilonPhi.core.estimator.tseriesEstimator import CTimeSeriesEstimator
        return CTimeSeriesEstimator(asset, asset.returns_type, asset.type).volatility().item()


    @staticmethod
    def get_historical_sharpe_ratio(asset):
        from epsilonPhi.core.estimator.tseriesEstimator import CTimeSeriesEstimator
        return CTimeSeriesEstimator(asset, asset.returns_type, asset.type).sharpe_ratio(asset.denominated_currency).item()


    @staticmethod
    def get_historical_value_at_risk(asset, confidence=0.99, horizon=1):
        from epsilonPhi.core.estimator.tseriesEstimator import CTimeSeriesEstimator
        return CTimeSeriesEstimator(asset, asset.returns_type, asset.type).var(
            confidence_level=confidence,
            horizon=horizon
        ).item()


    @staticmethod
    def get_historical_conditional_value_at_risk(asset, confidence=0.99, horizon=1):
        from epsilonPhi.core.estimator.tseriesEstimator import CTimeSeriesEstimator
        return CTimeSeriesEstimator(asset, asset.returns_type, asset.type).cvar(
            confidence_level=confidence,
            horizon=horizon
        ).item()


    @staticmethod
    def get_historical_probability_of_loss(asset, horizon=1):
        from epsilonPhi.core.estimator.tseriesEstimator import CTimeSeriesEstimator
        return CTimeSeriesEstimator(asset, asset.returns_type, asset.type).frequency_of_loss(horizon=horizon).item()


    @staticmethod
    def get_historical_worst_peak_to_trough(asset):
        from epsilonPhi.core.estimator.tseriesEstimator import CTimeSeriesEstimator
        return CTimeSeriesEstimator(asset, asset.returns_type, asset.type).worst_peak_to_trough().item()

    @staticmethod
    def get_historical_skewness(asset):
        from epsilonPhi.core.estimator.tseriesEstimator import CTimeSeriesEstimator
        return CTimeSeriesEstimator(asset, asset.returns_type, asset.type).skew().item()

    @staticmethod
    def get_historical_worst_period_return(asset, period=1):
        from epsilonPhi.core.estimator.tseriesEstimator import CTimeSeriesEstimator
        return CTimeSeriesEstimator(asset, asset.returns_type, asset.type).worst(period=period).item()

    @staticmethod
    def get_historical_best_period_return(asset, period=1):
        from epsilonPhi.core.estimator.tseriesEstimator import CTimeSeriesEstimator
        return CTimeSeriesEstimator(asset, asset.returns_type, asset.type).best(period=period).item()
