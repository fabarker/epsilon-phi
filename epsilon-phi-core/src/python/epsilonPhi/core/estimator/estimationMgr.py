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
    def get_return_betas(asset, normalized=True):
        from epsilonPhi.core.estimator.assetReturnEstimator import AssetReturnEstimator
        return AssetReturnEstimator.get_return_betas(asset, normalized)

    ######################

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
        return AssetRiskEstimator.get_betas_and_idio_risk(asset, hedging_ratio)

    @staticmethod
    def get_risk_factor_stdev(asset, hedging_ratio):
        from epsilonPhi.core.estimator.assetRiskEstimator import AssetRiskEstimator
        return AssetRiskEstimator.get_risk_factor_stdev(asset, hedging_ratio)



    ##################

    @staticmethod
    def get_historical_total_return(asset):
        from epsilonPhi.core.estimator.tseriesEstimator import CTimeSeriesEstimator
        return CTimeSeriesEstimator(asset, asset.returns_type, asset.type).get_historical_total_return()

    @staticmethod
    def get_historical_excess_return_df(asset):
        from epsilonPhi.core.estimator.tseriesEstimator import CTimeSeriesEstimator
        return CTimeSeriesEstimator(asset, asset.returns_type, asset.type).get_historical_excess_return_df(asset.denominated_currency)

    @staticmethod
    def get_historical_risk_premium(asset):
        from epsilonPhi.core.estimator.tseriesEstimator import CTimeSeriesEstimator
        return CTimeSeriesEstimator(asset, asset.returns_type, asset.type).get_historical_risk_premium(asset.denominated_currency)

    @staticmethod
    def get_historical_volatility(asset):
        from epsilonPhi.core.estimator.tseriesEstimator import CTimeSeriesEstimator
        return CTimeSeriesEstimator(asset, asset.returns_type, asset.type).get_historical_volatility()


    @staticmethod
    def get_historical_sharpe_ratio(asset):
        from epsilonPhi.core.estimator.tseriesEstimator import CTimeSeriesEstimator
        return CTimeSeriesEstimator(asset, asset.returns_type, asset.type).get_historical_sharpe_ratio(asset.denominated_currency)


    @staticmethod
    def get_historical_value_at_risk(asset, confidence=0.99, horizon=1):
        from epsilonPhi.core.estimator.tseriesEstimator import CTimeSeriesEstimator
        return CTimeSeriesEstimator(asset, asset.returns_type, asset.type).get_historical_value_at_risk(confidence_level=confidence,
                                                                                                           horizon=horizon)


    @staticmethod
    def get_historical_conditional_value_at_risk(asset, confidence=0.99, horizon=1):
        from epsilonPhi.core.estimator.tseriesEstimator import CTimeSeriesEstimator
        return CTimeSeriesEstimator(asset, asset.returns_type, asset.type).get_historical_conditional_value_at_risk(confidence_level=confidence,
                                                                                                                       horizon=horizon)


    @staticmethod
    def get_historical_probability_of_loss(asset, horizon=1):
        from epsilonPhi.core.estimator.tseriesEstimator import CTimeSeriesEstimator
        return CTimeSeriesEstimator(asset, asset.returns_type, asset.type).get_historical_probability_of_loss(horizon=horizon)


    @staticmethod
    def get_historical_worst_peak_to_trough(asset):
        from epsilonPhi.core.estimator.tseriesEstimator import CTimeSeriesEstimator
        return CTimeSeriesEstimator(asset, asset.returns_type, asset.type).get_historical_worst_peak_to_trough()


    @staticmethod
    def get_historical_equity_beta(asset):
        from epsilonPhi.core.estimator.tseriesEstimator import CTimeSeriesEstimator
        return CTimeSeriesEstimator(asset, asset.returns_type, asset.type).get_historical_equity_beta(currency=asset.denominated_currency)

    @staticmethod
    def get_historical_alpha_over_equity(asset):
        from epsilonPhi.core.estimator.tseriesEstimator import CTimeSeriesEstimator
        return CTimeSeriesEstimator(asset, asset.returns_type, asset.type).get_historical_alpha_over_equity(currency=asset.denominated_currency)

    @staticmethod
    def get_historical_skewness(asset):
        from epsilonPhi.core.estimator.tseriesEstimator import CTimeSeriesEstimator
        return CTimeSeriesEstimator(asset, asset.returns_type, asset.type).get_historical_skewness()

    @staticmethod
    def get_historical_worst_period_return(asset, period=1):
        from epsilonPhi.core.estimator.tseriesEstimator import CTimeSeriesEstimator
        return CTimeSeriesEstimator(asset, asset.returns_type, asset.type).get_historical_worst_period_return(period=period)

    @staticmethod
    def get_historical_best_period_return(asset, period=1):
        from epsilonPhi.core.estimator.tseriesEstimator import CTimeSeriesEstimator
        return CTimeSeriesEstimator(asset, asset.returns_type, asset.type).get_historical_best_period_return(period=period)

    @staticmethod
    def get_historical_real_return(asset):
        from epsilonPhi.core.estimator.tseriesEstimator import CTimeSeriesEstimator
        return CTimeSeriesEstimator(asset, asset.returns_type, asset.type).get_historical_real_return(asset.schema.currency)

    @staticmethod
    def get_historical_inflation_out_performance_frequency(asset, period):
        from epsilonPhi.core.estimator.tseriesEstimator import CTimeSeriesEstimator
        return CTimeSeriesEstimator(asset, asset.returns_type, asset.type).get_historical_inflation_out_performance_frequency(period=period)