from epsilonPhi.core.estimator.assetEstimatorInf import CAssetReturnEstimatorInf
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.config.configUtil import CAppConfig
import pandas as pd
import numpy as np
import math


class AssetReturnEstimator(CAssetReturnEstimatorInf):
    def __init__(self):
        super(AssetReturnEstimator, self).__init__()

    @staticmethod
    def get_risk_premium(asset):
        schema = asset.getSchema()
        hist_sharpe = schema.get_factor_panels().get_return_factor_Sharpe_ratios()
        betas = asset.get_return_betas()
        return np.mean(betas, axis=0) * hist_sharpe.conj().T * math.sqrt(schema.annualizing_factor)

    @staticmethod
    def get_excess_return_timeseries(asset):
        rfr = asset.get_risk_free_asset()

        common_dates = np.intersect1d(rfr.dates, asset.dates)
        factor = asset.select_subset_dates(common_dates) - rfr.select_subset_dates(common_dates).values
        factor.columns = pd.MultiIndex.from_tuples([(a, 'ER' if b == 'RI' else b) for a, b in factor.columns])
        return factor

    @staticmethod
    def get_historical_Sharpe_ratio(asset):
        factor = AssetReturnEstimator.get_excess_return_timeseries(asset)
        N = asset._schema.obs_per_year
        return (np.mean(factor.data) * N) / (np.std(factor.data) * math.sqrt(N))

    @staticmethod
    def get_historical_risk_premia(asset):
        factor = AssetReturnEstimator.get_excess_return_timeseries(asset)
        N = asset._schema.obs_per_year
        return np.mean(factor.data) * N

    @staticmethod
    def get_risk_premia_with_hedging(asset, hedging_option=None):

        schema = asset.getSchema()
        estimation_start_date = min(schema.dates)

        if hedging_option is None:
            hedging_option = asset.get_hedging_option()

        asset_name = asset.get_asset_name()
        return_mapping_info = SAAHedging.get_hedging_info(asset_name,
                                                        schema.currency,
                                                        schema.scope,
                                                        asset_time_series=asset,
                                                        hedging_option=hedging_option)

        unhedged_name = return_mapping_info.get_unhedged_symbol_for_return()
        hedged_name = return_mapping_info.get_hedged_symbol_for_return()
        hedged_return_proportion = return_mapping_info.get_hedging_Ratio_for_return()
        length_adjustment_scale = CAppConfig.get_config_util().get_length_adjustment(asset_name,
                                                                               schema.scope)

        # If we have no hedged name or hedged return proportion is 0, then just use the unhedged asset
        if not hedged_name or hedged_return_proportion == 0:
            unhedged_asset = schema.get_asset_mgr().get_asset_by_name(unhedged_name)
            risk_premia = unhedged_asset.get_risk_premia()
            data_length = unhedged_asset.get_data_length(estimation_start_date)
            historical_risk_premia = unhedged_asset.get_historical_risk_premia()

        # In this case we only use the hedge version of the time series
        # This could
        elif hedged_return_proportion == 1:
            hedged_asset = schema.get_asset_mgr().get_asset_by_name(hedged_name)
            risk_premia = hedged_asset.get_risk_premia()
            data_length = hedged_asset.get_data_length(estimation_start_date)
            historical_risk_premia = hedged_asset.get_historical_risk_premia()

        # Some combination of hedged an unhedged
        else:
            unhedged_asset = schema.get_asset_mgr().get_asset_by_name(unhedged_name)
            hedged_asset = schema.get_asset_mgr().get_asset_by_name(hedged_name)

            hdg_rp = hedged_asset.get_risk_premia()
            uhd_rp = unhedged_asset.get_risk_premia()
            risk_premia = (1-hedged_return_proportion) * uhd_rp + hedged_return_proportion * hdg_rp

            data_length = np.minimum(hedged_asset.get_data_length(estimation_start_date),
                                     unhedged_asset.get_data_length(estimation_start_date))

            historical_risk_premia = ((1-hedged_return_proportion) * unhedged_asset.get_historical_risk_premia() +
                                      hedged_asset.get_historical_risk_premia() * hedged_asset)

        if length_adjustment_scale and length_adjustment_scale != 0:
            data_length = data_length * length_adjustment_scale

        return risk_premia, data_length, historical_risk_premia

    @staticmethod
    def get_return_betas(asset, normalized=True):

        schema = asset.schema
        reg_result = None
        factor_panel = schema.get_factor_panels()

        orthog_list = factor_panel.get_factor_orthogonalier()
        estimation_config = CAppConfig.get_config_util().get_estimation_config('Default')

        if not orthog_list:
            orthog_list = []

        ts_class = CFactorUtil.asset_to_factor(asset)
        if ts_class is None or ts_class.is_empty():
            raise Exception('Error converting asset to factor for asset {}'.format(asset.get_name()))

        if asset.frequency == Frequency.MONTHLY:

            reg_result = CAppConfig.get_time_series_regression().rolling_regression(ts_class,
                                                                                    factor_panel.get_return_factor_dataframe(),
                                                                                    orthog_list,
                                                                                    estimation_config.rolling_window_size,
                                                                                    'rolling',
                                                                                    schema.start_date,
                                                                                    schema.end_date,
                                                                                    normalized,
                                                                                    Frequency.MONTHLY)

        if asset.frequency == Frequency.QUARTERLY:

            reg_result = CAppConfig.get_time_series_regression().rolling_regression(ts_class,
                                                                                    factor_panel.get_return_factor_dataframe(),
                                                                                    orthog_list,
                                                                                    estimation_config.expanding_window_size,
                                                                                    'expanding',
                                                                                    schema.start_date,
                                                                                    schema.end_date,
                                                                                    normalized,
                                                                                    Frequency.QUARTERLY)
        else:
            raise Exception("Frequency not supported {}".format(asset.frequency))

        if reg_result is None:
            raise Exception("Error - betas are zero")

        betas = reg_result.T

        if asset.get_asset_name() in CAppConfig.get_config_util().get_market_stress_beta():
           mkt_factor = factor_panel.get_market_factor()
           mkt_factor_index = factor_panel.get_return_factor_list().index(mkt_factor)

           stress_betas = CAppConfig.get_config_util().get_market_stress_beta()
           betas[:, mkt_factor_index] = betas[:, mkt_factor_index] * stress_betas[asset.get_asset_name()]

        return betas



if __name__ == "__main__":

    from epsilonPhi.core.schema.Schema import ContextCreator
    from epsilonPhi.core.asset.AssetMgr import CAssetMgr

    schema = ContextCreator(currency='GBP',
                            start_date='31-Dec-1999',
                            end_date='31-Dec-2022').create_context()

    assetMgr = CAssetMgr(schema)

    asset = assetMgr.get_asset_by_name('S&PCOMP')
    ER = AssetReturnEstimator.get_excess_return_timeseries(asset)





