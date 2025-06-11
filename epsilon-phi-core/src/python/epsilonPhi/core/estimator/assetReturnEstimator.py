from epsilonPhi.core.estimator.assetEstimatorInf import CAssetReturnEstimatorInf
from epsilonPhi.core.utils.DateUtils import DateUtils
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.config.configUtil import CAppConfig
import pandas as pd
import numpy as np
import math


class AssetReturnEstimator(CAssetReturnEstimatorInf):
    _cache = {}

    def __init__(self):
        super(AssetReturnEstimator, self).__init__()

    @staticmethod
    def calc_return_betas(asset, hedging_ratio=0.5, normalized=True):

        schema_currency = asset.schema.currency
        asset_in_schema_currency = asset.convert_asset_to_currency(schema_currency, hedging_ratio)

        model = CAppConfig.get_BaseModel()
        factor_df = asset_in_schema_currency.schema.get_return_factors_panel()

        rx = asset.get_excess_return_df()
        y, X = rx.intersect_over_date_range(factor_df)
        assert np.all(y.index == X.index), 'Error - date mismatch in regression'

        win = 60
        start = 60
        end = y.shape[0] + 1

        betas = np.full((end - win, 6), np.nan)
        for i in range(start, end):

            X_prime = X.iloc[i - win:i].copy()
            y_prime = y.iloc[i - win:i].copy()
            orth_X = model.regression.orthogonalize_columns(X_prime,
                                                            model.orthogonal_list)

            if normalized:
                stdev = np.std(orth_X, axis=0, ddof=1)
            else:
                stdev = 1

            eDfArray = X_prime / stdev
            _, b = model.regression.simple_regression_OLS_with_array(
                eDfArray.values,
                y_prime.values
            )

            betas[i - win] = b

        #regstats = model.regression.regress(
        #    X,
        #    y,
        #    orthogonalize_columns=model.orthogonal_list,
        #    normalize=normalized
        #)

        return pd.DataFrame(
            betas,
            index=factor_df.index[-betas.shape[0]:],
            columns=model.return_factor_list
        )

    @staticmethod
    def get_risk_premium(asset):
        hist_sharpe = CAppConfig.get_BaseModel().get_return_factor_Sharpe_ratios()
        betas = asset.get_return_betas().values
        return betas * hist_sharpe.values.T * math.sqrt(asset.schema.obs_per_year)

    @staticmethod
    def get_excess_return_timeseries(asset):
        rfr = asset.get_risk_free_asset()

        common_dates = np.intersect1d(rfr.dates, asset.dates)
        factor = asset.select_subset_dates(common_dates) - rfr.select_subset_dates(common_dates).values
        factor.columns = (factor.name[0], 'ER')
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
    def get_estimation_length(asset):
        common_dates = pd.to_datetime(np.intersect1d(asset.schema.get_return_factors_panel().dates,
                                      asset.get_excess_return_df().dates))
        return round((common_dates.max() - common_dates.min()).days / DateUtils.days_per_year, 4)


    @staticmethod
    def get_return_betas(asset, hedging_ratio, normalized=True):
        if (asset.name, hedging_ratio, normalized) not in AssetReturnEstimator._cache:
            AssetReturnEstimator._cache[(asset.name, hedging_ratio, normalized)] = (
                AssetReturnEstimator.calc_return_betas(
                asset,
                hedging_ratio,
                normalized)
            )
        return AssetReturnEstimator._cache[(asset.name, hedging_ratio, normalized)]




if __name__ == "__main__":

    from epsilonPhi.core.schema.Schema import ContextCreator
    from epsilonPhi.core.asset.AssetMgr import CAssetMgr

    schema = ContextCreator(currency='GBP',
                            start_date='30-Nov-1983',
                            end_date='31-Dec-2022').create_context()

    assetMgr = CAssetMgr(schema)

    asset = assetMgr.get_asset_by_name('S&PCOMP')
    ER = AssetReturnEstimator.get_excess_return_timeseries(asset)





