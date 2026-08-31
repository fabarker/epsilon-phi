from epsilonPhi.core.estimator.assetEstimatorInf import CAssetReturnEstimatorInf
from epsilonPhi.core.utils.DateUtils import DateUtils
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.config.configUtil import CAppConfig
import pandas as pd
import numpy as np
import math


class AssetReturnEstimator(CAssetReturnEstimatorInf):
    _cache = {}

    # Rolling-window normalisation statistics, shared across every asset.
    # See calc_return_betas: the statistic depends only on the factor window
    # and the model, so one asset's windows serve all the others.
    _window_cache = {}

    def __init__(self):
        super(AssetReturnEstimator, self).__init__()

    @staticmethod
    def calc_return_betas(asset, hedging_ratio=0.5, normalized=True):
        """Rolling 60-observation factor betas for *asset*.

        Results are bit-identical to the original implementation; only the
        work done to reach them changed:

        * The per-window normalisation statistic is memoised in
          ``_window_cache``. It is derived from the orthogonalised factor
          window, which depends on the model and the window's dates but NOT
          on the asset - the original recomputed it for every asset, and it
          dominated the profile (roughly two thirds of this function).
        * The ``convert_asset_to_currency`` call was removed. Its result was
          used only for ``.schema``, and ``Asset.convert_asset_to_currency``
          returns an asset carrying the *same* schema object, so
          ``asset.schema`` is the identical object (verified). The regression
          itself reads ``asset.get_excess_return_df()``, the unconverted
          asset.

        NOTE for the model owner: because the converted asset was discarded,
        ``hedging_ratio`` has never influenced this function's output -
        verified identical at ratios 0.0, 0.5 and 1.0. The parameter is kept
        for API compatibility. If return betas are meant to vary with the
        hedge (i.e. the regression should use the converted asset's excess
        returns), that is a model change, not a performance one, and the
        cache key in get_return_betas must gain the ratio back at the same
        time.
        """

        model = asset.schema.BaseModel
        factor_df = asset.schema.get_return_factors_panel()

        rx = asset.get_excess_return_df()
        y, X = rx.intersect_over_date_range(factor_df)
        assert np.all(y.index == X.index), 'Error - date mismatch in regression'

        win = 60
        start = 60
        end = y.shape[0] + 1

        model_key = model.__hash__()
        index = X.index
        X_values = X.values
        y_values = y.values

        betas = np.full((end - win, factor_df.shape[1]), np.nan)
        for i in range(start, end):

            lo, hi = i - win, i
            window_key = (model_key, index[lo], index[hi - 1], normalized)
            stdev = AssetReturnEstimator._window_cache.get(window_key)

            if stdev is None:
                if normalized:
                    orth_X = model.regression.orthogonalize_columns(
                        X.iloc[lo:hi].copy(),
                        model.orthogonal_list
                    )
                    stdev = np.std(orth_X, axis=0, ddof=1).values
                else:
                    stdev = 1
                AssetReturnEstimator._window_cache[window_key] = stdev

            _, b = model.regression.simple_regression_OLS_with_array(
                X_values[lo:hi] / stdev,
                y_values[lo:hi]
            )

            betas[i - win] = b

        return pd.DataFrame(
            betas,
            index=factor_df.index[-betas.shape[0]:],
            columns=factor_df.columns
        )

    @staticmethod
    def get_risk_premium(asset):
        hist_sharpe = asset.schema.BaseModel.get_return_factor_Sharpe_ratios()
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
    def _return_beta_cache_key(asset, normalized):
        """Key on everything calc_return_betas actually reads.

        The previous key was (name, hedging_ratio, normalized, model hash).
        That was wrong in both directions: it carried ``hedging_ratio``, which
        the computation does not use (so every hedging policy recomputed
        identical betas), and it omitted the schema, whose currency and date
        window the excess-return series does depend on.
        """
        schema = asset.schema
        return (
            asset.name,
            schema.currency,
            str(schema.start_date),
            str(schema.end_date),
            normalized,
            schema.BaseModel.__hash__(),
        )

    @staticmethod
    def get_return_betas(asset, hedging_ratio, normalized=True):

        key = AssetReturnEstimator._return_beta_cache_key(asset, normalized)
        if key not in AssetReturnEstimator._cache:
            AssetReturnEstimator._cache[key] = (
                AssetReturnEstimator.calc_return_betas(
                asset,
                hedging_ratio,
                normalized)
            )
        return AssetReturnEstimator._cache[key]




if __name__ == "__main__":

    from epsilonPhi.core.schema.Schema import ContextCreator
    from epsilonPhi.core.asset.AssetMgr import CAssetMgr

    schema = ContextCreator(currency='GBP',
                            start_date='30-Nov-1983',
                            end_date='31-Dec-2022').create_context()

    assetMgr = CAssetMgr(schema)

    asset = assetMgr.get_asset_by_name('S&PCOMP')
    ER = AssetReturnEstimator.get_excess_return_timeseries(asset)





