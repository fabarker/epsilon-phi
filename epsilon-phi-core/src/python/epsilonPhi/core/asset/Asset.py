from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType
from epsilonPhi.core.timeSeries.timeSeriesMain import CSlice, CTimeSeries
from epsilonPhi.core.asset.CAssetInf import CAssetInf, _assetmeta
from epsilonPhi.core.estimator.estimationMgr import EstimationMgr
from epsilonPhi.core.config.appConfig import CAppConfig
from epsilonPhi.core.asset.AssetMgr import CAssetMgr
from epsilonPhi.core.schema.Schema import CContext
from typing import Optional, Union
import numpy as np
import pandas as pd
import math
import copy

ts_type: TimeSeriesType = TimeSeriesType.LEVELS


class CAsset(CAssetInf, CSlice):
    _metadata = _assetmeta

    @property
    def _constructor(self):
        def _c(*args, **kwargs):
            return CAsset(*args, **kwargs).__finalize__(self)

        return _c

    def __init__(self,
                 data: Optional[Union[CTimeSeries, CSlice]],
                 schema: Optional[CContext] = None,
                 denominated_currency: Optional[str] = None,
                 exposure_currency: Optional[str] = None,
                 ts_hedge_ratio: Optional[Union[float, int]] = None,
                 returns_type: Optional[ReturnsType] = None,
                 ts_type: Optional[TimeSeriesType] = None,
                 **kwargs
                 ) -> None:

        series = CAssetMgr._prepare_dataframe_for_asset(schema, data)
        super(CAsset, self).__init__(data=series,
                                     returns_type=returns_type,
                                     ts_type=ts_type,
                                     **kwargs
                                     )

        self._denominated_currency = denominated_currency
        self._exposure_currency = exposure_currency
        self._schema = schema
        self._assetMgr = CAssetMgr(schema)
        self._reporting_name = None
        self._category = None
        self._hedge_ratio = None
        self._alpha = 0
        self._weight = None
        self._risk_premias_in_curr_env = None
        self._uncertainty = None

        if ((ts_hedge_ratio is not None) and
                (self.denominated_currency != self.exposure_currency)):
            self._ts_hedge_ratio = float(ts_hedge_ratio)
        else:
            self._ts_hedge_ratio = 0

    def deepcopy(self):
        return copy.deepcopy(self)

    ###############

    @property
    def denominated_currency(self):
        return self._denominated_currency

    @property
    def exposure_currency(self):
        return self._exposure_currency

    @property
    def obs_per_year(self):
        return self.frequency.obs_per_year()

    @property
    def frequency(self):
        return self.schema.frequency

    @property
    def schema(self):
        return self._schema

    @property
    def assetMgr(self):
        return self._assetMgr

    @property
    def hedging_ratio(self):
        return self._hedge_ratio

    @property
    def is_time_series_in_local_terms(self):
        return self.denominated_currency == self.exposure_currency

    @property
    def weight(self):
        return self._weight if isinstance(self._weight, float) else None

    @property
    def alpha(self):
        return self._alpha

    @property
    def reporting_name(self):
        return self._reporting_name

    @property
    def category(self):
        return self._category

    def set_reporting_info(self, reporting_name, category):
        self._reporting_name = reporting_name
        self._category = category

    ###############

    def copy_params(self, params):
        self.__dict__.update(params)

    def set_schema(self, schema):
        self._schema = schema
        self._assetMgr = CAssetMgr(self._schema)
        self._risk_premias_in_curr_env = None

    def set_currency_hedge_ratio(self, hedge_ratio: Optional[Union[float, int]]):
        if hedge_ratio is not None:
            assert 0 <= hedge_ratio <= 1, 'Error - Currency hedge ratio must be in interval [0,1]'
            assert isinstance(hedge_ratio, (float, int)), 'Error - Currency hedge ratio must be of type float or int'
        self._hedge_ratio = hedge_ratio

    def set_alpha(self, alpha):
        self._alpha = alpha

    def set_weight(self, weight):
        self._weight = float(weight)

    def get_asset_params(self):
        return {
            'denominated_currency': self.denominated_currency,
            'exposure_currency': self.exposure_currency,
            'ts_hedge_ratio': self._ts_hedge_ratio,
            'returns_type': self._returns_type,
            'ts_type': self._type,
        }

    def set_uncertainty(self, uncertainty):
        self._uncertainty = uncertainty

    ##################### Asset risk free rate ###########################

    def get_risk_free_asset(self):
        return self.assetMgr.get_risk_free_asset(self.denominated_currency)

    def get_risk_free_rate(self):
        return self._schema.risk_free_rate

    def get_medium_risk_free_rate(self):
        return self._schema.medium_term_risk_free_rate

    def get_current_risk_free_rate(self):
        return self._schema.curr_risk_free_rate

    ########### Historical Asset Class Performance Metrics ###############

    def get_historical_total_return(self, from_date=None, to_date=None):
        return EstimationMgr.get_historical_total_return(self[from_date:to_date])

    def get_historical_sharpe_ratio(self, from_date=None, to_date=None):
        return EstimationMgr.get_historical_sharpe_ratio(self[from_date:to_date])

    def get_historical_volatility(self, from_date=None, to_date=None):
        return EstimationMgr.get_historical_volatility(self[from_date:to_date])

    def get_historical_risk_premium(self, from_date=None, to_date=None):
        return EstimationMgr.get_historical_risk_premium(self[from_date:to_date])

    def get_excess_return_df(self, from_date=None, to_date=None):
        return EstimationMgr.get_excess_return_timeseries(self[from_date:to_date])

    def get_historical_excess_return_df(self, from_date=None, to_date=None):
        return EstimationMgr.get_historical_excess_return_df(self[from_date:to_date])

    def get_historical_value_at_risk(self, horizon=1, confidence=0.99, from_date=None, to_date=None):
        return EstimationMgr.get_historical_value_at_risk(
            self[from_date:to_date],
            horizon=horizon,
            confidence=confidence
        )

    def get_historical_conditional_value_at_risk(self, horizon=1, confidence=0.99, from_date=None, to_date=None):
        return EstimationMgr.get_historical_conditional_value_at_risk(
            self[from_date:to_date],
            horizon=horizon,
            confidence=confidence
        )

    def get_historical_probability_of_loss(self, horizon=1, from_date=None, to_date=None):
        return EstimationMgr.get_historical_probability_of_loss(self[from_date:to_date], horizon=horizon)

    def get_historical_worst_peak_to_trough(self, from_date=None, to_date=None):
        return EstimationMgr.get_historical_worst_peak_to_trough(self[from_date:to_date])

    def get_historical_skewness(self, from_date=None, to_date=None):
        return EstimationMgr.get_historical_skewness(self[from_date:to_date])

    def get_historical_worst_period_return(self, period=1, from_date=None, to_date=None):
        return EstimationMgr.get_historical_worst_period_return(self[from_date:to_date], period=period)

    def get_historical_best_period_return(self, period=1, from_date=None, to_date=None):
        return EstimationMgr.get_historical_best_period_return(self[from_date:to_date], period=period)

    ################### Factor Model Asset Metrics ###################

    def get_stressed_factor_based_risk_premium(self):

        from epsilonPhi.core.portfolio.SAAPortfolio import SAAPortfolio

        # Construct portfolio on the fly
        ptf = SAAPortfolio(
            self.name,
            self.schema,
        )

        # Add the asset we want the current environ risk premia for
        ptf.add_asset(self.deepcopy(), 1, self._hedge_ratio)

        # Get the stressed returns panel
        returns_panel = ptf.get_stressed_returns_panel()

        # Use Historical Sharpe Ratios
        factor_sharpes_uncapped = (self.schema.get_return_factors_sharpe_ratio_uncapped()
                                   / math.sqrt(self.schema.frequency.obs_per_year()))

        # Update Factor Sharpes
        normalized_factors = np.copy(returns_panel.factor_panel_normalized)
        renorm_factors = ((normalized_factors - np.mean(normalized_factors, axis=0)) +
                          factor_sharpes_uncapped.values.flatten())
        rp = (renorm_factors * returns_panel.stress_coeff_panels) @ returns_panel.betas
        return pd.Series(rp, name=self.name, index=returns_panel.dates)

    def get_stressed_factor_based_returns(self):
        risk_premium = self.get_stressed_factor_based_risk_premium()
        rfr = self.get_risk_free_asset().reindex(risk_premium.index)
        return risk_premium + rfr.values + (self.get_alpha() / self.schema.frequency.obs_per_year())

    def get_factor_backfilled_returns(self):
        hist = self.get_realized_return_time_series()
        fact = self.get_stressed_factor_based_returns()
        return pd.concat((fact[fact.index < hist.index.min()], hist)).sort_index()

    def get_risk_premias(self):
        return EstimationMgr.get_risk_premium(self)

    def get_risk_premia(self):
        return self.get_risk_premias().sum() + self.get_alpha()

    def get_risk_premias_in_current_environment(self):
        if self._risk_premias_in_curr_env is None:
            from epsilonPhi.core.portfolio.SAAPortfolio import SAAPortfolio

            # Construct portfolio on the fly
            ptf = SAAPortfolio(
                self.name,
                self.schema,
            )

            # Add the asset we want the current environ risk premia for
            ptf.add_asset(self.deepcopy(), 1, self._hedge_ratio)

            # Get the current environment risk premia from simulation module
            self._risk_premias_in_curr_env = ptf.get_current_env_risk_premias()
        return self._risk_premias_in_curr_env + self.get_alpha()

    def get_risk_premia_in_current_environment(self):
        return self.get_risk_premias_in_current_environment().sum()

    def get_total_return(self):
        return self.get_risk_premia() + self.get_risk_free_rate()

    def get_return_in_current_environment(self):
        return self.get_risk_premia_in_current_environment() + self.get_current_risk_free_rate()

    def get_medium_term_return(self):
        return self.get_risk_premia() + self.get_medium_risk_free_rate()

    def get_alpha(self):
        return self._alpha

    def get_sharpe_ratio(self):
        return (self.get_risk_premia()) / self.get_volatility()

    def get_sharpe_ratio_in_current_environment(self):
        return (self.get_risk_premia_in_current_environment()) / self.get_volatility()

    def get_return_betas(self, hedging_ratio=None, normalized=True):
        return self.get_rolling_return_betas(hedging_ratio, normalized).mean(axis=0)

    def get_rolling_return_betas(self, hedging_ratio=None, normalized=True):
        return EstimationMgr.get_return_betas(self, hedging_ratio if hedging_ratio is not None else self.hedging_ratio,
                                              normalized=normalized)

    def get_return_betas_not_normalized(self, hedging_ratio=None):
        return self.get_return_betas(hedging_ratio, normalized=False)

    def get_risk_betas(self):
        return EstimationMgr.get_risk_betas(self, self.hedging_ratio)

    def get_asset_risk_betas(self):
        return EstimationMgr.get_risk_betas(self, 1)

    def get_fx_risk_betas(self):
        return self.get_risk_betas() - self.get_asset_risk_betas()

    def get_fx_risk_decomposition(self):
        pass

    def get_beta_and_idio_risk(self, hedging_ratio=None):
        return EstimationMgr.get_beta_and_idio_variance(self,
                                                        hedging_ratio if hedging_ratio is not None else self.hedging_ratio)

    def get_variance(self, hedging_ratio=None):
        return self.get_systematic_variance(hedging_ratio) + self.get_idiosyncratic_variance(hedging_ratio)

    def get_systematic_variance(self, hedging_ratio=None):
        return EstimationMgr.get_systematic_variance(self,
                                                     hedging_ratio if hedging_ratio is not None else self.hedging_ratio)

    def get_idiosyncratic_variance(self, hedging_ratio=None):
        return EstimationMgr.get_idiosyncratic_variance(self,
                                                        hedging_ratio if hedging_ratio is not None else self.hedging_ratio)

    def get_volatility(self, hedging_ratio=None):
        return EstimationMgr.get_risk_factor_stdev(self,
                                                   hedging_ratio if hedging_ratio is not None else self.hedging_ratio)

    def get_data_length(self):
        return EstimationMgr.get_estimation_length(self)

    def get_residuals(self, hedging_ratio=None):
        return EstimationMgr.get_residuals(self, hedging_ratio if hedging_ratio is not None else self.hedging_ratio)

    def get_uncertainty(self):
        if self._uncertainty is None:
            return self.get_volatility() / np.sqrt(self.get_data_length())
        else:
            return self._uncertainty

    ##############

    def get_realized_return_time_series(self, hedging_ratio=None):
        return self.convert_asset_to_currency(self.schema.currency,
                                              hedging_ratio if hedging_ratio is not None else self.hedging_ratio)

    def get_realized_excess_return_time_series(self, hedging_ratio=None):
        return self.get_realized_return_time_series(
            hedging_ratio if hedging_ratio is not None else self.hedging_ratio).get_excess_return_df()

    def convert_asset_to_currency(self, currency, hedging_ratio):
        return self.assetMgr.convert_asset_to_currency(self, currency, hedging_ratio)

    def simulate(self, long_term_shocks=False):

        from epsilonPhi.core.portfolio.SAAPortfolio import SAAPortfolio

        # Construct portfolio on the fly
        ptf = SAAPortfolio(
            self.name,
            self.schema,
        )

        # Add the asset we want the current environ risk premia for
        ptf.add_asset(self.deepcopy(), 1, self._hedge_ratio)
        return ptf.get_portfolio_simulated_returns_panel(long_term_shocks=long_term_shocks)

    def brownian_bridge(self):
        pass

    def get_assumptions(self):
        return {
            'category': self._category,
            'reporting_name': self.reporting_name,
            'risk_premia': self.get_risk_premia(),
            'uncertainty': self.get_uncertainty(),
            'volatility': self.get_volatility(),
            'Sharpe ratio': self.get_sharpe_ratio(),
            'return': self.get_total_return(),
            'hedging ratio': self.hedging_ratio,
            'start date': self.index.min(),
            'end date': self.index.max()
        }


if __name__ == "__main__":

    import pandas as pd
    from epsilonPhi.core.schema.Schema import ContextCreator

    schema = ContextCreator(currency='USD',
                            start_date='30-Nov-1983',
                            end_date='31-Dec-2022').create_context()

    asset = schema.get_asset_from_name('LHYIELD_GE20')
    asset.set_currency_hedge_ratio(1)

    rp  = asset.get_risk_premia()
    asset.get_historical_risk_premium()
