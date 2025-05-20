import datetime
from dataclasses import dataclass
from typing import Optional, Any, List
import numpy as np

class Crisis:
    def __init__(
            self,
            name,
            start_date,
            end_date,
            stress_coefficient=0):

        self._name = name
        self._start_date = start_date
        self._end_date = end_date
        self._stress_coefficient = stress_coefficient
        self._total = 0

    @property
    def start_date(self):
        return self._start_date

    @property
    def end_date(self):
        return self._end_date

    @property
    def stress_coefficient(self):
        return self._stress_coefficient

    def set_stress_coefficient(self, stress_coefficient):
        self._stress_coefficient = stress_coefficient

class BootstrapIndicies(object):

    _short_term = None
    _medium_term = None
    _long_term = None

    def __init__(self):
        pass

    def get_short_term_indicies(self):
        return self._short_term

    def get_medium_term_indicies(self):
        return self._medium_term

    def get_long_term_indicies(self):
        return self._long_term

    def set_short_term_indicies(self, indicies):
        self._short_term = indicies

    def set_medium_term_indicies(self, indicies):
        self._medium_term = indicies

    def set_long_term_indicies(self, indicies):
        self._long_term = indicies


class PathsPanel:
    def __init__(self, panel):
        self._panel = panel

    def get_panel(self):
        return self._panel

class PathsShortBlockPanel:
    def __init__(self, paths, short_block_indicator):
        self._paths = paths
        self._short_block_indicator = short_block_indicator

    def get_paths(self):
        return self._paths

    def get_short_block_indicator(self):
        return self._short_block_indicator

@dataclass
class ReturnsPanel:
    factor_panel_normalized: Optional[Any] = None
    factor_panel_normalized_MT: Optional[Any] = None
    dates: Optional[Any] = None
    factor_contributions_panel: Optional[Any] = None
    factor_contributions_panel_MT: Optional[Any] = None
    rfr_values: Optional[Any] = None
    cpi_values: Optional[Any] = None
    return_factor_panel: Optional[Any] = None
    betas: Optional[Any] = None
    factor_sharpes: Optional[Any] = None
    factor_sharpes_MT: Optional[Any] = None
    factor_vols: Optional[Any] = None
    stress_coeff_panels: Optional[Any] = None
    stress_coeff_panel_MT: Optional[Any] = None
    alpha_total_monthly: Optional[Any] = None
    medium_term_risk_premia_5Y: Optional[Any] = None
    medium_term_risk_premia: Optional[Any] = None

    def get_risk_premia(self):
        return np.sum(self.factor_contributions_panel, axis=1)

    def get_ptf_systematic_index(self):
        return np.cumprod(1 + self.rfr_values + self.get_risk_premia() + self.alpha_total_monthly)

    def get_real_ptf_systematic_index(self):
        return self.get_ptf_systematic_index() / self.cpi_values

    def cumulative_factor_panel(self):
        return np.cumprod(1 + self.factor_contributions_panel, axis=0)


@dataclass
class FactorStressTestLosses:
    total: float
    real: Optional[float] = None
    factors: Optional[List[float]] = None
    cash_other: Optional[float] = None

@dataclass
class HistoricalStressTestLosses:
    total: float
    real: float
    cash_other: float
    start: datetime.datetime
    end: datetime.datetime


@dataclass
class FactorStressTestReal:
        total: float

@dataclass
class PerformanceVaRMetrics(object):
    def __init__(self, VaR, CVaR, PoL, confidence, loss):
        self._VaR = VaR
        self._CVaR = CVaR
        self._PoL = PoL
        self._confidence = confidence
        self._loss = loss

    @property
    def confidence(self):
        return self._confidence

    @property
    def loss(self):
        return self._loss

    @property
    def VaR(self):
        return self._VaR

    @property
    def CVaR(self):
        return self._CVaR

    @property
    def PoL(self):
        return self._PoL

    def get_VaR(self, idx):
        return self.VaR[idx]

    def get_CVaR(self, idx):
        return self.CVaR[idx]

    def get_PoL(self, idx):
        return self.PoL[idx]

    def __mul__(self, factor: float) -> "PerformanceVaRMetrics":
        return self.multiply(factor)

    def multiply(self, factor: float) -> "PerformanceVaRMetrics":
        """
        Returns a new PerformanceVaRMetrics object with VaR, CVaR, PoL, and loss
        scaled by the given factor.

        Args:
            factor (float): Multiplier to scale the risk metrics.

        Returns:
            PerformanceVaRMetrics: A new instance with scaled values.
        """
        return PerformanceVaRMetrics(
            VaR=self.VaR * factor,
            CVaR=self.CVaR * factor,
            PoL=self.PoL * factor,
            confidence=self.confidence,
            loss=self.loss
        )




class PerformanceVaRData:
    def __init__(self):
        self._one_month = None
        self._one_year = None
        self._three_year = None
        self._five_year = None

class PortfolioPaths:
    def __init__(
            self,
            systematic_panel=None,
            idio_panel=None,
            alpha_total_monthly=None
    ):

        self._systematic_panel = systematic_panel
        self._idio_panel = idio_panel
        self._alpha_total_monthly = alpha_total_monthly

    @property
    def systematic_panel(self):
        return self._systematic_panel

    @property
    def idiosyncratic_panel(self):
        return self._idio_panel

    @property
    def alpha_monthly_total(self):
        return self._alpha_total_monthly

    def set_systematic_panel(self, systematic_panel):
        self._systematic_panel = systematic_panel

    def set_idio_panel(self, idio_panel):
        self._idio_panel = idio_panel

    def set_alpha_total_monthly(self, alpha_total_monthly):
        self._alpha_total_monthly = alpha_total_monthly

    def get_systematic_panel(self):
        return self._systematic_panel

    def get_idio_panel(self):
        return self._idio_panel

    def get_alpha_total_monthly(self):
        return self._alpha_total_monthly

    def set_returns_panel(self, returns_panel):
        self._returns_panel = returns_panel

    def get_returns_panel(self):
        return self._returns_panel

class WealthFlows:

    def __init__(self, nominal=0, real=0, percent=0):
        self._nominal = nominal
        self._real = real
        self._percent = percent


    @property
    def nominal(self):
        return self._nominal

    @property
    def real(self):
        return self._real

    @property
    def percent(self):
        return self._percent


class WealthProjections:
    def __init__(self,
            nominal_values,
            total_returns_panel,
            inflation_paths,
            inflows,
            outflows,
            quantiles,
            frequency
    ):

        self._nominal_values = nominal_values
        self._total_returns_panel = total_returns_panel
        self._inflation_paths = inflation_paths
        self._inflows = inflows
        self._outflows = outflows
        self._quantiles = quantiles
        self._frequency = frequency

    @property
    def total_returns_panel(self):
        return self._total_returns_panel

    @property
    def frequency(self):
        return self._frequency

    @property
    def nominal_values(self):
        return self._nominal_values

    @property
    def real_values(self):
        return self._nominal_values / self.inflation_paths

    @property
    def inflation_paths(self):
        return self._inflation_paths

    @property
    def inflows(self):
        return self._inflows

    @property
    def outflows(self):
        return self._outflows

    @property
    def net_flows(self):
        return -self._outflows + self._inflows

    @property
    def quantiles(self):
        return self._quantiles

    def __calculate_quantiles(self, panel):
        return np.quantile(panel, self.quantiles, axis=1).T

    def get_inflation_quantiles(self):
        return self.__calculate_quantiles(self.inflation_paths)

    def get_nominal_quantiles(self):
        return self.__calculate_quantiles(self.nominal_values)

    def get_real_quantiles(self):
        return self.__calculate_quantiles(self.real_values)

    def get_nominal_inflow_quantiles(self):
        return self.__calculate_quantiles(self.inflows)

    def get_real_inflow_quantiles(self):
        return self.__calculate_quantiles(self.inflows/self.inflation_paths)

    def get_nominal_outflow_quantiles(self):
        return self.__calculate_quantiles(self.outflows)

    def get_real_outflow_quantiles(self):
        return self.__calculate_quantiles(self.outflows/self.inflation_paths)

    def get_net_flow_quantiles(self):
        return self.__calculate_quantiles(self.net_flows)

    def get_real_net_flow_quantiles(self):
        return self.__calculate_quantiles(self.net_flows/self.inflation_paths)





