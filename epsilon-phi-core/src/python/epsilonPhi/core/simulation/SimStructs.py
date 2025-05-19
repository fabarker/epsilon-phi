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




