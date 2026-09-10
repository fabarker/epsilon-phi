import numpy as np
import pandas as pd
from epsilonPhi.core.utils.SolverUtils import brentsmethod, brent
from epsilonPhi.core.utils.FrameUtils import FrameUtils
from epsilonPhi.core.lib.curve_fitting.cubic_spline.cubic_spline import cubic_spline as cubicspline
from epsilonPhi.core.lib.curve_fitting.cubic_spline.cubic_spline import calc_spline_params, piece_wise_spline, fit_cubic_spline, eval_cubic_spline, fit_spline
from epsilonPhi.core.dataModel.dataSources.curves.abstractCurve.AbstractCurve import AbstractCurve
from epsilonPhi.core.utils.NumpyUtils import NumpyUtils as npu
from scipy.optimize import brentq
from epsilonPhi.core.utils.FrameUtils import FrameUtils
from numba import jit
from tqdm import tqdm
from epsilonPhi.core.utils.OptionUtils import *
from abc import abstractmethod
from numba.typed import Dict
from epsilonPhi.core.dataModel.enums.ImpliedVolatility import *
from epsilonPhi.core.utils.MathUtils import cubic_spline, vanna_volga_1d
from epsilonPhi.core.dataModel.enums.ImpliedVolatility import MaturityType
from typing import Optional, Union

class AbstractInterpolator(object):

    def __init__(self,
                 market_df: pd.DataFrame,
                 **kwargs
                 ) -> None:

        # Properties of the surface fits
        self._fit = None
        self._fit_df = None
        self._fit_maturities = None

        # Properties of market data
        self._spot_rates = None
        self._funding_curve = None
        self._rate_curve = None
        self._forward_curve = None

        # set the market data frame
        self.set_market_df(market_df)

    @abstractmethod
    def fit_surface(self):
        pass

    @abstractmethod
    def solve_for_delta(self, pricing_dates, expiries, relative_strike):
        pass

    @abstractmethod
    def solve_for_strike(self, pricing_dates, expiries, relative_strike):
        pass

    @property
    def t(self):
        return np.unique(self._fit_df.columns.get_level_values(0))

    def ordinals(self):
        return DateUtils.to_ordinals(self._fit_df.index)

    def find_date_idx(self, dates):
        return DateUtils.find_first_date_loc(self._fit_df.index, dates)


    ######## Set Methods ########

    def set_market_df(self, market_df: pd.DataFrame)  -> None:

        # set market data frame
        self._market_df = market_df.copy().reset_index()

        if 's' in self._market_df.columns:
            s = self._market_df.get(['date', 's']).drop_duplicates(subset='date')
            s = s.set_index('date').sort_index()
            self.set_spot_rates(s)

    def set_deposit_rate_curve(self, deposit_rate_curve: AbstractCurve)  -> None:
        assert isinstance(deposit_rate_curve, AbstractCurve), 'Error - deposit_rate_curve must be type "AbstractCurve"'
        self._rate_curve = deposit_rate_curve

    def set_funding_rate_curve(self, funding_rate_curve: AbstractCurve)  -> None:
        assert isinstance(funding_rate_curve, AbstractCurve), 'Error - deposit_rate_curve must be type "AbstractCurve"'
        self._funding_curve = funding_rate_curve

    def set_spot_rates(self, spot_rates: Optional[Union[pd.DataFrame, pd.Series]])  -> None:
        self._spot_rates = spot_rates.resample('B').asfreq()

    def set_forward_rate_curve(self, forward_rate_curve: AbstractCurve)  -> None:
        assert isinstance(forward_rate_curve, AbstractCurve), 'Error - deposit_rate_curve must be type "AbstractCurve"'
        self._fwd_rate_curve = forward_rate_curve


    ######### Getter Methods ########

    def s(self):
        return self.get_s(self._fit_df.index).values.flatten()
    def rd(self):
        return self.get_rd(self._fit_df.index, self.t, is_stacked=False).values
    def rf(self):
        return self.get_rf(self._fit_df.index, self.t, is_stacked=False).values

    def get_f(self, pricing_dates, maturities, is_stacked=False):
        return self._fwd_rate_curve.get_curve(pricing_dates, maturities, is_stacked)

    def get_rd(self, pricing_dates, maturities, is_stacked=False):
        return self._rate_curve.get_curve(pricing_dates, maturities, is_stacked)

    def get_rf(self, pricing_dates, maturities, is_stacked=False):
        return self._funding_curve.get_curve(pricing_dates, maturities, is_stacked)

    def get_s(self, pricing_dates):
        return self._spot_rates.reindex(pricing_dates)

    def get_holidays(self):
        all_dates = self._spot_rates.dropna().resample('D').asfreq()
        return all_dates[all_dates.isna().values.flatten()].index

    def get_expiry_dates_from_settlement_dates_tenor(self, pricing_dates, tenors):
        return DateUtils.expiry_from_settlement(pricing_dates, tenors, self.get_holidays())
    def get_expiries_from_settlement_maturity_dates(self, pricing_dates, maturity_dates):
        return DateUtils.get_date_delta(pricing_dates, maturity_dates, True)
    def get_expiries_from_settlement_dates_tenors(self, pricing_dates, maturities):
        expiry_dates = self.get_expiry_dates_from_settlement_dates_tenor(pricing_dates, maturities)
        return self.get_expiries_from_settlement_maturity_dates(pricing_dates, expiry_dates)
    def get_expiry_dates_from_settlement_dates_expiries(self, pricing_dates, expiries):
        candidate = np.round(DateUtils.to_ordinals(pricing_dates) + expiries * 365, 0).astype(np.int32)
        return DateUtils.from_ordinals(DateUtils.shift_off_holidays(candidate,
                                                                    DateUtils.to_ordinals(self.get_holidays()),
                                                                    DateUtils.mat_to_Rdate(expiries)))
    def solve_for_tenor(self, pricing_dates, tenor, strike_reference, relative_strike):
        expiries = self.get_expiries_from_settlement_dates_tenors(pricing_dates, tenor)
        return self.solve_for_expiries(pricing_dates, expiries, strike_reference, relative_strike)
    def solve_for_expiry_dates(self, pricing_dates, expiry_dates, strike_reference, relative_strike):
        expiries = self.get_expiries_from_settlement_maturity_dates(pricing_dates, expiry_dates)
        return self.solve_for_expiries(pricing_dates, expiries, strike_reference, relative_strike)
    def solve_for_expiries(self, pricing_dates, expiries, strike_reference, relative_strike):

        if strike_reference in [StrikeReference.DELTA, StrikeReference.DELTA.value]:
            return self.solve_for_delta(pricing_dates, expiries, relative_strike)
        elif strike_reference in [StrikeReference.STRIKE_PRICE, StrikeReference.STRIKE_PRICE.value]:
            return self.solve_for_strike(pricing_dates, expiries, relative_strike)
        else:
            raise ValueError('Error')

    def get_ivols(self, pricing_dates, strike_reference, relative_strike, maturity_type, maturities):

        if not DateUtils.is_iterable(pricing_dates):
           pricing_dates = pd.to_datetime([pricing_dates])

        maturities = np.ravel([maturities])
        relative_strike = np.ravel([relative_strike])
        if ((pricing_dates.size != maturities.size) and
                (pricing_dates.size != relative_strike.size) and (maturities.size == relative_strike.size)):
            pricing_dates, maturities, relative_strike = (
                npu.flat_meshgrid(pricing_dates, maturities, relative_strike))

        if (pricing_dates.size == maturities.size and
            relative_strike.size == 1):
            relative_strike = np.ravel([relative_strike] * pricing_dates.size)

        assert pricing_dates.size == maturities.size, 'Error - inconsistent dates and maturities'
        assert pricing_dates.size == relative_strike.size, 'Error - inconsistent dates and strikes'

        if maturity_type in [MaturityType.MATURITY_STRING, MaturityType.MATURITY_STRING.value]:
           sig = self.solve_for_tenor(pricing_dates, maturities, strike_reference, relative_strike)
        elif maturity_type in [MaturityType.EXPIRY_DATE, MaturityType.EXPIRY_DATE.value]:
           sig = self.solve_for_expiry_dates(pricing_dates, maturities, strike_reference, relative_strike)
        elif maturity_type in [MaturityType.YEARFRAC, MaturityType.YEARFRAC.value]:
           sig = self.solve_for_expiries(pricing_dates, maturities, strike_reference, relative_strike)
        else:
            raise ValueError('Error: maturity_type {} not supported'.format(maturity_type))

        # Get the expiry date
        sig['expiry'] = self.get_expiry_dates_from_settlement_dates_expiries(pricing_dates, np.ravel(sig.t))
        return sig.copy()


if __name__ == "__main__":

        from epsilonPhi.core.dataModel.dataSources.impliedVolatility.Models import *
        from epsilonPhi.core.dataModel.dataSources.impliedVolatility.VolSurfaceMgr import VolSurfaceMgr

        underlier = 'GBPUSD'
        vsm = VolSurfaceMgr(underlier,
                             pricing_location='NYC',
                             strike_reference=StrikeReference.DELTA)


        ### Get ivols
        ivols = vsm.get_ivols()

        s = vsm.get_spot_prices()
        rd = vsm.get_rate_curve()
        rf = vsm.get_funding_curve()
