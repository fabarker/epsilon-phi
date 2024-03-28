from epsilonPhi.core.dataModel.dataSources.impliedVolatility.VolSurfaceMgr import VolSurfaceMgr
from epsilonPhi.core.dataModel.enums.ImpliedVolatility import Interpolator
from epsilonPhi.core.dataModel.dataSources.impliedVolatility.interpolators.gaussianKernelSmoother.GaussianKernel import GaussianKernel
from epsilonPhi.core.dataModel.dataSources.impliedVolatility.interpolators.vannaVolga.VannaVolga import VannaVolga
from epsilonPhi.core.dataModel.dataSources.impliedVolatility.interpolators.cubicSpline.CubicSpline import CubicSpline
from epsilonPhi.core.dataModel.enums.ImpliedVolatility import *
import pandas as pd
import numpy as np

class AbstractVolSurface(object):
    _cache = {}

    def __init__(self,
                 underlier,
                 strike_reference=None,
                 interpolation_method=None,
                 pricing_location=None):

        self._mgr = VolSurfaceMgr(underlier,
                                  pricing_location,
                                  strike_reference)
        self.set_interpolator(interpolation_method)


    @property
    def dates(self):
        return self.raw.index
    @property
    def strike_reference(self):
        return self._mgr.strike_reference
    @property
    def security(self):
        return self._mgr.underlier
    @property
    def raw(self):
        return self._mgr._ivol_cache[self.security].copy()
    @property
    def rate_curve(self):
        return self._mgr._rate_curve
    @property
    def funding_curve(self):
        return self._mgr._funding_curve
    @property
    def spot_prices(self):
        return self._mgr._spot_prices
    @property
    def delta_type(self):
        return deltaConvention.get(self.security)

    def set_interpolator(self, interpolator):
        if interpolator in [ Interpolator.VANNA_VOLGA, Interpolator.VANNA_VOLGA.value ]:
           self.interpolator = VannaVolga(self.raw,
                                          self.spot_prices,
                                          self.rate_curve,
                                          self.funding_curve,
                                          deltaConvention.get(self.security))

        elif interpolator in [ Interpolator.GAUSSIAN_KERNEL_SMOOTHING,
                               Interpolator.GAUSSIAN_KERNEL_SMOOTHING.value ]:
           self.interpolator = GaussianKernel(self.raw,
                                              self.spot_prices,
                                              self.rate_curve,
                                              self.funding_curve)

        elif interpolator in [Interpolator.CUBIC_SPLINE,
                               Interpolator.CUBIC_SPLINE.value]:
            self.interpolator = CubicSpline(self.raw,
                                            self.spot_prices,
                                            self.rate_curve,
                                            self.funding_curve)
        else:
            raise ValueError('Error - Interpolation scheme {} not supported'.format(interpolator))

    def get_spot_prices(self, dates):
        return self.spot_prices.reindex(dates)

    def get_forward_prices(self, dates, maturities):
        return self._mgr.get_forward_prices(dates, maturities)

    def get_risk_free(self, dates, maturities):
        return self.rate_curve.get_curve(dates, maturities)

    def get_funding_rate(self, dates, maturities):
        return self.funding_curve.get_curve(dates, maturities)

    def get_ivols(self, relative_strike, maturity, maturity_type, pricing_dates=None, strike_reference=None):

        if pricing_dates is None:
           pricing_dates = pd.to_datetime(np.unique(self.dates))

        if strike_reference is None:
           strike_reference = self.strike_reference

        return self.interpolator.get_ivols(pd.to_datetime(pricing_dates),
                                           strike_reference=strike_reference,
                                           relative_strike=np.array(relative_strike),
                                           maturity_type=maturity_type,
                                           maturities=np.array(maturity))

    def get_option_price(self):
        pass

    def get_bsdelta(self):
        pass

    def get_bsgamma(self):
        pass

    def get_bsvega(self):
        pass

    def get_bsvolga(self):
        pass

    def get_bsvanna(self):
        pass

    def get_bstheta(self):
        pass

    def get_omega(self):
        pass

    def get_lambda(self):
        pass

    def simulate_vol_surface(self):
        pass

    def simulate_option_prices(self):
        pass

    def get_straddle_prices(self):
        pass

    def get_strangle_prices(self):
        pass

    def get_risk_reversal_prices(self):
        pass

    def get_butterfly_prices(self):
        pass





if __name__ == "__main__":

    self = AbstractVolSurface('GBPUSD',
                              strike_reference=StrikeReference.DELTA,
                              interpolation_method=Interpolator.CUBIC_SPLINE)

    _DELTAS = [-0.5]
    _MATURITIES = ['1m']

    pricing_dates = pd.date_range('01-01-2022', '31-12-2022', freq='B')
    vols_t = self.get_ivols(strike_reference=StrikeReference.DELTA,
                            relative_strike=_DELTAS,
                            maturity_type=MaturityType.MATURITY_STRING,
                            maturity=_MATURITIES,
                            pricing_dates=pricing_dates)
    vols_t['open_dates'] = vols_t['date']


    pd_ = pd.to_datetime(vols_t.get('date').values) + pd.offsets.BDay(1)
    ed_ = pd.to_datetime(vols_t.get('expiry').values)
    vols_T = self.get_ivols(pricing_dates=pd_,
                            strike_reference=StrikeReference.STRIKE_PRICE,
                            relative_strike=np.asarray(vols_t['k']),
                            maturity_type=MaturityType.EXPIRY_DATE,
                            maturity=ed_)

    vols_T['expiry'] = vols_t['expiry']
    vols_T['x'] = vols_t['x']
    vols_T['open_dates'] = vols_t['open_dates']
    sigs = pd.concat((vols_t.dropna(), vols_T.dropna()), axis=0).dropna()
    sigs['elapsed'] = (sigs.get('date') - sigs.get('open_dates'))

    sigs = pd.concat((sig_t, sig_T), axis=0)


    # Open Date
    # Expiration Date
    # Pricing Date
    # Strike
    # Implied Volatility






