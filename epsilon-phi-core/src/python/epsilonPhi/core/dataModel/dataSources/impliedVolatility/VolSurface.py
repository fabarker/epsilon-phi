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
    def underlier(self):
        return self._mgr.underlier
    @property
    def raw(self):
        return self._mgr._ivol_cache[self.underlier].copy()
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
        return deltaConvention.get(self.underlier)

    def set_interpolator(self, interpolator):
        if interpolator in [ Interpolator.VANNA_VOLGA, Interpolator.VANNA_VOLGA.value ]:
           self.interpolator = VannaVolga(self.raw,
                                          self.spot_prices,
                                          self.rate_curve,
                                          self.funding_curve,
                                          deltaConvention.get(self.underlier))
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
        return self.spot_prices.loc[dates]

    def get_forward_prices(self, dates, maturities):
        return self._mgr.get_forward_prices(dates, maturities)

    def get_risk_free(self, dates, maturities):
        return self.rate_curve.get_curve(dates, maturities)

    def get_funding_rate(self, dates, maturities):
        return self.funding_curve.get_curve(dates, maturities)

    def get_keys(self, pricing_dates, relative_strike, maturity):

        if (pricing_dates.shape != relative_strike.shape) & (pricing_dates.shape != maturity.shape):
            _d, _k, _m = np.meshgrid(pricing_dates, relative_strike, maturity)
            return list(zip(_d, _k, _m))
        else:
            return list(zip(pricing_dates, relative_strike, maturity))

    def load_ivols(self, pricing_dates, strike_reference, relative_strike, maturity):
        _vols = self.interpolator.get_ivols(pricing_dates,
                                            strike_reference=strike_reference,
                                            relative_strike=relative_strike,
                                            maturities=maturity)


    def get_ivols(self, relative_strike, maturity, pricing_dates=None, strike_reference=None):

        if pricing_dates is None:
           pricing_dates = pd.to_datetime(np.unique(self.dates))

        if strike_reference is None:
           strike_reference = self.strike_reference

        return self.interpolator.get_ivols(pricing_dates,
                                           strike_reference=strike_reference,
                                           relative_strike=relative_strike,
                                           maturities=maturity)

    def get_option_prices(self,
                          pricing_dates,
                          strikes,
                          maturities):
        pass

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

    _DELTAS = [-0.1, -0.25, -0.5, -0.75, -0.9]
    _MATURITIES = [1/12, 3/12, 6/12, 9/12, 12/12]

    pricing_dates = pd.date_range('01-01-1996', '31-12-2022', freq='B')

    vols_t = self.get_ivols(strike_reference=StrikeReference.DELTA,
                            relative_strike=_DELTAS,
                            maturity=_MATURITIES,
                            pricing_dates=pricing_dates)

    from epsilonPhi.core.utils.DateUtils import DateUtils
    str_ = DateUtils.mat_to_Rdate(vols_t.get('t').values)

    DateUtils.get_expiry_date(vols_t.get('date')[0], 6/6)

    day_to_expiry = (vols_t.get('t') * 365).astype(np.int64)
    vols_t['expiry'] = vols_t.get('date') + pd.to_timedelta(day_to_expiry, 'D')

    pd_ = vols_t.get('date') + pd.to_timedelta(1, 'D')
    td_ = (vols_t['expiry'] - pd_).dt.days.values / 365


    vols_T = self.get_ivols(pricing_dates=pd_,
                            strike_reference=StrikeReference.STRIKE_PRICE,
                            relative_strike=np.asarray(vols_t['k']),
                            maturity=td_)

    vols_T['expiry'] = vols_t['expiry']
    vols_T['x'] = vols_t['x']

    sig_t = vols_t.set_index(['date', 'k', 'expiry'], drop=True).dropna().get('mid')
    sig_T = vols_T.set_index(['date', 'k', 'expiry'], drop=True).dropna().get('mid')


    sigs = pd.concat((sig_t, sig_T), axis=0)






