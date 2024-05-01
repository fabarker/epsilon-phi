from epsilonPhi.core.dataModel.dataSources.impliedVolatility.VolSurfaceMgr import VolSurfaceMgr
from epsilonPhi.core.dataModel.dataSources.impliedVolatility.interpolators.cubicSpline.Spline import Spline
from epsilonPhi.core.dataModel.dataSources.impliedVolatility.interpolators.rolloos.Rolloos import Rolloos
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.dataModel.enums.ImpliedVolatility import *
from epsilonPhi.core.dataModel.enums.ImpliedVolatility import *
from epsilonPhi.core.utils.DateUtils import DateUtils
from epsilonPhi.core.utils.OptionUtils import *
import pandas as pd
import numpy as np

mgr = SessionMgr()

class AbstractVolSurface(object):
    _cache = {}

    def __init__(self,
                 underlier,
                 strike_reference=StrikeReference.DELTA,
                 interpolation_method=Interpolator.CUBIC_SPLINE,
                 pricing_location=None):


        self._underlier = underlier
        self._strike_reference = strike_reference
        self._bid_ask_vol_spread = 0.3/100
        self._pricing_location = pricing_location

        # Load the underlying data
        self.set_vol_surface_manager()
        self.set_interpolator(interpolation_method)

    @staticmethod
    def get_volatility_surface(underlier, **kwargs):

        asset_class = mgr.get_underlier_asset_class(underlier)
        if asset_class == 'FX':
           return FXVolSurface(underlier, **kwargs)
        else:
           return AbstractVolSurface(underlier, **kwargs)


    @property
    def pricing_location(self):
        return self._pricing_location
    @property
    def common_dates(self):
        return np.intersect1d(np.unique(self.market_data.index),
                              self.spot_prices.dropna().index)
    @property
    def strike_reference(self):
        return self._strike_reference
    @property
    def underlier(self):
        return self._underlier
    @property
    def security(self):
        return self.underlier
    @property
    def market_data(self):
        return self._mgr.get_ivols()
    @property
    def rate_curve(self):
        return self._mgr.get_rate_curve()
    @property
    def funding_curve(self):
        return self._mgr.get_funding_curve()
    @property
    def forward_curve(self):
        return self._mgr.get_forward_curve()
    @property
    def spot_prices(self):
        return self._mgr.get_spot_prices()
    @property
    def delta_type(self):
        return deltaConvention.get(self.security)

    def set_pricing_location(self, pricing_location):
        self._pricing_location = pricing_location
        self.load_vol_surface_manager()

    def set_vol_surface_manager(self, vol_surface_manager=None):

        if vol_surface_manager is None:
           self.load_vol_surface_manager()
        else:
           self._mgr = vol_surface_manager

    def load_vol_surface_manager(self):
        self._mgr = VolSurfaceMgr(self.underlier,
                                  self.pricing_location,
                                  self.strike_reference)


    def set_bid_ask_vol_spreads(self, bid_ask_vol):
        self._bid_ask_vol_spread = bid_ask_vol

    def set_interpolator(self, interpolator):

        if interpolator == Interpolator.CUBIC_SPLINE:
            self.interpolator = Spline(self.market_data)
        elif interpolator == Interpolator.ROLLOOS:
            self.interpolator = Rolloos(self.market_data)
        else:
            raise ValueError('Error - Interpolation scheme {} not supported'.format(interpolator))

        # set deposit rates
        self.interpolator.set_deposit_rate_curve(self.rate_curve)
        self.interpolator.set_funding_rate_curve(self.funding_curve)

    def get_spot_prices(self, dates):
        return self.spot_prices.reindex(dates)

    def get_forward_prices(self, dates, maturities, is_stacked=False):
        r = self.get_risk_free(dates, maturities, is_stacked)
        q = self.get_funding_rate(dates, maturities, is_stacked)
        s = self.get_spot_prices(dates)
        return s.values * np.exp(-q * maturities) / np.exp(-r * maturities)

    def get_hedge_instrument_prices(self, dates, t):

        if 'forward' in self.delta_type.name.lower():
            return self.get_forward_prices(dates, t, is_stacked=True)
        else:
            return self.get_spot_prices(dates)

    def get_risk_free_rate_curve(self, currency):
        return self._mgr.get_risk_free_rate_curve(currency)

    def get_risk_free(self, dates, maturities, is_stacked=False):
        return self.rate_curve.get_curve(dates, maturities, is_stacked)

    def get_funding_rate(self, dates, maturities, is_stacked=False):
        return self.funding_curve.get_curve(dates, maturities, is_stacked)

    def get_ivols(self, pricing_dates, strike_reference, relative_strike, maturity, maturity_type):
        return self.interpolator.get_ivols(pricing_dates,
                                           strike_reference=strike_reference,
                                           relative_strike=relative_strike,
                                           maturity_type=maturity_type,
                                           maturities=maturity)

    def get_fixed_strike_implied_vols(self, dates, strikes, time_to_maturity):
        sig = self.get_ivols(dates, StrikeReference.STRIKE_PRICE, strikes, time_to_maturity, MaturityType.EXPIRY_DATE)
        _sig = sig.set_index('date').groupby(['k', 'expiry'], group_keys=False, dropna=False).apply(lambda x: x.sort_index().ffill())
        reindexed = _sig.reset_index(drop=False).set_index(['date', 'k', 'expiry']).loc[zip(dates, strikes, time_to_maturity)]
        return reindexed.get('mid').values

    def get_option_strike_prices(self, dates, strike_reference, relative_strike, maturity, maturity_type):
        sig = self.get_ivols(dates, strike_reference, relative_strike, maturity, maturity_type)
        return sig.dropna()

    def get_option_prices(self, open_dates, strike_reference, relative_strike, maturity_type, maturities, option_type):

        if isinstance(option_type, OptionTypes):
           option_type = option_type.value

        # 1. Get the Implied Vols at Open
        open_vols = self.get_ivols(open_dates, strike_reference, relative_strike, maturities, maturity_type).dropna()
        dates, k, open_date, expiry_date, t = self.get_contract_pricing_dates(open_vols.date, open_vols.expiry, open_vols.k)

        i = self.get_fixed_strike_implied_vols(dates, k, expiry_date)
        q = self.get_funding_rate(dates, t, True).values
        r = self.get_risk_free(dates, t, True).values
        s = self.get_spot_prices(dates).values
        f = self.get_forward_prices(dates, t, True).values
        h = self.get_hedge_instrument_prices(dates, t)

        p = self.get_mid_premium(s, t, k, r, q, i, option_type)
        b = self.get_bid_premium(s, t, k, r, q, i, option_type)
        a = self.get_ask_premium(s, t, k, r, q, i, option_type)

        cp = self.get_cash_premium(s, t, k, r, q, i, option_type)

        d = self.get_bsdelta(s, t, k, r, q, i, self.delta_type.value, option_type)
        g = self.get_bsgamma(s, t, k, r, q, i)
        v = self.get_bsvega(s, t, k, r, q, i)
        va = self.get_bsvanna(s, t, k, r, q, v)
        vo = self.get_bsvolga(s, t, k, r, q, v)
        theta = self.get_bstheta(s, t, k, r, q, i, option_type)

        # Put the intrinsic value in at the maturity date
        intrinsic = self.get_intrinsic_value(s, k, option_type)
        p[dates == expiry_date] = intrinsic[dates == expiry_date]

        df_ = pd.DataFrame(np.column_stack((i, q, r, s, f, p, d, g, v, t, k, b, a, h, cp, va, vo, theta)), index=[dates, k, open_date, expiry_date])
        df_.columns = ['i','q','r','s', 'f', 'p','d','g','v','t', 'k', 'b', 'a', 'h', 'cp', 'va', 'vo', 'theta']
        df_.index.names = ['date','k','open','expiry']
        df_ = df_[open_date < max(dates)]
        return df_[~df_.index.get_level_values('date').weekday.isin([5, 6])]

    def get_single_contract_pricing_dates(self, start_date, end_date):
        return self.common_dates[(self.common_dates >= start_date) &
                                 (self.common_dates <= end_date)]

    def get_contract_pricing_dates(self, start_date, end_date, strike, capped_forward_pricing_date=None):

        @njit(cache=True, fastmath=True)
        def get_contract_pricing_info(start, end, strikes, cap):

            res = list()
            for t in range(len(start)):
                pricing_dates = np.arange(start[t], cap[t]+1)
                res.extend((np.column_stack((pricing_dates,
                                            np.full(len(pricing_dates), strikes[t]),
                                            np.full(len(pricing_dates), start[t]),
                                            np.full(len(pricing_dates), end[t])))))
            return res

        if capped_forward_pricing_date is None:
           capped_forward_pricing_date = end_date

        specs = get_contract_pricing_info(DateUtils.to_ordinals(start_date),
                                          DateUtils.to_ordinals(end_date),
                                          np.ravel(strike),
                                          DateUtils.to_ordinals(capped_forward_pricing_date))

        date, k, open, close = np.hsplit(np.vstack(specs), 4 )
        ttm = (close - date) / 365

        _common = DateUtils.to_ordinals(self.common_dates)
        idx = (np.isin(date, _common))
        return (DateUtils.from_ordinals(date[idx]),
                k[idx],
                DateUtils.from_ordinals(open[idx]),
                DateUtils.from_ordinals(close[idx]),
                np.ravel(ttm[idx]))

    def get_cash_premium(self, s, t, k, r, q, v, option_type):
        return self.get_midprice(s, t, k, r, q, v, option_type)
    def get_mid_premium(self, s, t, k, r, q, v, option_type):
        return self.get_midprice(self, s, t, k, r, q, v, option_type) / s
    def get_bid_premium(self, s, t, k, r, q, v, option_type):
        return self.get_bidprice(self, s, t, k, r, q, v, option_type) / s
    def get_ask_premium(self, s, t, k, r, q, v, option_type):
        return self.get_askprice(self, s, t, k, r, q, v, option_type) / s
    def get_midprice(self, s, t, k, r, q, v, option_type):
        return self.get_bsprice(s, t, k, r, q, v, option_type)
    def get_bidprice(self, s, t, k, r, q, v, option_type):
        return self.get_bsprice(s, t, k, r, q, v-(self._bid_ask_vol_spread/2), option_type)
    def get_askprice(self, s, t, k, r, q, v, option_type):
        return self.get_bsprice(s, t, k, r, q, v+(self._bid_ask_vol_spread/2), option_type)
    def get_intrinsic_value(self, s, k, option_type):
        return np.clip(option_type * (s - k), 0, np.inf)
    def get_bsprice(self, s, t, k, r, q, v, option_type):
        return blsvalue(s, t, k, r, q, v, option_type)
    def get_bsdelta(self, s, t, k, r, q, v, delta_type, option_type):
        return nb_delta(s, t, k, r, q, v, delta_type, option_type)
    def get_bsgamma(self, s, t, k, r, q, v):
        return bs_gamma(s, t, k, r, q, v)
    def get_bsvega(self, s, t, k, r, q, v):
        return bs_vega(s, t, k, r, q, v)
    def get_bsvolga(self, s, t, k, r, q, v):
        return bs_volga(s, t, k, r, q, v)
    def get_bsvanna(self, s, t, k, r, q, v):
        return bs_vanna(s, t, k, r, q, v)
    def get_bstheta(self, s, t, k, r, q, v, option_type):
        return bs_theta(s, t, k, r, q, v, option_type)
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

    def get_atm_vols(self):
        pass



class FXVolSurface(AbstractVolSurface):
    def __init__(self,
                 currency_pair,
                 interpolation_method=Interpolator.ROLLOOS,
                 pricing_location='NYC'):

        if '/' in currency_pair:
           self._foreign_currency, self._domestic_currency = currency_pair.split('/')
           self._bbid = self._foreign_currency + self._domestic_currency
           self._currency_pair = currency_pair
        else:
           self._bbid = currency_pair
           self._foreign_currency = self._bbid[0:3]
           self._domestic_currency = self._bbid[3:]
           self._currency_pair = self._foreign_currency + '/' + self._domestic_currency

        super(FXVolSurface, self).__init__(self._bbid ,
                                           StrikeReference.DELTA,
                                           interpolation_method,
                                           pricing_location)


    @property
    def premium_currency(self):
        return prem_currency.get(self._bbid)

    def get_domestic_pips_price(self, s, t, k, r, q, v, option_type):
        return blsvalue(s, t, k, r, q, v, option_type)

    def get_domestic_premium_price(self, s, t, k, r, q, v, option_type):
        return blsvalue(s, t, k, r, q, v, option_type) / k

    def get_foreign_pips_price(self, s, t, k, r, q, v, option_type):
        vdf = self.get_domestic_pips_price(s, t, k, r, q, v, option_type)
        return vdf / (s * k)

    def get_foreign_premium_price(self, s, t, k, r, q, v, option_type):
        vdf = self.get_domestic_pips_price(s, t, k, r, q, v, option_type)
        return vdf / s

    def get_ask_premium(self, s, t, k, r, q, v, option_type):

        if self.premium_currency == self._foreign_currency:
            return self.get_foreign_premium_price(s, t, k, r, q, v + (self._bid_ask_vol_spread / 2), option_type)
        else:
            return self.get_domestic_premium_price(s, t, k, r, q, v + (self._bid_ask_vol_spread / 2), option_type)

    def get_bid_premium(self, s, t, k, r, q, v, option_type):

        if self.premium_currency == self._foreign_currency:
            return self.get_foreign_premium_price(s, t, k, r, q, v-(self._bid_ask_vol_spread/2), option_type)
        else:
            return self.get_domestic_premium_price(s, t, k, r, q, v - (self._bid_ask_vol_spread / 2), option_type)

    def get_mid_premium(self, s, t, k, r, q, v, option_type):

        if self.premium_currency == self._foreign_currency:
            return self.get_foreign_premium_price(s, t, k, r, q, v, option_type)
        else:
            return self.get_domestic_premium_price(s, t, k, r, q, v, option_type)
    def get_bsprice(self, s, t, k, r, q, v, option_type):
        pips_v = blsvalue(s, t, k, r, q, v, option_type)
        if self.premium_currency == self._foreign_currency:
            return pips_v / (s * k)
        else:
            return pips_v

    def get_bsgamma(self, s, t, k, r, q, v):
        pips_gamma = bs_gamma(s, t, k, r, q, v)
        if self.premium_currency == self._foreign_currency:
            return pips_gamma / (s * k)
        else:
            return pips_gamma

    def get_vanna(self):
        pips_vanna = bs_vanna(s, t, k, r, q, v)
        if self.premium_currency == self._foreign_currency:
            pips_vega = bs_vega(s, t, k, r, q, v)
            return pips_vanna - (pips_vega / s)
        else:
            return pips_vanna

    def get_intrinsic_value(self, s, k, option_type):
        pip_vf = np.clip(option_type * (s - k), 0, np.inf)
        if self.premium_currency == self._foreign_currency:
            return pip_vf / s
        else:
            return pip_vf / k

    def get_cash_premium(self, s, t, k, r, q, v, option_type):
        if self.premium_currency == self._foreign_currency:
            return self.get_foreign_premium_price(s, t, k, r, q, v, option_type)
        else:
            return self.get_foreign_premium_price(s, t, k, r, q, v, option_type) * k

    def get_hedge_instrument_prices(self, dates, t):
        return super(FXVolSurface, self).get_hedge_instrument_prices(dates, t)

if __name__ == "__main__":

    self = FXVolSurface('USD/JPY')
    dates = pd.date_range('2012-12-31', '31-12-2022', freq='B')
    prices = self.get_option_prices(dates=np.intersect1d(dates, self.common_dates),
                                    strike_reference=StrikeReference.DELTA,
                                    relative_strike=-0.25,
                                    maturity_type=MaturityType.YEARFRAC,
                                    maturities=1/12,
                                    option_type=OptionTypes.EUROPEAN_PUT)







