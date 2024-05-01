import numpy as np
from epsilonPhi.core.dataModel.dataSources.impliedVolatility.VolSurface import AbstractVolSurface
from epsilonPhi.core.dataModel.enums.ImpliedVolatility import *
from epsilonPhi.core.lib.smoothing.GaussianKernel import smooth_2d
from scipy.optimize import lsq_linear
from epsilonPhi.core.utils.DateUtils import DateUtils
from epsilonPhi.core.utils.OptionUtils import *
from epsilonPhi.core.utils.FrameUtils import FrameUtils
import numpy as np
from numba import njit
import pandas as pd
from scipy.stats import t as tstudent

class SurfaceStatArb(object):

    _DELTAS = [-0.10, -0.25, -0.5, -0.75, -0.90]
    _MATURITIES = np.array([1, 3, 6, 9, 12]) / 12

    def __init__(self,
                 underlier,
                 start_date,
                 end_date,
                 interpolator=Interpolator.CUBIC_SPLINE):

        self._underlier = underlier
        self._vs = AbstractVolSurface.get_volatility_surface(self._underlier,
                                                             interpolation_method=interpolator)

        self._start_date = pd.to_datetime(start_date)
        self._end_date = pd.to_datetime(end_date)
        self.load_data()

    def get_ivols(self, dates, strike_reference, relative_strike, maturities, maturity_type):
        return self._vs.get_ivols(dates, strike_reference, relative_strike, maturities, maturity_type).dropna()

    def get_option_prices(self, dates, strike_reference, relative_strike, maturity_type, maturities, option_type):

        ivols = self.get_ivols(dates, strike_reference, relative_strike, maturities, maturity_type).dropna()

        capped_pricing_dates = self._vs.interpolator.get_expiry_dates_from_settlement_dates_tenor(ivols.date, '1m')
        dates, k, open_date, expiry_date, t = self._vs.get_contract_pricing_dates(ivols.date, ivols.expiry, ivols.k, capped_pricing_dates)

        i = self._vs.get_fixed_strike_implied_vols(dates, k, expiry_date)
        q = self._vs.get_funding_rate(dates, t, True).values
        r = self._vs.get_risk_free(dates, t, True).values
        s = self._vs.get_spot_prices(dates).values
        f = self._vs.get_forward_prices(dates, t, True).values
        h = self._vs.get_hedge_instrument_prices(dates, t)

        p = self._vs.get_mid_premium(s, t, k, r, q, i, option_type)
        b = self._vs.get_bid_premium(s, t, k, r, q, i, option_type)
        a = self._vs.get_ask_premium(s, t, k, r, q, i, option_type)

        cp = self._vs.get_cash_premium(s, t, k, r, q, i, option_type)

        d = self._vs.get_bsdelta(s, t, k, r, q, i, self._vs.delta_type.value, option_type)
        g = self._vs.get_bsgamma(s, t, k, r, q, i)
        v = self._vs.get_bsvega(s, t, k, r, q, i)
        va = self._vs.get_bsvanna(s, t, k, r, q, v)
        vo = self._vs.get_bsvolga(s, t, k, r, q, v)
        theta = self._vs.get_bstheta(s, t, k, r, q, i, option_type)

        # Put the intrinsic value in at the maturity date
        intrinsic = self._vs.get_intrinsic_value(s, k, option_type)
        p[dates == expiry_date] = intrinsic[dates == expiry_date]

        df_ = pd.DataFrame(np.column_stack((i, q, r, s, f, p, d, g, v, t, k, b, a, h, cp, va, vo, theta)),
                           index=[dates, k, open_date, expiry_date])
        df_.columns = ['i', 'q', 'r', 's', 'f', 'p', 'd', 'g', 'v', 't', 'k', 'b', 'a', 'h', 'cp', 'va', 'vo', 'theta']
        df_.index.names = ['date', 'k', 'open', 'expiry']
        df_ = df_[open_date < max(dates)]
        return df_[~df_.index.get_level_values('date').weekday.isin([5, 6])]


    def load_data(self):

        # Load Market Data
        locs = ((self._vs.common_dates >= self._start_date) &
                (self._vs.common_dates <= self._end_date))

        market_data = self.get_option_prices(self._vs.common_dates[locs],
                                             StrikeReference.DELTA,
                                             self._DELTAS,
                                             MaturityType.YEARFRAC,
                                             self._MATURITIES,
                                             -1)




if __name__ == "__main__":

    import pandas as pd
    import numpy as np
    from epsilonPhi.core.utils.DateUtils import DateUtils

    SD = pd.to_datetime('31-Dec-2021')
    ED = pd.to_datetime('31-Dec-2023')
    self = SurfaceStatArb('GBPUSD', SD, ED)


