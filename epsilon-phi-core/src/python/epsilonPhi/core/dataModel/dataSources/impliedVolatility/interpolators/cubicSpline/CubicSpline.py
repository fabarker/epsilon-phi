import numpy as np
import pandas as pd
from epsilonPhi.core.utils.FrameUtils import FrameUtils
from epsilonPhi.core.lib.curve_fitting.cubic_spline.cubic_spline import cubic_spline as cubicspline
from epsilonPhi.core.dataModel.dataSources.curves.abstractCurve.AbstractCurve import AbstractCurve
from epsilonPhi.core.utils.NumpyUtils import NumpyUtils as npu
from numba import jit
from tqdm import tqdm
from epsilonPhi.core.utils.OptionUtils import *
from epsilonPhi.core.dataModel.enums.ImpliedVolatility import *
from epsilonPhi.core.utils.MathUtils import cubic_spline
from epsilonPhi.core.dataModel.enums.ImpliedVolatility import MaturityType


class CubicSpline(object):
    def __init__(self, ivols, s, rd, rf):

        # Set ivols panel..
        self.set_ivols(ivols)

        # Sset spot rates
        self.set_spot_prices(s)

        # Set interest and funding rate curves
        self.set_risk_free_rate_curve(rd)
        self.set_funding_rate_curve(rf)
        self.set_forward_prices()
        self._x = None


    def set_ivols(self, ivols):

        # Process the ivols dataframe
        _ivols = ivols.reset_index()

        cols = _ivols.columns
        if 'relative_strike' in cols:
            _ivols = _ivols[_ivols.columns.difference(['relative_strike'])]

        self._strike_reference = _ivols.columns.difference(['date', 'mid', 't', 'k', 'exp']).values.item()
        _ivols = _ivols.rename(columns={self._strike_reference: 'x'})
        _dduped = _ivols.drop_duplicates(subset=['x', 't', 'date'])

        # Set the dataframe in the object
        self._ivols = _dduped.set_index('date').pivot(columns=['t', 'x']).sort_index()


    def set_spot_prices(self, df):
        self._spot = df.loc[self.dates].to_frame()

    def set_risk_free_rate_curve(self, rd):
        self._rate_curve = rd
        self.set_risk_free_rate()

    def set_funding_rate_curve(self, rf):
        self._funding_curve = rf
        self.set_funding_rate()

    def set_forward_prices(self):
        self._f = (self._spot.values *
                   np.exp(-1 * self.get_rd(self.dates, np.unique(self.mat)) * np.unique(self.mat).reshape(1, -1)) /
                   np.exp(-1 * self.get_rf(self.dates, np.unique(self.mat)) * np.unique(self.mat).reshape(1, -1)))
        self._forward_curve = AbstractCurve(self._f)

    def set_risk_free_rate(self):
        self._rd = self._rate_curve.get_curve(self.dates, self.t[0,:])

    def set_funding_rate(self):
        self._rf = self._funding_curve.get_curve(self.dates, self.t[0,:])

    @property
    def k_ref(self):
        return self._strike_reference
    @property
    def T(self):
        return len(self.dates)
    @property
    def dates(self):
        return self.ivols.index
    @property
    def ivols(self):
        return self._ivols.copy()
    @property
    def sig(self):
        return self.ivols.get('mid')
    @property
    def x(self):
        return self.strikes.repeat(self.T, axis=0)
    @property
    def strikes(self):
        return self.ivols.columns.get_level_values(self._strike_reference).values[None, :]
    @property
    def mat(self):
        return self.ivols.columns.get_level_values('t').values[None, :]
    @property
    def t(self):
        return self.mat.repeat(self.T, axis=0)
    @property
    def k(self):
        return self.ivols.get('k')
    @property
    def rf(self):
        return self._rf.values
    @property
    def rd(self):
        return self._rd.values
    @property
    def s(self):
        return self._spot.values
    @property
    def f(self):
        return self._f.values

    def get_f(self, pricing_dates, maturities, is_stacked=False):
        return self._forward_curve.get_curve(pricing_dates, maturities, is_stacked)
    def get_rd(self, pricing_dates, maturities, is_stacked=False):
        return self._rate_curve.get_curve(pricing_dates, maturities, is_stacked)
    def get_rf(self, pricing_dates, maturities, is_stacked=False):
        return self._funding_curve.get_curve(pricing_dates, maturities, is_stacked)
    def get_s(self, pricing_dates):
        return self._spot.reindex(pricing_dates)

    def get_log_moneyness(self, sig):
        _s = self.get_s(sig.get('date')).values.flatten().astype(np.float64)
        return np.log(sig.get('k')/_s).to_frame('x')

    def get_deltas(self, sig):

        _t = sig.get('t').values.flatten().astype(np.float64)
        _s = self.get_s(sig.get('date')).values.flatten().astype(np.float64)
        _v = sig.get('mid').values.astype(np.float64)
        _k = sig.get('k').values.astype(np.float64)

        rd = self.get_rd(sig.get('date'), _t, True).values.astype(np.float64)
        rf = self.get_rf(sig.get('date'), _t, True).values.astype(np.float64)

        res = nb_delta(_s, _t, _k, rd, rf, _v, 2, -1)
        return pd.DataFrame(np.round(res, 5), columns=['x']).replace(0, np.nan)

    def get_ivols(self, pricing_dates, strike_reference, relative_strike, maturity_type, maturities):

        maturities = np.asarray(maturities)
        relative_strike = np.asarray(relative_strike)

        if (pricing_dates.shape != maturities.shape) or (pricing_dates.shape != relative_strike.shape):
            pricing_dates, maturities, relative_strike = (
                npu.flat_meshgrid(pricing_dates, maturities, relative_strike))

        assert pricing_dates.shape == maturities.shape, 'Error - dimension mis-match'
        assert pricing_dates.shape == relative_strike.shape, 'Error - dimension mis-match'

        if maturity_type in [MaturityType.MATURITY_STRING, MaturityType.MATURITY_STRING.value]:
           matdates = DateUtils.expiry_from_settlement(pricing_dates, maturities)
           mat = DateUtils.get_date_delta(pricing_dates, matdates, True)
           ivols = self.load_ivols(pricing_dates, mat, relative_strike, strike_reference)
           ivols['expiry'] = matdates
           ivols['mats'] = maturities
           return ivols
        elif maturity_type in [MaturityType.EXPIRY_DATE, MaturityType.EXPIRY_DATE.value]:
           mat = DateUtils.get_date_delta(pricing_dates, maturities, True)
           ivols = self.load_ivols(pricing_dates, mat, relative_strike, strike_reference)
           ivols['expiry'] = maturities
           return ivols
        elif maturity_type in [MaturityType.YEARFRAC, MaturityType.YEARFRAC.value]:
           return self.load_ivols(pricing_dates, maturities, relative_strike, strike_reference)

    def load_ivols(self, pricing_dates, maturities, relative_strike, strike_reference):

        # Interpolate along the time direction - Flat Forward Interpolation.
        _sigs = self.interpolate_maturities(maturities, pricing_dates)
        _sigs['k'] = self.get_strikes(_sigs, self._strike_reference)

        # Compute the surface values for the strike reference we care about
        if strike_reference in [StrikeReference.DELTA, StrikeReference.DELTA.value]:
            _sigs['xk'] = self.get_deltas(_sigs)
        elif strike_reference in [StrikeReference.STRIKE_PRICE, StrikeReference.STRIKE_PRICE.value]:
            _sigs['xk'] = self.get_log_moneyness(_sigs)
            relative_strike = np.log(relative_strike/self.get_s(pricing_dates).values.flatten())
        else:
            raise ValueError('Error - strike reference {} not supported'.format(strike_reference))

        vk = _sigs.set_index(['date', 't']).pivot(columns='x').reindex(zip(pricing_dates, maturities))
        v = vk.get('mid').sort_index(axis=1)
        k = vk.get('xk').sort_index(axis=1)

        _ivols = self.interpolate(v, k, relative_strike)
        _ivols['k'] = self.get_strikes(_ivols, strike_reference)
        return _ivols.copy()


    def get_strikes(self, ivols, strike_reference):

        # Compute the surface values for the strike reference we care about
        if strike_reference in [StrikeReference.DELTA, StrikeReference.DELTA.value]:
            return self.get_strikes_from_delta(ivols)
        elif strike_reference in [StrikeReference.STRIKE_PRICE, StrikeReference.STRIKE_PRICE.value]:
            return np.exp(ivols.get('x')) * self.get_s(ivols.get('date')).values.flatten()
        else:
            raise ValueError('Error - strike reference {} not supported'.format(strike_reference))

    def get_strikes_from_delta(self, df_):

        _v = df_.get('mid').values.astype(np.float64)
        _x = df_.get('x').values.astype(np.float64)
        _s = self.get_s(df_.get('date')).values.ravel().astype(np.float64)
        _t = df_.get('t').values.astype(np.float64)

        rd = self.get_rd(df_.get('date'), _t, True).values.astype(np.float64)
        rf = self.get_rf(df_.get('date'), _t, True).values.astype(np.float64)
        ot = np.sign(_x).astype(np.int64)
        k = nb_strike(_s, _t, rd, rf, ot, _x, 2, _v)
        return pd.DataFrame(k, index=df_.index, columns=['k'])


    @staticmethod
    @njit([float64[:](float64[:,:], float64[:,:], float64[:])], cache=True)
    def spline_interp(x, y, xp):

        xp_ = np.asarray(xp, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        z = np.full(xp.shape, np.nan)
        for t in range(len(xp)):
            x_ = x[t, ~np.isnan(x[t, :])]
            y_ = y[t, ~np.isnan(y[t, :])]

            if len(x_) == len(y_):

                _loc = np.argsort(x_)

                x_sorted = x_[_loc]
                y_sorted = y_[_loc]
                x_hat = xp_[t]

                l_X = len(x_)
                if l_X > 3:
                    z.flat[t] = cubicspline(x_sorted, y_sorted, x_hat).item()
                elif l_X > 2:
                    z.flat[t] = quad_poly_regression(x_sorted, y_sorted, x_hat)
                elif np.any(np.abs(x_hat - x_sorted) < 1e-9):
                    z.flat[t] = y_sorted[np.abs(x_hat - x_sorted) < 1e-9].item()
                else:
                    z.flat[t] = np.nan
        return z


    def interpolate(self, v, k, x):

        # Target Observations...
        fy = np.asarray(v, dtype=np.float64)
        fx = np.asarray(k, dtype=np.float64)
        xp = np.asarray(x, dtype=np.float64)

        res = np.column_stack((CubicSpline.spline_interp(fx, fy, xp), x))
        res[(res[:, 0] <= 0) | (res[:, 0] > np.nanmax(fy) * 5), 0] = np.nan
        return pd.DataFrame(res, index=v.index, columns=['mid', 'x']).reset_index()


    def interpolate_maturities(self, maturities, pricing_dates=None):

        if maturities is None:
           return

        if pricing_dates is None:
           _df = self.sig
        else:
           _df = self.sig.reindex(np.unique(pricing_dates)).dropna(how='all', axis=0)

        res = FrameUtils.rowise_flat_forward_interpolation_on_groups(_df,
                                                                      np.unique(maturities),
                                                                      x_lev='t',
                                                                      group='x').dropna(how='all', axis=1)

        # Drop Columns
        _abs_x = np.abs(res.columns.get_level_values('x'))
        _x_drop = np.setdiff1d(_abs_x, [0.75, 0.5, 0.25])
        drop_cols = (np.array(res.columns.get_level_values('t')) < 1/12) & _abs_x.isin(_x_drop)
        res_ = res.iloc[:, ~drop_cols]
        return res_.stack(level=[0, 1]).to_frame('mid').reset_index(drop=False)



if __name__ == "__main__":

        from epsilonPhi.core.dataModel.dataSources.impliedVolatility.Models import *
        from epsilonPhi.core.dataModel.dataSources.impliedVolatility.VolSurfaceMgr import VolSurfaceMgr

        underlier = 'GBPUSD'
        vsm = VolSurfaceMgr(underlier,
                             pricing_location='NYC',
                             strike_reference=StrikeReference.DELTA)


        ### Get ivols
        ivols = vsm._ivol_cache[underlier].copy()

        s = vsm._spot_prices
        rd = vsm._rate_curve
        rf = vsm._funding_curve

        self = CubicSpline(ivols,
                           s,
                           rd,
                           rf)

        strikes = [1.19200405116997]
        OD =  pd.to_datetime('25/05/2023')
        PDs = pd.to_datetime(['25/05/2023',
              '26/05/2023',
              '30/05/2023',
              '31/05/2023',
              '01/06/2023',
              '02/06/2023',
              '05/06/2023',
              '06/06/2023',
              '07/06/2023',
              '08/06/2023',
              '09/06/2023',
              '12/06/2023',
              '13/06/2023',
              '14/06/2023',
              '15/06/2023',
              '16/06/2023',
              '20/06/2023',
              '21/06/2023',
              '22/06/2023',
              '23/06/2023',
              '26/06/2023',
              '27/06/2023'])

        mat = 1/12
        elapsed = ((PDs - OD)/np.timedelta64(1, 'D')).values.astype(np.int64)/365
        ttm = np.maximum(0, mat - elapsed)
        fxivols = self.get_ivols(pd.to_datetime(PDs),
                                 strike_reference=StrikeReference.STRIKE_PRICE,
                                 relative_strike=np.array(strikes),
                                 maturity_type=MaturityType.YEARFRAC,
                                 maturities=np.array(ttm))

        _vols = fxivols.set_index(['date', 't'], drop=True).loc[list(zip(PDs, ttm))]

        import utils_find_1st.find_1st


