from scipy.optimize import brentq
from epsilonPhi.core.utils.OptionUtils import *
from numba import int32
from epsilonPhi.core.dataModel.enums.ImpliedVolatility import *
from epsilonPhi.core.dataModel.dataSources.impliedVolatility.interpolators.abstractInterpolator.AbstractInterpolator import AbstractInterpolator


class Rolloos(AbstractInterpolator):

    def __init__(self,
                 market_df: pd.DataFrame,
                 **kwargs):

        super(Rolloos, self).__init__(market_df=market_df,
                                     **kwargs)
        self.fit_surface()

    def get_X1(self):
        mkt_data = self._market_df.set_index(['date', 't', 'relative_strike'])
        X1 = (mkt_data.get('ve') - mkt_data.get('mid') * mkt_data.get('vo')).dropna()
        return X1.unstack(level='relative_strike').sort_index()

    def get_X2(self):
        mkt_data = self._market_df.set_index(['date', 't', 'relative_strike'])
        X2 = 0.5 * mkt_data.get('vo').dropna()
        return X2.unstack(level='relative_strike').sort_index()

    def get_X3(self):
        mkt_data = self._market_df.set_index(['date', 't', 'relative_strike'])
        X3 = (mkt_data.get('s') * mkt_data.get('va')).dropna()
        return X3.unstack(level='relative_strike').sort_index()

    def get_Y(self):
        mkt_data = self._market_df.set_index(['date', 't', 'relative_strike'])
        Y = (mkt_data.get('ve') * mkt_data.get('mid') -
                0.5 * mkt_data.get('vo') * np.power(mkt_data.get('mid'), 2)).dropna()
        return Y.unstack(level='relative_strike').sort_index()

    def fit_surface(self):

        print('Fitting volatility surface')
        X1 = self.get_X1().values.astype(np.float64)
        X2 = self.get_X2().values.astype(np.float64)
        X3 = self.get_X3().values.astype(np.float64)
        y = self.get_Y()

        @njit([float64[:, :](float64[:, :], float64[:, :], float64[:, :], float64[:, :])],
              cache=True)
        def estimate_params(y, x1, x2, x3):

            T, K = y.shape
            b = np.full((T, 3), np.nan)
            for t in range(T):

                cols = y[t, :] > 0
                if np.sum(cols) > 2:

                    theta = y[t, cols]
                    X = np.column_stack((x1[t, cols],
                                         x2[t, cols],
                                         x3[t, cols],
                                         ))

                    XtX = np.dot(X.T, X)
                    XtY = np.dot(X.T, theta)
                    betas = np.linalg.solve(XtX, XtY)
                    b[t,:] = betas
            return b

        betas = estimate_params(y.values, X1, X2, X3)
        _fit_df = pd.DataFrame(betas, index=y.index).dropna().unstack(level=1).sort_index()

        # say no to maturities less than one week
        keep_mats = _fit_df.columns.get_level_values(1) > 2 / 365
        _fit_df = _fit_df.iloc[:, keep_mats]

        self._fit_maturities = np.unique(_fit_df.columns.get_level_values(1)).astype(np.float64)
        K = len(self._fit_maturities)
        N = len(np.unique(_fit_df.columns.get_level_values(0)))

        self._fit = _fit_df.values.reshape((_fit_df.shape[0], N, K)).swapaxes(2, 1)
        self._fit_df = _fit_df.swaplevel(1, 0, axis=1).sort_index(level=0, axis=1)

    def solve_for_strike(self, dates, expiries, strikes, option_type=-1):

        s =  np.asarray(self.get_s(dates)).flat
        rf = np.asarray(self.get_rf(dates, expiries, True))
        rd = np.asarray(self.get_rd(dates, expiries, True))

        @njit([float64[:](float64[:], float64[:,:], float64[:,:], float64[:,:,:], float64[:], float64[:], float64[:], int32[:])])
        def fast_fit(s, r, q, fy, ft, m, x, id):

            sigs = np.full(len(id), np.nan)
            for k in range(len(id)):
                i = id[k]
                if i > 0:
                    try:
                        sigs.flat[k] = solve_for_sig_strike_Rolloos(s[i],
                                                                    r[i],
                                                                    q[i],
                                                                    fy[i],
                                                                    ft,
                                                                    m[k],
                                                                    x[k])
                    except:
                        pass
            return sigs

        ids = self.find_date_idx(dates)
        sigs = fast_fit(self.s(), self.rd(), self.rf(),  self._fit, self.t, expiries, strikes, ids)
        sigs[sigs <= 0] = np.nan

        df = pd.DataFrame((sigs, expiries, strikes), index=['mid', 't', 'k']).T
        df['x'] = nb_delta(s, expiries, strikes, rd, rf, sigs, 2, option_type)
        df['date'] = dates
        return df


    def solve_for_delta(self, dates, maturities, deltas, delta_type=DeltaType.FORWARD_DELTA.value):

        s =  np.asarray(self.get_s(dates)).flat
        rf = np.asarray(self.get_rf(dates, maturities, True))
        rd = np.asarray(self.get_rd(dates, maturities, True))

        @njit([float64[:](float64[:], float64[:,:], float64[:,:], float64[:,:,:], float64[:], float64[:], float64[:], int32[:])])
        def fast_fit(s, r, q, fy, ft, m, x, id):

            sigs = np.full(len(id), np.nan)
            for k in range(len(id)):
                i = id[k]
                if i > 0:
                    sigs.flat[k] = solve_for_sigma_delta_Rolloos(s[i],
                                                                 r[i],
                                                                 q[i],
                                                                 fy[i],
                                                                 ft,
                                                                 m[k],
                                                                 x[k])
            return sigs

        ids = self.find_date_idx(dates)
        sigs = fast_fit(self.s(), self.rd(), self.rf(),  self._fit, self.t, maturities, deltas, ids)
        sigs[sigs <= 0] = np.nan

        df = pd.DataFrame((sigs, maturities, deltas), index=['mid', 't', 'x']).T
        df['k'] = nb_strike(s, maturities, rd, rf, np.sign(deltas).astype(np.int32), deltas, 2, sigs)
        df['date'] = dates
        return df


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

        self = Rolloos(ivols)
        self.set_deposit_rate_curve(rd)
        self.set_funding_rate_curve(rf)

        sigs = ivols[ivols.t == ivols.t.unique()[1]]
        D = np.unique(sigs.index)[-400:]

        k = sigs.loc[D].get('k') * 0.9999
        d = pd.to_datetime(k.index)
        t = ivols.t.unique()[2]

        fxivols = self.get_ivols(d[-1],
                                 strike_reference=StrikeReference.DELTA,
                                 relative_strike=np.arange(-0.9, -0.1, 0.01),
                                 maturity_type=MaturityType.YEARFRAC,
                                 maturities=[7/365, 1/12, 1])

        m = fxivols.set_index(['t','x']).get('mid').unstack(level=0)