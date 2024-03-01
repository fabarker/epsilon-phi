import pandas as pd

from epsilonPhi.core.dataModel.alchemist.SessionManager import *
from epsilonPhi.core.dataModel.dataSources.curves.interestRateCurve.IRCurve import IRCurve
from epsilonPhi.core.dataModel.dataSources.curves.abstractCurve.AbstractCurve import AbstractCurve
from epsilonPhi.core.utils.OptionUtils import *
from numba import jit
from epsilonPhi.core.dataModel.enums.Database import PriceQuote
from epsilonPhi.core.dataModel.enums.ImpliedVolatility import StrikeReference
from tqdm import tqdm

sessionMgr = SessionMgr()
session = sessionMgr.getSessionFactory()

_DROP_COLS = ['uid', 'pricing_location', 'relative_strike', 'strike_reference', 'tenor', 'security']

# Default interpolation points for converting between strike reference space
_DELTA_POINTS = np.arange(0.05, 1, 0.05)
_MONEYNESS_POINTS = np.arange(50, 155, 5)/100
_ZSCORE_POINTS = np.arange(-3, 3.5, 0.5)
_CONVEXITY_MNY_POINTS = np.arange(-3, 3.5, 0.1)


class VolSurfaceMgr(object):
    _raw_cache = {}
    _ivols = {}

    def __init__(self, 
                 underlier,
                 pricing_location=None,
                 strike_reference=None):

        # Optional Parameters
        self._underlier = underlier
        self._pricing_location = pricing_location
        self._strike_reference = strike_reference

        # Properties of the raw data
        self._maturities = None
        self._currency = None

        # Spec for the ivols
        self._spec = None
        self._datasource = None

        # Asset Prices and Interest/Funding Rates
        self._interest_rate_curve = None
        self._funding_rate_curve = None
        self._spot_prices = None

        # Validate the target surface. i.e make sure
        # we have the data
        self._validate_underlier()

        # Load Data
        self._load_vol_surface_spec()
        self._load_ivols()


    @property
    def pricing_location(self):
        return self._pricing_location
    @property
    def strike_reference(self):
        return self._strike_reference
    @property
    def security(self):
        return self.underlier
    @property
    def underlier(self):
        return self._underlier
    @property
    def ticker(self):
        return self._spec.get('ticker')
    @property
    def exchange(self):
        return self._pricing_location
    @property
    def currency(self):
        return self._spec.get('currency')
    @property
    def name(self):
        return self._spec.get('name')
    @property
    def region(self):
        return self._spec.get('region')
    @property
    def category(self):
        return self._spec.get('category')
    @property
    def delta_convention(self):
        return deltaConvention.get(self.underlier, DeltaType.SPOT_DELTA)
    @property
    def maturities(self):
        return self._maturities

    @property
    def ds(self):
        if self._datasource is None:
            from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
            self._datasource = GlobalDataSource()
        return self._datasource

    def _load_vol_surface_spec(self):
        self._spec = sessionMgr.get_ivol_spec(self._underlier)

    def _load_spot_prices(self):

        if self.category == 'FX':
           spt = self.ds.get_fx_spot_rates(self.ticker, PriceQuote.BID.value)
        elif self.category in ['Equity Index', 'ETF']:
           spt = self.ds.get_time_series_data_from_ticker(self.ticker, cols='PI')
        elif self.category == 'Future':
           spt = self.ds.get_front_futures_continuous_series_settlement_price(self.ticker[0:3])
        else:
            raise ValueError('Error - category {} not recognized'.format(self.category))

        spt.columns = [self.security]
        self._spot_prices = spt.copy()

    def get_spot_prices(self, dates=None):
        if self._spot_prices is None:
           self._load_spot_prices()

        if dates is None:
            return self._spot_prices.copy()
        else:
            return self._spot_prices.loc[dates].values.flatten()

    def get_forward_prices(self, pricing_dates, maturities):

        s = self.get_spot_prices(pricing_dates)
        if isinstance(maturities, (pd.DataFrame, pd.Series)):
           maturities = maturities.values

        rf = self.get_funding_rates(pricing_dates, maturities)
        rd = self.get_interest_rates(pricing_dates, maturities)
        return s * np.exp(-rf * maturities) / np.exp(-rd * maturities)

    def _load_interest_rate_curve(self):
        self._interest_rate_curve = IRCurve(region=self.region, type=['Interbank', 'Deposit'])

    def get_funding_rate_curve(self, maturities=None, stack=False):

        if self._funding_rate_curve is None:
            self._load_funding_rate_curve()

        if stack:
            return self._funding_rate_curve.get_curve(maturities).stack()
        else:
            return self._funding_rate_curve.get_curve(maturities)

    def get_interest_rate_curve(self, maturities=None, stack=False):

        if self._interest_rate_curve is None:
            self._load_interest_rate_curve()

        if stack:
            return self._interest_rate_curve.get_curve(maturities).stack()
        else:
            return self._interest_rate_curve.get_curve(maturities)

    def _load_funding_rate_curve(self):

        if self.category == 'FX':
            if self.ticker[0:3] == 'EUR':
               fx = 'DEM'
            else:
               fx = self.ticker[0:3]
            rf = IRCurve(currency=fx, type=['Interbank', 'Deposit']).get_curve()
        elif self.category in ['Equity Index', 'ETF']:
            rf = self.ds.get_time_series_data_from_ticker(self.ticker, cols='DY')
            rf = rf.resample('B').ffill() / 100
            rf.columns = [0]
        elif self.category == 'Future':
            rf = self.ds.get_front_futures_continuous_series_settlement_price(self.ticker[0:3])
        else:
            raise ValueError('Error - category {} not recognized'.format(self.category))
        self._funding_rate_curve = AbstractCurve(rf)

    def get_interest_rates(self, pricing_dates, maturities):
        if isinstance(maturities, (pd.DataFrame, pd.Series)):
           maturities = maturities.values

        _curve = self.get_interest_rate_curve(np.unique(maturities), stack=True)
        idx = pd.MultiIndex.from_tuples(list(zip(pricing_dates, maturities)))
        return _curve.loc[idx].values.flatten()

    def get_funding_rates(self, pricing_dates, maturities):
        if isinstance(maturities, (pd.DataFrame, pd.Series)):
           maturities = maturities.values

        _curve = self.get_funding_rate_curve(np.unique(maturities), stack=True)
        idx = pd.MultiIndex.from_tuples(list(zip(pricing_dates, maturities)))
        return _curve.loc[idx].values.flatten()

    def get_ivols(self):
        if self.underlier not in self._ivols.keys():
            self._load_ivols()
        return self._ivols.get(self.underlier)

    def _process_spot_strike_reference_ivol_data(self, frame):
        return self.get_strikes_from_moneyness(frame)

    def _process_forward_strike_reference_ivol_data(self):
        pass

    def _process_delta_strike_reference_ivol_data(self):
        pass





    def _load_ivols(self):

        if self.underlier not in self._ivols.keys():

            # Load raw ivols data from database
            df = sessionMgr.get_ivols(self._underlier, self._pricing_location, index='date')

            # pricing location
            self._pricing_location = df['pricing_location'].unique().item()

            # Insert the maturities
            df['t'] = DateUtils.Rdate_to_mat(df['tenor'].values)
            self._maturities = np.sort(df.t.unique())
            _strike_references = np.unique(df.get('strike_reference').values)

            # Estimate strikes from moneyness
            _spt_idx = df.strike_reference.values == 'spot'
            df.loc[_spt_idx, 'k'] = self.get_strikes_from_moneyness(df['relative_strike'][_spt_idx])

            # Estimate strikes from forwards
            _fwd_idx = df.strike_reference.values == 'forward'
            df.loc[_fwd_idx, 'k'] = self.get_strikes_from_forward_moneyness(df['relative_strike'][_fwd_idx],
                                                                            df['t'][_fwd_idx])

            # Estimates strikes from deltas
            _del_idx = df.strike_reference.values == 'delta'
            df.iloc[_del_idx, -1] = self.get_strikes_from_deltas(df['mid'][_del_idx],
                                                                 df['t'][_del_idx],
                                                                 df['relative_strike'][_del_idx],
                                                                 self.delta_convention)

            # Estimate the strikes for delta neutral points
            _DN_idx = df.relative_strike == -999
            df.iloc[_DN_idx, -1] = self.get_strikes_from_atm_delta_neutral(df['mid'][_DN_idx],
                                                                           df['t'][_DN_idx],
                                                                           self.delta_convention)

            # Estimate all deltas from the strikes (We may need this for Vanna-Volga Interpolation)
            self._ivols[self.underlier] = self.convert_to_strike_reference(df, self.strike_reference)

    def get_atm_delta_neutral_deltas(self, ivols, t, deltaTypeValue, option_type_value):
        rf = self.get_forward_prices(ivols.index, t)
        return delta_from_delta_neutral_straddle_quote(t,
                                                       rf,
                                                       ivols,
                                                       deltaTypeValue,
                                                       option_type_value)

    def get_strikes_from_atm_delta_neutral(self, ivols, t, deltaTypeValue):
        s = self.get_spot_prices(ivols.index)
        rf = self.get_funding_rates(ivols.index, t)
        rd = self.get_interest_rates(ivols.index, t)
        return atm_delta_neutral_strike(s,
                                        t,
                                        rd,
                                        rf,
                                        ivols,
                                        deltaTypeValue)

    def get_strikes_from_forward_moneyness(self, x, t):
        return x * self.get_forward_prices(x.index, t)

    def get_strikes_from_moneyness(self, x, s=None):
        if s is None:
           s = self.get_spot_prices(x.index)
        return x * s

    def get_moneyness_from_strikes(self, k, s=None):
        if s is None:
           s = self.get_spot_prices(x.index)
        return k / s

    def get_strikes_from_log_moneyness(self, lnx, t, f=None):
        if f is None:
           f = self.get_forward_prices(lnx.index, t)
        return f * np.exp(lnx)

    def get_log_moneyness_from_strikes(self, k, t, f=None):
        if f is None:
           f = self.get_forward_prices(k.index, t)
        return np.log(k / f)

    def get_deltas_from_strikes(self, ivols, t, k, delta_type, option_type):

        s = self.get_spot_prices(ivols.index)
        rf = self.get_funding_rates(ivols.index, t)
        rd = self.get_interest_rates(ivols.index, t)
        return fast_delta(s,
                          t,
                          k,
                          rd,
                          rf,
                          ivols,
                          delta_type,
                          option_type)

    def get_strikes_from_deltas(self, ivols, t, delta, delta_type):

        s = self.get_spot_prices(ivols.index)
        rf = self.get_funding_rates(ivols.index, t)
        rd = self.get_interest_rates(ivols.index, t)
        return solve_for_strike(s,
                                t,
                                rd,
                                rf,
                                np.sign(delta),
                                delta,
                                delta_type,
                                ivols)

    def get_convexity_moneyness_from_strikes(self, ivols, k, t):

        # Compute the convexity adjusted moneyness measure
        sigsq = np.power(ivols, 2)
        lnm = self.get_log_moneyness_from_strikes(k, t)
        zp = lnm + 0.5 * sigsq * t
        return zp / (ivols * np.sqrt(t))

    def get_strikes_from_convexity_moneyness(self, ivols, x, t):

        # Compute the convexity adjusted moneyness measure
        f = self.get_forward_prices(ivols.index, t)
        sigsq = np.power(ivols, 2)
        return f * np.exp(x * (ivols * np.sqrt(t)) - 0.5 * sigsq * t)

    def get_zscore_from_strikes(self, ivols, k, t):
        lmn = self.get_log_moneyness_from_strikes(k, t)
        return lmn / (ivols * np.sqrt(t))

    def get_strikes_from_zscore(self, ivols, x, t):
        f = self.get_forward_prices(ivols.index, t)
        return f * np.exp(x * (ivols * np.sqrt(t)))

    def __solve_for_deltas(self, _df, opt_type, mny_type, dlt_type):

        if dlt_type == 1:
           x = self.get_spot_prices(_df.index)
        elif dlt_type == 2:
           x = self.get_forward_prices(_df.index, _df['t'])
        else:
            raise ValueError('Error')

        lnm = np.log(_df['k'].values / x)
        if np.sign(mny_type) == 0:
            idx = lnm == 0
        elif np.sign(mny_type) == -1:
            idx = lnm < 0
        elif np.sign(mny_type) == 1:
            idx = lnm > 0
        else:
            raise ValueError('Error')

        _tmp = _df.iloc[idx, :].copy()
        _tmp['x'] = self.get_deltas_from_strikes(_tmp['mid'], _tmp['t'], _tmp['k'], dlt_type, opt_type).values
        return _tmp.copy()

    def convert_to_strike_reference(self, df, strike_reference):

        if strike_reference in [StrikeReference.SPOT_DELTA, StrikeReference.FORWARD_DELTA]:
            return self.convert_to_delta_strike_reference(df, strike_reference)
        if strike_reference in [StrikeReference.MONEYNESS]:
            return self.convert_to_moneyness_strike_reference(df)
        if strike_reference in [StrikeReference.Z_SCORE]:
            return self.convert_to_zscore_strike_reference(df)
        if strike_reference in [StrikeReference.CONVEXITY_MN]:
            return self.convert_to_convexity_moneyness_strike_reference(df)
        else:
            return df.copy()

    def convert_to_zscore_strike_reference(self, df):
        raise ValueError('Not Implemented')

    def convert_to_convexity_moneyness_strike_reference(self, df):

        # Compute the convexity moneyness measures for the data we have
        df['x'] = self.get_convexity_moneyness_from_strikes(df['mid'], df['k'], df['t'])
        ivols = self.interpolate(df, strikes=_CONVEXITY_MNY_POINTS, maturities=df['t'])
        ivols['k'] = self.get_strikes_from_convexity_moneyness(ivols['mid'], ivols['x'], ivols['t'])
        return ivols.copy()

    def convert_to_moneyness_strike_reference(self, df):

        # Compute the moneyness measures for the data we have
        df['x'] = df['k'] / self.get_spot_prices(df.index)
        ivols = self.interpolate(df, strikes=_MONEYNESS_POINTS, maturities=df['t'])
        ivols['k'] = self.get_strikes_from_moneyness(ivols['x'])
        return ivols.copy()

    def convert_to_delta_strike_reference(self, df, strike_reference):

         # Compute Deltas for the quotes we have
         x = self.get_deltas_from_strikes(df['mid'], df['t'], df['k'], strike_reference.value, 1)
         df['x'] = x.round(3)
         interp_ = df.reset_index(drop=False).drop_duplicates(subset=['date', 't', 'x']).set_index('date')

         # Interpolate the deltas to points in the cross-section
         ivols = self.interpolate(interp_, strikes=_DELTA_POINTS, maturities=df['t'])
         # Compute the new impled strikes
         ivols['k'] = self.get_strikes_from_deltas(ivols['mid'], ivols['t'], ivols['x'], strike_reference.value)
         # Express and OTM Puts and Calls
         ivols.loc[ivols.x > 0.5, 'x'] = ivols.loc[ivols.x > 0.5, 'x'] - 1
         return ivols


    @staticmethod
   # @jit(nopython=True, fastmath=True, cache=True)
    def interpolate_single_date(mat, unique_m, x, unique_x, mids):

        N = len(mids)

        # get the std of the reference on each date
        sig_x = np.std(x)
        h_x = 1.0 * np.power(4.0 / 3.0, 1.0 / 5.0) * sig_x / np.power(N, 1.0 / 5.0)

        # do the same in the maturity direction
        sig_lm = np.log(mat).std()
        h_m = 2.0 * np.power(4.0 / 3.0, 1.0 / 5.0) * sig_lm / np.power(N, 1.0 / 5.0) * 0.1

        lnmat_ = np.log(mat).reshape((-1, 1))

        m_diff = np.abs(lnmat_ - np.log(unique_m.repeat(N).reshape((unique_m.size, N)).T)) / h_m
        x_diff = np.abs(x.reshape((-1, 1)) - unique_x.repeat(N).reshape((unique_x.size, N)).T) / h_x

        x_wts = np.exp(-1.0 * np.power(x_diff, 2.0) / 2.0)
        x_wts = np.expand_dims(x_wts, -1).repeat(unique_m.size).reshape((x_wts.shape[0], x_wts.shape[1], unique_m.size))

        m_wts = np.exp(-1.0 * np.power(m_diff, 2.0) / 2.0).reshape((m_diff.shape[0], 1, m_diff.shape[1]))
        m_wts_ = np.full(x_wts.shape, np.nan) * np.nan
        for dim in range(x_wts.shape[1]):
            m_wts_[:, dim, :] = m_wts[:, 0, :]

        wts = (x_wts * m_wts) / np.sum(x_wts * m_wts_, axis=0)
        ivol = mids.repeat(wts.shape[1] * unique_m.size).reshape(m_wts_.shape)
        ivols = np.sum(wts * ivol, axis=0).flatten()
        return ivols

    def interpolate(self, _df, maturities, strikes, dates=None):

        _df = _df.reset_index(drop=False).set_index('date')
        if dates is None:
           dates = _df.index.unique()

        unique_d = np.unique(dates)
        unique_m = np.unique(maturities).reshape(1, -1).astype(float)
        unique_x = np.unique(strikes).reshape(1, -1).astype(float)

        NX = len(unique_x.flatten())
        NM = len(unique_m.flatten())
        ND = len(unique_d.flatten())

        t_vec = _df['t'].values.astype(float)
        x_vec = _df['x'].values.astype(float)
        v_vec = _df['mid'].values.astype(float)

        interp = np.full(shape=(NX*NM, ND), fill_value=np.nan)
        for t in tqdm(range(ND), desc="Interpolating Vol Surface"):
            idx = (unique_d[t] == _df.index)
            T = ND - 1
            interp[:, t] = self.interpolate_single_date(t_vec[idx],
                                                         unique_m,
                                                         x_vec[idx],
                                                         unique_x,
                                                         v_vec[idx])

        idx_m = unique_x.T.repeat(NM, axis=0).flatten()
        idx_x = unique_m.repeat(NX, 0).flatten()

        vols = pd.DataFrame(interp, columns=unique_d)
        vols.index = pd.MultiIndex.from_tuples(list(zip(idx_m, idx_x)), names=['x','t'])
        vols.columns.names = ['date']
        return vols.stack().reorder_levels([2, 0, 1]).to_frame('mid').reset_index(drop=False).set_index('date')

    def _validate_underlier(self):
        if not session.query(exists().where(ImpliedVolatility.security == self.underlier)).scalar():
            raise ValueError('Error: {} ticker not supported'.format(self._underlier))


if __name__ == "__main__":

        from epsilonPhi.core.dataModel.dataSources.impliedVolatility.Models import *

        underlier = 'EURUSD'
        self = VolSurfaceMgr(underlier, pricing_location='NYC', strike_reference=StrikeReference.SPOT_DELTA)

        df = sessionMgr.get_ivols(self._underlier, self._pricing_location)
        df = df.set_index('date', drop=True)
        df['t'] = DateUtils.Rdate_to_mat(df.tenor)

        df = df[df.relative_strike.isin([-999, -0.25, 0.25])].reset_index(drop=False)
        df_ = df.set_index(['date','t','relative_strike'])
        sigmas_ = df_[df_.index.duplicated()].get('mid').unstack(level=[1, 2]).sort_index()


        ########### 1. Interpolate to the Horizons we want ##########

        from epsilonPhi.core.utils.FrameUtils import FrameUtils
        horizons = np.cumsum([1/365] * 31)
        horizon_vols = FrameUtils.rowise_linear_interpolate_on_groups(sigmas_,
                                                                      horizons,
                                                                      0,
                                                                      'relative_strike')

        ############ 2. Get Strikes for Horizon Vols ###############





        ############ 3. Interpolate the cross sections ############







        #########

        df_t['rd'] = self.get_interest_rates(df_t.index, df_t.t)
        df_t['rf'] = self.get_funding_rates(df_t.index, df_t.t)
        df_t['F'] = self.get_forward_prices(df_t.index, df_t.t)
        df_t['S'] = self.get_spot_prices(df_t.index)
        df_t = df_t[~df_t.relative_strike.isin([1, 0.45, 0.55])]
        df_t['t'] = DateUtils.Rdate_to_mat(df_t.get('tenor'))
        df_t = df_t.drop_duplicates(['relative_strike','tenor'])
        df_t.loc[df_t.relative_strike < 0, 'relative_strike'] = df_t.loc[df_t.relative_strike < 0, 'relative_strike'] + 1
        df_t.loc[df_t.relative_strike == -998, 'relative_strike'] = 0.5

        _s = df_t.set_index(['t', 'relative_strike']).get('mid').unstack(level=0)
        _s = _s.sort_index(axis=1)

        # 1d polynomial regression
        y = _s.get(1).values
        x = np.array(_s.index)

        # Fit the Polynomial Regression
        udel = np.linspace(0.1, 0.9, 1000)
        fit = polynomial_regression_1d(x, y, np.linspace(0.1, 0.9, 1000))
        fit_ = gaussian_kernel_smoother_1d(x, y, np.linspace(0.1, 0.9, 1000), 0.2)


        df_t['k'] = solve_for_strike(df_t['S'],
                                      df_t['t'],
                                      df_t['rd'],
                                      df_t['rf'],
                                      1,
                                      df_t['relative_strike'],
                                      1,
                                      df_t['mid']).values

        idx_L = np.logical_or((df_t['relative_strike'] == -0.25), (df_t['relative_strike'] == 0.75))
        idx_U = np.logical_or((df_t['relative_strike'] == 0.25), (df_t['relative_strike'] == -0.75))
        idx_A = df_t['relative_strike'] == 0.5

        kput_ = df_t[idx_L][['mid','k','t','F']].sort_values('t')
        katm_ = df_t[idx_A][['mid', 'k', 't']].sort_values('t')
        kcal_ = df_t[idx_U][['mid', 'k', 't']].sort_values('t')

        F = kput_['F'].values
        t = kput_['t'].values

        ks = np.linspace(df_t['k'].min(), df_t['k'].max(), 1000)


        # strike, maturity pairs

        sig_x = vanna_volga_2d(F,
                               ks,
                               t,
                               kput_.get('k'),
                               katm_.get('k'),
                               kcal_.get('k'),
                               kput_.get('mid'),
                               katm_.get('mid'),
                               kcal_.get('mid'))

        # Fit the 1-d Gaussian Kernel
        from matplotlib import pyplot as plt  # Change - use least squares

        plt.scatter(x, y)
        u = np.linspace(0.1, 0.9, 1000)
        plt.plot(u, fit)
        plt.show()

        plt.plot(u, fit_)
        plt.show()

