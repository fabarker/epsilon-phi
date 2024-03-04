import pandas as pd

from epsilonPhi.core.dataModel.alchemist.SessionManager import *
from epsilonPhi.core.dataModel.dataSources.curves.interestRateCurve.IRCurve import IRCurve
from epsilonPhi.core.dataModel.dataSources.curves.abstractCurve.AbstractCurve import AbstractCurve
from epsilonPhi.core.utils.OptionUtils import *
from epsilonPhi.core.utils.FrameUtils import FrameUtils
from epsilonPhi.core.dataModel.enums.Database import PriceQuote
from tqdm import tqdm

sessionMgr = SessionMgr()
session = sessionMgr.getSessionFactory()

_DROP_COLS = ['uid', 'pricing_location', 'relative_strike', 'strike_reference', 'tenor', 'security']

_RELATIVE_STRIKE = 'relative_strike'
_STRIKE_REFERENCE = 'strike_reference'
_TENOR = 'tenor'
_MATURITY = 't'

# Default interpolation points for converting between strike reference space
_DELTA_POINTS = np.arange(0.05, 1, 0.05)
_MONEYNESS_POINTS = np.arange(50, 155, 5)/100
_ZSCORE_POINTS = np.arange(-3, 3.5, 0.5)
_CONVEXITY_MNY_POINTS = np.arange(-3, 3.5, 0.1)


class VolSurfaceMgr(object):
    _raw_cache = {}
    _ivol_cache = {}

    def __init__(self, 
                 underlier,
                 pricing_location=None,
                 strike_reference=None):

        # Optional Parameters
        self._underlier = underlier
        self._pricing_location = pricing_location
        self._strike_reference = strike_reference

        # Properties of the raw data
        self._currency = None

        # Spec for the ivols
        self._spec = None
        self._datasource = None

        # Asset Prices and Interest/Funding Rates
        self._interest_rate_curve = None
        self._funding_rate_curve = None
        self._spot_prices = None
        self._put_deltas = None
        self._call_deltas = None

        # Validate the target surface. i.e make sure
        # we have the data
        self._validate_underlier()

        # Load Data
        self._load_vol_surface_spec()
        self._load_ivols()

    @property
    def dates(self):
        return pd.to_datetime(self.ivols.index)
    @property
    def pricing_location(self):
        return self._pricing_location
    @property
    def strike_reference(self):
        return self._strike_reference
    @property
    def ivols(self):
        return self.get_ivols()
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
    def mids(self):
        return self.ivols.get('mid')
    @property
    def maturities(self):
        return self.ivols.get(_MATURITY)
    @property
    def strikes(self):
        return self.ivols.get('k')
    @property
    def spot_prices(self):
        return self.get_spot_prices(self.dates)
    @property
    def forward_prices(self):
        return self.get_forward_prices(self.dates,
                                       self.maturities)
    @property
    def risk_free_rates(self):
        return self.get_interest_rates(self.dates,
                                       self.maturities)
    @property
    def funding_rates(self):
        return self.get_funding_rates(self.dates,
                                      self.maturities)
    @property
    def strike_references(self):
        return self.ivols.get(_STRIKE_REFERENCE)
    @property
    def relative_strikes(self):
        return self.ivols.get(_RELATIVE_STRIKE)
    @property
    def put_deltas(self):
        return self.get_put_deltas()
    @property
    def call_deltas(self):
        return self.get_call_deltas()
    @property
    def straddle_deltas(self):
        return self.call_deltas + self.put_deltas
    @property
    def ds(self):
        if self._datasource is None:
            from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
            self._datasource = GlobalDataSource()
        return self._datasource

    ################ LOAD METHODS ############

    def _load_vol_surface_spec(self):
        self._spec = sessionMgr.get_ivol_spec(self._underlier)

    def _load_interest_rate_curve(self):
        self._interest_rate_curve = IRCurve(region=self.region, type=['Interbank', 'Deposit'])

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

    def _load_ivols(self):

        if self.underlier not in self._ivol_cache.keys():

            # Load raw ivols data from database
            df = sessionMgr.get_ivols(self.underlier,
                                      self.pricing_location,
                                      index='date')

            # Estimate strikes from spot moneyness
            _spt_idx = df[_STRIKE_REFERENCE].values == 'spot'
            df.loc[_spt_idx, 'k'] = self.get_strikes_from_moneyness(df[_RELATIVE_STRIKE][_spt_idx])

            # Estimate strikes from forward moneyness
            _fwd_idx = df[_STRIKE_REFERENCE].values == 'forward'
            df.loc[_fwd_idx, 'k'] = self.get_strikes_from_forward_moneyness(df[_RELATIVE_STRIKE][_fwd_idx],
                                                                            df[_MATURITY][_fwd_idx])
            # Estimates strikes from deltas
            _del_idx = df[_STRIKE_REFERENCE].values == 'delta'
            df.loc[_del_idx, 'k'] = self.get_strikes_from_deltas(df['mid'][_del_idx],
                                                                 df[_MATURITY][_del_idx],
                                                                 df[_RELATIVE_STRIKE][_del_idx],
                                                                 self.delta_convention)
            # Estimate the strikes for delta neutral points
            _DN_idx = df.relative_strike == -999
            df.loc[_DN_idx, 'k'] = self.get_strikes_from_atm_delta_neutral(df['mid'][_DN_idx],
                                                                           df[_MATURITY][_DN_idx],
                                                                           self.delta_convention)


            self._ivol_cache[self.underlier] = df.copy()
            self._ivol_cache[self.underlier][self.strike_reference] = self.get_strike_reference(self.strike_reference)


    ############### Public Getter Methods ###############

    def get_put_deltas(self):
        if self._put_deltas is None:
           self._put_deltas = self.get_deltas_from_strikes(self.mids,
                                                       self.maturities,
                                                       self.strikes,
                                                       self.delta_convention, -1)
        return self._put_deltas.copy()

    def get_call_deltas(self):
        if self._call_deltas is None:
           self._call_deltas = self.get_deltas_from_strikes(self.mids,
                                                            self.maturities,
                                                            self.strikes,
                                                            self.delta_convention, 1)
        return self._call_deltas.copy()

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


    def get_interest_rates(self, pricing_dates, maturities):
        if isinstance(maturities, (pd.DataFrame, pd.Series)):
           maturities = maturities.values

        _curve = self.get_interest_rate_curve(np.unique(maturities), stack=True)
        idx = pd.MultiIndex.from_tuples(list(zip(pricing_dates, maturities)))
        return _curve.loc[zip(pricing_dates, maturities)].values.flatten()

    def get_funding_rates(self, pricing_dates, maturities):
        if isinstance(maturities, (pd.DataFrame, pd.Series)):
           maturities = maturities.values

        _curve = self.get_funding_rate_curve(np.unique(maturities), stack=True)
        return _curve.loc[zip(pricing_dates, maturities)].values.flatten()

    def get_ivols(self):
        if self.underlier not in self._ivol_cache.keys():
            self._load_ivols()
        return self._ivol_cache.get(self.underlier)

    def get_atm_delta_neutral_deltas(self, ivols, t, deltaTypeValue, option_type_value):

        # Get Forward Prices
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
        return np.round(fast_delta(s,
                                   t,
                                   k,
                                   rd,
                                   rf,
                                   ivols,
                                   delta_type,
                                   option_type), 3)

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

    def get_strike_reference(self, strike_reference):

        if strike_reference in [StrikeReference.DELTA]:
            return self.get_delta_strike_reference()
        if strike_reference in [StrikeReference.MONEYNESS]:
            return self.get_moneyness_strike_reference()
        if strike_reference in [StrikeReference.Z_SCORE]:
            return self.get_zscore_strike_reference()
        if strike_reference in [StrikeReference.CONVEXITY_MN]:
            return self.get_convexity_moneyness_strike_reference()
        else:
            return df.copy()

    def get_zscore_strike_reference(self):
        raise ValueError('Not Implemented')

    def get_convexity_moneyness_strike_reference(self):
        return self.get_convexity_moneyness_from_strikes(self.mids,
                                                         self.strikes,
                                                         self.maturities)
    def get_log_moneyness_strike_reference(self):
        return np.log(self.strikes / self.spot_prices)

    def get_moneyness_strike_reference(self):
        return np.exp(self.get_log_moneyness_strike_reference())

    def get_delta_strike_reference(self):
        return self.get_deltas_from_strikes(self.mids,
                                            self.maturities,
                                            self.strikes,
                                            self.delta_convention, 1)


    def get_vanna_volga_pillars(self):
        pillars = (self.ivols[self.ivols[_RELATIVE_STRIKE].isin([-0.25, -999, 0.25])].
                get(['mid','k','relative_strike','t'])).drop_duplicates()
        return pillars.reset_index(drop=False).set_index(['date','t']).pivot(columns='relative_strike')

    def build_smile_vanna_volga(self,
                                strike_reference_points=[-0.1, -0.25, 0.5, 0.25, 0.1],
                                strike_reference_type=None):

        if strike_reference_type is None:
           strike_reference_type = StrikeReference.DELTA

        vv_points = self.get_vanna_volga_pillars()

        pricing_dates = vv_points.index.get_level_values('date')
        M = np.array(vv_points.index.get_level_values('t'))
        F = self.get_forward_prices(pricing_dates, M)

        if strike_reference_type in [StrikeReference.DELTA,
                                     StrikeReference.DELTA.value]:

            sig = vv_points.get('mid').get(-999).values
            L = F * np.exp((-3 * sig * np.sqrt(M)))
            U = F * np.exp((+3 * sig * np.sqrt(M)))
            ks = np.linspace(L, U, 100).T

            sig_x = vanna_volga_2d(F,
                                   ks,
                                   M,
                                   vv_points.get('k').get(-0.25).values,
                                   vv_points.get('k').get(-999).values,
                                   vv_points.get('k').get(0.25).values,
                                   vv_points.get('mid').get(-0.25).values,
                                   vv_points.get('mid').get(-999).values,
                                   vv_points.get('mid').get(0.25).values)


            sig_x = pd.DataFrame(sig_x, index=vv_points.index).droplevel(level=1)
            p_deltas = self.get_deltas_from_strikes(sig_x,
                                                    M,
                                                    ks,
                                                    self.delta_convention,
                                                    -1)

        elif strike_reference_type in [StrikeReference.MONEYNESS,
                                         StrikeReference.MONEYNESS.value]:
            pass


        elif strike_reference_type in [StrikeReference.LOG_MONEYNESS,
                                       StrikeReference.LOG_MONEYNESS.value]:
            pass

        elif strike_reference_type in [StrikeReference.Z_SCORE,
                                       StrikeReference.Z_SCORE.value]:
            pass

        elif strike_reference_type in [StrikeReference.CONVEXITY_MN,
                                       StrikeReference.CONVEXITY_MN.value]:
            pass


        elif strike_reference_type in ['absolute']:
            pass


    def interpolate_term_structure(self, ivols, maturities):
        group = np.setdiff1d(ivols.columns.names, 't').item()
        return FrameUtils.rowise_linear_interpolate_on_groups(ivols,
                                                              maturities,
                                                              x_lev='t',
                                                              group=group)

    def interpolate_strike_reference_dimension(self):
        pass



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
        self = VolSurfaceMgr(underlier,
                             pricing_location='NYC',
                             strike_reference=StrikeReference.DELTA)


        ### Get ivols
        ivols = self.get_ivols()
        ivols = ivols.get(['mid', 'k', 't'])
        self.build_smile_vanna_volga()



        horizon_vols.columns.names = ['relative_strike', 't']
        horizon_ks = horizon_vols.stack(level=[0,1]).reset_index().set_index('date')
        horizon_ks = horizon_ks.rename(columns={0:'mid'})
        # Estimate the strikes for delta neutral points

        _DN_idx = horizon_ks.relative_strike == -999
        horizon_ks.loc[_DN_idx, 'k'] = self.get_strikes_from_atm_delta_neutral(horizon_ks['mid'][_DN_idx],
                                                                          horizon_ks['t'][_DN_idx],
                                                                          self.delta_convention)

        horizon_ks.loc[~_DN_idx, 'k'] = self.get_strikes_from_deltas(horizon_ks['mid'][~_DN_idx],
                                                             horizon_ks['t'][~_DN_idx],
                                                             horizon_ks['relative_strike'][~_DN_idx],
                                                             self.delta_convention)

        vv_params = horizon_ks.reset_index(drop=False).set_index(['date','relative_strike','t']).unstack(level=1)
        f = self.get_forward_prices(vv_params.index.get_level_values(0),
                                    np.array(vv_params.index.get_level_values(1)))

        t = np.array(vv_params.index.get_level_values(1))

        kput = vv_params.get('k').get(-0.25).values
        kcall = vv_params.get('k').get(0.25).values
        katm = vv_params.get('k').get(-999).values

        k_prime = f.reshape(-1, 1).repeat(100, 1) * np.linspace(0.95, 1, 100).reshape(1, -1).repeat(f.shape[0], 0)

        sigput = vv_params.get('mid').get(-0.25).values
        sigcal = vv_params.get('mid').get(0.25).values
        sigatm = vv_params.get('mid').get(-999).values

        _vv = vanna_volga_2d(f, k_prime, t, kput, katm, kcall, sigput, sigatm, sigcal)
        res = pd.DataFrame(_vv, index=vv_params.index)
        res.index = res.index.get_level_values(0)

        d = self.get_deltas_from_strikes(res,
                                         t,
                                         k_prime,
                                     1,
                                         -1)

        locs = np.min(np.abs(d - -0.25), axis=1).reshape(-1,1) == np.abs(d - -0.25)


        #######




        ########### 1. Interpolate to the Horizons we want ##########

        from epsilonPhi.core.utils.FrameUtils import FrameUtils
        horizons = np.cumsum([1/365] * 31)
        horizon_vols = FrameUtils.rowise_linear_interpolate_on_groups(ivols,
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

