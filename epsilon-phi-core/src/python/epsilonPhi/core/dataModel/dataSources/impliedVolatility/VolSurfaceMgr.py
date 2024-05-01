from epsilonPhi.core.dataModel.alchemist.SessionManager import *
from epsilonPhi.core.dataModel.dataSources.curves.interestRateCurve.IRCurve import IRCurve
from epsilonPhi.core.dataModel.dataSources.curves.abstractCurve.AbstractCurve import AbstractCurve
from epsilonPhi.core.utils.OptionUtils import *
from epsilonPhi.core.dataModel.enums.Database import PriceQuote
from epsilonPhi.core.dataModel.enums.Database import Provider, PricingLocation, PriceQuote

sessionMgr = SessionMgr()
session = sessionMgr.getSessionFactory()

_RELATIVE_STRIKE = 'relative_strike'
_STRIKE_REFERENCE = 'strike_reference'
_TENOR = 'tenor'
_MATURITY = 't'

class VolSurfaceMgr(object):

    _raw_cache = {}
    _ivol_cache = {}
    _VOL_TOL = 0.02

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
        self._rate_curve = None
        self._funding_curve = None
        self._spot_prices = None

        self._s = None
        self._rd = None
        self._rf = None
        self._f = None
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
    def exchange(self):
        return self._pricing_location
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
        return deltaConvention.get(self.underlier, DeltaType.FORWARD_DELTA)
    @property
    def mids(self):
        return self.ivols.get('mid').values.reshape(-1, 1)
    @property
    def sig(self):
        return self.ivols.get('mid').values.reshape(-1, 1)
    @property
    def t(self):
        return self.ivols.get(_MATURITY).values.reshape(-1, 1)
    @property
    def k(self):
        return self.ivols.get('k').values.reshape(-1, 1)
    @property
    def s(self):
        if self._s is None:
            self._s = self.get_spot_prices(self.dates).values.reshape(-1, 1)
        return self._s
    @property
    def f(self):
        if self._f is None:
            self._f = self.get_forward_prices(self.dates, self.t).values.reshape(-1, 1)
        return self._f
    @property
    def rd(self):
        if self._rd is None:
            self._rd = self.get_interest_rates(self.dates, self.t).values.reshape(-1, 1)
        return self._rd
    @property
    def rf(self):
        if self._rf is None:
            self._rf = self.get_funding_rates(self.dates, self.t).values.reshape(-1, 1)
        return self._rf
    @property
    def strike_references(self):
        return self.ivols.get(_STRIKE_REFERENCE)
    @property
    def relative_strikes(self):
        return self.ivols.get(_RELATIVE_STRIKE)

    @property
    def ds(self):
        if self._datasource is None:
            from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
            self._datasource = GlobalDataSource()
        return self._datasource

    ################ LOAD METHODS ############

    def _load_ivols(self):

        if self.underlier not in self._ivol_cache.keys():

            # Load raw ivols data from database
            df = sessionMgr.get_ivols(self.underlier,
                                      self.pricing_location,
                                      index='date')

            # throw away any vols that fail the tolerence
            df = df[df.mid > self._VOL_TOL]

            if 'k' not in df.columns:
                # Estimate strikes from spot moneyness
                _spt_idx = df[_STRIKE_REFERENCE].values == 'spot'
                df.loc[_spt_idx, 'k'] = self.get_strikes_from_moneyness(df[_RELATIVE_STRIKE][_spt_idx])

                # Estimate strikes from forward moneyness
                _fwd_idx = df[_STRIKE_REFERENCE].values == 'forward'
                df.loc[_fwd_idx, 'k'] = self.get_strikes_from_forward_moneyness(df[_RELATIVE_STRIKE][_fwd_idx],
                                                                                df[_MATURITY][_fwd_idx])
                # Estimates strikes from deltas
                _del_idx = (df[_STRIKE_REFERENCE].values == 'delta') & (df[_RELATIVE_STRIKE] > -999)
                df.loc[_del_idx, 'k'] = self.get_strikes_from_deltas(df['mid'][_del_idx],
                                                                     df[_MATURITY][_del_idx],
                                                                     df[_RELATIVE_STRIKE][_del_idx],
                                                                     self.delta_convention)
                # Estimate the strikes for delta neutral points
                _DN_idx = df.relative_strike == -999
                df.loc[_DN_idx, 'k'] = self.get_strikes_from_atm_delta_neutral(df['mid'][_DN_idx],
                                                                               df[_MATURITY][_DN_idx],
                                                                               self.delta_convention)



            self._ivol_cache[self.underlier] = df[['mid','t','k','relative_strike','tenor']].copy()
            self._ivol_cache[self.underlier]['s'] = self.s
            self._ivol_cache[self.underlier]['ve'] = self.get_bsm_vega()
            self._ivol_cache[self.underlier]['vo'] = self.get_bsm_volga()
            self._ivol_cache[self.underlier]['va'] = self.get_bsm_vanna()
            self._ivol_cache[self.underlier]['put_delta'] = self.get_deltas(-1)
            self._ivol_cache[self.underlier]['call_delta'] = self.get_deltas(1)
            self._ivol_cache[self.underlier][self.strike_reference] = self.get_strike_reference(self.strike_reference)


    def _load_vol_surface_spec(self):
        self._spec = sessionMgr.get_ivol_spec(self._underlier)

    def _load_spot_prices(self):

        if self.category == 'FX':
           self.ds._fx_curve.set_provider(Provider.GS)
           self.ds._fx_curve.set_pricing_location(PricingLocation.NEW_YORK)
           spt = self.ds.get_fx_spot_rates(self.ticker, PriceQuote.MID.value)
        elif self.category in ['Equity Index', 'ETF']:
           spt = self.ds.get_time_series_data_from_ticker(self.ticker, cols='PI')
        elif self.category == 'Future':
           spt = self.ds.get_front_futures_continuous_series_settlement_price(self.ticker[0:3])
        else:
            raise ValueError('Error - category {} not recognized'.format(self.category))

        self._spot_prices = pd.Series(spt.iloc[:,0], name=self.security)

    def _load_rate_curve(self):
        rf = IRCurve(region=self.region, type=['Interbank', 'Deposit']).get_curve()
        self._rate_curve = AbstractCurve(rf)

    def get_risk_free_rate_curve(self, currency):
        rate_curve = IRCurve(currency=currency,
                     type=['Interbank', 'Deposit']).get_curve()
        return AbstractCurve(rate_curve)

    def get_rate_curve(self):
        if self._rate_curve is None:
            self._load_rate_curve()
        return self._rate_curve

    def get_funding_curve(self):
        if self._funding_curve is None:
            self._load_funding_curve()
        return self._funding_curve

    def _load_funding_curve(self):

        if self.category == 'FX':
            if self.ticker[0:3] == 'EUR':
               fx = 'DEM'
            else:
               fx = self.ticker[0:3]
            rf = IRCurve(currency=fx, type=['Interbank', 'Deposit']).get_curve()
        elif self.category in ['Equity Index', 'ETF']:
            rf = self.ds.get_time_series_data_from_ticker(self.ticker, cols='DY')
            rf = pd.concat([rf] * 2, axis=1).resample('B').ffill() / 100
            rf.columns = [0, 10]
        elif self.category == 'Future':
            rf = self.ds.get_front_futures_continuous_series_settlement_price(self.ticker[0:3])
        else:
            raise ValueError('Error - category {} not recognized'.format(self.category))
        self._funding_curve = AbstractCurve(rf)



    ############### Public Getter Methods ###############

    def get_invalid_maturity_dates(self):

        if self._spot_prices is None:
            self._load_spot_prices()
        all_dates = self._spot_prices.resample('D').asfreq()
        return all_dates[all_dates.isna()].index

    def get_ivols(self):
        if self.underlier not in self._ivol_cache.keys():
            self._load_ivols()
        return self._ivol_cache.get(self.underlier)


    def get_spot_prices(self, dates=None):

        if self._spot_prices is None:
           self._load_spot_prices()

        if dates is None:
            return self._spot_prices.copy()
        else:
            return self._spot_prices.reindex(dates)

    def get_interest_rates(self, pricing_dates, maturities, is_stacked=True):
        if self._rate_curve is None:
            self._load_rate_curve()
        return self._rate_curve.get_curve(pricing_dates, np.array(maturities), is_stacked)

    def get_funding_rates(self, pricing_dates, maturities, is_stacked=True):
        if self._funding_curve is None:
            self._load_funding_curve()
        return self._funding_curve.get_curve(pricing_dates, maturities, is_stacked)

    def get_forward_prices(self, pricing_dates, maturities):
        mat = np.array(maturities).flatten()
        _s = self.get_spot_prices(pricing_dates).values
        _rf = self.get_funding_rates(pricing_dates, maturities)
        _rd = self.get_interest_rates(pricing_dates, maturities)
        return _s * np.exp(-_rf * mat) / np.exp(-_rd * mat)

    def get_bsm_vega(self):
        return black_vega(self.f,
                          self.t,
                          self.k,
                          self.rd,
                          self.sig)

    def get_bsm_volga(self):
        return black_volga(self.f,
                           self.t,
                           self.k,
                           self.rd,
                           self.sig)

    def get_bsm_vanna(self):
        return black_vanna(self.f,
                           self.t,
                           self.k,
                           self.rf,
                           self.sig)

    def get_strikes_from_deltas(self, ivols, t, delta, delta_type):

        if ivols.size > 0:
            s = self.get_spot_prices(ivols.index).values.astype(np.float64)
            rf = self.get_funding_rates(ivols.index, t).values.astype(np.float64)
            rd = self.get_interest_rates(ivols.index, t).values.astype(np.float64)
            t = t.values.astype(np.float64)
            deltas = delta.values.astype(np.float64)
            return solve_for_strike(s, t, rd, rf, np.sign(deltas).astype(np.int64), deltas, delta_type.value, ivols.values.astype(np.float64))

    def get_strikes_from_atm_delta_neutral(self, ivols, t, deltaType):
        if ivols.size > 0:
            s = self.get_spot_prices(ivols.index).values.astype(np.float64)
            rf = self.get_funding_rates(ivols.index, t).values.astype(np.float64)
            rd = self.get_interest_rates(ivols.index, t).values.astype(np.float64)
            t = t.values.astype(np.float64)
            return atm_delta_neutral_strike(s,
                                            t,
                                            rd,
                                            rf,
                                            ivols.values.astype(np.float64),
                                            deltaType.value)

    def get_strikes_from_forward_moneyness(self, x, t):
        if x.size > 0:
            return x * self.get_forward_prices(x.index, t).droplevel(1)

    def get_strikes_from_moneyness(self, x):
        if x.size > 0:
            return x * self.get_spot_prices(x.index).values.flatten()

    def get_strikes_from_log_moneyness(self, x, t):
        if x.size > 0:
            return self.get_forward_prices(x.index, t).droplevel(1) * np.exp(x)

    def get_atm_delta_neutral_deltas(self, ivols, t, deltaTypeValue, option_type_value):

        # Get Forward Prices
        rf = self.get_forward_prices(ivols.index, t)
        return delta_from_delta_neutral_straddle_quote(t,
                                                       rf,
                                                       ivols,
                                                       deltaTypeValue,
                                                       option_type_value)

    def get_log_moneyness(self):
        return np.log(self.k / self.f)

    def get_deltas(self, option_type):
        return nb_delta(self.s,
                        self.t,
                        self.k,
                        self.rd,
                        self.rf,
                        self.sig,
                        self.delta_convention.value,
                        int(option_type))

    def get_convexity_moneyness(self):

        # Compute the convexity adjusted moneyness measure
        sigsq = np.power(self.sig, 2)
        lnm = self.get_log_moneyness()
        zp = lnm + 0.5 * sigsq * self.t
        return zp / (self.sig * np.sqrt(self.t))

    def get_zscore(self):
        lmn = self.get_log_moneyness()
        return lmn / (self.sig * np.sqrt(self.t))

    def get_moneyness(self):
        return np.exp(self.get_log_moneyness())

    def get_strike_reference(self, strike_reference):

        if strike_reference in [StrikeReference.DELTA]:
            return self.get_deltas(-1)
        if strike_reference in [StrikeReference.MONEYNESS]:
            return self.get_moneyness()
        if strike_reference in [StrikeReference.LOG_MONEYNESS]:
            return self.get_log_moneyness()
        if strike_reference in [StrikeReference.Z_SCORE]:
            return self.get_zscore()
        if strike_reference in [StrikeReference.CONVEXITY_MN]:
            return self.get_convexity_moneyness()
        else:
            raise ValueError('Error - strike_reference not supported')

    def _validate_underlier(self):
        if not session.query(exists().where(ImpliedVolatility.security == self.underlier)).scalar():
            raise ValueError('Error: {} ticker not supported'.format(self._underlier))

    def get_atm_vols(self,  maturity):

        t = np.maximum(DateUtils.Rdate_to_mat(maturity),
                       DateUtils.Rdate_to_mat('1m'))
        tau = DateUtils.mat_to_Rdate(t)

        _ivols = self.get_ivols()
        atms = _ivols[self.get_ivols().relative_strike.isin([-999])]
        atm_vols =  atms[atms.tenor.isin(tau)].get('mid')

        name = self.security + 'V' + maturity.upper()
        return atm_vols[~atm_vols.index.duplicated()].sort_index().to_frame(name)

    def get_delta_vols(self, delta, maturity):

        t = np.maximum(DateUtils.Rdate_to_mat(maturity),
                       DateUtils.Rdate_to_mat('1m'))
        tau = DateUtils.mat_to_Rdate(t)

        _ivols = self.get_ivols()
        vols = _ivols[self.get_ivols().relative_strike.isin([delta])]
        sigs = vols[vols.tenor.isin(tau)].get('mid')
        return sigs[~sigs.index.duplicated()].sort_index()

    def get_risk_reversal(self, delta, maturity):

        P = self.get_delta_vols(-delta, maturity)
        C = self.get_delta_vols(delta, maturity)

        common = np.intersect1d(P.index, C.index)
        rr = C.loc[common] - P.loc[common].values

        name = self.security + str(int(delta * 100)) + 'R' + maturity.upper()
        return rr.to_frame(name)

    def get_butterfly(self, delta, maturity):

        P = self.get_delta_vols(-delta, maturity)
        C = self.get_delta_vols(delta, maturity)
        A = self.get_atm_vols(maturity)

        common = np.intersect1d(np.intersect1d(P.index, C.index), A.index)

        vwb = (0.5 * (C.loc[common] + P.loc[common].values)) - A.loc[common].values.flatten()
        name = self.security + str(int(delta * 100)) + 'B' + maturity.upper()
        return vwb.to_frame(name)



if __name__ == "__main__":

        from epsilonPhi.core.dataModel.dataSources.impliedVolatility.Models import *

        underlier = 'USDJPY'
        self = VolSurfaceMgr(underlier,
                             pricing_location='NYC',
                             strike_reference=StrikeReference.DELTA)

        _DELTAS = [0.1, 0.25]
        _MATS = ['ON', '1W', '2W', '3W', '1M', '2M', '3M', '6M', '9M', '1Y']

        vols = pd.DataFrame()
        for mat in _MATS:
            for delta in _DELTAS:
                bf = self.get_butterfly(delta, mat)
                rr = self.get_risk_reversal(delta, mat)
                vols = pd.concat((vols, rr, bf), axis=1)
            vols = pd.concat((vols, self.get_atm_vols(mat)), axis=1)

        vols.mul(100).dropna().to_csv('/Users/francisbarker/Desktop/ivol cache/FX VOLS/' + underlier + '.csv')






