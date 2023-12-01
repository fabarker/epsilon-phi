from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries, TimeSeriesType, ReturnsType
from epsilonPhi.core.lib.Decorators import SingletonDecorator
from epsilonPhi.core.dataModel.enums.Database import Provider, PricingLocation, PriceQuote
from epsilonPhi.core.dataModel.enums.Composites import CompositeFXRates
from sqlalchemy import distinct, not_
import pandas as pd

@SingletonDecorator
class FXCurveManager(object):

    _DEFAULT_PRICING_LOCATION = PricingLocation.LONDON.value
    _DEFAULT_PROVIDERS = [Provider.WMR.value,
                          Provider.REFINITIV.value,
                          Provider.BBI.value]

    _fx_cache = pd.DataFrame()
    _cached_pairs = list()

    _fx_curve_cache = dict()

    _sessionMgr = SessionMgr()
    _session = SessionMgr().getSessionFactory()

    def __init__(self):
        self.__set_fx_spec()

    def __set_fx_spec(self):
        q = self._session.query(FXRateSpec.provider,
                                FXRateSpec.bbid,
                                FXRateSpec.maturity,
                                FXRateSpec.uid)
        self.__spec = self._sessionMgr.query_format_df(q).set_index('uid', drop=True)

    @property
    def fx_spec(self):
        return self.__spec

    @property
    def _cached_fx_curves(self):
        return self._fx_curve_cache.keys()

    def reset_cache(self):
        self._fx_cache = pd.DataFrame()
        self._cached_pairs = list()

    def get_bbid_providers(self, bbid):
        bbid = bbid.replace('/', '')
        return list(self.fx_spec[self.fx_spec.bbid == bbid].provider.unique())

    def get_bbid_maturities(self, bbid):
        bbid = bbid.replace('/', '')
        return list(self.fx_spec[self.fx_spec.bbid == bbid].maturity.unique())

    def get_bbid_spec(self, bbid: str, provider=None):
        q = self._session.query(FXRateSpec).filter(FXRateSpec.bbid == bbid)

        if provider is not None:
           q = q.filter(FXRateSpec.provider.in_(provider))
        return self._sessionMgr.query_format_df(q)

    def get_fx_uid_spec(self, uid):
        return self.__spec.loc[[uid]].reset_index()

    def get_uids_from_bbids(self, bbids):
        if isinstance(bbids, str):
            bbids = [bbids]
        return [x[0] for x in self._session.query(FXRateSpec.uid).filter(FXRateSpec.bbid.in_(bbids))]

    @staticmethod
    def parse_reverse_currency_pair(bbid):
        return bbid[3:] + bbid[0:3]

    def __load_fx_series_df_from_uids(self, uids):
        if isinstance(uids, int):
           uids = [uids]
        q = self._session.query(FXRate.uid,
                                FXRate.date,
                                FXRate.EB.label(PriceQuote.BID.value),
                                FXRate.ER.label(PriceQuote.MID.value),
                                FXRate.EO.label(PriceQuote.ASK.value),
                                FXRate.pricing_location).filter(FXRate.uid.in_(uids))
        df_ = self._sessionMgr.query_format_df(q).set_index('uid', drop=True)
        self._fx_cache = pd.concat((self._fx_cache, df_), axis=0)

    def get_fx_series_df_from_uid(self, uid, pricing_location=None):

        if self._fx_cache.size > 0 and uid in self._fx_cache.index:
           fx_ = self._fx_cache.loc[uid].set_index('pricing_location', drop=False)

           if pricing_location is not None:
               return fx_.loc[pricing_location].set_index('date', drop=True).drop(columns='pricing_location')
           else:
               return fx_.set_index('date', drop=True)
        else:
            self.__load_fx_series_df_from_uids(uid)
            return self.get_fx_series_df_from_uid(uid, pricing_location)

    def get_fx_time_series_from_uid(self, uid, pricing_location=None):

        fx_df = self.get_fx_series_df_from_uid(uid, pricing_location).dropna(axis=1, how='all')
        fx_spec = self.get_fx_uid_spec(uid)
        fx_spec['pricing_location'] = pricing_location

        atts = pd.concat([fx_spec.T] * fx_df.shape[1], axis=1)
        atts.columns = fx_df.columns

        cols = pd.concat((atts, fx_df.columns.to_frame().T), axis=0)
        cols.index = atts.index.to_list() + ['quote']

        series = CTimeSeries(fx_df, ts_type=TimeSeriesType.LEVELS)
        series.columns = pd.MultiIndex.from_frame(cols.T)
        series.columns.names = fx_spec.columns.to_list() + ['quote']
        return series

    def __construct_fx_curve_single_currency_USD_cross(self,
                                                       bbid,
                                                       provider,
                                                       pricing_location):

        print('Constructing {} {} {} FX Curve...'.format(bbid, provider, pricing_location))


        rvs_bbid = self.parse_reverse_currency_pair(bbid)
        bbid_spec = pd.concat((self.get_bbid_spec(bbid, provider),
                               self.get_bbid_spec(rvs_bbid, provider)), axis=0).set_index('maturity', drop=True)
        self.cache_fx_curve_data_single_currency_pair(bbid)

        if bbid_spec.size == 0:
            print('Error - currency pair {} or {} not in database and curve cannot be constructed'.format(bbid, rvs_bbid))
            return pd.DataFrame()

        unique_mats = np.unique(bbid_spec.index)

        curve_df = CTimeSeries(ts_type=TimeSeriesType.LEVELS, returns_type=ReturnsType.SIMPLE )
        for mat in unique_mats:
            mat_spec = bbid_spec.loc[[mat]].set_index('uid', drop=True)

            mat_df = CTimeSeries(ts_type=TimeSeriesType.LEVELS, returns_type=ReturnsType.SIMPLE)
            for uid in mat_spec.index:
                uid_spec = mat_spec.loc[uid]
                _rates = self.get_fx_time_series_from_uid(uid, pricing_location=pricing_location)

                # shift provider to level 0 in the columns
                _rates.columns = \
                    _rates.columns.reorder_levels(['provider'] + list(np.setdiff1d(_rates.columns.names, 'provider')))

                if uid_spec.bbid == self.parse_reverse_currency_pair(bbid):
                    _rates = 1 / _rates
                    _rates = _rates.rename(
                        columns={PriceQuote.ASK.value: PriceQuote.BID.value, PriceQuote.BID.value: PriceQuote.ASK.value,
                                 uid_spec.bbid: bbid})

                mat_df = mat_df.concat(_rates)

            for single_provider in provider:
                tmp = mat_df.get(single_provider, CTimeSeries(columns=mat_df.columns, returns_type=ReturnsType.SIMPLE, ts_type=TimeSeriesType.LEVELS))
                tmp._added_attributes = tmp._added_attributes.get(single_provider, pd.DataFrame(columns=mat_df.columns))

                tmp.columns = tmp.columns.droplevel('uid')
                tmp._added_attributes.columns = tmp.columns

                df_p = tmp.T.groupby(lambda x: x).first().T.dropna(how='all')
                curve_df = curve_df.combine_left(df_p)

        curve_df.columns = pd.MultiIndex.from_tuples(curve_df.columns)
        curve_df.columns.names = ['bbid', 'maturity', 'pricing_location', 'quote']

        rvs_curve = self.reverse_fx_curve(curve_df)
        rvs_bbid = self.parse_reverse_currency_pair(bbid)

        self._save_currency_curve_to_pickles(curve_df, bbid, provider, pricing_location)
        self._save_currency_curve_to_pickles(rvs_curve, rvs_bbid, provider, pricing_location)
        self.reset_cache()

        id = self._get_pickle_uid(bbid, provider, pricing_location)
        rvs_id = self._get_pickle_uid(rvs_bbid, provider, pricing_location)

        self._fx_curve_cache[id] = curve_df.copy()
        self._fx_curve_cache[rvs_id] = rvs_curve.copy()

    def _save_currency_curve_to_pickles(self, curve_df, bbid, provider, pricing_location):
        uid = self._get_pickle_uid(bbid, provider, pricing_location)

        print('Saving {} to pickles'.format(uid))
        self._sessionMgr.pickle_and_save_to_database(curve_df, uid)

    def is_pickled(self,
                   currency_pair,
                   provider=None,
                   pricing_location=None):

        uid = self._get_pickle_uid(currency_pair,
                                   provider,
                                   pricing_location)
        return self._sessionMgr.is_pickled(uid)

    def _get_pickle_uid(self, currency_pair, provider, pricing_location):
        return str((currency_pair.replace('/',""), provider, pricing_location))

    def get_fx_curve_USD_cross(self, bbid, provider, pricing_location):
        assert 'USD' in bbid, 'Error - function only supports USD cross exchange rates'
        key = str((bbid, provider, pricing_location))
        if key not in self._cached_fx_curves:
           self.__load_fx_curve_USD_cross(bbid, provider, pricing_location)
        return self._fx_curve_cache.get(key, pd.DataFrame()).dropna(how='all')

    def __load_fx_curve_USD_cross(self, bbid, provider, pricing_location):

        # First check if the curve is pickled, if not construct is
        if not self.is_pickled(bbid, provider, pricing_location):
            self.__construct_fx_curve_single_currency_USD_cross(bbid, provider, pricing_location)
        else:
            self._load_fx_curve_from_pickles(bbid, provider, pricing_location)

    def _load_fx_curve_from_pickles(self, bbid, provider, pricing_location):

        id = self._get_pickle_uid(bbid, provider, pricing_location)
        self._fx_curve_cache[id] = self._sessionMgr.load_pickle_from_database(id)

        rvs_bbid = self.parse_reverse_currency_pair(bbid)
        rvs_id = self._get_pickle_uid(rvs_bbid, provider, pricing_location)
        self._fx_curve_cache[rvs_id] =  self.reverse_fx_curve(self._fx_curve_cache[id])

    def reverse_fx_curve(self, fx_curve):

        copy_curve = fx_curve.copy()
        bbid = fx_curve.columns.get_level_values('bbid').unique()[0]
        rvs_df = 1 / copy_curve
        rvs_df = rvs_df.rename(
            columns={PriceQuote.ASK.value: PriceQuote.BID.value, PriceQuote.BID.value: PriceQuote.ASK.value},
            level='quote')
        return rvs_df.rename(columns={bbid: self.parse_reverse_currency_pair(bbid)}, level='bbid')

    def cache_fx_curve_data_single_currency_pair(self, bbid):

        if bbid not in self._cached_pairs:
            rvs_bbid = self.parse_reverse_currency_pair(bbid)
            uids = self.get_uids_from_bbids([bbid, rvs_bbid])
            self.__load_fx_series_df_from_uids(uids)
            self._cached_pairs.extend([bbid, rvs_bbid])

    def get_xUSD_carry(self, currency):

        from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
        gds = GlobalDataSource()

        if self.is_composite_currency(currency):
            self.construct_composite_forward_rates(currency)

        # Construct the regions forward rates from interest rate carry and index implied spot rates
        subregion = self._sessionMgr.get_region_from_currency(currency)
        f_rf = gds.get_interest_rates_for_region(subregion, ['ON', '1m', '3m']).mean(axis=1)
        d_rf = gds.get_interest_rates_for_region('United States', ['ON', '1m', '3m']).mean(axis=1).resample('B').asfreq()
        carry = d_rf.subtract_over_common_dates(f_rf).to_frame(currency)

        # Get the carry from the database, if it exists, we use it where data is available
        fwd_carry = gds.get_fx_carry([currency + 'USD'], '1m', 'mid')
        fwd_carry_copy = fwd_carry.copy()
        fwd_carry = pd.DataFrame()
        if fwd_carry.size > 0:
            fwd_carry.columns = carry.columns
            fx_carry = pd.concat((carry[carry.index < fwd_carry.index.min()],
                                      fwd_carry), axis=0).sort_index()
        else:
            fx_carry = carry.copy()
            fx_carry.columns = spt.columns
        return fx_carry.copy(), fwd_carry_copy

    def construct_composite_forward_rates(self, currency):

        from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
        from epsilonPhi.core.dataModel.dataSources.riskFreeRates.RiskFreeRates import MSCIActivityPanel
        gds = GlobalDataSource()

        if not self.is_composite_currency(currency):
            raise ValueError('Error - currency {} is not a defined composite fx'.format(currency))

        region = self._sessionMgr.get_region_from_currency(currency)
        currency_activity = MSCIActivityPanel.get_activity_panel_single_index(region)
        unique_currencies = np.unique(currency_activity.columns.get_level_values('Currency'))

        carry_rates = CTimeSeries(ts_type=TimeSeriesType.RETURNS, returns_type=ReturnsType.SIMPLE)
        carry_chk = CTimeSeries(ts_type=TimeSeriesType.RETURNS, returns_type=ReturnsType.SIMPLE)
        MVs = CTimeSeries(ts_type=TimeSeriesType.LEVELS, returns_type=ReturnsType.SIMPLE)
        for region_currency in unique_currencies:
            print(region_currency)

            # Get the regions carry rate
            carry, fwd_carry = self.get_xUSD_carry(region_currency)
            carry.columns = [region_currency]

            currency_carry = pd.concat((carry, fwd_carry), axis=1)
            currency_carry.columns = pd.MultiIndex.from_tuples([(region_currency, 'Rates'), (region_currency, 'Forwards')])
            carry_chk = carry_chk.concat(currency_carry)

            # We need to multiply the forward rates by the Market Value in the Index
            ticker = self._get_regional_equity_index(region_currency, 'USD')
            MV = gds.get_time_series_data_from_ticker(ticker, cols='MV', ts_type=TimeSeriesType.LEVELS).dropna()

            # List of dates where the region was included in the parent index
            ccy_locs = currency_activity.iloc[:, currency_activity.columns.get_level_values('Currency') == region_currency].any(axis=1)
            locs = ccy_locs[ccy_locs]
            MV.columns = pd.MultiIndex.from_tuples([(region_currency, locs.first_valid_index())])

            # Find the intersect of carry, market values and locs
            common_dates = np.intersect1d(np.intersect1d(carry.index, MV.index), locs.index)

            carry_rates = carry_rates.concat(carry.loc[common_dates])
            MVs = MVs.concat(MV.loc[common_dates])

        wts = MVs / MVs.sum(axis=1, skipna=True).to_frame('MV').values
        fx_carry = carry_rates.multiply_over_common_dates(wts.get(carry_rates.columns)).sum(axis=1)

        spt = self.get_msci_composite_xUSD_spot_rates(currency)
        fwds = spt.multiply_over_common_dates(np.exp(fx_carry.to_frame('carry') * (1 / 12)))
        return fwds

    def get_msci_composite_xUSD_fwd_rates(self, currency):

        if self.is_composite_currency(currency):
           return  self.construct_composite_forward_rates(currency)

        # Get the spot rate for the currency region
        spt = self.get_msci_composite_xUSD_spot_rates(currency)
        fx_carry = self.get_xUSD_carry(currency)

        fwds = spt.multiply_over_common_dates(np.exp(fx_carry * (1 / 12)))
        return fwds.copy()

    def is_composite_currency(self, currency):
        return currency in CompositeFXRates.composite_currencies

    def is_EURO_legacy(self, currency):
        return self._sessionMgr.is_EURO_legacy(currency)

    def _get_regional_equity_index(self, exposure_currency, denominated_currency):
        region = self._sessionMgr.get_region_from_currency(exposure_currency)

        if (self.is_EURO_legacy(exposure_currency) and
                (exposure_currency == denominated_currency)):
            exposure_currency = 'EUR'
            denominated_currency = 'EUR'
        elif self.is_EURO_legacy(exposure_currency):
            exposure_currency = 'EUR'

        q = self._session.query(EquityIndexSpec.ticker).filter(EquityIndexSpec.exposure_currency == exposure_currency,
                                                                  EquityIndexSpec.denominated_currency == denominated_currency,
                                                                  EquityIndexSpec.region == region,
                                                                  EquityIndexSpec.provider == 'MSCI',
                                                                  not_(EquityIndexSpec.name.like('%Hedge%')),
                                                                  not_(EquityIndexSpec.name.like('%Hedged%')),
                                                                  not_(EquityIndexSpec.name.like('%Growth%')),
                                                                  not_(EquityIndexSpec.name.like('%Value%')))
        if denominated_currency == 'USD':
            q = q.filter(EquityIndexSpec.ticker.like('%$'))
        elif exposure_currency == denominated_currency:
            q = q.filter(EquityIndexSpec.ticker.like('%L'))
        return q.scalar()


    def get_msci_composite_xUSD_spot_rates(self, currency):


        local = self._get_regional_equity_index(currency, currency)
        if not local:
            return CTimeSeries(ts_type=TimeSeriesType.LEVELS)

        base = currency
        if self.is_EURO_legacy(currency):
           currency = 'EUR'

        from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
        gds = GlobalDataSource()

        dollar = local[0:-1] + '$'
        Local = gds.get_time_series_data_from_ticker(local, cols='PI')
        Foreign = gds.get_time_series_data_from_ticker(dollar, cols='PI')

        fx = Foreign.division_over_common_dates(Local)
        fx.columns = [currency + 'USD']
        rebase = 1

        #db_spts = gds.get_fx_spot_rates([currency + 'USD'], 'mid')
        #if db_spts.size > 0:
        #    rebase = (db_spts.loc[max(np.intersect1d(fx.index, db_spts.index))].values /
        #          fx.loc[max(np.intersect1d(fx.index, db_spts.index))].values)
        #else:
        #    rebase = 1

        rebased_fx = fx * rebase
        rebased_fx.columns = [base + 'USD']
        return rebased_fx


if __name__ == "__main__":

    from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
    from epsilonPhi.core.utils.FrameUtils import FrameUtils
    from epsilonPhi.core.dataModel.dataSources.riskFreeRates.RiskFreeRates import MSCIActivityPanel

    mgr = FXCurveManager()

    currency = 'WLD'
    spt = mgr.get_msci_composite_xUSD_spot_rates('ACW')
    fwd = mgr.get_msci_composite_xUSD_fwd_rates('ACW')



    panel = MSCIActivityPanel.get_activity_panel_single_index('AC World')
    currencies = panel.columns.get_level_values('Currency')
    gds = GlobalDataSource()

    mgr = FXCurveManager()
    failed = list()

    fxdf = CTimeSeries(ts_type=TimeSeriesType.LEVELS)
    for currency in np.unique(currencies):

        if currency == 'USD':
            continue

        if currency == 'QAD':
            currency = 'QAR'

        print(currency)

        df_fwds = mgr.get_msci_implied_xUSD_fwd_rates(currency)
        fxdf = fxdf.concat(df_fwds)

    xrates = ['USD' + x for x in currencies]
    xrate_spt = gds.get_fx_spot_rates(xrates, 'mid')
    self = FXCurveManager()