from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.dataModel.dataSources.fxCurve.FXCurveMgr import FXCurveManager
from epsilonPhi.core.dataModel.enums.Database import Provider, PricingLocation, PriceQuote
from epsilonPhi.core.lib.Decorators import SingletonDecorator
from epsilonPhi.core.utils.FrameUtils import FrameUtils
import pandas as pd
import itertools
import copy
import datetime as dt


@SingletonDecorator
class FXCurve(object):

    _fx_cache = pd.DataFrame()
    _DEFAULT_PRICING_LOCATION = PricingLocation.LONDON
    _DEFAULT_PROVIDERS = (Provider.WMR, Provider.REFINITIV, Provider.BBI)

    def __init__(self, provider=None, pricing_location=None):

        self._fxCurveMgr = FXCurveManager()
        if provider is not None:
           self.set_provider(provider)
        else:
            self.set_provider(self._DEFAULT_PROVIDERS)

        if pricing_location is not None:
            self.set_pricing_location(pricing_location)
        else:
            self.set_pricing_location(self._DEFAULT_PRICING_LOCATION)


    @staticmethod
    def parse_reverse_currency_pair(bbid):
        return bbid[3:] + bbid[0:3]

    @property
    def cached_bbids(self):
        if self._fx_cache.size > 0:
            return list(np.unique(self._fx_cache.columns.get_level_values('bbid')))
        else:
            return self._fx_cache.columns
    @property
    def provider(self):
        return [x.value for x in self.__provider]

    @property
    def pricing_location(self):
        return self.__pricing_location

    def reset_cache(self):
        self._fx_cache = pd.DataFrame()

    def set_provider(self, provider):

        if not DateUtils.is_iterable(provider):
           provider = [provider]
        self.__provider = tuple(provider)
        self.reset_cache()

    def set_pricing_location(self, pricing_location):
        if isinstance(pricing_location, PricingLocation):
            self.__pricing_location = pricing_location.value
        else:
            self.__pricing_location = pricing_location
        self.reset_cache()

    def __get_fx_curve(self, bbid):
        return self._fx_cache.iloc[:, self._fx_cache.columns.get_level_values('bbid') == bbid].dropna(how='all')

    def get_fx_curves(self, currency_pairs):

        if isinstance(currency_pairs, str):
            currency_pairs = [currency_pairs]

        fx_pairs = [pd.DataFrame()]
        for fx_pair in currency_pairs:
            fx_pairs.extend([self.get_fx_curve_single_currency(fx_pair)])
        return pd.concat(fx_pairs, axis=1)

    def get_fx_curve_single_currency(self, currency_pair):

        bbid = currency_pair.replace('/', '')
        if bbid in self.cached_bbids:
            return self.__get_fx_curve(bbid)
        elif self.parse_reverse_currency_pair(bbid) in self.cached_bbids:
            rvs_df = 1 / self.__get_fx_curve(self.parse_reverse_currency_pair(bbid))
            rvs_df = rvs_df.rename(columns={PriceQuote.ASK.value: PriceQuote.BID.value, PriceQuote.BID.value:  PriceQuote.ASK.value}, level='quote')
            rvs_df = rvs_df.rename(columns={self.parse_reverse_currency_pair(bbid): bbid}, level='bbid')
            return rvs_df.copy()
        else:
            self.__load_fx_curve_single_currency(bbid)
            return self.__get_fx_curve(bbid)

    @staticmethod
    def multiply_curves(curve_1, curve_2):

        bbid_1 = curve_1.columns.get_level_values('bbid').unique().all()
        bbid_2 = curve_2.columns.get_level_values('bbid').unique().all()
        bbid = bbid_1[0:3] + bbid_2[3:]

        common_dates = np.intersect1d(curve_1.index, curve_2.index)

        bse_prime = curve_1.loc[common_dates]
        ctr_prime = curve_2.loc[common_dates]
        bse_prime.columns = bse_prime.columns.droplevel('bbid')
        ctr_prime.columns = ctr_prime.columns.droplevel('bbid')

        common_cols = np.intersect1d(bse_prime.columns,
                                     ctr_prime.columns)

        cross_df = bse_prime[common_cols] * ctr_prime[common_cols]
        cross_df.columns = pd.MultiIndex.from_tuples([(bbid,) + x for x in cross_df.columns])
        cross_df.columns.names = copy.deepcopy(curve_1.columns.names)
        return cross_df.copy()

    def __load_fx_curve_single_currency(self, bbid: str):

        bbid = bbid.replace('/', '')
        if bbid.find('USD') < 0:
            bse = bbid[0:3] + 'USD'
            ctr = 'USD' + bbid[3:]

            bse_fx = self.get_fx_curve_single_currency(bse)
            ctr_fx = self.get_fx_curve_single_currency(ctr)
            curve_df = self.multiply_curves(bse_fx, ctr_fx)
            self._fx_cache = pd.concat((self._fx_cache, curve_df), axis=1)
        else:
            curve_df = self._fxCurveMgr.get_fx_curve_USD_cross(bbid, self.provider, self.pricing_location)
            self._fx_cache = pd.concat((self._fx_cache, curve_df), axis=1)

    def get_spot_rates(self, bbids, quote='mid'):
        if isinstance(bbids, str):
            bbids = [bbids]
        return self.get_forward_rates(bbids, '0m', quote)

    def get_forward_rates(self, bbids, tenor='1m', quote='mid'):
        if isinstance(bbids, str):
            bbids = [bbids]
        fx_curves = self.get_fx_curves(bbids)
        fwd_df = FrameUtils.select_subset_level(fx_curves, 'maturity', tenor)
        return FrameUtils.select_subset_level(fwd_df, 'quote', quote).dropna(how='all')

    def get_forward_prices(self, bbids, pricing_date, maturity_date, quote='mid'):

        fx_curves = self.get_fx_curves(bbids)

        fx_curves = FrameUtils.drop_subset_level(fx_curves, 'maturity', ['ON', 'SW', 'TN'])
        pricing_date = pd.to_datetime(pricing_date)
        maturity_date = pd.to_datetime(maturity_date)

        if isinstance(quote, str):
            quote = np.array([quote])

        fx_curve = FrameUtils.select_subset_level(fx_curves, 'quote', quote).dropna(how='all')
        df = fx_curve.reindex(pricing_date)

        level_number = df.columns._get_level_number('maturity')
        mats = DateUtils.Rdate_to_mat(df.columns._levels[level_number])
        df.columns = df.columns.set_levels(mats, level='maturity', verify_integrity=False)

        # Time to maturity
        tau = DateUtils.get_date_delta(pricing_date, maturity_date, True)

        # build levels
        level_ints = np.setdiff1d(np.array(range(len(df.columns._levels))), level_number)
        levels = [np.array(df.columns._levels[x]) for x in level_ints]
        levels.insert(level_number, tau)

        # Create a new list with unique combinations
        cols = list(itertools.product(*levels))
        vals = pd.DataFrame(np.ones((df.shape[0], len(cols))) * np.nan, columns=pd.MultiIndex.from_tuples(cols), index=df.index)
        df_prime = pd.concat((df, vals.loc[:, ~vals.columns.isin(df.columns)]), axis=1)

        sorted_df = df_prime.T.sort_index(level='maturity').T
        ccys = sorted_df.columns.get_level_values('bbid').unique()
        types = sorted_df.columns.get_level_values('quote').unique()

        interp_df = pd.DataFrame()
        for ccy in ccys:
            for type in types:
                idxs = np.logical_and(sorted_df.columns.get_level_values('bbid') == ccy,
                               sorted_df.columns.get_level_values('quote') == type)
                # Interpolate curves
                fwd_curve = np.exp(np.log(sorted_df.iloc[:, idxs].interpolate(method='linear', axis=1)))

                # Extract the observations we need
                locs = list(zip(pricing_date, itertools.product([ccy], tau, [type])))
                fwds = pd.DataFrame([fwd_curve.loc[x] for x in locs], index=pricing_date)
                fwds.columns = pd.MultiIndex.from_tuples([(ccy, type)])

                interp_df = pd.concat((interp_df, fwds), axis=1).ffill()
        return interp_df.copy()

    def get_carry(self, bbids, tenors='1m', quotes='mid'):

        # Get the spot and forward rates
        spts = self.get_spot_rates(bbids, quotes)
        fwds = self.get_forward_rates(bbids, tenors, quotes)

        # find the dates that are common between both dataframes
        common_dates = np.intersect1d(spts.index, fwds.index)

        # align the dataframes and take log ratios
        cols = fwds.columns.set_levels(['0m'] * fwds.shape[1], level='maturity', verify_integrity=False)
        carry = np.log(fwds.loc[common_dates] / spts.get(cols).loc[common_dates].values)
        mult = np.reshape(1/DateUtils.Rdate_to_mat(carry.columns.get_level_values('maturity').values), (1, fwds.shape[1]))

        # annualize the carry rates
        return carry * np.repeat(mult, carry.shape[0], axis=0)


if __name__ == "__main__":

    curve = FXCurve()

    pricing_date = dt.date(year=2022, month=12, day=31)
    maturity_date = dt.date(year=2023, month=1, day=30)
    df_ = curve.get_forward_prices('AUD/USD', pricing_date, maturity_date, quote='mid')


