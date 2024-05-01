from epsilonPhi.core.dataModel.enums.TimeSeries import ReturnsType, TimeSeriesType
from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.dataModel.dataSources.curves.fxCurve.FXCurveMgr import FXCurveManager
from epsilonPhi.core.dataModel.enums.Database import Provider, PricingLocation, PriceQuote
from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries
from epsilonPhi.core.lib.Decorators import SingletonDecorator
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.utils.FrameUtils import FrameUtils
from epsilonPhi.core.utils.DateUtils import Offsets
import pandas as pd
import itertools
import copy
import warnings

warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')


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
        #return self._fx_cache.iloc[:, self._fx_cache.columns.get_level_values('bbid') == bbid].dropna(how='all')
        return FrameUtils.select_subset_level(self._fx_cache, 'bbid', bbid).dropna(how='all')

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

    def get_forward_contract_return(self, bbid, pricing_dates, maturity_date, quote='mid'):

        prices_s = self.get_forward_prices(bbid, pricing_dates[0:-1], maturity_date, quote)
        prices_s = prices_s.reset_index(level='maturity_dates', drop=True).sort_index()

        assert np.all(maturity_date >= pricing_dates[1:]), 'Error - maturity date must fall on or after end pricing date'
        prices_t = self.get_forward_prices(bbid, pricing_dates[1:]  , maturity_date, quote)
        prices_t = prices_t.reset_index(level='maturity_dates', drop=True).sort_index()
        df_ = pd.DataFrame(np.log(prices_t.values / prices_s.values), index=prices_t.index, columns=[bbid])
        return df_.sort_index()

    def hedge_time_series_from_local(self, time_series, from_currency, to_currency, hedge_ratio, hedge_frequency=Frequency.BUSINESS_MONTHLY):

        if from_currency == to_currency:
            return time_series

        if abs(hedge_ratio) < DateUtils.ERROR_TOLERANCE:
            unhdgd = self.unhedged_time_series(time_series, from_currency, to_currency).get_returns()
            return unhdgd.to_time_series_type(time_series.type)

        lvls = time_series.get_levels()
        maturity_date = lvls.dates[0:-1] + Offsets.getOffset(hedge_frequency, 1)

        fwd_returns = self.get_forward_contract_return(from_currency + to_currency, lvls.dates, maturity_date)
        unhedged = self.unhedged_time_series(time_series, from_currency, to_currency).get_returns(ReturnsType.LOG)
        hdg = unhedged.subtract_over_common_dates(hedge_ratio * fwd_returns).dropna()
        return hdg.get_returns(time_series.returns_type).to_time_series_type(time_series.type)


    def hedge_time_series(self,
                          time_series,
                          target_currency,
                          target_hedge_ratio,
                          denominated_currency,
                          exposure_currency=None,
                          current_hedge_ratio=None,
                          hedge_frequency=Frequency.BUSINESS_MONTHLY):

        lvls = time_series.get_levels()
        maturity_date = lvls.dates[0:-1] + Offsets.getOffset(hedge_frequency, 1)

        # If the exposure and denominated currency match, we have local index
        if exposure_currency == denominated_currency or exposure_currency is None:
            return self.hedge_time_series_from_local(time_series, denominated_currency, target_currency, target_hedge_ratio, hedge_frequency)

        # Otherwise there has already been some fx translations applied
        assert current_hedge_ratio is not None, 'Error - time series has already undergone fx translations and must provide its current hedge ratio'
        fx_conversion = self.unhedged_time_series(time_series, denominated_currency, target_currency)
        fx_conversion = fx_conversion.get_returns(ReturnsType.LOG)
        fwds_exp_den = current_hedge_ratio * self.get_forward_contract_return(exposure_currency + denominated_currency, lvls.dates, maturity_date)
        fwds_exp_tar = target_currency * self.get_forward_contract_return(exposure_currency + target_currency, lvls.dates, maturity_date)
        fx_conversion = fx_conversion.addition_over_common_dates(fwds_exp_den).subtract_over_common_dates(fwds_exp_tar)
        return fx_conversion.get_returns(time_series.returns_type).to_time_series_type(time_series.type)

    def unhedged_time_series(self, time_series, from_currency, to_currency):

        if from_currency == to_currency:
           return time_series

        spt = self.get_spot_rates(from_currency + to_currency, quote='mid')

        lvls = time_series.get_levels()
        return lvls.multiply_over_common_dates(spt).to_time_series_type(time_series.type)

    def get_forward_prices_old(self, bbids, pricing_date, maturity_date, quote='mid'):

        if not DateUtils.is_iterable(bbids):
           bbids = [bbids]

        fx_curves = self.get_fx_curves(bbids)
        fx_curves.columns = fx_curves.columns.droplevel('pricing_location')

        fx_curves = FrameUtils.drop_subset_level(fx_curves, 'maturity', ['ON', 'SW', 'TN'])
        pricing_date = pd.to_datetime(pricing_date)
        maturity_date = pd.to_datetime(maturity_date)

        if isinstance(quote, str):
            quote = np.array([quote])

        if isinstance(pricing_date, pd.Timestamp):
           pricing_date = [pricing_date]

        fx_curve = FrameUtils.select_subset_level(fx_curves, 'quote', quote).dropna(how='all')
        df = fx_curve.reindex(pricing_date.unique()).rename_axis(index=fx_curve.index.name)

        level_number = df.columns._get_level_number('maturity')
        mats = DateUtils.Rdate_to_mat(df.columns.get_level_values('maturity'))
        df = FrameUtils.set_levels(df, level_values=mats, level_name='maturity')

        # Time to maturity
        tau = DateUtils.get_date_delta(pricing_date, maturity_date, True)

        # build levels
        level_ints = np.setdiff1d(np.array(range(len(df.columns._levels))), level_number)
        levels = [np.array(df.columns._levels[x]) for x in level_ints]
        levels.insert(level_number, np.unique(tau))

        # Create a new list with unique combinations
        cols = list(itertools.product(*levels))
        vals = pd.DataFrame(np.ones((df.shape[0], len(cols))) * np.nan, columns=pd.MultiIndex.from_tuples(cols), index=df.index)
        df_prime = pd.concat((df, vals.loc[:, ~vals.columns.isin(df.columns)]), axis=1)
        df_prime.columns.names = df.columns.names

        sorted_df = FrameUtils.sort_by_level(df_prime, 'maturity').replace(0, np.nan)
        ccys = sorted_df.columns.get_level_values('bbid').unique()
        types = sorted_df.columns.get_level_values('quote').unique()

        interp_df = pd.DataFrame()
        for ccy in ccys:

            print('Calculating forward prices for {}'.format(ccy))
            for type in types:
                xs_df = sorted_df.xs(key=(ccy, type), level=('bbid', 'quote'), axis=1, drop_level=False)
                xs_df = xs_df.iloc[:, xs_df.columns.get_level_values('maturity') <= np.max(tau)]

                fwd_curve = np.exp(np.log(xs_df).interpolate(method='linear', axis=1, limit_area='inside'))

                df_vec = FrameUtils.vectorize(fwd_curve, 'Price')
                locs = list(zip(itertools.cycle([ccy]), pricing_date, tau, itertools.cycle([type])))
                fwds = df_vec.loc[locs]
                fwds.index = [maturity_date, pricing_date]
                fwds.index.names = ['maturity_dates', 'pricing_dates']
                fwds = fwds.sort_index(level=['maturity_dates', 'pricing_dates'])
                fwds.columns = pd.MultiIndex.from_tuples([(ccy, type)])

                # Extract the observations we need
                #locs = list(zip(pricing_date, itertools.product([ccy], tau, [type])))
                #fwds = pd.DataFrame([fwd_curve.loc[x] for x in locs], index=[maturity_date, pricing_date])
                #fwds.index.names = ['maturity_dates','pricing_dates']
                #fwds = fwds.sort_index(level=['maturity_dates', 'pricing_dates'])
                #fwds.columns = pd.MultiIndex.from_tuples([(ccy, type)])

                interp_df = pd.concat((interp_df, fwds), axis=1).ffill()
        return interp_df.copy()

    def get_forward_prices(self, bbids, pricing_date, maturity_date, quote='mid'):

        if not DateUtils.is_iterable(bbids):
           bbids = [bbids]

        fx_curves = self.get_fx_curves(bbids)
        fx_curves.columns = fx_curves.columns.droplevel('pricing_location')

        fx_curves = FrameUtils.drop_subset_level(fx_curves, 'maturity', ['ON', 'SW', 'TN'])
        pricing_date = pd.to_datetime(pricing_date)
        maturity_date = pd.to_datetime(maturity_date)

        if isinstance(quote, str):
            quote = np.array([quote])

        if isinstance(pricing_date, pd.Timestamp):
           pricing_date = [pricing_date]

        fx_curve = FrameUtils.select_subset_level(fx_curves, 'quote', quote).dropna(how='all')
        df = fx_curve.reindex(pricing_date).rename_axis(index=fx_curve.index.name)

        mats = DateUtils.Rdate_to_mat(df.columns.get_level_values('maturity'))
        df = FrameUtils.set_levels(df, level_values=mats, level_name='maturity')

        # Time to maturity
        tau = DateUtils.get_date_delta(pricing_date, maturity_date, True)

        ccys = df.columns.get_level_values('bbid').unique()
        types = df.columns.get_level_values('quote').unique()

        interp_df = pd.DataFrame()
        for ccy in ccys:

            print('Calculating forward prices for {}'.format(ccy))
            for type in types:
                xs_df = df.xs(key=(ccy, type), level=('bbid', 'quote'), axis=1, drop_level=False)
                xs_df = FrameUtils.sort_by_level(xs_df, level_name='maturity')

                assert np.all(np.sort(xs_df.columns.get_level_values('maturity'))
                              == xs_df.columns.get_level_values('maturity')), 'Error - maturities not sorted'


                mats = xs_df.columns.get_level_values('maturity').values.reshape(1,-1)
                maturity_mat = mats.repeat(xs_df.index.size, axis=0)
                T = tau.reshape(-1, 1).repeat(maturity_mat.shape[1], axis=1)

                LB_locs = ((maturity_mat <= T) & (~xs_df.isna().values))
                UB_locs = ((maturity_mat > T) & (~xs_df.isna().values))

                LB = np.nanmax(maturity_mat * np.where(LB_locs, LB_locs, np.nan), axis=1)
                UB = np.nanmin(maturity_mat * np.where(UB_locs, UB_locs, np.nan), axis=1)

                y_LB = np.log(xs_df.values[np.where(np.isnan(LB), 0, LB).reshape(-1, 1) == maturity_mat])
                y_UB = np.log(xs_df.values[np.where(np.isnan(UB), 0, UB).reshape(-1, 1) == maturity_mat])

                W = (tau - UB) / (LB - UB)
                fwd_mat = np.exp(W * y_LB + (1 - W) * y_UB)
                fwd_mat[LB == tau] = np.exp(y_LB[LB == tau])

                fwds = pd.DataFrame(fwd_mat, index=[maturity_date, pricing_date])
                fwds.index.names = ['maturity_dates', 'pricing_dates']
                fwds = fwds.sort_index(level=['maturity_dates', 'pricing_dates'])
                fwds.columns = pd.MultiIndex.from_tuples([(ccy, type)])

                interp_df = pd.concat((interp_df, fwds), axis=1).ffill()
        return interp_df.copy()

    def get_carry(self, bbids, tenors='1m', quotes='mid'):

        # Get the spot and forward rates
        spts = self.get_spot_rates(bbids, quotes)
        fwds = self.get_forward_rates(bbids, tenors, quotes)

        if spts.size > 0 and fwds.size > 0:

            # find the dates that are common between both dataframes
            common_dates = np.intersect1d(spts.index, fwds.index)

            # align the dataframes and take log ratios
            cols = fwds.columns.set_levels(['0m'] * fwds.shape[1], level='maturity', verify_integrity=False)
            carry = np.log(fwds.loc[common_dates] / spts.get(cols).loc[common_dates].values)
            mult = np.reshape(1/DateUtils.Rdate_to_mat(carry.columns.get_level_values('maturity').values), (1, fwds.shape[1]))

            # annualize the carry rates
            return carry * np.repeat(mult, carry.shape[0], axis=0)
        else:
            return CTimeSeries(columns=[bbids],
                               returns_type=ReturnsType.SIMPLE,
                               ts_type=TimeSeriesType.LEVELS)


if __name__ == "__main__":


    curve = FXCurve(provider=Provider.GS, pricing_location=PricingLocation.NEW_YORK)
    fx = curve.get_fx_curve_single_currency('EUR/USD')



