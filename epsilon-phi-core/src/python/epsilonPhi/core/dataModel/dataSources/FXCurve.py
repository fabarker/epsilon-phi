from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.lib.Decorators import SingletonDecorator
from epsilonPhi.core.utils.FrameUtils import FrameUtils
import pandas as pd
import itertools
import copy


@SingletonDecorator
class FXCurve(object):
    _fx_cache = pd.DataFrame()
    _SUPPORTED_PROVIDERS = dict()

    def __init__(self):
        self._session = SessionMgr().getSessionFactory()

    @staticmethod
    def parse_reverse_currency_pair(bbid):
        return bbid[3:] + bbid[0:3]

    @property
    def cached_bbids(self):
        if self._fx_cache.size > 0:
            return list(np.unique(self._fx_cache.columns.get_level_values('bbid')))
        else:
            return self._fx_cache.columns

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
            rvs_df = rvs_df.rename(columns={'ask': 'bid', 'bid': 'ask'}, level='quote')
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
        if bbid not in self.cached_bbids:

            if bbid.find('USD') < 0:
                bse = bbid[0:3] + 'USD'
                ctr = 'USD' + bbid[3:]

                bse_fx = self.get_fx_curve_single_currency(bse)
                ctr_fx = self.get_fx_curve_single_currency(ctr)
                curve_df = self.multiply_curves(bse_fx, ctr_fx)
                self._fx_cache = pd.concat((self._fx_cache, curve_df), axis=1)
                return

            # LOAD RAW DATA FROM DATABASE
            fx_info = FXRate.get_spec_df_from_bbids([bbid, self.parse_reverse_currency_pair(bbid)], index_col='uid')
            if fx_info.size == 0:
                raise ValueError('Error - currency pair {} or {} not in database and curve cannot be constructed'.format(bbid, self.parse_reverse_currency_pair(bbid)))

            fx_df = FXRate.getDataframe(uids=fx_info.index, index_col='uid')

            curve_df = pd.DataFrame()
            unique_mats = np.unique(fx_info.maturity)
            for mat in unique_mats:
                df_info = fx_info[fx_info.maturity == mat]

                mat_df = pd.DataFrame()
                for uid in df_info.index:
                    if df_info.loc[uid].bbid == self.parse_reverse_currency_pair(bbid):
                        rates = 1 / fx_df.loc[uid].set_index('date', drop=True)
                        rates = rates.rename(columns={'ask': 'bid', 'bid': 'ask'})
                    else:
                        rates = fx_df.loc[uid].set_index('date', drop=True)
                    rates.columns = pd.MultiIndex.from_tuples(
                        [(bbid, mat, df_info.loc[uid].provider) + (x,) for x in rates.columns])
                    rates.columns.names = df_info.columns.append(pd.Index(['quote']))
                    mat_df = pd.concat((mat_df, rates), axis=1)

                provider_rank = ['WM/Refinitiv', 'Refinitiv', 'Barclays Bank PLC', 'GTIS - FTID/TR']
                providers = [x for x in provider_rank if x in mat_df.columns.get_level_values('provider')]
                fx_rates = pd.DataFrame()

                if len(providers) > 0:
                    for provider in providers:
                        df_temp = mat_df.iloc[:, mat_df.columns.get_level_values('provider') == provider].dropna(how='all')
                        df_temp.columns = df_temp.columns.droplevel('provider')
                        df_p = df_temp.groupby(lambda x: x, axis=1).first()
                        fx_rates = fx_rates.combine_first(df_p)
                    fx_rates.columns = pd.MultiIndex.from_tuples(fx_rates.columns)
                    fx_rates.columns.names = mat_df.columns.droplevel('provider').names

                curve_df = pd.concat((curve_df, fx_rates), axis=1).sort_index()
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
                locs = list(zip(pricing_dates, itertools.product([ccy], tau, [type])))
                fwds = pd.DataFrame([fwd_curve.loc[x] for x in locs], index=pricing_dates)
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
    pricing_dates = pd.date_range('31-Dec-2021', '30-Dec-2022')
    maturity_dates = pd.to_datetime(['30-Dec-2022'] * pricing_dates.__len__())

    fwds = curve.get_forward_prices(['GBPUSD','EURUSD'], pricing_dates, maturity_dates, quote=['mid'])
    spts = curve.get_spot_rates('GBPUSD')

    spt_fwd = pd.concat((spts.reindex(fwds.index).ffill(), fwds), axis=0)
