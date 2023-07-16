from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.lib.Decorators import SingletonDecorator
from epsilonPhi.core.utils.FrameUtils import FrameUtils
import pandas as pd
import itertools
import copy


@SingletonDecorator
class YieldCurve(object):
    _yield_cache = pd.DataFrame()
    _session = SessionMgr().getSessionFactory()

    def __init__(self, currency):
        self._session = SessionMgr().getSessionFactory()

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



if __name__ == "__main__":

    _session = SessionMgr().getSessionFactory()
    tickers = session

    specs = _session.query(InterestRateSpec.ticker).filter_by(currency='USD').all()

