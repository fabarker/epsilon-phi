from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries, TimeSeriesType
from epsilonPhi.core.lib.Decorators import SingletonDecorator
from sqlalchemy import distinct
import pandas as pd

@SingletonDecorator
class FXCurveManager(object):

    _fx_cache = pd.DataFrame()
    _cached_pairs = list()

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
                                FXRate.EB,
                                FXRate.ER,
                                FXRate.EO,
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

    def cache_fx_curve_data_single_currency_pair(self, bbid):

        if bbid not in self._cached_pairs:
            rvs_bbid = self.parse_reverse_currency_pair(bbid)
            uids = self.get_uids_from_bbids([bbid, rvs_bbid])
            self.__load_fx_series_df_from_uids(uids)
            self._cached_pairs.extend([bbid, rvs_bbid])

if __name__ == "__main__":
    self = FXCurveManager().get_fx_series_df_from_uid(1921)