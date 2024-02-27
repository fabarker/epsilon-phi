from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.dataModel.alchemist.DataModel import YieldCurve, YieldCurveSpec
import numpy as np
import pandas as pd

class YieldCurveMgr(object):
    _cache = {}

    def __init__(self):
        self._sessionMgr = SessionMgr()
        self._session = self._sessionMgr.getSessionFactory()

    def get_yield_curve_info_for_region(self, region):
        return self._sessionMgr.get_yield_curve_tickers_for_region(region)

    def load_yield_curve_dataframe(self, region):

        if region not in self._cache.keys():
            info = self.get_yield_curve_info_for_region(region)
            q = self._session.query(YieldCurve).filter(YieldCurve.uid.in_(list(info.index)))
            res = self._sessionMgr.query_format_df(q).set_index(['uid', 'date'])
            df = pd.concat([res.loc[x].get('RY').to_frame(float(info.loc[x].maturity))
                        for x in np.unique(res.index.get_level_values(0))], axis=1).sort_index()

            unique_mats = np.sort(np.unique(df.columns))
            df_ = pd.concat([df.get([x]).mean(axis=1).to_frame(x)
                                             for x in unique_mats], axis=1)/100
            # Insert all dates
            self._cache[region] = df_.resample('B').asfreq().ffill()

    def get_yield_curve_dataframe(self, region):

        if region not in self._cache.keys():
           self.load_yield_curve_dataframe(region)
        return self._cache.get(region)


if __name__ == "__main__":

    self = YieldCurveMgr()
    reg = 'United States'
    df = self.get_yield_curve_dataframe(reg)



