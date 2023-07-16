import time
from enum import Enum
from functools import lru_cache

import pandas as pd
from gs_quant.data import Dataset
from gs_quant.session import GsSession, Environment
from epsilonPhi.core.lib.Decorators import SingletonDecorator
from epsilonPhi.core.env.Env import GSQ_CLIENT_ID, GSQ_CLIENT_SECRET

class Datasets(Enum):

    FX_IVOL = 'FXIVOL_V2_PREMIUM'
    FX_SPOT = 'FXSPOT_V2_PREMIUM'
    FX_FORWARD_POINTS = 'FXFORWARDPOINTS_V2_PREMIUM'

    HFRI = 'HFRRETURNS'

    EQ_IVOL_TENOR = 'EDRVOL_PERCENT_INTERNAL'
    EQ_IVOL_EXPIRY = 'EDRVOL_PERCENT_EXPIRY_INTERNAL'

    REF_EQUITY_FUTURES_EOD = 'TREOD'

@SingletonDecorator
class GSQuantManager(object):
    def __init__(self):
        self.initialize()

    def initialize(self):
        GsSession.use(Environment.PROD,
                      GSQ_CLIENT_ID,
                      GSQ_CLIENT_SECRET,
                      ('read_product_data','run_analytics',))

    @staticmethod
    def getDataset(DatasetEnum):
        return Dataset(DatasetEnum.value)
    @staticmethod
    def getData(DatasetEnum, tickers, from_date, to_date):
        ds = GSQuantManager.getDataset(DatasetEnum)
        return ds.get_data(from_date=from_date,
                           to_date=to_date,
                           assetId=[tickers],
                           limit=50)

    @staticmethod
    @lru_cache(maxsize=128)
    def _get_coverage(ds, include_history=True):
        print("Fetching coverage", end=" ")
        cov = ds.get_coverage(include_history=include_history)
        print("[DONE]")
        return cov

# class FXIVOL_V2_PREMIUM(GSQuantManager):
#     _DATASET = Dataset('FXIVOL_V2_PREMIUM')
#     _CACHE = dict()
#
#     def __init__(self):
#         pass
#
#     def get_data(self, start_date, bbids, tenors, deltas, putcall):
#         pass
#
#     @staticmethod
#     def get_coverage():
#
#         cov = GSQuantManager._get_coverage(FXIVOL_V2_PREMIUM._DATASET)
#         coverage = pd.DataFrame([x.split() for x in cov.name])
#         coverage.columns = ['type', 'currency', 'tenor', 'delta', 'putcall']
#         coverage['assetID'] = cov.assetId
#         coverage.drop(columns='type')
#         coverage = coverage.set_index('assetID')
#         return coverage.copy()



if __name__ == "__main__":

    import os
    import datetime, os
    import numpy as np

    def _nest_list(flat_list, nested_size):
        return [flat_list[i:i + nested_size] for i in range(0, len(flat_list), nested_size)]

    ds = Dataset('FXSPOT_V2_PREMIUM')
    cov = GSQuantManager()._get_coverage(ds)
    cov = cov.rename(columns={'name':'names'})

    save_dir = r'C:\Users\fabar\Documents\Data\gsquant\fx_spot_fwds'

    USD_list = list()
    for idx, row in cov.iterrows():
        if 'USD' in row.names:
           USD_list.extend([row.assetId])

    chunks = _nest_list(USD_list, nested_size=20)
    frames = [pd.DataFrame()]
    print("Reading Data:", end=" ")
    for ch in chunks:
        print("#", end="")
        res = ds.get_data(start=datetime.date(year=1980, month=12, day=31), end=datetime.date.today(), assetId=ch, pricing_location=['LDN','NYC'])
        con_pd = res.reset_index(drop=False).set_index('assetId')
        uniqueIDs = np.unique(con_pd.index)

        for id in uniqueIDs:
            save_path = os.path.join(save_dir, id + '.csv')
            con_pd.loc[id].reset_index(drop=False).set_index('date').to_csv(save_path)

