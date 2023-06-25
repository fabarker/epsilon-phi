from enum import Enum
from functools import lru_cache
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
    gsq = GSQuantManager()
    ds = Dataset('EDRVOL_PERCENT_INTERNAL')
    cov = gsq._get_coverage(ds)

    import datetime
    import pandas as pd
    from gs_quant.api.gs.assets import GsAssetApi
    ticker = 'MA4B66MW5E27U8P32SB'
    assets = GsAssetApi.get_many_assets(["id"], limit=10000, id=[ticker])
    data = ds.get_data(datetime.date(2019, 6, 3), assetId=[ticker], limit=50)
    print(data.head())  # peek at first few rows of data


    import datetime
    import pandas as pd
    # ds = Dataset('FXIVOL_V2_PREMIUM')
    # cov = gsq._get_coverage(ds)
    # coverage = pd.DataFrame([x.split() for x in cov.name])
    # coverage.columns = ['type','currency','tenor','delta','putcall']
    # coverage['assetID'] = cov.assetId
    # coverage.drop(columns='type')
    # coverage = coverage.set_index('assetID')
    #
    # from gs_quant.api.gs.assets import GsAssetApi
    # assets = GsAssetApi.get_many_assets(["id"], limit=10000, id=['MA000H1QRPSBXH8S'])
    #
    # data = ds.get_data(datetime.date(2019, 6, 3), assetId=["MA569C4VG2TC7Y8V", "MAG88ES0BEWNQCHV", "MA4XVXGQ63957V6F"], limit=50)
    # print(data.head())  # peek at first few rows of data
    #
    #
    # ds = Dataset('EDRVOL_PERCENT_INTERNAL')
    # data = ds.get_data(datetime.date(2004, 3, 1), assetId=["MA4B66MW5E27UADJ5FE", "MAKEKJNSGN4H5V2T", "MA4B66MW5E27UAN26Y8"], limit=50)
    # print(data.head())  # peek at first few rows of data
