import time
from enum import Enum
from functools import lru_cache
import pandas as pd
from gs_quant.data import Dataset
from gs_quant.session import GsSession, Environment
from epsilonPhi.core.lib.Decorators import SingletonDecorator
from epsilonPhi.core.env.Env import GSQ_CLIENT_ID, GSQ_CLIENT_SECRET
from gs_quant.data import DataMeasure, DataFrequency, Dataset, AssetMeasure
from datetime import date
import datetime
from typing import List
from gs_quant.data import DataContext
from gs_quant.markets.securities import SecurityMaster, AssetIdentifier, ExchangeCode
from gs_quant.api.gs.assets import GsAssetApi
import gs_quant.timeseries as ts
from gs_quant.timeseries.measures import VolReference


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

    _cache = {}
    def __init__(self):
        self.initialize()

    def initialize(self):
        GsSession.use(Environment.PROD,
                      GSQ_CLIENT_ID,
                      GSQ_CLIENT_SECRET,
                      ('read_product_data'))

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

    @staticmethod
    def _resolve_identifier(identifier: List[str]) -> tuple:
        response = GsAssetApi.resolve_assets(
            identifier=identifier,
            fields=['name', 'id', 'type', 'ticker', 'isin', 'bbid', 'gsid', 'exchange'],
            limit=1,
        )

        res = response[identifier]
        if len(res) > 0:
           flds = res[0].keys()

           if 'id' in flds:
               return res[0].get('id'), AssetIdentifier.MARQUEE_ID
           if 'ticker' in flds:
               return res[0].get('ticker'), AssetIdentifier.TICKER
           elif 'bbid' in flds:
               return res[0].get('bbid'), AssetIdentifier.BLOOMBERG_ID
           elif 'isin' in flds:
               return res[0].get('isin'), AssetIdentifier.ISIN
           else:
               raise ValueError('Error - not compatible fields')
        else:
            raise ValueError('Error - identifier {} not recognised'.format(identifier))

    @staticmethod
    def get_security_tick_trade_data(security, month, year):

        security = GSQuantManager().get_security(security)
        SD = datetime.datetime(year=year, month=month, day=1)
        ED = SD + pd.tseries.offsets.MonthEnd(1)

        prices = security.get_data_series(DataMeasure.TRADE_PRICE,
                                         frequency=DataFrequency.REAL_TIME,
                                         start=SD,
                                         end=ED.to_pydatetime())
        return prices.to_frame(security.name)


    @staticmethod
    def get_security(asset_id):
        if asset_id not in GSQuantManager()._cache.keys():
            id, id_type = GSQuantManager()._resolve_identifier(asset_id)
            GSQuantManager()._cache[asset_id] = SecurityMaster.get_asset(id, id_type=id_type)
        return GSQuantManager()._cache[asset_id]


    @staticmethod
    def get_data_context(start_date, end_date):
        return DataContext(start=pd.to_datetime(start_date),
                           end=pd.to_datetime(end_date))

    @staticmethod
    def get_ivol(asset_id,
                 start_date,
                 end_date=date.today(),
                 tenor='1m',
                 vol_reference='spot',
                 relative_strike=100):


        asset = GSQuantManager().get_security(asset_id)
        context = GSQuantManager().get_data_context(start_date, end_date)
        with context:
            res = ts.implied_volatility(asset,
                                        tenor=tenor,
                                        strike_reference=VolReference(vol_reference.lower()),
                                        relative_strike=relative_strike)
            df_ = res.to_frame(asset)
            df_.columns = pd.MultiIndex.from_tuples([(asset_id, tenor, vol_reference, relative_strike)])
            df_.columns.names = ['id', 'tenor', 'vol_reference', 'ref_strike']
        return df_.copy()

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
    import pandas as pd

    _SAVE_PATH = r'/Users/francisbarker/Data/SPX Vols'

    gsq = GSQuantManager()
    spx = gsq.get_security('SPX')

    strikes = np.array(range(40, 180, 5))
    tenors = ['2m']

    df_ = pd.DataFrame()
    for tenor in tenors:
        for k in strikes:
            print(str(k) + ' ' + tenor)
            try:
                res = gsq.get_ivol('SPX', '31-Dec-2004',  tenor=tenor, vol_reference='normalized', relative_strike=k)
                res.to_csv(os.path.join(_SAVE_PATH, str(k) + '_' + tenor + '.csv'))
            except:
                pass

    tenors = ['1m', '2m', '3m', '6m', '1y']
    #relative_strike = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    ds = Dataset('EDRVOL_PERCENT_INTERNAL')

    _all_vols = pd.DataFrame()
    for tenor in tenors:
        print(tenor)
        res_1 = ds.get_data(start=datetime.date(year=1999, month=1, day=1), end=datetime.date(year=2009, month=12, day=31),
                              assetId='MA4B66MW5E27U8P32SB',
                              strikeReference='forward',
                              tenor=tenor)
        res_2 = ds.get_data(start=datetime.date(year=2010, month=1, day=1), end=datetime.date.today(),
                              assetId='MA4B66MW5E27U8P32SB',
                              strikeReference='spot',
                              tenor=tenor)

        res = pd.concat((res_1, res_2), axis=0)
        _all_vols = pd.concat((_all_vols, res), axis=0)



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


df = pd.read_csv('/Users/francisbarker/Desktop/SPX Normalized Vols.csv')
df.date = pd.to_datetime(df.date, format='%d/%m/%Y')
df = df.set_index('date', drop=True)

spx = gsq.get_security('SPX')
prices = spx.get_data_series(DataMeasure.CLOSE_PRICE,
                             frequency=DataFrequency.DAILY,
                             start='1990-12-31',
                             end=df.index.max().strftime("%Y-%m-%d"))
prices = prices.resample('D').asfreq().ffill()
prices = prices.loc[df.index].to_frame('spot')
vols = pd.concat((df, prices), axis=1)

df = vols.to_csv('/Users/francisbarker/Desktop/SPX Normalized Vols.csv')



