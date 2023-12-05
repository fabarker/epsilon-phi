import numpy as np
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.utils.FrameUtils import FrameUtils
from epsilonPhi.core.dataModel.dataSources.vendor.Bloomberg import Bloomberg
from epsilonPhi.core.dataModel.dataSources.vendor.Datastream import pyDatastream
import pandas as pd
import datetime as dt

sessionMgr = SessionMgr()
session = SessionMgr().getSessionFactory()
gds = GlobalDataSource()

data = pd.read_csv('USD Rates.csv', index_col=0, header=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
unique_tickers = data.columns.get_level_values('ticker').unique()

df_ = pd.DataFrame()
for col in set(unique_tickers):
    print(col)

    idx = data.columns.get_level_values('ticker') == col
    sset = data.iloc[:, idx]

    ccy = sset.columns.get_level_values('domestic_currency').unique()[0]
    mat = sset.columns.get_level_values('maturity').unique()[0]

    if ccy not in ['USD']:

        q = session.query(FXRateSpec.ticker).filter(FXRateSpec.provider.in_(['Refinitiv','WM/Refinitiv']),
                                             FXRateSpec.bbid.like('%' + ccy + '%'),
                                             FXRateSpec.bbid.like('%USD%'),
                                             FXRateSpec.maturity == mat)
        res_labels = sessionMgr.query_format_df(q)
        res = gds.get_dataframe_from_tickers(res_labels.values.flatten())

        min_date = np.min([res.get(x)[np.intersect1d(res.get(x).columns, ['EB', 'EO', 'ER'])].dropna().index.min() for x in res_labels.ticker.values ])
        sset.columns = FrameUtils.add_index_to_multi_index(sset.columns, min_date, 'start')
        df_ = pd.concat((df_, sset), axis=1)

df_.to_csv('USDx rates from GBP.csv')
