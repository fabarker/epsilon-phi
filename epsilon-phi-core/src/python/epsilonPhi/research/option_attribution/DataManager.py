import numpy as np
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.alchemist.DataModel import InterestRateSpec, TimeSeriesSpec
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.utils.DateUtils import DateUtils
import pandas as pd
from tqdm import tqdm
import scipy.optimize as optimize

tqdm.pandas()

gds = GlobalDataSource()
mgr = SessionMgr()
session = mgr.getSessionFactory()


## Load Implied Vol Data

def getDataManager():

    df = pd.read_csv('/Users/francisbarker/Data/SPX Normalized Vols.csv', index_col=0)
    df.index = pd.to_datetime(df.index, format='%d/%m/%Y')

    ## Load Spot Data

    spt = gds.get_dataframe_from_uid(18405, cols='PI')
    S = spt.reindex(df.index)
    S.columns = ['spot']
    df = pd.concat((df, S), axis=1)
    df['relative_strike'] = np.log(df.get('absoluteStrike') / df.get('spot'))


    ## Load Risk Free Rate Data

    tickers = ['ECUSD1W','ECUSD1M','ECUSD3M','ECUSD6M','ECUSD1Y']

    rfrs = pd.DataFrame()
    for ticker in tickers:
        uid = session.query(TimeSeriesSpec.uid).filter(TimeSeriesSpec.ticker == ticker).scalar()
        mat = session.query(InterestRateSpec.maturity).filter(InterestRateSpec.uid == uid).scalar()
        rfr = gds.get_dataframe_from_tickers(ticker, cols='IR')
        rfr = rfr.iloc[:,0]
        rfr = rfr.to_frame(DateUtils.Rdate_to_mat(mat))
        rfrs = pd.concat((rfrs, rfr), axis=1)

    rfrs[9/12] = np.nan
    rfrs = rfrs.get(rfrs.columns.sort_values())
    rfrs = rfrs.interpolate(axis=1) / 100
    rfrs.columns = ['1w', '1m', '3m', '6m', '9m', '1y']
    rfrs = rfrs.reindex(df.index.unique()).ffill()
    rfrs = rfrs.unstack().reset_index(drop=False)
    rfrs.columns = ['tenor', 'date', 'rate']

    rfrs_ = rfrs.set_index(['date', 'tenor'])

    ##

    df_ = df.reset_index(drop=False).set_index(['date', 'tenor'])
    df_['rate'] = rfrs_.loc[df_.index].values
    df_['ttm'] = DateUtils.Rdate_to_mat(df_.index.get_level_values('tenor'))
    df_ = df_.reset_index(drop=False)
    return df_
