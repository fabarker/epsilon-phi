from epsilonPhi.core.dataModel.dataSources.vendor.Bloomberg import Bloomberg
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.dataModel.alchemist.DataModel import *
from sqlalchemy import distinct
import pandas as pd
import os


nan = pd.pandas._libs.tslibs.nattype.NaTType
session = SessionMgr().getSessionFactory()

_DATA_PATH = r'C:\Users\fabar\OneDrive\Desktop\data\factors'
_FILE_PATH = os.path.join(_DATA_PATH, 'AQR and FF Factors.xlsx')
_SHEETNAME = 'FACTORS'

df = pd.read_excel(_FILE_PATH, _SHEETNAME, index_col=0, header=[0,1,2,3,4,5,6,7,8])
for col in df.columns:
    col_ = df.get([col]).dropna()
    ticker = col_.columns.get_level_values('ticker')[0]

    if not Bloomberg.is_ticker_in_database(ticker):

        try:
            spec = FactorSpec()
            spec.category = col_.columns.get_level_values('category')[0]
            spec.datasource = col_.columns.get_level_values('datasource')[0]
            spec.provider = col_.columns.get_level_values('provider')[0]
            spec.factor = col_.columns.get_level_values('factor')[0]
            spec.name = col_.columns.get_level_values('name')[0]
            spec.region = col_.columns.get_level_values('region')[0]
            spec.symbol = col_.columns.get_level_values('ticker')[0]
            spec.ticker = col_.columns.get_level_values('ticker')[0]
            spec.universe = col_.columns.get_level_values('universe')[0]
            spec.uid = Bloomberg.get_max_uid() + 1
            session.add_all([spec])
            session.commit()
        except:
            session.rollback()
            print('Error - could not add time series spec info for ticker {}'.format(ticker))
        finally:
            session.close()

        uid = Bloomberg.get_uid_from_ticker(ticker)
        if uid:
            dt_df = col_.copy()
            dt_df.columns = ['XR']
            dt_df = dt_df.reset_index(drop=False)
            dt_df.columns = ['date', 'XR']
            dt_df['uid'] = uid
            dt_df = dt_df.drop_duplicates(subset='date', keep='first')

            dt_df.to_sql(name='factor',
                         con=SessionMgr().getEngine(),
                         if_exists='append',
                         index=False)
            print('Data added for {}'.format(ticker))

