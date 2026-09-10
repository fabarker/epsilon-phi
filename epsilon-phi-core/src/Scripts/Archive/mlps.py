from epsilonPhi.core.dataModel.dataSources.vendor.Bloomberg import Bloomberg
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.dataModel.alchemist.DataModel import *
from sqlalchemy import distinct
import pandas as pd
import os


nan = pd.pandas._libs.tslibs.nattype.NaTType
session = SessionMgr().getSessionFactory()

_SHEETNAME = 'Sheet1'
df = pd.read_excel(r'/Users/francisbarker/Desktop/OIS.xlsx',
                   'OIS', header=[0])

df.to_sql(name='interest_rate',
          con=SessionMgr().getEngine(),
          if_exists='append',
          index=False)
