import pandas as pd
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.dataModel.alchemist.DataModel import *

mgr = SessionMgr()
session = mgr.getSessionFactory()

info = pd.read_excel('/Users/francisbarker/Desktop/updates.xlsx', sheet_name='updates', index_col=0)

for uid, row in info.iterrows():

    db_spec = session.query(FXRateSpec).filter(FXRateSpec.uid == uid).scalar()
    db_spec.bbid = row.to
    db_spec.foreign_currency = row.to[:3]
    db_spec.domestic_currency = row.to[3:]
    db_spec.name = row.to + ' ' + ' '.join(db_spec.name.split()[1:])
    session.commit()

session.close()