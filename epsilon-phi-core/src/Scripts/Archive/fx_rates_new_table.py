from epsilonPhi.core.dataModel.alchemist.SessionManager import *
import numpy as np
from sqlalchemy import distinct

nan = pd.pandas._libs.tslibs.nattype.NaTType
session = SessionMgr().getSessionFactory()

uid = 4234
q_old = session.query(FXRate).filter(FXRate.uid.in_([uid]))
d_old = SessionMgr().query_format_df(q_old)

q_new = session.query(FXRateNew).filter(FXRateNew.uid.in_([uid]))
d_new = SessionMgr().query_format_df(q_new)

#%%

OLD_TABLE_UIDS = session.query(distinct(FXRate.uid)).all()
NEW_TABLE_UIDS = session.query(distinct(FXRateNew.uid)).all()

missing_uids = [ int(x) for x in np.setdiff1d(OLD_TABLE_UIDS, NEW_TABLE_UIDS) ]
q = session.query(TimeSeriesSpec).filter(TimeSeriesSpec.uid.in_(missing_uids))
missing_info = SessionMgr().query_format_df(q)

ctr = 0
N = len(missing_uids)
data_table_name = 'fx_rates_new'
for uid in missing_uids:
    q = session.query(FXRate).filter(FXRate.uid.in_([uid]))
    dta = SessionMgr().query_format_df(q)
    dta.to_sql(name=data_table_name, con=SessionMgr().getEngine(), if_exists='append', index=False)
    print('{} added to database table'.format(uid))
    ctr = ctr + 1

data_table_name = 'fx_rates_new'

