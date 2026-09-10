from epsilonPhi.core.dataModel.alchemist.SessionManager import *
from sqlalchemy import create_engine, exists, distinct
import pandas as pd

sessionMgr = SessionMgr()
session = sessionMgr.getSessionFactory()

db = pd.read_excel('/Users/francisbarker/Desktop/FX Vol Checks.xlsx', sheet_name=None)
df_db = db.get('Database').set_index('uid')
df_gs = db.get('Goldman').set_index('assetID')

_unique_securities = np.unique([x.split(' ')[0] for x in df_db.get('name').values])

for _security in _unique_securities:
    q = session.query(ImpliedVolatility).filter(ImpliedVolatility.security ==
                                                _security)

    df = sessionMgr.query_format_df(q)
    df = df.set_index('uid')

    # build new table
    df_new = pd.DataFrame()

    unique_uids = df.index.unique()
    for uid in unique_uids:

        subset_uid_ = df.loc[uid].copy()
        _sr = subset_uid_.get('strike_reference').unique().item()
        _rs = subset_uid_.get('relative_strike').unique().item()
        _tn = subset_uid_.get('tenor').unique().item()
        _cy = subset_uid_.get('security').unique().item()

        _info = df_db.loc[uid]
        ccy, tenor, delta, opttype, _, _ = _info.get('name').split(' ')
        _name = _info.get('name')

        if delta == 'Spot':
           str_ref = delta.lower()
           rel_str = '100'
        elif delta == 'ATMF':
           str_ref = 'forward'
           rel_str = '100'
        elif delta == 'DN':
           str_ref = 'delta'
           rel_str = 'DN'
        else:
           str_ref = 'delta'
           if opttype == 'Put':
               rel_str = '-' + delta.replace('D', '')
           elif opttype == 'Call':
               rel_str = delta.replace('D', '')

        if _sr != str_ref:
            print('Strike Reference for {} incorrect, reads {} but should be {}'.format(_name, _sr, str_ref))
            subset_uid_['strike_reference'] = str_ref


        if _rs != rel_str:
            print('Relative Strike for {} incorrect, reads {} but should be {}'.format(_name, _rs, rel_str))
            subset_uid_['relative_strike'] = rel_str

        if _tn != tenor:
            print('Tenor for {} incorrect, reads {} but should be {}'.format(_name, _tn, tenor))
            subset_uid_['tenor'] = tenor

        if _cy != ccy:
            print('Currency for {} incorrect, reads {} but should be {}'.format(_name, _cy, ccy))
            subset_uid_['security'] = ccy

        df_new = pd.concat((df_new, subset_uid_), axis=0)

    assert df_new.shape == df.shape
    df_new = df_new.reset_index(drop=False)
    try:
        df_new.to_sql(name='implied_volatility_new',
                      con=sessionMgr.getEngine(),
                      if_exists='append',
                      index=False,
                      chunksize=10000)
    except:
        session.rollback()
        print('Error - could not add time series spec info for ticker {}'.format(_security))
    finally:
        session.close()



