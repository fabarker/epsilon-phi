import datetime
import os, sys
import pandas as pd
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.dataModel.dataSources.vendor.GSQuant import GSQuantManager, Dataset, Datasets
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.utils.ListUtils import ListUtils
from sqlalchemy import distinct, func

sessionMgr = SessionMgr()
fac = sessionMgr.getSessionFactory()
gsq = GSQuantManager()
gds = GlobalDataSource()

ds = Dataset('FXIVOL_V2_PREMIUM')

##

def get_uids(tickers):
    res = fac.query(TimeSeriesSpec.uid, TimeSeriesSpec.symbol).filter(TimeSeriesSpec.symbol.in_(list(tickers))).all()
    return pd.DataFrame(res).set_index('symbol', drop=True)

def get_rows(tickers):
    results = pd.DataFrame()
    uids = get_uids(tickers)
    for uid in list(uids.values):
        print(uid)
        query = fac.query(ImpliedVolatilityNew).filter_by(uid=int(uid[0])).first()
        tmp = pd.DataFrame(query.__dict__.values(), index=query.__dict__.keys(), columns=[uid]).T
        results = pd.concat((results, tmp), axis=1)
    return results

def _get_latest_observation_date_from_tickers(tickers):
    res = fac.query(TimeSeriesSpec.symbol,
                        func.max(ImpliedVolatilityNew.date)).join(ImpliedVolatilityNew,
                                                         TimeSeriesSpec.uid == ImpliedVolatilityNew.uid). \
        filter(TimeSeriesSpec.symbol.in_(list(tickers))). \
        group_by(TimeSeriesSpec.symbol).all()
    return pd.DataFrame(res, columns=['symbol', 'start']).set_index('symbol')

def get_gs_fxivol_specs():
    sql = "SELECT * FROM `epsilon-phi-dev`.time_series_spec WHERE provider = 'GS' AND category = 'Implied Volatility' AND ticker NOT LIKE '%:%';"
    return pd.read_sql(sql, con=fac.get_bind()).set_index('uid')

def get_current_data(uid):
    info = fac.query(ImpliedVolatilityNew).filter(ImpliedVolatilityNew.uid == uid)
    return sessionMgr.query_format_df(info)

spec = get_gs_fxivol_specs()
_list = ListUtils._nest_list(list(spec.ticker), 500)
uids = get_uids(list(spec.ticker))

df_ = pd.DataFrame()
for l in _list:
    _last_date = _get_latest_observation_date_from_tickers(l)
    res = ds.get_data(start=pd.to_datetime(_last_date.min().values[0]).date(), end=datetime.date.today(),
                      assetId=l, pricing_location=['LDN', 'NYC'])
    for ticker in l:
        print(ticker)
        tmp = res[res.assetId == ticker]
        query = fac.query(ImpliedVolatilityNew).filter_by(uid=int(uids.loc[ticker].values[0])).first()
        trimmed = tmp[tmp.index > pd.to_datetime(_last_date.loc[ticker].values[0])][['pricingLocation', 'impliedVolatility']]

        frame = trimmed.rename(columns={'pricingLocation': 'pricing_location', 'impliedVolatility': 'mid'})
        frame['uid'] = query.uid
        frame['strike_reference'] = query.strike_reference
        frame['relative_strike'] = query.relative_strike
        frame['tenor'] = query.tenor
        frame['security'] = query.security
        df_ = pd.concat((df_, frame.reset_index()), axis=1).reset_index(drop=True)


df_.to_sql(name='implied_volatility_new',
            con=SessionMgr().getEngine(),
            if_exists='append',
            index=False)
print('Data appended to table impliedVolatilityNew')





