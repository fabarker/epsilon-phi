import datetime
import os, sys
import pandas as pd
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.dataModel.dataSources.vendor.GSQuant import GSQuantManager, Dataset, Datasets
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource

sessionMgr = SessionMgr()
fac = sessionMgr.getSessionFactory()
gsq = GSQuantManager()
gds = GlobalDataSource()

ds = Dataset('FXIVOL_V2_PREMIUM')

##

def get_gs_fxivol_specs():
    sql = "SELECT * FROM `epsilon-phi-dev`.time_series_spec WHERE provider = 'GS' AND category = 'Implied Volatility' AND ticker NOT LIKE '%:%';"
    return pd.read_sql(sql, con=fac.get_bind()).set_index('uid')


def get_current_data(uid):
    info = fac.query(ImpliedVolatilityNew).filter(ImpliedVolatilityNew.uid == uid)
    return sessionMgr.query_format_df(info)


spec = get_gs_fxivol_specs()

ctr = 0
for id, row in spec.iterrows():
    print(spec.shape[0] - ctr)
    ctr += 1

    symbol = row.ticker

    last_date = sessionMgr.get_latest_observation_date_from_tickers(symbol)
    if last_date < datetime.datetime.today():
        from_date = sessionMgr.get_latest_observation_date_from_tickers(symbol)
        to_date = datetime.datetime.today()

        res = ds.get_data(start=from_date.date(), end=to_date.date(),
                          assetId=symbol, pricing_location=['LDN', 'NYC'])
        res.index = pd.to_datetime(res.index)

        df_ = gds.get_dataframe_from_uid(id).get(id)
        df_.index = pd.to_datetime(df_.index)
        idx = df_.reset_index().set_index(['date','pricing_location']).index

        res_ = res.reset_index().set_index(['date', 'pricingLocation'])
        keep_rows = res_.index.difference(idx)

        db_data = res_.loc[keep_rows].reset_index()
        unduped = db_data.drop_duplicates(subset=['date', 'pricingLocation'])

        for_db = unduped[['date', 'pricingLocation', 'impliedVolatility']]
        frame = for_db.rename(columns={'pricingLocation': 'pricing_location', 'impliedVolatility': 'mid'})
        frame['uid'] = id
        frame['strike_reference'] = df_.strike_reference.unique()[0]
        frame['relative_strike'] = df_.relative_strike.unique()[0]
        frame['tenor'] = df_.tenor.unique()[0]
        frame['security'] = df_.security.unique()[0]

        if frame.size > 0:
            frame.to_sql(name='implied_volatility_new',
                           con=SessionMgr().getEngine(),
                           if_exists='append',
                           index=False)
            print('Data appended to table ' + 'fx_rates' + ' for time series with ticker {}'.format(
                symbol))





