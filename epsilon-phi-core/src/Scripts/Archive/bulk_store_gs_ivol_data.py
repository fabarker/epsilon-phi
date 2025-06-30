from epsilonPhi.core.dataModel.dataSources.vendor.Bloomberg import Bloomberg
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.dataModel.alchemist.DataModel import *
from sqlalchemy import distinct
import pandas as pd
import os


nan = pd.pandas._libs.tslibs.nattype.NaTType
session = SessionMgr().getSessionFactory()

df_ = pd.read_csv('/Users/francisbarker/Desktop/SPX Data/SPX Vols by Moneyness.csv')
df = df_.set_index('uid', drop=True)
unique_uids = df.index.unique()

for uid in unique_uids:
    _subset = df.loc[uid]

    _ts = _subset[['date', 'pricing_location', 'strike_reference', 'relative_strike', 'tenor', 'mid', 'security']]
    _info = _subset[['ticker', 'name', 'region', 'provider', 'datasource', 'symbol', 'frequency']].drop_duplicates()
    _info['category'] = 'Implied Volatility'

    if not Bloomberg.is_ticker_in_database(_info.ticker.values[0]):
        _spec = _info.reset_index(drop=False).T.to_dict().get(0)

        try:
            spec = TimeSeriesSpec()
            spec.category = _spec.get('category')
            spec.datasource = 'GS'
            spec.provider = 'GS'
            spec.name = _spec.get('name')
            spec.region = _spec.get('region')
            spec.symbol = _spec.get('symbol')
            spec.ticker = _spec.get('ticker')
            spec.uid = _spec.get('uid')
            spec.frequency = _spec.get('frequency')
            session.add_all([spec])
            session.commit()
        except:
            session.rollback()
            print('Error - could not add time series spec info for ticker {}'.format(_info.ticker.values[0]))
        finally:
            session.close()

    if Bloomberg.is_ticker_in_database(_info.ticker.values[0]):

        uid = Bloomberg.get_uid_from_ticker(_info.ticker.values[0])
        dt_df = _ts.reset_index(drop=False)
        if uid:
            dt_df['uid'] = uid
            dt_df = dt_df[['uid','date','pricing_location','strike_reference','relative_strike','tenor','mid','security']]
            dt_df['relative_strike'] = dt_df['relative_strike'] * 100
            dt_df['pricing_location'] = 'CBOE'
            dt_df.date = pd.to_datetime(dt_df.date, format='%d/%m/%Y')

            dt_df = dt_df.drop_duplicates(subset='date', keep='first')
            try:
                dt_df.to_sql(name='implied_volatility',
                                      con=SessionMgr().getEngine(),
                                      if_exists='append',
                                      index=False)
                print('Data added for {}'.format(_info.ticker.values[0]))
            except:
                print('Error - could not add time series data info for ticker {}'.format(_info.ticker.values[0]))











