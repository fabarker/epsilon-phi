import pandas as pd
import numpy as np
from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr

session = SessionMgr().getSessionFactory()

path = '/Users/francisbarker/Library/Mobile Documents/com~apple~CloudDocs/Data/Datastream/FX/Delta One/DS FX Series.xlsm'
info = pd.read_excel(path, sheet_name='FX Info', index_col='Symbol')
info = info.drop_duplicates()

in_db = session.query(FXRateSpec.ticker).filter(FXRateSpec.provider != 'BBG').all()
in_db = [x[0] for x in in_db]

#
missing = np.setdiff1d(info.index.values.flatten(), in_db)
df = info.loc[missing]
#
# #df = pd.DataFrame()
# #for ccy in np.unique(currencies):
# #    df = pd.concat((df, raw[raw.ISO.apply(lambda x: ccy in x and 'USD' in x)]), axis=0)
# sources = ['Refinitiv', 'Barclays Bank PLC', 'WM/Refinitiv']
# subset_sources = df[df.Source.apply(lambda x: x in sources)]
# mats = ['0m', '1m', '2m', '3m', '6m', '1y', '1w', '9m', '2w', '3w','12m']
# subset_mats = subset_sources[subset_sources.Maturity.apply(lambda x: x in mats)]
#
# tickers = subset_mats.index
# df = pyDatastream.fetch(tickers, fields=['EO','EB'], from_date=dt.date(day=31, month=12, year=1950), frequency='D')
#
# df_idx = df.set_index('level_0', drop=True).dropna()
# df_idx.columns = ['Date','EO','EB']
# df_idx['ER'] = ( df_idx['EO'] + df_idx['EB'] ) / 2
# df_idx.index.name = 'ticker'
#
# unique_tickers = np.unique(df_idx.index)
# dta = pd.DataFrame()
# for ticker in unique_tickers:
#     sub = df_idx.loc[ticker].set_index('Date', drop=True)
#     sub.columns = pd.MultiIndex.from_tuples([(ticker, x) for x in sub.columns])
#     dta = pd.concat((dta, sub), axis=1)

#%%

import os
path = '/Users/francisbarker/Library/Mobile Documents/com~apple~CloudDocs/Data/Datastream/FX/Delta One/'
workbooks = os.listdir(path)
dta = {}
for wkbk in workbooks:
    if '.csv' in wkbk:
        #df_new = pd.read_csv(os.path.join(path, wkbk), sheet_name=wkbk.replace('.csv', ''), header=[0, 1], index_col=0)
        df_new = pd.read_csv(os.path.join(path, wkbk), header=[0, 1], index_col=0)
        dta[wkbk] = df_new.copy()

info = pd.read_excel(os.path.join(path, 'DS FX Series.xlsm'), sheet_name='FX Info', index_col='Symbol')
info = info.drop_duplicates()
df_all = pd.DataFrame()
for key in dta.keys():
    df_all = pd.concat((df_all, dta.get(key)), axis=1)

tickers = np.unique(df_all.columns.get_level_values(0))
for ticker in tickers:

        df_ticker = df_all[[ticker]].dropna(how='all')
        df_ticker.columns = df_ticker.columns.get_level_values(1)
        df_ticker = df_ticker.rename(columns={'EO': 'ask', 'EB': 'bid', 'ER': 'mid'}).reset_index(drop=False)
        info_ticker = info.loc[ticker]

        if not Bloomberg.is_ticker_in_database(ticker):
            try:
                fx_spec = FXRateSpec()
                fx_spec.bbid = info_ticker.ISO
                fx_spec.category = 'FX'
                fx_spec.domestic_currency = info_ticker.Domestic
                fx_spec.foreign_currency = info_ticker.Foreign
                fx_spec.maturity = info_ticker.Maturity
                fx_spec.name = info_ticker['Full Name']
                fx_spec.provider = info_ticker.Source
                fx_spec.region = info_ticker.get('Region Name')
                fx_spec.ticker = ticker
                fx_spec.uid = Bloomberg.get_max_uid() + 1

                session.add_all([fx_spec])
                session.commit()
            except:
                session.rollback()
                print('Error - could not add time series spec info for ticker {}'.format(ticker))
            finally:
                session.close()

        df = df_ticker.copy().dropna(how='all').set_index('Date', drop=True)
        df.index.name = 'date'
        df.index = pd.to_datetime(df.index)
        df['uid'] = Bloomberg.get_uid_from_ticker(ticker)
        df['last'] = df['mid']
        df = df.reset_index(drop=False)


        df_prime = pd.read_sql('SELECT date FROM fx_rates where uid ="' +
                               str(Bloomberg.get_uid_from_ticker(ticker)) + '"',
                               SessionMgr().getEngine())

        df_sql = df[~df['date'].isin(df_prime['date'])]
        if df_sql.size > 0:
            df_sql.to_sql(name='fx_rates',
                          con=SessionMgr().getEngine(),
                          if_exists='append',
                          index=False)
            print('Data appended to table interest_rates for time series with ticker {}'.format(ticker))
        else:
            print('No data for add for time series with ticker {}'.format(ticker))


df_all.to_clipboard()

