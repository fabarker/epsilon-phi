import pandas as pd
import numpy as np
import os, sys
from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.dataModel.dataSources.Bloomberg import Bloomberg

session = SessionMgr().getSessionFactory()

# Load data and info from csv file
path = '/Users/francisbarker/Library/Mobile Documents/com~apple~CloudDocs/Data/Datastream/FX/Delta One'


info = pd.read_csv(os.path.join(path, 'Refinitiv Info.csv'), index_col='Symbol')
data = pd.read_csv(os.path.join(path, 'Refinitiv Rates.csv'))

keep_cols = np.array(['Unnamed' not in x for x in data.columns])
data = data.iloc[:, keep_cols]
data = data.set_index('Code', drop=True)
data.index = pd.to_datetime(data.index)

unique_tickers = info.index.unique()

#for ticker in unique_tickers:
#    if Bloomberg.is_ticker_in_database(ticker):
#        uid = Bloomberg.get_uid_from_ticker(ticker)
#        session.query(FXRate).filter(FXRate.uid == uid).delete()
#        session.commit()


for ticker in unique_tickers:
    info_t = info.loc[ticker]

    cols = data.iloc[:, np.array([ticker in x for x in data.columns])]
    if cols.size > 0:

        if not Bloomberg.is_ticker_in_database(ticker):
            try:
                fx_spec = FXRateSpec()
                fx_spec.bbid = info_t.ISO
                fx_spec.category = 'FX'
                fx_spec.domestic_currency = info_t.Domestic
                fx_spec.foreign_currency = info_t.Foreign
                fx_spec.maturity = info_t.Maturity
                fx_spec.name = info_t['Full Name']
                fx_spec.provider = info_t.Source
                fx_spec.region = info_t.get('Region Name')
                fx_spec.ticker = ticker
                fx_spec.datasource = 'Datastream'
                fx_spec.uid = Bloomberg.get_max_uid() + 1

                session.add_all([fx_spec])
                session.commit()
            except:
                session.rollback()
                print('Error - could not add time series spec info for ticker {}'.format(ticker))
            finally:
                session.close()

        df = cols.copy().dropna(how='all')
        df.index.name = 'date'
        df.index = pd.to_datetime(df.index)

        df = df.rename(columns={ticker + '(ER)': 'mid'})
        df = df.rename(columns={ticker + '(EB)': 'bid'})
        df = df.rename(columns={ticker + '(EO)': 'ask'})

        # add the mid
        if df.shape[1] == 2 and np.all([x in ['bid', 'ask'] for x in df.columns]):
            df['mid'] = 0.5 * (df['bid'] + df['ask'])

        # add the bid
        if df.shape[1] == 2 and np.all([x in ['mid', 'ask'] for x in df.columns]):
            df['bid'] = 2 * df['mid'] - df['ask']

        # add the ask
        if df.shape[1] == 2 and np.all([x in ['mid', 'bid'] for x in df.columns]):
            df['ask'] = 2 * df['mid'] - df['bid']

        if np.isin('mid', df.columns):
            df['last'] = df['mid']

        assert np.all(np.isin(df.columns, ['bid', 'mid', 'ask', 'last'])), 'Error in column name'

        df['uid'] = Bloomberg.get_uid_from_ticker(ticker)
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

print('No data for add for time series with ticker {}'.format(ticker))





