from epsilonPhi.core.dataModel.dataSources.vendor.Bloomberg import Bloomberg
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.dataModel.alchemist.DataModel import *
from sqlalchemy import distinct
import pandas as pd
import os


nan = pd.pandas._libs.tslibs.nattype.NaTType
session = SessionMgr().getSessionFactory()

_DATA_PATH = os.path.join(os.environ.get('HOMEDRIVE'), os.environ.get('HOMEPATH'), 'Documents', 'Data')
_FOLDER_PATH = os.path.join(_DATA_PATH, 'gsquant', 'equity')

xl_spec = pd.read_excel(os.path.join(_FOLDER_PATH, 'Spec.xlsx'), sheet_name='Spec')
dbase_tickers = [ x[0] for x in session.query(distinct(TimeSeriesSpec.ticker)).all() ]

for _, row in xl_spec.iterrows():

    if row.ticker not in dbase_tickers:

        data_path = os.path.join(_FOLDER_PATH, row.Name, row.save_file)
        if os.path.exists(data_path + '.csv'):
            dt_df = pd.read_csv(data_path + '.csv')
            dt_df['pricing_location'] = row.pricing_location
            dt_df['security'] = row.security
            dt_df = dt_df.rename(columns={'impliedVolatility':'mid',
                                          'strikeReference':'strike_reference',
                                          'relativeStrike':'relative_strike'})

            try:
                spec = TimeSeriesSpec()
                spec.category = 'Implied Volatility'
                spec.datasource = 'GS'
                spec.provider = 'GS'
                spec.name = row.long_name
                spec.region = row.region
                spec.symbol = row.symbol
                spec.ticker = row.ticker
                spec.uid = Bloomberg.get_max_uid() + 1
                session.add_all([spec])
                session.commit()
            except:
                session.rollback()
                print('Error - could not add time series spec info for ticker {}'.format(row.ticker))
            finally:
                session.close()

            uid = Bloomberg.get_uid_from_ticker(row.ticker)
            if uid:
                dt_df['uid'] = uid
                dt_df = dt_df[['uid','date','pricing_location','strike_reference','relative_strike','tenor','mid','security']]
                dt_df['relative_strike'] = dt_df['relative_strike'] * 100
                dt_df = dt_df.drop_duplicates(subset='date', keep='first')
                dt_df.to_sql(name='implied_volatility',
                                  con=SessionMgr().getEngine(),
                                  if_exists='append',
                                  index=False)
                print('Data added for {}'.format(row.ticker))









