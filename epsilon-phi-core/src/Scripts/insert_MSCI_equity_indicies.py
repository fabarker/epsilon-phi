from epsilonPhi.core.dataModel.dataSources.vendor.Bloomberg import Bloomberg
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.dataModel.alchemist.DataModel import *
from sqlalchemy import distinct
import pandas as pd
import os


nan = pd.pandas._libs.tslibs.nattype.NaTType
session = SessionMgr().getSessionFactory()

_SHEETNAME = 'Sheet1'
df = pd.read_excel(r'/Users/francisbarker/Desktop/EM EQUITY.xlsx',
                   _SHEETNAME, index_col=0, header=[0,1,2,3,4,5,6,7,8,9,10])

unique_tickers = np.unique(df.columns.get_level_values(0))

for ticker in unique_tickers:

        col_ = df.get(ticker).dropna(how='all')

        if not Bloomberg.is_ticker_in_database(ticker):

            try:
                spec = EquityIndexSpec()
                spec.category = col_.columns.get_level_values('category')[0]
                spec.datasource = col_.columns.get_level_values('datasource')[0]
                spec.denominated_currency = col_.columns.get_level_values('denominated_currency')[0]
                spec.exposure_currency = col_.columns.get_level_values('exposure_currency')[0]
                spec.hedge_ratio = int(col_.columns.get_level_values('hedge_ratio')[0])
                spec.name = col_.columns.get_level_values('name')[0]
                spec.provider = col_.columns.get_level_values('provider')[0]
                spec.region = col_.columns.get_level_values('region')[0]
                spec.symbol = col_.columns.get_level_values('symbol')[0]
                spec.ticker = ticker
                spec.uid = int(Bloomberg.get_max_uid() + 1)

                session.add_all([spec])
                session.commit()
            except:
                session.rollback()
                print('Error - could not add time series spec info for ticker {}'.format(ticker))
            finally:
                session.close()

        try:
            uid = Bloomberg.get_uid_from_ticker(ticker)
            db_dta = pd.read_sql('SELECT * FROM equity_index WHERE uid = "' + str(uid) + '";',
                                 con=SessionMgr().getEngine())

            # prepare excel data
            dt_df = col_.copy().dropna(how='all')
            dt_df.index.name = 'date'
            cols = col_.columns.get_level_values('datatype')
            cols.name = None
            dt_df.columns = cols
            dt_df = dt_df.reset_index(drop=False)
            dt_df['uid'] = uid
            dt_df = dt_df.drop_duplicates(subset='date', keep='first')

            if db_dta.size > 0:
                db_dates = pd.to_datetime(db_dta.get('date').values)
                xl_dates = pd.to_datetime(dt_df.get('date').values)
                missing = np.setdiff1d(xl_dates, db_dates)
            else:
                missing = pd.to_datetime(dt_df.get('date').values)

            if len(missing) > 0:
                for_db = dt_df.set_index('date', drop=True).reindex(missing)
                for_db = for_db[['uid','RI']]
                for_db.index.name = 'date'
                for_db = for_db.reset_index(drop=False)
                for_db.to_sql(name='equity_index',
                              con=SessionMgr().getEngine(),
                              if_exists='append',
                              index=False)
                print('Data added for {}'.format(ticker))
            else:
                print('No data to add for ticker {}'.format(ticker))
        except:
            print('Error - could not add time series data info for ticker {}'.format(ticker))

