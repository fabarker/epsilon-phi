from epsilonPhi.core.dataModel.dataSources.vendor.Bloomberg import Bloomberg
import os
from epsilonPhi.core.dataModel.alchemist.SessionManager import *
import numpy as np

nan = pd.pandas._libs.tslibs.nattype.NaTType
session = SessionMgr().getSessionFactory()

_DATA_PATH = os.path.join(os.environ.get('HOMEDRIVE'), os.environ.get('HOMEPATH'), 'Documents', 'Data')

session = SessionMgr().getSessionFactory()

#%%

data_table_name = 'implied_volatility'
spec_table_name = 'implied_volatility_spec'

folder_name = r'gsquant\fxivol'
info_workbook_name = 'fxivol.xlsx'
info_sheetname = 'FX Info'

df_info = pd.read_excel(os.path.join(_DATA_PATH, folder_name, info_workbook_name), sheet_name=info_sheetname)
df_info = df_info.set_index('datasource_ticker', drop=True)

for ticker, row in df_info.iterrows():
    dta_path = os.path.join(_DATA_PATH, folder_name, row.pricing_location, ticker + '.csv')
    if os.path.isfile(dta_path):

        df = pd.read_csv(dta_path, index_col=0)
        if df.size > 0:

            df.index = pd.to_datetime(df.index)
            if not Bloomberg.is_ticker_in_database(ticker):
                try:
                    spec = TimeSeriesSpec()
                    spec.category = row.category
                    spec.datasource = row.datasource
                    spec.name = row.long_name
                    spec.provider = row.provider
                    spec.region = row.region
                    spec.ticker = ticker

                    spec.uid = Bloomberg.get_max_uid() + 1
                    session.add_all([spec])
                    session.commit()
                except:
                    session.rollback()
                    print('Error - could not add time series spec info for ticker {}'.format(ticker))
                finally:
                    session.close()

            uid = Bloomberg.get_uid_from_ticker(ticker)
            if uid:
                df['uid'] = uid
                df = df.reset_index(drop=False)
                df['date'] = pd.to_datetime(df['date'].values)

                df_prime = pd.read_sql('SELECT date FROM ' + data_table_name + ' where uid ="' + str(Bloomberg.get_uid_from_ticker(ticker)) + '" AND pricing_location = "' + row.pricing_location + '"',
                                           SessionMgr().getEngine())

                sqldates = np.setdiff1d(pd.to_datetime(df['date'].values), pd.to_datetime(df_prime['date']))
                if len(sqldates) > 0:
                    df_sql = df[df['date'].isin(sqldates)]

                    df_sql = df_sql.dropna(how='all', axis=1)
                    if df_sql.size > 0:

                        df_sql = df_sql[['uid', 'date','impliedVolatility']]
                        df_sql.columns = ['uid', 'date','mid']
                        df_sql['relative_strike'] = row.relative_strike
                        df_sql['strike_reference'] = row.strike_reference

                        df_sql['underlier_ticker'] = row.underlier_name
                        df_sql['pricing_location'] = row.pricing_location
                        df_sql['tenor'] = row.tenor
                        df_sql = df_sql.drop_duplicates(subset='date', keep='first')

                        df_sql.to_sql(name=data_table_name,
                                      con=SessionMgr().getEngine(),
                                      if_exists='append',
                                      index=False)
                        print('Data appended to table ' + data_table_name + ' for time series with ticker {}'.format(
                                ticker))
                else:
                    print('No data for add for time hedge_fund_index with ticker {}'.format(ticker))





