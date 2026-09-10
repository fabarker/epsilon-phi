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

folder_name = 'gsquant'
info_workbook_name = 'IVol Info.xlsx'
info_sheetname = 'Sheet3'

df_info = pd.read_excel(os.path.join(_DATA_PATH, folder_name, info_workbook_name), sheet_name=info_sheetname)
df_info = df_info.set_index('ticker', drop=True)

sub_folders = os.listdir(os.path.join(_DATA_PATH, folder_name))
sub_folders = [ x for x in sub_folders if 'xlsx' not in x ]

for sub_folder in sub_folders:
    data_folder = os.path.join(_DATA_PATH, folder_name, sub_folder)
    data_files = os.listdir(data_folder)

    for file in data_files:
        dta_path = os.path.join(data_folder, file)

        if os.path.isfile(dta_path):

            df = pd.read_csv(dta_path, index_col=0)
            df_info_series = df_info[df_info.save_file == file.replace('.csv','')]

            if df.size > 0 and df_info.size > 0:

                df.index = pd.to_datetime(df.index)
                ticker = df_info_series.index[0]

                if not Bloomberg.is_ticker_in_database(df_info_series.index[0]):
                    try:
                        spec = ImpliedVolatilitySpec()
                        spec.asset_class = df_info_series.get('asset_class').values[0]
                        spec.category = df_info_series.get('category').values[0]
                        spec.datasource = df_info_series.get('datasource').values[0]
                        spec.name = df_info_series.get('long_name').values[0]
                        spec.provider = df_info_series.get('provider').values[0]
                        spec.region = df_info_series.get('region').values[0]
                        spec.strike_reference = df_info_series.get('relative_strike').values[0]
                        spec.relative_strike = float(df_info_series.get('strike_reference').values[0])
                        spec.tenor = df_info_series.get('tenor').values[0]
                        spec.ticker = df_info_series.index[0]
                        spec.underlier = df_info_series.get('Name').values[0]

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

                    df_prime = pd.read_sql('SELECT date FROM ' + data_table_name + ' where uid ="' +
                                           str(Bloomberg.get_uid_from_ticker(ticker)) + '"',
                                           SessionMgr().getEngine())

                    sqldates = np.setdiff1d(pd.to_datetime(df['date'].values), pd.to_datetime(df_prime['date']))
                    if len(sqldates) > 0:
                        df_sql = df[df['date'].isin(sqldates)]

                        df_sql = df_sql.dropna(how='all', axis=1)
                        if df_sql.size > 0:

                            df_sql = df_sql[['uid', 'date', 'strikeReference', 'relativeStrike', 'tenor', 'impliedVolatility']]
                            df_sql.columns = ['uid', 'date', 'strike_reference', 'relative_strike', 'tenor', 'mid']
                            df_sql['relative_strike'] = df_sql['relative_strike'] * 100
                            df_sql['underlier'] = df_info_series.get('Name').values[0]
                            df_sql['asset_class'] = df_info_series.get('asset_class').values[0]
                            df_sql = df_sql.drop_duplicates(subset='date', keep='first')


                            df_sql.to_sql(name=data_table_name,
                                          con=SessionMgr().getEngine(),
                                          if_exists='append',
                                          index=False)
                            print('Data appended to table ' + data_table_name + ' for time series with ticker {}'.format(
                                ticker))
                    else:
                        print('No data for add for time hedge_fund_index with ticker {}'.format(ticker))





