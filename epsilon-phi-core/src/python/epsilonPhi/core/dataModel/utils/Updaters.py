import pandas as pd

from epsilonPhi.core.dataModel.dataSources.Bloomberg import Bloomberg, ISO_to_region
from epsilonPhi.core.utils.ExcelUtils import ExcelUtils
import os
from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.dataModel.alchemist.SessionManager import *
from datetime import datetime
import numpy as np
from dateutil import parser

nan = pd.pandas._libs.tslibs.nattype.NaTType
session = SessionMgr().getSessionFactory()

_DATA_PATH = os.path.join(os.environ.get('HOMEDRIVE'), os.environ.get('HOMEPATH'), 'Documents', 'Data')


class implied_volatility(Bloomberg):

    _BVOLS_PATH = '/Users/francisbarker/Library/Mobile Documents/com~apple~CloudDocs/Data/Bloomberg/FX/Vols'
    _WING_BVOLS_PATH = os.path.join(_BVOLS_PATH, 'Wing Vols.xlsx')
    _ATM_BVOL_PATH = os.path.join(_BVOLS_PATH, 'ATM Vols.xlsx')

    @staticmethod
    def update_fx_atm_bvols():

        raw = implied_volatility.bdh_load_from_excel(implied_volatility._ATM_BVOL_PATH)
        unique_tickers = raw.columns.get_level_values(0).unique()

        for ticker in unique_tickers:

            # If the time series does not exists in the time series spec, then add it
            if not implied_volatility.is_ticker_in_database(ticker):
                implied_volatility.insert_time_series_spec_from_fx_BVOL_ticker(ticker)

            df = raw.loc[:, raw.columns.get_level_values(0) == ticker].copy().dropna(how='all')
            df.columns = [x.lower().replace('px_','') for x in df.columns.get_level_values(-1)]
            df.index.name = 'date'

            pair, tenor = ticker.split()[0].split('V')

            if tenor.lower() == 'on':
                tenor = '1d'

            d_ccy = pair[3:]
            f_ccy = pair[:3]

            df['strike_reference'] = 'ATM'
            df['tenor'] = tenor.lower()
            df['domestic_currency'] = d_ccy
            df['foreign_currency'] = f_ccy
            df['currency'] = pair
            df['uid'] = Bloomberg.get_uid_from_ticker(ticker)
            df = df.reset_index(drop=False)

            df_prime = pd.read_sql('SELECT date FROM implied_volatility where uid ="' +
                                         str(Bloomberg.get_uid_from_ticker(ticker)) + '"',
                                         SessionMgr().getEngine())

            df_sql = df[~df['date'].isin(df_prime['date'])]
            if df_sql.size > 0:
                df_sql.to_sql(name='implied_volatility',
                          con=SessionMgr().getEngine(),
                          if_exists='append',
                          index=False)
            else:
                print('No data for add for time series with ticker {}'.format(ticker))


    @staticmethod
    def load_wing_implied_vols_from_excel():

        raw = ExcelUtils.xlsread_sheets(implied_volatility._WING_BVOLS_PATH)
        # We may have a tickers sheet, if so, remove it
        if 'Tickers' in raw.keys():
            del raw['Tickers']

        df = pd.DataFrame()
        for sheet in raw.keys():

            processed_df = implied_volatility.process_wing_implied_vols(raw.get(sheet))
            df = pd.concat((df, processed_df), axis=1)
        return df.copy()


    @staticmethod
    def process_wing_implied_vols(df):
        df = df.replace({pd.NaT: np.nan}).dropna(how='all')

        unique_quote_types = np.unique(df.values[df.apply(lambda x:
                                                          x.astype(str).str.contains('BID|MID|ASK'))])
        frame = pd.DataFrame()
        for price_quote in unique_quote_types:
            processed_df = implied_volatility.process_wing_implied_vols_single_quote_type(df, price_quote)
            processed_df.columns = pd.MultiIndex.from_tuples([x.replace('DP','P').replace('DC','C').split()[0:3] + [price_quote.lower()] + [x] for x in processed_df.columns])
            frame = pd.concat((frame, processed_df), axis=1)

        #df_dta = pd.DataFrame()
        #unique_maturities = frame.columns.get_level_values(0).unique()
        #for mat in unique_maturities:
        #    df_mat = frame.get(mat).copy()
        #    if 'ask' in df_mat.columns and 'bid' in df_mat.columns:
        #        df_mat['mid'] = df_mat[['ask', 'bid']].mean(axis=1, skipna=True)
        #    df_mat.columns = pd.MultiIndex.from_tuples([(mat, x) for x in df_mat.columns])
        #    df_dta = pd.concat((df_dta, df_mat), axis=1)
        return frame.copy()

    @staticmethod
    def process_wing_implied_vols_single_quote_type(df, price_quote):

        prc_locs = df.apply(lambda x: x.astype(str).str.contains(price_quote))
        keep_cols = np.array(df.columns[prc_locs.any(axis=0)])
        df = df.loc[~prc_locs.any(axis=1), :].loc[:, np.insert(keep_cols, 0, 0)]
        date_locs = df.applymap(lambda x: isinstance(x, pd.Timestamp) or isinstance(x, datetime))

        date_col = date_locs.any(axis=0)
        assert date_col.sum() == 1, 'Error - cant support multiple date columns'

        date_col = int(date_col.index[date_col].to_numpy())
        df[date_col] = df[date_col].apply(lambda x: parser.parse(x) if isinstance(x, str) else x)
        date_rows = df[date_col].apply(lambda x: isinstance(x, datetime) and x is not pd.NaT)

        frame = df.loc[date_rows, :]
        frame.columns = df.loc[~date_rows, :].values.flatten()
        frame = frame.rename(columns={frame.columns[date_col]:'date'})
        frame = frame.set_index('date', drop=True)
        return frame.sort_index()

    @staticmethod
    def update_fx_wing_bvols():

        fxivols = implied_volatility.load_wing_implied_vols_from_excel()

        unique_tickers = fxivols.columns.get_level_values(-1).unique()
        for ticker in unique_tickers:

            # If the time series does not exists in the time series spec, then add it
            if not implied_volatility.is_ticker_in_database(ticker):
                implied_volatility.insert_time_series_spec_from_fx_BVOL_ticker(ticker)

            # Insert Time Series
            assert implied_volatility.is_ticker_in_database(ticker), 'Error ticker {} not in TimeSeriesSpec'.format(ticker)
            df = fxivols.loc[:, fxivols.columns.get_level_values(-1) == ticker].copy()
            df.columns = df.columns.get_level_values(-2)

            if 'bid' in df.columns and 'ask' in df.columns:
                df['mid'] = df.mean(axis=1).copy()

            pair, tenor, strike, _, _, _ = ticker.split()
            d_ccy = pair[3:]
            f_ccy = pair[:3]
            delta, put_call = strike.split('D')

            if 'p' in put_call.lower():
                df['strike_reference'] = '-' + delta
            else:
                df['strike_reference'] = delta

            df['tenor'] = tenor.lower()
            df['domestic_currency'] = d_ccy
            df['foreign_currency'] = f_ccy
            df['currency'] = pair
            df['uid'] = Bloomberg.get_uid_from_ticker(ticker)
            df = df.reset_index(drop=False)

            df_prime = pd.read_sql('SELECT date FROM implied_volatility where uid ="' +
                                         str(Bloomberg.get_uid_from_ticker(ticker)) + '"',
                                         SessionMgr().getEngine())

            df_sql = df[~df['date'].isin(df_prime['date'])]
            if df_sql.size > 0:
                df_sql.to_sql(name='implied_volatility',
                          con=SessionMgr().getEngine(),
                          if_exists='append',
                          index=False)
            else:
                print('No data for add for time series with ticker {}'.format(ticker))

    @staticmethod
    def is_wing_bvol_ticker(ticker):
        return 'VOL BVOL Curncy'.lower() in ticker.lower()


    @staticmethod
    def insert_time_series_spec_from_fx_BVOL_ticker(ticker):

        if not Bloomberg.is_ticker_in_database(ticker):
            try:
                uid = Bloomberg.get_max_uid() + 1
                region = ISO_to_region.region_from_iso(ticker.split()[0][3:6])
                if implied_volatility.is_wing_bvol_ticker(ticker):
                    name = ticker.replace('VOL BVOL Curncy','Implied Volatility')
                else:
                    name = ticker.split()[0][0:6] + ' ' + ticker.split()[0][7:] + ' ATM Implied Volatility'
                spec = TimeSeriesSpec(uid=uid, name=name, category='FX', provider='BBG', ticker=ticker, region=region)
                session.add(spec)
                session.commit()
            except:
                session.rollback()
                print('Error - could not add time series spec info for ticker {}'.format(ticker))
            finally:
                session.close()


class interest_rates(Bloomberg):

    _RATES_PATH = '/Users/francisbarker/Library/Mobile Documents/com~apple~CloudDocs/Data/Bloomberg/Rates/'

    @staticmethod
    def load_data_from_bbg_bdh(workbook_name):

        files = os.listdir(interest_rates._RATES_PATH)
        assert workbook_name + '.xlsx' in files,  'Error - workbook not found'
        fullfile_path = os.path.join(interest_rates._RATES_PATH, workbook_name + '.xlsx')
        return Bloomberg.bdh_load_from_excel(fullfile_path)

    @staticmethod
    def update_rates_from_bdh(workbook_name):

        rates = interest_rates.load_data_from_bbg_bdh(workbook_name)
        tickers = rates.columns.get_level_values('TICKER')
        for ticker in tickers:

            sub = rates.iloc[:, tickers == ticker].dropna()
            sub = sub.loc[:, ~sub.columns.duplicated()]
            is_present = interest_rates.is_ticker_in_database(ticker)

            if not is_present:
                interest_rates.insert_time_series_spec(sub)

            # Insert Time Series
            assert interest_rates.is_ticker_in_database(ticker), 'Error ticker {} not in TimeSeriesSpec'.format(ticker)

            df = pd.concat([sub.copy()] * 2, axis=1)
            df.columns = ['last','mid']
            df['uid'] = interest_rates.get_uid_from_ticker(ticker)

            if 'date' in df.index.name.lower():
                df.index.name = 'date'
            else:
                raise ValueError('Error - index must contain dates')
            df = df.reset_index(drop=False)

            df_prime = pd.read_sql('SELECT date FROM interest_rates where uid ="' +
                                         str(interest_rates.get_uid_from_ticker(ticker)) + '"',
                                         SessionMgr().getEngine())

            df_sql = df[~df['date'].isin(df_prime['date'])]
            if df_sql.size > 0:
                df_sql.to_sql(name='interest_rates',
                          con=SessionMgr().getEngine(),
                          if_exists='append',
                          index=False)
                print('Data appended to table interest_rates for time series with ticker {}'.format(ticker))
            else:
                print('No data for add for time series with ticker {}'.format(ticker))


    @staticmethod
    def insert_time_series_spec(df):

        tmp = dict(zip(df.columns.names, df.columns.values[0]))
        if not Bloomberg.is_ticker_in_database(tmp.get('TICKER')):
            try:
                uid = Bloomberg.get_max_uid() + 1

                ir_spec = InterestRateSpec(uid=uid, currency=tmp.get('CRNCY'), type=tmp.get('CURVE'), provider='BBG',
                                           category='Interest Rate')
                ir_spec.region = tmp.get('REGION')
                ir_spec.ticker = tmp.get('TICKER')
                ir_spec.name = tmp.get('LONG_COMP_NAME')
                ir_spec.maturity = tmp.get('MATURITY').lower()

                session.add_all([ir_spec])
                session.commit()
            except:
                session.rollback()
                print('Error - could not add time series spec info for ticker {}'.format(tmp.get('TICKER')))
            finally:
                session.close()


if __name__ == "__main__":


    from epsilonPhi.core.dataModel.dataSources.Bloomberg import Bloomberg
    session = SessionMgr().getSessionFactory()

    data_table_name = 'interest_rate'
    spec_table_name = 'interest_rate_spec'

    folder_name = r'rates\short rates'
    info_workbook_name = 'Spec.xlsx'
    info_sheetname = 'Spec'
    data_folder = os.path.join(_DATA_PATH, folder_name, 'Data Repository')

    df_info = pd.read_excel(os.path.join(_DATA_PATH, folder_name, info_workbook_name), sheet_name=info_sheetname)
    df_info = df_info.set_index('ticker', drop=True)
    dir_list = os.listdir(os.path.join(_DATA_PATH, folder_name, 'Data Repository'))

    for dir_name in dir_list:

        dta_path = os.path.join(data_folder, dir_name)
        if os.path.isfile(dta_path):

            df_raw = pd.read_csv(dta_path, index_col=0, header=[1,2,3,4,5])
            keep_cols = np.array(['ERROR' not in x for x in df_raw.columns.get_level_values('Name')])
            df = df_raw.iloc[:, keep_cols].dropna(how='all', axis=0)
            if df.size > 0:

                ticker = df.columns.get_level_values('MNEM').unique()[0]
                row = df_info.loc[ticker]

                df.columns = df.columns.get_level_values('DATATYPE')
                df = df.applymap(lambda x: np.nan if isinstance(x, str) and '$$ER:' in x else x).dropna(how='all',axis=0)
                df.index = pd.to_datetime(df.index)
                df.index.name = 'date'

                if not Bloomberg.is_ticker_in_database(ticker):
                    try:
                        spec = EquitySpec()
                        spec.ISIN = row['ISIN CODE']
                        spec.SEDOL = row['SEDOL CODE']
                        spec.category = row.category
                        spec.datasource = row.datasource
                        spec.denominated_currency = row.Currency
                        spec.exchange = row.Exchange
                        spec.exchange_code = row['BOURSE CODE']
                        spec.exchange_mnemonic = row['BOURSE MNEMONIC']
                        spec.exposure_currency = row.Currency
                        spec.hedge_ratio = 0
                        spec.industry = row.Industry
                        spec.industry_group = row['Industry Group']
                        spec.name = row.long_name
                        spec.provider = row.provider
                        spec.region = row.region
                        spec.sector = row.Sector
                        spec.sub_industry = row['Sub Industry']
                        spec.symbol = row.ticker
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

                    df_prime = pd.read_sql('SELECT date FROM ' + data_table_name + ' where uid ="' +
                                           str(Bloomberg.get_uid_from_ticker(ticker)) + '"',
                                           SessionMgr().getEngine())

                    sqldates = np.setdiff1d(pd.to_datetime(df['date'].values), pd.to_datetime(df_prime['date']))
                    if len(sqldates) > 0:
                        df_sql = df[df['date'].isin(sqldates)]

                        df_sql = df_sql.dropna(how='all', axis=1)
                        if df_sql.size > 0:
                            df_sql.to_sql(name=data_table_name,
                                          con=SessionMgr().getEngine(),
                                          if_exists='append',
                                          index=False)
                            print('Data appended to table ' + data_table_name + ' for time series with ticker {}'.format(ticker))
                    else:
                        print('No data for add for time hedge_fund_index with ticker {}'.format(ticker))




    session = SessionMgr().getSessionFactory()
















