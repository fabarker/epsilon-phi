from epsilonPhi.core.dataModel.dataSources.vendor.Bloomberg import Bloomberg, ISO_to_region
from epsilonPhi.core.dataModel.dataSources.vendor.Datastream import pyDatastream
from epsilonPhi.core.dataModel.alchemist.SessionManager import *
from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.utils.ExcelUtils import ExcelUtils
from epsilonPhi.core.utils.ListUtils import ListUtils
from datetime import timedelta
from sqlalchemy import distinct, func
from datetime import datetime
from dateutil import parser
import pandas as pd
import numpy as np
import os
import platform

nan = pd.pandas._libs.tslibs.nattype.NaTType
session = SessionMgr().getSessionFactory()

f_path_ = os.path.abspath(__file__)
if 'windows' in platform.system().lower():
    DIR_ = os.path.join(f_path_[:f_path_.find('epsilon-phi-core\\') + len('epsilon-phi-core\\')],
                        'src\\resources\\database\\dataloader\\logs')
else:
    DIR_ = os.path.join(f_path_[:f_path_.find('epsilon-phi-core/')+ len('epsilon-phi-core/')],
                        'src/resources/database/dataloader/logs')

_ERROR_PATH_DS = os.path.join(DIR_, 'DS')
_ERROR_PATH_GS = os.path.join(DIR_, 'GS')

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

class Updater:

    _session = SessionMgr()
    _session_factory = _session.getSessionFactory()

    @staticmethod
    def _get_bind():
        return Updater._session.getSessionFactory().get_bind()

    @staticmethod
    def update_table_data(table_name):
        from sqlalchemy import distinct
        table_obj = Updater._session.fetch_model_class_from_table_name(table_name)

        if table_obj is None:
            print('No table {} in database'.format(table_name))
            return

        uids = [x[0] for x in Updater._session_factory.query(distinct(table_obj.uid)).all()]
        unique_sources = Updater._session_factory.query(distinct(TimeSeriesSpec.datasource)).filter(TimeSeriesSpec.uid.in_(uids)).all()
        for source in unique_sources:
            Updater._update_table_data_single_datasource(table_obj, source[0])

    @staticmethod
    def _update_table_data_single_datasource(table, datasource):
        if datasource.lower() == 'datastream':
            DATASTREAM_UPDATER(table).run()
        else:
            print('datasource {} not recognised'.format(datasource))

    @staticmethod
    def _get_time_series_spec_for_table_datasource(table, datasource):
        uids = [x[0] for x in Updater._session_factory.query(distinct(table.uid)).all()]
        series_specs = pd.DataFrame(Updater._session_factory.query(TimeSeriesSpec.uid,
                                                                   TimeSeriesSpec.symbol,
                                                                   TimeSeriesSpec.frequency).filter(TimeSeriesSpec.datasource == datasource,
                                                                                                 TimeSeriesSpec.uid.in_(uids)).all())
        return series_specs.set_index('symbol', drop=True)

class GSQUANT_UPDATER(Updater):

    _datasource = 'GS'

    def __init__(self, table):
        super(GSQUANT_UPDATER, self).__init__()
        self._table = table
        self._table_name = self._table.__table__.name
        self._tbl_cols = Updater._session.get_all_columns_in_table(table.__table__.name)
        self._setup()

    def _setup(self):
        self._spec = self._get_time_series_spec_for_table_datasource(self._table, self._datasource)
        self._nested_list = ListUtils._nest_list(self._spec.index, 100)
        self._error_dir = os.path.join(_ERROR_PATH_GS, self._table_name)
        if not os.path.isdir(self._error_dir):
            os.mkdir(self._error_dir)

    def get_tickers_from_dataset(self, dataset):
        pass

    def get_dataset_tickers(self):
        pass

    def update_fx_ivols(self):
        pass

    def update_fx_spots(self):
        pass

    def update_fx_forwards(self):
        pass







class DATASTREAM_UPDATER(Updater):
    _DATASTREAM_DATATYPES = ['529E', 'APC', 'CX', 'DIEP', 'DIPE', 'DM', 'DY', 'EB', 'EO', 'EPS', 'EPS1FD12', 'ER', 'IB',
                             'IN', 'IO', 'IR', 'ISIN', 'L', 'MV', 'NOSH', 'OI', 'PA', 'PB', 'PE', 'PH', 'PI', 'PL', 'PO', 'PS',
                             'PTBV', 'RI', 'RY', 'SEDOL', 'VM', 'VO', 'WC01001', 'X', 'YTW', 'x']

    _datasource = 'Datastream'
    _use_API = True

    def __init__(self, table, chunk_size=100):
        super(DATASTREAM_UPDATER, self).__init__()
        self._table = table
        self._table_name = self._table.__table__.name
        self._tbl_cols = Updater._session.get_all_columns_in_table(table.__table__.name)
        self._flds = np.intersect1d(DATASTREAM_UPDATER._DATASTREAM_DATATYPES, self._tbl_cols)

        if self._table_name == 'equity_index':
           self._flds = np.append(self._flds, ['DSDY','DSRI'])

        self._chunk_size = chunk_size
        self._setup()

    def _get_frequency_from_tickers(self, tickers):
        pass

    def _get_latest_observation_date_from_tickers(self, tickers):
        session = self._session_factory
        res = session.query(TimeSeriesSpec.symbol,
                            func.max(self._table.date)).join(self._table,
                                                             TimeSeriesSpec.uid == self._table.uid).\
                                                             filter(TimeSeriesSpec.symbol.in_(list(tickers))).\
                                                             group_by(TimeSeriesSpec.symbol).all()
        return pd.DataFrame(res, columns=['symbol', 'start']).set_index('symbol')


    @staticmethod
    def _get_datastream_frequency(frequency):
        if frequency in ['B','BD','D']:
           return 'D'
        elif frequency in ['M','MS','BM']:
            return 'M'
        else:
            return frequency

    def _filter_tickers_by_end_date(self, spec):

        DB_LD = self._get_latest_observation_date_from_tickers(spec.index)
        DS_LD = pyDatastream.get_latest_date_from_tickers(list(DB_LD.index))
        DS_LD = DS_LD.replace('NA', pd.to_datetime(datetime.date(datetime.today())))

        if 'MS' in spec.frequency.unique():
            DS_LD.loc[spec.loc[spec.frequency == 'MS'].index] = (DS_LD.loc[spec.loc[spec.frequency == 'MS'].index] +
                                                                 pd.tseries.offsets.MonthBegin(-1))

        EDs = pd.concat((DB_LD, DS_LD), axis=1)
        keep = EDs.index[(EDs.diff(axis=1).dropna(axis=1) > timedelta(1)).values.flatten()]
        return DB_LD.loc[keep]

    def _setup(self):

        self._error_dir = os.path.join(_ERROR_PATH_DS, self._table_name)
        if not os.path.isdir(self._error_dir):
            os.mkdir(self._error_dir)

        spec = self._get_time_series_spec_for_table_datasource(self._table,
                                                                     self._datasource)
        tickers = self._filter_tickers_by_end_date(spec)
        _spec = pd.concat((spec.loc[tickers.index], tickers), axis=1)
        _spec.index.name = 'symbol'
        self._spec = _spec.reset_index(drop=False).set_index('symbol')
        self._frequencies = _spec.reset_index(drop=False).set_index('frequency')
        self._unique_frequencies = np.unique(self._frequencies.index)

    def _query_datastream(self, query, freq, SD):

        f = DATASTREAM_UPDATER._get_datastream_frequency(freq)
        return pyDatastream.fetch(query,
                                 fields=list(self._flds),
                                 from_date=SD,
                                 frequency=f)

    def run(self):
        for freq in self._unique_frequencies:
            F = self._frequencies.loc[freq]
            if F.ndim == 1:
                self.run_single_frequency(F.symbol, freq, F.start)
            else:
                self.run_single_frequency(list(F.symbol), freq, F.start.min())

    def run_single_frequency(self, ticker_list, freq, SD):

        _nested_list = ListUtils._nest_list(ticker_list, self._chunk_size)

        ctr = 1
        for list in _nested_list:
            print(len(_nested_list) - ctr)
            ctr += 1

            df_ = self._query_datastream(list, freq, SD)

            df_dtbs = pd.DataFrame()
            unique_tickers = df_.index.get_level_values(0).unique()
            for symbol in unique_tickers:
                processed = self._process_frame(df_.loc[symbol].copy(), symbol, freq)
                df_dtbs = pd.concat((df_dtbs, processed), axis=0)


            if df_dtbs.size > 0:
                df_sql = df_dtbs.reset_index(drop=True).dropna(how='all', axis=1)
                if df_sql.size > 0:
                   self.insert(df_sql)
                else:
                    pd.DataFrame(list).to_csv(os.path.join(self._error_dir, datetime.now().strftime("%m%d%Y %H%M%S%z")))
                    print('No data for add for time series ' + self._table_name)
            else:
                pd.DataFrame(list).to_csv(os.path.join(self._error_dir, datetime.now().strftime("%m%d%Y %H%M%S%z")))
                print('No data for add for time series ' + self._table_name)

    def _process_frame(self, df, symbol, freq):

        # drop any rows that are all names
        df_ = df.dropna(how='all', axis=0)

        # rename the index to match the date col of the table
        df_.index.name = 'date'

        if freq.upper() == 'MS':
            df_.index = pd.to_datetime(df_.index) + pd.tseries.offsets.MonthBegin(-1)
        else:
            df_.index = pd.to_datetime(df_.index)

        # add the series uid to the table
        uid = int(self._spec.loc[symbol].uid)
        uids = pd.DataFrame([uid] * df_.shape[0], columns=['uid'], index=df_.index)
        df_ = pd.concat((uids, df_), axis=1)
        df_ = df_.reset_index(drop=False)

        cols = np.setdiff1d(self._tbl_cols, df_.columns)
        for col in cols:
            _txt = 'SELECT DISTINCT(' + col + ') FROM ' + self._table_name + ' WHERE uid = ' + str(
                self._spec.loc[symbol].uid)
            df_[col] = self._session_factory.execute(text(_txt)).first()[0]

        # now drop any columns with all nans
        tbl_df = df_.dropna(how='all', axis=1)

        if tbl_df.size == 0:
            pd.DataFrame([symbol + ' NOT UPDATED']).to_csv(os.path.join(self._error_dir, symbol.replace(':', '_') +
                                                                        datetime.now().strftime(" %m%d%Y %H%M%S%z")))
            return pd.DataFrame()

        # Check if we have DSRI, DSDY in the frame
        if 'DSRI' in tbl_df.columns:
            if 'RI' in tbl_df.columns:
                tbl_df = tbl_df.drop(columns='DSRI')
            else:
                tbl_df = tbl_df.rename(columns={'DSRI': 'RI'})

        if 'DSDY' in tbl_df.columns:
            if 'DY' in tbl_df.columns:
                tbl_df = tbl_df.drop(columns='DSDY')
            else:
                tbl_df = tbl_df.rename(columns={'DSDY': 'DY'})

        tmp = tbl_df.set_index('date', drop=True)
        q_dts = Updater._session_factory.query(self._table.date).filter(self._table.uid == uid)
        dbs_dates = Updater._session.query_format_df(q_dts)

        keep_dates = np.setdiff1d(pd.to_datetime(tmp.index),
                                  pd.to_datetime(dbs_dates.values.flatten()))
        df_dbs = tmp.loc[keep_dates].reset_index(drop=False)
        if df_dbs.size == 0:
                pd.DataFrame([symbol + ' NOT UPDATED']).to_csv(os.path.join(self._error_dir, symbol.replace(':', '_') + datetime.now().strftime(" %m%d%Y %H%M%S%z")))
        return df_dbs.copy()

    def insert(self, df):
        try:
            df.to_sql(name=self._table_name, con=Updater._session.getEngine(), if_exists='append', index=False)
            print('Data appended to table ' + self._table_name)
        except:
            print('Error adding data to database {}'.format(df.columns))
            df.to_csv(os.path.join(self._error_dir, datetime.now().strftime("%m%d%Y %H%M%S%z")))


if __name__ == "__main__":

    table_names = ['interest_rate','fx_rates','equity_index',
                   'bond_index','commodity_index','hedge_fund_index','yield_curve','future']

    for table in table_names:
        Updater.update_table_data(table)




