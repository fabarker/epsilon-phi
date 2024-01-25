import datetime

from pydatastream import Datastream as pyds
from epsilonPhi.core.utils.ListUtils import ListUtils as lutils
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.env.Env import DS_USERNAME, DS_PASSWORD
from typing import Optional, Union
import numpy as np
import pandas as pd
import datetime as dt
import time, os
import subprocess
from epsilonPhi.core.lib.Decorators import SingletonDecorator

#_DATA_PATH = os.path.join(os.environ.get('HOMEDRIVE'), os.environ.get('HOMEPATH'), 'Documents', 'Data')

class DatatypeMapper(object):

    @staticmethod
    def datasource_to_database_mapping(source_datatype):
        if source_datatype.upper() in ['YTM','RY','YTW','IY','RA']:
            return 'RY'
        if source_datatype.upper() in ['RI']:
            return 'RI'
        if source_datatype.upper() in ['IN']:
            return 'IN'
        if source_datatype.upper() in ['CX']:
            return 'CX'
        if source_datatype.upper() in ['DM','DU']:
            return 'DM'
        else:
            raise ValueError('Error - datafield {} not supported')

class Freq(object):

    freq = dict()
    freq[0] = 'D'
    freq[1] = 'W'
    freq[2] = 'M'
    freq[3] = 'Q'
    freq[4] = 'Y'

    @staticmethod
    def freq_from_enum(frequency: Frequency):
        return Freq.freq[frequency.value]

class pyDatastream(object):


    _cache = None
    _CREDENTIALS = list()
    _CREDENTIALS.extend([('ZGOL433','ATLAS802'),
                         ('ZGSC304','TORCH863'),
                         ('ZGOL188','YOUNG902'),
                         ('ZGOL865','SOUTH366'),
                         ('ZGDP339','ALPHA198')])

    @staticmethod
    def pyds(raise_on_error=False):

        if pyDatastream._cache is None:
            pyDatastream._cache = pyds(username=DS_USERNAME,
                        password=DS_PASSWORD,
                        proxy=None,
                        raise_on_error=raise_on_error)
        return pyDatastream._cache

    @staticmethod
    def get_usage(months=12):
        return pyDatastream.pyds().usage_statistics(months=months)

    @staticmethod
    def get_max_instruments_per_call(data_types):

        MAX_INSTRUMENTS = 50
        MAX_DATATYPES = 50
        MAX_PROD = 100

        if data_types is None:
            return int(MAX_INSTRUMENTS)

        if isinstance(data_types, str):
           data_types = [datetime]

        N = len(data_types)
        if N == 0:
           return int(MAX_INSTRUMENTS)
        elif N > MAX_DATATYPES:
           raise ValueError('Error - number of datatypes exceeds maximum datatypes')

        max_instruments = MAX_PROD // N
        return int(min(MAX_INSTRUMENTS, max_instruments))

    @staticmethod
    def fetch(tickers: Optional[Union[list, np.array, str]] = None,
              fields: Optional[Union[list, np.array, str]] = None,
              from_date: Optional[dt.date] = None,
              to_date: Optional[dt.date] = None,
              frequency: Optional[str] = None) -> pd.DataFrame:

        N = pyDatastream.get_max_instruments_per_call(fields)
        chunks = lutils._nest_list([tickers] if isinstance(tickers, str) else list(tickers), N)

        frames = pd.DataFrame()
        for chunk in chunks:
            res = pyDatastream.pyds(raise_on_error=False).fetch(chunk,
                                                                fields=fields,
                                                                date_from=from_date,
                                                                date_to=to_date,
                                                                freq=frequency,
                                                                always_multiindex=True)
            frames = pd.concat((frames, res))
        return frames

    @staticmethod
    def fetch_static(tickers: Optional[Union[list, np.array, str]] = None,
                     fields: Optional[Union[list, np.array, str]] = None) -> pd.DataFrame:

        N = pyDatastream.get_max_instruments_per_call(fields)
        chunks = lutils._nest_list([tickers] if isinstance(tickers, str) else list(tickers), N)

        frames = pd.DataFrame()
        for chunk in chunks:
            res = pyDatastream.pyds(raise_on_error=False).fetch(chunk,
                                                                fields=fields,
                                                                static=True)
            frames = pd.concat((frames, res))
        return frames

    @staticmethod
    def get_futures_meta(tickers: Optional[Union[list, np.array, str]] = None):
        return pyDatastream.fetch_static(tickers,
                                    ['FLOT','FEX','EXCODE','EXNAME','EXDSCD','EXMNEM','ISONAME','ISOCUR','DS.EXPNAME','NAME','FUTBDATE','TICKS','TICKV',
                                           'TCYCLE','TYPE','FISN','UNITS','FUI','MIFUNAC','CFI','GEOG','GEOL','GEOLC','GEOLN'])



    @staticmethod
    def get_currency_ISO_from_tickers(tickers: Union[list, str]) -> pd.DataFrame:
        chunks = lutils._nest_list([tickers] if isinstance(tickers, str) else list(tickers), 50)
        frames = pd.DataFrame()
        for chunk in chunks:
            res = pyDatastream.pyds().fetch(chunk,
                                            fields='ISOCUR',
                                            static=True)
            frames = pd.concat((frames, res))
        return frames

    @staticmethod
    def get_bond_cusips_from_tickers(tickers: Union[list, str]) -> pd.DataFrame:
        chunks = lutils._nest_list([tickers] if isinstance(tickers, str) else list(tickers), 50)
        frames = pd.DataFrame()
        for chunk in chunks:
            res = pyDatastream.pyds().fetch(chunk,
                                            fields='LOC',
                                            static=True)
            frames = pd.concat((frames, res))
        return frames

    @staticmethod
    def get_name_from_tickers(tickers: Union[list, str]) -> pd.DataFrame:
        chunks = lutils._nest_list([tickers] if isinstance(tickers, str) else list(tickers), 50)
        frames = pd.DataFrame()
        for chunk in chunks:
            res = pyDatastream.pyds().fetch(chunk,
                                            fields='NAME',
                                            static=True)
            frames = pd.concat((frames, res))
        return frames

    @staticmethod
    def get_source_from_tickers(tickers: Union[list, str]) -> pd.DataFrame:
        chunks = lutils._nest_list([tickers] if isinstance(tickers, str) else list(tickers), 50)
        frames = pd.DataFrame()
        for chunk in chunks:
            res = pyDatastream.pyds().fetch(chunk,
                                            fields='DS.SRCE',
                                            static=True)
            frames = pd.concat((frames, res))
        return frames

    @staticmethod
    def get_latest_date_from_tickers(tickers: Union[list, str]) -> pd.DataFrame:
        chunks = lutils._nest_list([tickers] if isinstance(tickers, str) else list(tickers), 50)
        frames = pd.DataFrame()
        for chunk in chunks:
            res = pyDatastream.pyds().fetch(chunk,
                                            fields='TIME',
                                            static=True)
            frames = pd.concat((frames, res))
        return frames

    @staticmethod
    def get_contract_size_from_tickers(tickers: Union[list, str]) -> pd.DataFrame:
        chunks = lutils._nest_list([tickers] if isinstance(tickers, str) else list(tickers), 50)
        frames = pd.DataFrame()
        for chunk in chunks:
            res = pyDatastream.pyds().fetch(chunk,
                                            fields='FLOT',
                                            static=True)
            frames = pd.concat((frames, res))
        return frames

    @staticmethod
    def get_tick_size_from_tickers(tickers: Union[list, str]) -> pd.DataFrame:
        chunks = lutils._nest_list([tickers] if isinstance(tickers, str) else list(tickers), 50)
        frames = pd.DataFrame()
        for chunk in chunks:
            res = pyDatastream.pyds().fetch(chunk,
                                            fields='TICKS',
                                            static=True)
            frames = pd.concat((frames, res))
        return frames

    @staticmethod
    def get_tick_value_from_tickers(tickers: Union[list, str]) -> pd.DataFrame:
        chunks = lutils._nest_list([tickers] if isinstance(tickers, str) else list(tickers), 50)
        frames = pd.DataFrame()
        for chunk in chunks:
            res = pyDatastream.pyds().fetch(chunk,
                                            fields='TICKV',
                                            static=True)
            frames = pd.concat((frames, res))
        return frames

    @staticmethod
    def get_region(tickers):
        pass

class closeExcel:
    @staticmethod
    def kill_all_excel_instances():
        subprocess.call(["taskkill", "/f", "/im", "EXCEL.EXE"])


class request(object):
    def __init__(self,
                 tickers,
                 datatypes,
                 start_date="",
                 end_date="",
                 freq=""):

        self.requestData = [None] * 10
        self.requestData[0] = 'TSL'
        self.requestData[1] = 'RCF:MNEM,DATATYPE,NAME,SECD,ISIN,CODON,ISOCUR'

        self.set_tickers(tickers)
        self.set_datatypes(datatypes)

        self.requestData[4] = start_date
        self.requestData[5] = end_date
        self.requestData[6] = freq
        self.requestData[7] = ""
        self.requestData[8] = 7

    def set_tickers(self, tickers):
        if isinstance(tickers, str):
            self.requestData[2] = tickers
        else:
            self.requestData[2] = ','.join(list(tickers))

    def set_datatypes(self, datatypes):
        if isinstance(datatypes, str):
            self.requestData[3] = datatypes
        else:
            self.requestData[3] = ','.join(list(datatypes))


dfo_path = r'/src/resources/templates/DFORequest.xlsm'
refinitive_run = r'"C:\Users\fabar\AppData\Local\Refinitiv\Refinitiv Workspace\RefinitivWorkspace.exe" --excel'

@SingletonDecorator
class pyDatastreamFO(object):
    _cache = []

    def __init__(self, requests=None):

        closeExcel.kill_all_excel_instances()
        subprocess.run(refinitive_run, shell=True)
        time.sleep(10)
        subprocess.run(dfo_path, shell=True)
        time.sleep(10)

        import win32com.client as win32
        workbook = win32.GetObject(dfo_path)
        workbook.Application.Visible = True
        self.app = win32.Dispatch("Excel.Application")
        self.workbook = self.app.ActiveWorkbook
        self.app.DisplayAlerts = False
        self.reset_requests()

        if requests:
            self.append_requests(requests)


    def reset_requests(self):
        self.requestData = list()

    def append_requests(self, requests):
        if isinstance(requests, request):
            requests = [requests]
        for req in requests:
            self.requestData.extend([req.requestData])

    def query(self):
        res = [pd.DataFrame()]
        for req in self.requestData:
            res.extend([self.post(req)])
        return pd.concat(res, axis=1)

    def post(self, request):
        try:
            df = pd.DataFrame(self.app.Run('QueryDS', [request.requestData]))
            return df.set_index(0, drop=True).replace('=NA()', np.nan).replace('', np.nan)
        except:
            return pd.DataFrame()

    @staticmethod
    def query_with_data_dump(request, save_folder=None):

        if save_folder is None:
           save_folder = os.path.join(_DATA_PATH, 'Data Repository')

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        save_path = request.requestData[2].replace(':','_')
        fullfile_save = os.path.join(save_folder, save_path + '.csv')

        pydfo = pyDatastreamFO()
        res = pydfo.post(request)

        if res.size > 0:
            res.dropna(how='all', axis=0).to_csv(fullfile_save)
            print('Data saved for for tickers {}'.format(request.requestData[2].replace(',','|')))
        else:
            print('No data returned for tickers {}'.format(request.requestData[2].replace(',','|')))

    @staticmethod
    def query_datastream(request):
        pydfo = pyDatastreamFO()
        return pydfo.post(request)


if __name__ == "__main__":


    ticker = 'SPX03244160C'
    fields = ['DZ','FV','GM','VL','OI','PA','PB','P','OX','TA','VG','VM']

    from_date = datetime.date(year=1969, month=12, day=31)
    frame = pyDatastream.fetch([ticker], fields, from_date=from_date, frequency='D')


    from epsilonPhi.core.dataModel.dataSources.vendor.Bloomberg import Bloomberg
    from epsilonPhi.core.dataModel.alchemist.DataModel import *
    from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr

    sessionMgr = SessionMgr()
    session = SessionMgr().getSessionFactory()

    fullfile = '/Users/francisbarker/Desktop/Trend Following/Moskowitz, Ooi and Pedersen.xlsx'
    futures_spec = pd.read_excel(fullfile, sheet_name='Series to Add', index_col=0)
    tickers = np.unique(futures_spec.index)

    flds = ['L','OI','PH','PL','PS','PO','VM']
    for ticker in tickers:

        tmp_spec = futures_spec.loc[ticker]
        frame = pyDatastream.fetch([ticker], flds, from_date=from_date, frequency='D')
        frame = frame.dropna(how='all')

        if frame.size > 0:

            try:

                spec = FutureSpec()
                spec.name = tmp_spec.get('long_name')

                spec.denominated_currency = tmp_spec.get('denominated_currency')
                spec.exposure_currency = tmp_spec.get('exposure_currency')
                spec.ticker = tmp_spec.get('ticker')
                spec.provider = tmp_spec.get('provider')
                spec.frequency = tmp_spec.get('frequency')
                spec.category = tmp_spec.get('category')
                spec.contract_size = float(tmp_spec.get('contract_size'))
                spec.tick_size = float(tmp_spec.get('tick_size'))
                spec.tick_value = float(tmp_spec.get('tick_value'))
                spec.symbol = tmp_spec.get('symbol')
                spec.security = tmp_spec.get('security')
                spec.security_name = tmp_spec.get('security_name')
                spec.security_type = tmp_spec.get('security_type')
                spec.security_unit = tmp_spec.get('security_unit')
                spec.future = tmp_spec.get('future')
                spec.hedge_ratio = int(tmp_spec.get('hedge_ratio'))

                spec.cycle = tmp_spec.get('cycle')
                spec.datasource = tmp_spec.get('datasource')
                spec.exchange = tmp_spec.get('exchange')
                spec.exchange_name = tmp_spec.get('exchange_name')
                spec.position_forward = int(tmp_spec.get('position_forward'))
                spec.region = tmp_spec.get('region')
                spec.start_date = tmp_spec.get('start_date').strftime("%Y-%m-%d %H:%M:%S")
                spec.ticker = ticker

                spec.uid = Bloomberg.get_max_uid() + 1

                session.add_all([spec])
                session.commit()

                df = frame.loc[ticker].dropna(how='all', axis=0)
                df.index.name = 'date'
                df = df.reset_index(drop=False)
                uid = Bloomberg.get_uid_from_ticker(ticker)

                for_db = df.copy()
                for_db['uid'] = uid
                for_db.to_sql(name='future',
                              con=SessionMgr().getEngine(),
                              if_exists='append',
                              index=False)

                print('Time series {} added'.format(ticker))
            except:
                session.rollback()
                print('Error - could not add time series for ticker {}'.format(ticker))
            finally:
                session.close()
    session.close()




