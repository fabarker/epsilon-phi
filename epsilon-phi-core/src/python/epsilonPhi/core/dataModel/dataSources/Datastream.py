from pydatastream import Datastream as pyds
from epsilonPhi.core.utils.ListUtils import ListUtils as lutils
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.env.Env import DS_USERNAME, DS_PASSWORD
from typing import Optional, Union
import numpy as np
import pandas as pd
import datetime as dt
import time, os
import win32com.client as win32
import subprocess
from epsilonPhi.core.lib.Decorators import SingletonDecorator

_DATA_PATH = os.path.join(os.environ.get('HOMEDRIVE'), os.environ.get('HOMEPATH'), 'Documents', 'Data')

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

    _CREDENTIALS = list()
    _CREDENTIALS.extend([('ZGOL433','ATLAS802'),
                         ('ZGSC304','TORCH863'),
                         ('ZGOL188','YOUNG902'),
                         ('ZGOL865','SOUTH366'),
                         ('ZGOL865','SOUTH366')])

    @staticmethod
    def pyds():
        return pyds(username=DS_USERNAME,
                    password=DS_PASSWORD,
                    proxy=None,
                    raise_on_error=True)

    @staticmethod
    def get_usage(months=12):
        return pyDatastream.pyds().usage_statistics(months=months)

    @staticmethod
    def fetch(tickers: Optional[Union[list, np.array, str]] = None,
              fields: Optional[Union[list, np.array, str]] = None,
              from_date: Optional[dt.date] = None,
              to_date: Optional[dt.date] = None,
              frequency: Optional[Frequency] = None) -> pd.DataFrame:

        chunks = lutils._nest_list([tickers] if isinstance(tickers, str) else list(tickers), 50)
        frames = pd.DataFrame()
        for chunk in chunks:
            try:
                res = pyDatastream.pyds().fetch(chunk,
                                                fields = fields,
                                                date_from= from_date,
                                                date_to = to_date,
                                                freq = frequency,
                                                always_multiindex=False).reset_index()
                frames = pd.concat((frames, res))
            except:
                pass
        return frames

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


dfo_path = r'C:\Users\fabar\Repos\epsilon-phi\epsilon-phi-core\src\resources\templates\DFORequest.xlsm'
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


if __name__ == "__main__":

    from epsilonPhi.core.dataModel.dataSources.Bloomberg import Bloomberg, ISO_to_region
    def run_updates():
        datafields = ['X','RI','IO','IB','IR']

        # Hedge Funds
        folder_name = r'C:\Users\fabar\Documents\Data\rates\short rates'
        info_workbook_name = 'Spec.xlsx'
        info_sheetname = 'Spec'
        save_folder = os.path.join(_DATA_PATH, folder_name, 'Data Repository')

        df_info = pd.read_excel(os.path.join(_DATA_PATH, folder_name, info_workbook_name), sheet_name=info_sheetname)
        Tickers = df_info['ticker'].values.flatten()

        ctr = 0
        for ticker in Tickers:
            if ctr == 300:
                pyDatastreamFO.cleanup()
                closeExcel.kill_all_excel_instances()
                time.sleep(30)
                ctr = 0

            if not os.path.isfile(os.path.join(save_folder, ticker.replace(':','_') + '.csv')):
                r = request(ticker, datafields, start_date='BDATE', freq='Daily')
                pyDatastreamFO().query_with_data_dump(r, save_folder)
                ctr += 1
    run_updates()


