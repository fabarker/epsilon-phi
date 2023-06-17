import os, time
import pandas as pd
import win32com.client as win32
import subprocess
import numpy as np

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

class pyDFO(object):
    path = r'C:\Users\fabar\Repos\epsilon-phi\epsilon-phi-core\src\resources\templates\DFORequest.xlsm'
    refinitive_run = r'"C:\Users\fabar\AppData\Local\Refinitiv\Refinitiv Workspace\RefinitivWorkspace.exe" --excel'

    def __init__(self):

        subprocess.run(pyDFO.refinitive_run, shell=True)
        time.sleep(5)
        subprocess.run(pyDFO.path, shell=True)
        time.sleep(20)

        workbook = win32.GetObject(pyDFO.path)
        workbook.Application.Visible = False
        self.app = win32.Dispatch("Excel.Application")
        self.workbook = self.app.ActiveWorkbook
        self.app.DisplayAlerts = False
        self.reset_requests()

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
            df = pd.DataFrame(self.app.Run('QueryDS', [request]))
            return df.set_index(0, drop=True).replace('=NA()', np.nan)
        except:
            return pd.DataFrame()


date_path = r'C:\Users\fabar\Documents'
path_to_folder = os.path.join(date_path)
workbook_name = 'Bank of America ML.xlsx'

df_info = pd.read_excel(os.path.join(path_to_folder, workbook_name), 'Info')
tickers = pd.DataFrame(df_info.ticker.values.flatten()).dropna().values.flatten()
# < 2009 and Alive only
# series = alive[alive['Hist.'] <= 2010]
FIELDS = ['DM','RI','RY','CX']

ds = pyDFO()
for f in tickers:
     print(f)

     save_path = os.path.join(path_to_folder, 'Data', f.replace('.', '') + '.csv')
     if not os.path.isfile(save_path):

         r = request(f, FIELDS, start_date='31/12/1969', freq='Daily')
         ds.requestData = list()
         ds.append_requests(r)
         res = ds.query()

         if res.size > 0:
            save_path = os.path.join(path_to_folder, 'Data', f.replace('.','') + '.csv')
            res.dropna(how='all', axis=0).to_csv(save_path)

# for f in walk:
#     print(f)
#     df = pd.read_csv(f)
#     if df.size > 0 and df.loc[np.any(df == 'MNEM', axis=1), :].size > 1:
#         df = df.set_index(df.columns[0], drop=True)
#         df = df.dropna(axis=0, how='all')
#         ticker = df.loc['MNEM'].unique()[0]
#
#         locs = np.array([ ':' in x for x in df.index ])
#         idx = np.asarray(df.index)
#         idx[locs] = pd.to_datetime(idx[locs])
#         df.index = idx
#
#         r = request(ticker, ['DIEP','DIPE','PA','PB','PL','PH'], start_date='31/12/1969', freq='Daily')
#         ds.requestData = list()
#
#         ds.append_requests(r)
#         res = ds.query()
#
#         df.columns = np.array(range(len(df.columns)))
#         res.columns = res.columns + max(df.columns)
#         res_df = pd.concat((df, res), axis=1).dropna(axis=0, how='all')
#         res_df.to_csv(f)

closeExcel.kill_all_excel_instances()

