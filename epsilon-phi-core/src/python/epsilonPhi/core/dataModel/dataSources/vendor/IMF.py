from imfpy import searches, retrievals, tools
from imfpy.retrievals import dots
from weo import download, WEO, all_releases
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
import requests
import pandas as pd
import numpy as np
import datetime as dt
from weo import download, WEO

#mapper_df = pd.read_excel(r'/Users/francisbarker/Repositories/Python/epsilon-phi/epsilon-phi-core/src/resources/templates/IMF Data Mapper.xlsx', sheet_name='IMF Mapping', index_col=0)
mgr = SessionMgr()
session = mgr.getSessionFactory()

class IMFQuery(object):
    _BASE_URL = r'http://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/IFS/'
    _mapper = dict()

    _mapper['FITB_PA'] = 'TBIMF'
    _mapper['FID_PA'] = 'DRIMF'
    _mapper['FPOLM_PA'] = 'PRIMF'
    _mapper['FIDR_PA'] = 'DPIMF'

    @staticmethod
    def get_ticker(region, ticker):
        code = IMFQuery.get_country_code_from_name(region)

        if ticker in IMFQuery._mapper.keys():
            return code + IMFQuery._mapper.get(ticker)
        else:
            raise ValueError('Error - ticker {} not recognised')

    @staticmethod
    def get_rate_type_from_ticker(ticker):
        if ticker.upper() == 'FITB_PA':
           return 'Treasury'
        elif ticker.upper() == 'FID_PA':
           return 'Policy'
        elif ticker.upper() == 'FPOLM_PA':
           return 'Policy'
        elif ticker.upper() == 'FIDR_PA':
           return 'Deposit'
        else:
            raise ValueError('Error {} not suppported'.format(ticker))

    @staticmethod
    def get_rate_name_from_ticker(ticker):
        if ticker.upper() == 'FITB_PA':
           return 'Treasury Bill'
        elif ticker.upper() == 'FID_PA':
           return 'Discount Rate'
        elif ticker.upper() == 'FPOLM_PA':
           return 'Policy Rate'
        elif ticker.upper() == 'FIDR_PA':
           return 'Deposit Rate'
        else:
            raise ValueError('Error {} not suppported'.format(ticker))

    @staticmethod
    def get_rate_tenor_from_ticker(ticker):
        if ticker.upper() == 'FITB_PA':
           return '3m'
        elif ticker.upper() == 'FID_PA':
           return 'ON'
        elif ticker.upper() == 'FPOLM_PA':
           return 'ON'
        elif ticker.upper() == 'FIDR_PA':
           return np.nan
        else:
            raise ValueError('Error {} not suppported'.format(ticker))

    @staticmethod
    def get_currency_iso(region):
        return mgr.get_currency_from_region(region)

    @staticmethod
    def get_country_code_from_name(region):
        return mgr.get_imf_code_from_region(region)

    @staticmethod
    def get_all_country_codes():
        return searches.country_codes().set_index('Country', drop=True)

    @staticmethod
    def get_region_inflation_forecast(region):
        latest_release = all_releases()[-1]
        path, url = download(latest_release[0], latest_release[1])
        iso3 = WEO(path).iso_code3(region)
        return WEO(path).inflation().get(iso3) / 100

    @staticmethod
    def get_region_interest_rates(region):

        code = IMFQuery.get_country_code_from_name(region)
        currency = IMFQuery.get_currency_iso(region)

        year = str(dt.date.today().year)

        url = 'http://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/IFS/M.{}.FPOLM_PA+FID_PA+FITB_PA.?startPeriod=1900&endPeriod={}'.format(code, year)
        data = requests.get(url).json()
        series_list = data.get('CompactData').get('DataSet').get('Series')

        if series_list is None:
           print('No data returned for {}'.format(region))
           return

        if not isinstance(series_list, list):
           series_list = [series_list]

        df_ = pd.DataFrame()
        for s in series_list:
            if isinstance(s.get('Obs'), list):
                try:
                    new_df = pd.DataFrame(s.get('Obs'))
                    new_df = new_df[['@TIME_PERIOD','@OBS_VALUE']]
                    dbticker = IMFQuery.get_ticker(region, s.get('@INDICATOR'))
                    name = IMFQuery.get_rate_name_from_ticker(s.get('@INDICATOR'))
                    type = IMFQuery.get_rate_type_from_ticker(s.get('@INDICATOR'))
                    mat = IMFQuery.get_rate_tenor_from_ticker(s.get('@INDICATOR'))

                    new_df.columns = ['date', dbticker]
                    new_df = new_df.set_index('date', drop=True)
                    new_df.index = pd.to_datetime([x + '-01' for x in new_df.index])
                    col = (dbticker, region + ' ' + name, region, 'Interest Rate', 'IMF', 'IMF', s.get('@INDICATOR'), currency, mat, type, 'IR')
                    new_df.columns = pd.MultiIndex.from_tuples([col])

                    df_ = pd.concat((df_, new_df), axis=1)
                except:
                    print('failed {} {}'.format(region, s.get('@INDICATOR')))
        return df_.copy()


if __name__ == "__main__":

    regions = ['United States']
    for region in regions:
        print(region)
        rfr = IMFQuery.get_region_inflation_forecast(region)
        rfrs = pd.concat((rfrs, rfr), axis=1)
    rfrs.to_clipboard()





