from imfpy import searches, retrievals, tools
from imfpy.retrievals import dots
import requests
import pandas as pd
import numpy as np
import datetime as dt


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
           return 'Trsy'
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
           return '1m'
        elif ticker.upper() == 'FID_PA':
           return 'ON'
        elif ticker.upper() == 'FPOLM_PA':
           return 'ON'
        elif ticker.upper() == 'FIDR_PA':
           return np.nan
        else:
            raise ValueError('Error {} not suppported'.format(ticker))


    @staticmethod
    def get_country_code_from_name(country_name):
        res = searches.country_search(country_name).get('Country Code', pd.DataFrame())
        if res.size == 1:
           return res.values[0]
        else:
           return None

    @staticmethod
    def get_all_country_codes():
        return searches.country_codes().set_index('Country', drop=True)

    @staticmethod
    def get_region_interest_rates(region):
        code = IMFQuery.get_country_code_from_name(region)

        year = str(dt.date.today().year)
        url = 'http://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/IFS/M.{}.FPOLM_PA+FID_PA+FITB_PA+FIDR_PA.?startPeriod=1900&endPeriod={}'.format(code, year)
        data = requests.get(url).json()
        series_list = data.get('CompactData').get('DataSet').get('Series')

        df_ = pd.DataFrame()
        for s in series_list:
            if isinstance(s.get('Obs'), list):
                new_df = pd.DataFrame(s.get('Obs'))
                new_df = new_df[['@TIME_PERIOD','@OBS_VALUE']]
                dbticker = IMFQuery.get_ticker(region, s.get('@INDICATOR'))
                name = IMFQuery.get_rate_name_from_ticker(s.get('@INDICATOR'))
                type = IMFQuery.get_rate_type_from_ticker(s.get('@INDICATOR'))
                mat = IMFQuery.get_rate_tenor_from_ticker(s.get('@INDICATOR'))

                new_df.columns = ['date', dbticker]
                new_df = new_df.set_index('date', drop=True)
                new_df.index = pd.to_datetime([x + '-01' for x in new_df.index])
                col = (dbticker, region + ' ' + name, region, 'Interest Rate', 'IMF', 'IMF', s.get('@INDICATOR'), np.nan, mat, type, 'IR')
                new_df.columns = pd.MultiIndex.from_tuples([col])

                df_ = pd.concat((df_, new_df), axis=1)
        return df_.copy()


if __name__ == "__main__":

    import time
    regions_df = pd.read_excel(r'C:\Users\fabar\OneDrive\Desktop\IMF Regions.xlsx', index_col=0)
    regions = regions_df.values.flatten()

    rfrs = pd.DataFrame()
    failed = list()
    for region in regions:
        time.sleep(5)
        try:
            rfr = IMFQuery.get_region_interest_rates(region)
            rfrs = pd.concat((rfrs, rfr), axis=1)
        except:
            print('Error pulling rates data for region {}'.format(region))
            failed.extend([region])
    rfrs.to_clipboard()





