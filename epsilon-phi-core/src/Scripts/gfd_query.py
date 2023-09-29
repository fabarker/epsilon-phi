import requests
import pandas as pd
import datetime
import os
import getpass

__author__ = 'barkfr'

class GlobalFinancialData(object):

    _url = 'https://api.globalfinancialdata.com/login/'

    def __init__(self, username=None, password=None):

        self._username = username
        self._password = password
        self._parameters = {'username': username,
                            'password': password}
        self.get_token()

    def get_token(self):

        resp = requests.post(url, data=parameters)

        # check for unsuccessful API returns
        if resp.status_code != 200:
            raise ValueError('GFD API request failed with HTTP status code %s' % resp.status_code)

        json_content = resp.json()
        os.environ['GFD_API_TOKEN'] = json_content['token'].strip('"')


    def is_token_valid(self):
        pass

    def get_dfd_data(self,
                    symbol,
                    periodicity='Daily',
                    total_return =False,
                    close_only=False,
                    start_date='',
                    end_date=''):

        if isinstance(symbol, str):
           symbol = [symbol]
        symbol = ','.join(symbol)

        pars = {}
        pars['token'] = os.environ['GFD_API_TOKEN']
        pars['symbol'] = symbol
        pars['periodicity'] = periodicity
        pars['totalreturn'] = total_return
        pars['closeonly'] = close_only
        pars['startdate'] = start_date
        pars['endate'] = end_date

        # series API call
        r = requests.post(self.url, data=pars)
        # print(r.content)

        # extract pricing data and assigns it to a pandas dataframe. export data to a CSV file in the local directory
        data = pd.DataFrame(r.json()['price_data'])
        return  data[['series_id', 'date', 'open', 'high', 'low', 'close', 'openint', 'volume', 'total_return']]

if __name__ == "__main__":

    gfd = GlobalFinancialData('francis.barker@gs.com', 'isg123')