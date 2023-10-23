import pandas as pd
import requests
from io import StringIO
import datetime as dt

LIQUIDITY_FACTOR_URL = 'https://finance.wharton.upenn.edu/~stambaug/liq_data_1962_2022.txt'

class Liquidity(object):
    _df = None

    @staticmethod
    def get_liquidity_factor():
        df = Liquidity.get_df()
        return df[df.get('factor') > -99].get('factor').to_frame('PSLIQ')

    @staticmethod
    def get_aggregate_liquidity():
        df = Liquidity.get_df()
        return df.get('agg_liquidity').to_frame('AGGLIQ')

    @staticmethod
    def get_liquidity_innovations():
        df = Liquidity.get_df()
        return df.get('liquidity_innovations').to_frame('LIQINO')

    @staticmethod
    def get_df():
        if Liquidity._df is None:
           Liquidity.load_df()
        return Liquidity._df.copy()

    @staticmethod
    def load_df():

        if Liquidity._df is None:
            response = requests.get(LIQUIDITY_FACTOR_URL)
            data = StringIO('\n'.join(response.text.split('\n')[11:]))
            df = pd.read_csv(data, delim_whitespace=True, header=None,
                         names=['date', 'agg_liquidity', 'liquidity_innovations', 'factor'])
            df.date = df.date.apply(lambda x: dt.date(year=int(str(x)[0:4]), month=int(str(x)[4:]), day=1))
            df.date = df.date + pd.tseries.offsets.BMonthEnd(1)
            Liquidity._df = df.set_index('date', drop=True).copy()

    def replicate_factor(self):
        pass

if __name__ == "__main__":

    df = Liquidity.get_df()