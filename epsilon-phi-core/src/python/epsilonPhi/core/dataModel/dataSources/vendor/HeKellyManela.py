import pandas as pd
import requests
from io import StringIO
import datetime as dt

CAPITAL_RATIO_FACTOR_URL = "https://zhiguohe.net/wp-content/uploads/2024/07/He_Kelly_Manela_Factors_monthly_240723.csv"

class Intermediary(object):
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
    def get_capital_risk_factor():
        df = Intermediary.get_df()
        return df.get('liquidity_innovations').to_frame('LIQINO')

    @staticmethod
    def get_df():
        if Intermediary._df is None:
           Intermediary.load_df()
        return Intermediary._df.copy()

    @staticmethod
    def load_df():

        if Intermediary._df is None:
            df = pd.read_csv(CAPITAL_RATIO_FACTOR_URL, index_col=0)
            df.index = pd.to_datetime(df.index, format="%Y%m") + pd.tseries.offsets.BMonthEnd(0)
            Intermediary._df = df.copy()


def replicate_factor(self):
        pass

if __name__ == "__main__":

    df = Intermediary.get_df()