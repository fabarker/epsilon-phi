import pandas as pd
import requests
from io import StringIO
import datetime as dt

CAPITAL_RATIO_FACTOR_URL = "https://zhiguohe.net/wp-content/uploads/2024/07/He_Kelly_Manela_Factors_monthly_240723.csv"

class Intermediary(object):
    _df = None


    @staticmethod
    def get_capital_risk_factor():
        df = Intermediary.get_df()
        return df.get('intermediary_capital_risk_factor').to_frame('CAPITAL_RISK_FACTOR')

    @staticmethod
    def get_intermediary_capital_ratio():
        df = Intermediary.get_df()
        return df.get('intermediary_capital_ratio').to_frame('CAPITAL_RATIO')

    @staticmethod
    def get_intermediary_value_weighted_investment_return():
        df = Intermediary.get_df()
        return df.get('intermediary_value_weighted_investment_return').to_frame('LEVFAC')

    @staticmethod
    def get_intermediary_leverage_ratio_squared():
        df = Intermediary.get_df()
        return df.get('intermediary_leverage_ratio_squared').to_frame('LEV_RATIO_SQ')

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
            df.index.names = ["date"]
            Intermediary._df = df.copy()


if __name__ == "__main__":

    df = Intermediary.get_df()