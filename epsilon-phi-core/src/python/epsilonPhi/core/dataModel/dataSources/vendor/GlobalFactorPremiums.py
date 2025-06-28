import pandas as pd
import numpy as np

GLOBAL_FACTOR_PREMIUMS_URL = 'https://datarepository.eur.nl/ndownloader/files/26879918'


class GlobalFactorPremiums(object):
    _df = None

    @staticmethod
    def get_df():
        if GlobalFactorPremiums._df is None:
           GlobalFactorPremiums.load_df()
        return GlobalFactorPremiums._df.copy()

    @staticmethod
    def get_CARRY():
        df = GlobalFactorPremiums.get_df()
        return df.get('Carry')

    @staticmethod
    def get_TREND():
        df = GlobalFactorPremiums.get_df()
        return df.get('Trend')

    @staticmethod
    def get_MOMENTUM():
        df = GlobalFactorPremiums.get_df()
        return df.get('Momentum')

    @staticmethod
    def get_VALUE():
        df = GlobalFactorPremiums.get_df()
        return df.get('Value')

    @staticmethod
    def get_SEASONAL():
        df = GlobalFactorPremiums.get_df()
        return df.get('Seasonal')

    @staticmethod
    def get_BETTING_AGAINST_BETA():
        df = GlobalFactorPremiums.get_df()
        return df.get('Beting-Against-Beta (BAB)')

    @staticmethod
    def load_df():

        if GlobalFactorPremiums._df is None:
            df = (pd.read_excel(
                GLOBAL_FACTOR_PREMIUMS_URL,
                sheet_name='Data BSV_JFE')
                  .dropna(axis=1, how='all')
                  .dropna(axis=0, how='all')
                  )

            row, col = np.where(df.apply(lambda x: x == "Year"))
            raw = df.iloc[int(row+1):, int(col+2):].values
            cols = pd.DataFrame(df.values[int(row)-1:int(row+1), int(col+2):]).T.ffill()

            index = df.iloc[int(row+1):, int(col):int(col+2)].copy()
            index.columns = ['Year', 'Month']

            # Convert to numeric (in case they are strings)
            index['Year'] = pd.to_numeric(index['Year'], errors='coerce')
            index['Month'] = pd.to_numeric(index['Month'], errors='coerce')

            # Create datetime with day=1
            index['date'] = pd.to_datetime(dict(
                year=index['Year'],
                month=index['Month'],
                day=1)
            ) + pd.tseries.offsets.BMonthEnd(0)

            new_df = pd.DataFrame(
                raw,
                columns=pd.MultiIndex.from_frame(cols),
                index=index['date'])
            new_df.columns.names = ["factor", "asset_class"]

            # Save the cleaned dataframe
            GlobalFactorPremiums._df = new_df.copy()


if __name__ == "__main__":

    df = GlobalFactorPremiums.get_df()
    df = GlobalFactorPremiums.get_TREND()