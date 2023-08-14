from epsilonPhi.core.utils.ExcelUtils import ExcelUtils
from epsilonPhi.core.utils.FrameUtils import FrameUtils
from sqlalchemy import func
from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from enum import Enum
import numpy as np
import re

nan = pd.pandas._libs.tslibs.nattype.NaTType
session = SessionMgr().getSessionFactory()

class ISO_to_region(object):
    map = {'USD':'United States', 'EUR':'European Union', 'GBP':'United Kingdom', 'JPY':"Japan", 'DKK': 'Denmark',
           'CAD':'Canada', 'SEK':'Sweden', 'CHF':'Switzerland', 'AUD': 'Australia', 'NOK':'Norway', 'NZD':'New Zealand'}

    @staticmethod
    def region_from_iso(currency_iso):
        return ISO_to_region.map.get(currency_iso)
class TickerUtils(Enum):
    regex = r'[A-Z]{2} Equity| Curncy| Comdty| Index|_|[A-Z]{2}\d{3}'


class Bloomberg(object):
    pass
    """
    This class manages data sourced from bloomberg.
    
    Currently supports
    1. Data in excel pulled from BBG via bdh function
    2. Data from bloomberg API (blpapi)

    Returns:
           dict: [(Ticker, field)] = {data: pd.Dataframe, longName: str}
    """

    @staticmethod
    def bdh_load_from_excel(fullfile_path):

        raw = ExcelUtils.xlsread_sheets(fullfile_path)
        df = pd.DataFrame()
        for sheet in raw.keys():
            new_df = Bloomberg.process_xls_bdh(raw.get(sheet))
            new_df.columns = FrameUtils.add_index_to_multi_index(new_df.columns, [sheet]*new_df.shape[1], 'SHEET')
            df = pd.concat((df, new_df), axis=1)
        return df

    @staticmethod
    def process_xls_bdh(df: pd.DataFrame) -> pd.DataFrame:

        df = df.dropna(how='all', axis=1)
        row, col = np.where(df == 'Dates')
        if row.size > 0:
            dta = df.iloc[row[0]+1:,:]
            dta = dta.set_index(0, drop=True)
            dta.index.name = 'Dates'
            dta.index = pd.to_datetime(dta.index)
        else:
            raise ValueError('Error - cant find Dates row in sheet...')

        attributes = [x[0].upper() for x in df.iloc[0:row[0], col].dropna().values if x[0].upper() not in ['START DATE','END DATE']]

        atts = list()
        for att in attributes:
            att_row, att_col = np.where(df == att)
            atts.extend([list(df.iloc[att_row[0], att_col[0] + 1:].values)])
        dta.columns = pd.MultiIndex.from_tuples(list(zip(*atts)))
        dta.columns = dta.columns.set_names(attributes)

        if 'CRNCY' in dta.columns.names:
            regions = [ISO_to_region.region_from_iso(x) for x in dta.columns.get_level_values('CRNCY')]
            dta.columns = FrameUtils.add_index_to_multi_index(dta.columns, regions, 'REGION')

        if 'FIELD' in dta.columns.names:
            flds = dta.columns.get_level_values('FIELD')
            bbg_flds = [x.replace(' Price', '').replace('PX_', '').lower() for x in flds]
            dta.columns = FrameUtils.add_index_to_multi_index(dta.columns.droplevel('FIELD'), bbg_flds, 'FIELD')
        return dta.sort_index()

    @staticmethod
    def get_ticker_fld_locs(arr) -> pd.DataFrame:
        if not isinstance(arr, pd.DataFrame):
            arr = pd.DataFrame(arr)
        return arr.applymap(lambda x: isinstance(x, str) and
                                     bool(re.search(TickerUtils.regex.value, x)))

    @staticmethod
    def get_max_uid():
        return session.query(func.max(TimeSeriesSpec.uid)).scalar()

    @staticmethod
    def get_uid_from_ticker(ticker):
        res = session.query(TimeSeriesSpec).filter(TimeSeriesSpec.ticker == ticker).first()
        return res.uid if res else None

    @staticmethod
    def is_ticker_in_database(ticker):
        return bool(Bloomberg.get_uid_from_ticker(ticker))




if __name__ == "__main__":

    path = '/Users/francisbarker/Library/Mobile Documents/com~apple~CloudDocs/Data/Bloomberg/Rates/Rates.xlsx'
    df = Bloomberg.bdh_load_from_excel()











