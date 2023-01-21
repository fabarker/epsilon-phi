from pydatastream import Datastream as pyds
from epsilonPhi.utils.ListUtils import ListUtils as lutils
from epsilonPhi.dataModel.enums.FrequencyType import Frequency
from typing import Optional, Union
import numpy as np
import pandas as pd
import datetime as dt

class pyDatastream(object):

    _CREDENTIALS = list()
    # user name and password here
    _CREDENTIALS.extend([('','')])

    @staticmethod
    def pyds():
        return pyds(username=pyDatastream._CREDENTIALS[0][0],
                    password=pyDatastream._CREDENTIALS[0][1],
                    proxy=None)

    @staticmethod
    def fetch(tickers: Optional[Union[list, np.array, str]] = None,
              fields: Optional[Union[list, np.array, str]] = None,
              from_date: Optional[dt.datetime] = None,
              to_date: Optional[dt.datetime] = None,
              frequency: Optional[Frequency] = None) -> pd.DataFrame:

        chunks = lutils._nest_list([tickers] if isinstance(tickers, str) else list(tickers), 50)
        frames = pd.DataFrame()
        for chunk in chunks:
            res = pyDatastream.pyds().fetch(chunk,
                                            fields = fields,
                                            date_from= from_date,
                                            date_to = to_date,
                                            freq = frequency,
                                            always_multiindex=False).reset_index()
            frames = pd.concat((frames, res))
        return frames

    @staticmethod
    def get_currency_ISO_from_tickers(tickers: Union[list, str]) -> pd.DataFrame:
        chunks = lutils._nest_list([tickers] if isinstance(tickers, str) else list(tickers), 50)
        frames = pd.DataFrame()
        for chunk in chunks:
            res = pyDatastream.pyds().fetch(chunk,
                                            fields = 'ISOCUR',
                                            static = True)
            frames = pd.concat((frames, res))
        return frames

    @staticmethod
    def get_region(tickers):
        pass


if __name__ == "__main__":
    df = pyDatastream.fetch(['MSUSAML','MSUTDKL'],
                       fields='RI',
                       from_date=dt.date(day=1, month=12, year=2001),
                       to_date=dt.date(day=1, month=12, year=2022),
                       frequency='D')