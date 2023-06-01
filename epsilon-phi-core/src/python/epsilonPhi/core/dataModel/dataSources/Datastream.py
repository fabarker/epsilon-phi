from pydatastream import Datastream as pyds
from epsilonPhi.core.utils.ListUtils import ListUtils as lutils
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.env.Env import DS_USERNAME, DS_PASSWORD
from typing import Optional, Union
import numpy as np
import pandas as pd
import datetime as dt

class Freq(object):

    freq = dict()
    freq[0] = 'D'
    freq[1] = 'W'
    freq[2] = 'M'
    freq[3] = 'Q'
    freq[4] = 'Y'

    @staticmethod
    def freq_from_enum(frequency: Frequency):
        return Freq.freq[frequency.value]

class pyDatastream(object):

    _CREDENTIALS = list()
    _CREDENTIALS.extend([('ZGOL433','ATLAS802'),
                         ('ZGSC304','TORCH863'),
                         ('ZGOL188','YOUNG902'),
                         ('ZGOL865','SOUTH366')])

    @staticmethod
    def pyds():
        return pyds(username=DS_USERNAME,
                    password=DS_PASSWORD,
                    proxy=None,
                    raise_on_error=True)

    @staticmethod
    def get_usage(months=12):
        return pyDatastream.pyds().usage_statistics(months=months)

    @staticmethod
    def fetch(tickers: Optional[Union[list, np.array, str]] = None,
              fields: Optional[Union[list, np.array, str]] = None,
              from_date: Optional[dt.date] = None,
              to_date: Optional[dt.date] = None,
              frequency: Optional[Frequency] = None) -> pd.DataFrame:

        chunks = lutils._nest_list([tickers] if isinstance(tickers, str) else list(tickers), 50)
        frames = pd.DataFrame()
        for chunk in chunks:
            try:
                res = pyDatastream.pyds().fetch(chunk,
                                                fields = fields,
                                                date_from= from_date,
                                                date_to = to_date,
                                                freq = frequency,
                                                always_multiindex=False).reset_index()
                frames = pd.concat((frames, res))
            except:
                pass
        return frames

    @staticmethod
    def get_currency_ISO_from_tickers(tickers: Union[list, str]) -> pd.DataFrame:
        chunks = lutils._nest_list([tickers] if isinstance(tickers, str) else list(tickers), 50)
        frames = pd.DataFrame()
        for chunk in chunks:
            res = pyDatastream.pyds().fetch(chunk,
                                            fields='ISOCUR',
                                            static=True)
            frames = pd.concat((frames, res))
        return frames

    @staticmethod
    def get_region(tickers):
        pass


if __name__ == "__main__":

    usage = pyDatastream.get_usage()

    df = pyDatastream.fetch(['MSUSAML','MSUTDKL'],
                       from_date=dt.date(day=31, month=12, year=1990),
                       frequency='M')