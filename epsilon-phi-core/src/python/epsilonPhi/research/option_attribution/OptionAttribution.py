import pandas as pd
import numpy as np


class OptionAttributer(object):

    _DATA_PATH = '/Users/francisbarker/Desktop/SPX Options 1.csv'
    _DAYS_PER_YEAR = 365.25
    def __init__(self, start_date, end_date):

        self._start_date = pd.to_datetime(start_date)
        self._end_date = pd.to_datetime(end_date)
        self._process_data()

    @property
    def dates(self):
        return self._ds.index.get_level_values('date').unique()
    @property
    def unique_strikes(self):
        return np.array(self._ds.index.get_level_values('Z').unique()).reshape(1, -1)
    @property
    def unique_maturities(self):
        return np.array(self._ds.index.get_level_values('T').unique()).reshape(1, -1)
    @property
    def spot(self):
        return self._ds.get('s').reindex(self.dates).to_frame()
    @property
    def sig(self):
        return self._ds.get('sig').reindex(self.dates).to_frame()
    @property
    def dsig(self):
        return
    @property
    def sig_sq(self):
        return np.power(self.sig, 2)
    @property
    def A(self):
        return

    def _process_data(self):

        _df = pd.read_csv('/Users/francisbarker/Desktop/SPX Options 1.csv')
        _df['date'] = pd.to_datetime(_df['date'])
        _df = _df.set_index(['date','Z','T'], drop=True)
        self._ds = _df.copy()

if __name__ == "__main__":
    self = OptionAttributer('31-Dec-1990', '31-Dec-2025')
