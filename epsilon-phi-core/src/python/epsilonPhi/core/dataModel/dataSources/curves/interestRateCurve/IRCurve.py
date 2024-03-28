from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries, TimeSeriesType, ReturnsType
from epsilonPhi.core.dataModel.dataSources.curves.abstractCurve.AbstractCurve import AbstractCurve
from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.utils.FrameUtils import FrameUtils
import pandas as pd
import warnings

warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')

class IRCurve(AbstractCurve):

    _rate_cache = dict()
    _sessionMgr = SessionMgr()
    _session = _sessionMgr.getSessionFactory()

    def __init__(self, currency=None, region=None, type=None):

        if currency is None and region is None:
            raise ValueError('Error - must specify either currency or region')

        if currency is None:
            self._currency = self._sessionMgr.get_currency_from_region(region)
        else:
            self._currency = currency

        if region is None:
            self._region = self._sessionMgr.get_region_from_currency(currency)
        else:
            self._region = region

        self._type = type

        df = self.get_interest_rates()
        super(IRCurve, self).__init__(df)

    @property
    def maturities(self):
        return self._sessionMgr.get_interest_rate_maturities_for_region(self._region)

    def load_interest_rates(self):
        df_rfrs = self._sessionMgr.get_interest_rates_for_region(self._region, self.maturities)

        df = pd.DataFrame()
        for col in df_rfrs.columns:
            df_col = df_rfrs.get(col).dropna().to_frame(col).resample('B').ffill() / 100
            df = pd.concat((df, df_col), axis=1)

        df.columns.names = ['uid', 'ticker', 'tenor', 'type']
        mats = DateUtils.Rdate_to_mat(df.columns.get_level_values('tenor'))
        ts = CTimeSeries(df, ts_type=TimeSeriesType.RETURNS, returns_type=ReturnsType.SIMPLE)
        ts.columns = FrameUtils.add_level(df.columns, mats, 'maturity')

        if self._type is not None:
            _cols = np.isin(ts.columns.get_level_values('type'), self._type)
            ts = ts.iloc[:, _cols].dropna(how='all', axis=1)
        _rates = ts.T.groupby(level='maturity').mean().T
        IRCurve._rate_cache[self._region] = _rates.sort_index(axis=1)

    def get_interest_rates(self):
        if self._region not in IRCurve._rate_cache.keys():
            self.load_interest_rates()
        return IRCurve._rate_cache.get(self._region)

    def get_interest_rate_curve(self, maturities=None):
        return self.get_curve(maturities)


if __name__ == "__main__":

    self = IRCurve(region='United States', type=['Interbank'])
    curve = self.get_curve()

    np.timedelta64
