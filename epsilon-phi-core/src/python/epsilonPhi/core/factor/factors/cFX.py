from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.dataModel.enums.Database import PriceQuote
from epsilonPhi.ep_strategies.fx.factor import Factor as Carry
from epsilonPhi.ep_strategies.fx.factor import Signals
from epsilonPhi.core.factor.Factor import CFactor
from epsilonPhi.core.dataModel.alchemist.DataModel import *
import datetime as dt
from epsilonPhi.core.dataModel.enums.Factor import FACTOR
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource

ts_type: TimeSeriesType = TimeSeriesType.LEVELS
gds = GlobalDataSource()

sessionMgr = SessionMgr()
session = sessionMgr.getSessionFactory()

class cFX(CFactor):

    _CURRENCIES = ['EUR', 'DEM', 'CHF', 'NZD', 'AUD', 'CAD', 'GBP', 'NLG', 'BEF', 'ATS', 'FRF', 'ESP', 'ITL', 'NOK', 'DKK', 'SEK']
    _BASE_CURRENCY = 'USD'
    _CURRENCY_PAIRS = [x + '/USD' for x in _CURRENCIES]
    _START_DATE = dt.date(year=1983, month=10, day=31)
    _END_DATE = dt.date(year=2023, month=10, day=31)
    _STRATEGY_TYPE = 'HML'

    def __init__(self,
                 data,
                 ts_type=TimeSeriesType.RETURNS,
                 returns_type=ReturnsType.SIMPLE,
                 **kwargs):

        super(cFX, self).__init__(data=data,
                                  ts_type=ts_type,
                                  returns_type=returns_type,
                                  **kwargs)

    @staticmethod
    def construct_factor(frequency):

        signal_df = Signals.get_CAR(cFX._CURRENCY_PAIRS, '1m')
        CAR = Carry(cFX._START_DATE, cFX._END_DATE, frequency)

        # Set the strategy parameters
        CAR.rebalancing_frequency = Frequency.BUSINESS_MONTHLY
        CAR.set_signal(signal_df)
        CAR.price_quote_type = PriceQuote.BID
        CAR.number_of_portfolios = 5
        CAR.run_strategy()

        df_lvls = CAR.PnLCurve.get(cFX._STRATEGY_TYPE).fillna(1)
        df_lvls.name = FACTOR.CARRY_GLOBAL_ISG.name
        return df_lvls.get_returns()


if __name__ == "__main__":

    df = cFX.construct_factor(Frequency.BUSINESS_MONTHLY)

    