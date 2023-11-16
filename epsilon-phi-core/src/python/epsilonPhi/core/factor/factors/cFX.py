from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.dataModel.enums.Database import PriceQuote
from epsilonPhi.ep_strategies.fx.factor import Factor as Carry
from epsilonPhi.ep_strategies.fx.factor import Signals
from epsilonPhi.core.dataModel.alchemist.DataModel import *
import datetime as dt
from epsilonPhi.core.factor.Factor import CConstructedFactor
from epsilonPhi.core.dataModel.enums.Factor import FACTOR
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource

ts_type: TimeSeriesType = TimeSeriesType.LEVELS
gds = GlobalDataSource()

sessionMgr = SessionMgr()
session = sessionMgr.getSessionFactory()

class cFX(CConstructedFactor):

    _CURRENCIES = ['EUR', 'DEM', 'CHF', 'NZD', 'AUD', 'CAD', 'GBP', 'NLG', 'BEF', 'ATS', 'FRF', 'ESP', 'ITL', 'NOK', 'DKK', 'SEK']
    _BASE_CURRENCY = 'USD'
    _CURRENCY_PAIRS = [x + '/USD' for x in _CURRENCIES]
    _START_DATE = dt.date(year=1983, month=10, day=31)
    _END_DATE = dt.date(year=2023, month=10, day=31)
    _STRATEGY_TYPE = 'HML'

    def __init__(self, schema):
        super(CConstructedFactor, self).__init__(schema=schema)
        self.construct_factor(FACTOR.CARRY_GLOBAL_ISG)

    def get_carry(self, currency_pairs):
        return Signals.get_CAR(currency_pairs, '1m')

    def construct_factor(self, name):

        signal_df = self.get_carry(self._CURRENCY_PAIRS)
        CAR = Carry(self._START_DATE, self._END_DATE, self._schema.frequency)

        # Set the strategy parameters
        CAR.rebalancing_frequency = Frequency.BUSINESS_MONTHLY
        CAR.set_signal(signal_df)
        CAR.price_quote_type = PriceQuote.BID
        CAR.number_of_portfolios = 5
        CAR.run_strategy()

        df_ = CAR.PnLCurve.get(cFX._STRATEGY_TYPE).fillna(1).pct_change().dropna().to_frame('FX')
        self._cast_derived_class(df_)


if __name__ == "__main__":
    from epsilonPhi.core.schema.Schema import ContextCreator

    schema = ContextCreator(currency='GBP',
                            start_date='31-Dec-1999',
                            end_date='31-Dec-2022').create_context()

    self = cFX(schema)

    