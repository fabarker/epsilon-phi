from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.dataModel.dataSources.vendor.PastorStambaugh import Liquidity
from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.factor.Factor import CConstructedFactor
from epsilonPhi.core.dataModel.enums.Factor import FACTOR
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource

ts_type: TimeSeriesType = TimeSeriesType.LEVELS
gds = GlobalDataSource()
_EXCHANGE_CODES = ['NY', 'AX']

sessionMgr = SessionMgr()
session = sessionMgr.getSessionFactory()

class cLiquidity(CConstructedFactor):

    def __init__(self, schema):
        super(CConstructedFactor, self).__init__(schema=schema)
        self.construct_factor(FACTOR.LIQUIDITY_US_PASTOR_STAMBAUGH)

    def get_equity_tickers(self):
        q = session.query(EquitySpec.ticker).filter(EquitySpec.exchange_code.in_(_EXCHANGE_CODES))
        res = sessionMgr.query_format_df(q).values
        return list(sessionMgr.query_format_df(q).values.flatten())

    def get_non_traded_liquidity_factor_innovations(self):
        return Liquidity.get_liquidity_innovations()

    def get_Pastor_Stambaugh_liquidity_factor(self):
        return Liquidity.get_liquidity_factor()

    def get_data_for_regressions(self, dates):
        pass

    def replicate_factor(self):

        LIQ = self.get_non_traded_liquidity_factor_innovations()

        # 3 Year Rolling Regressions
        T = len(LIQ)
        for i in range(T - 3*12):
            X = LIQ[i:i+3*12]
            y = self.get_data_for_regressions(X.index)

    def construct_factor(self, name):

        df_ = self.get_Pastor_Stambaugh_liquidity_factor()
        df_.columns = ['Liquidity']
        self._cast_derived_class(df_)

if __name__ == "__main__":
    from epsilonPhi.core.schema.Schema import ContextCreator

    schema = ContextCreator(currency='GBP',
                            start_date='31-Dec-1999',
                            end_date='31-Dec-2022').create_context()

    self = cLiquidity(schema)
