from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.dataModel.dataSources.vendor.PastorStambaugh import Liquidity
from epsilonPhi.core.timeSeries.timeSeriesMain import CSlice
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.dataModel.enums.Factor import FACTOR
from epsilonPhi.core.factor.Factor import CFactor
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource

ts_type: TimeSeriesType = TimeSeriesType.LEVELS
gds = GlobalDataSource()
_EXCHANGE_CODES = ['NY', 'AX']

sessionMgr = SessionMgr()
session = sessionMgr.getSessionFactory()

class cLiquidity(CFactor):

    def __init__(self,
                 data,
                 ts_type=TimeSeriesType.RETURNS,
                 returns_type=ReturnsType.SIMPLE,
                 **kwargs):

        super(cLiquidity, self).__init__(data=data,
                                         ts_type=ts_type,
                                         returns_type=returns_type,
                                         **kwargs)

    @staticmethod
    def get_equity_tickers():
        q = session.query(EquitySpec.ticker).filter(EquitySpec.exchange_code.in_(_EXCHANGE_CODES))
        res = sessionMgr.query_format_df(q).values
        return list(sessionMgr.query_format_df(q).values.flatten())

    @staticmethod
    def get_non_traded_liquidity_factor_innovations():
        return Liquidity.get_liquidity_innovations()

    @staticmethod
    def get_Pastor_Stambaugh_liquidity_factor():
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

    @staticmethod
    def construct_factor(frequency):

        if frequency.obs_per_year() > Frequency.MONTHLY.obs_per_year():
           raise ValueError('Error - PS Liquidity only available at monthly or higher frequencies')

        df_ = cLiquidity.get_Pastor_Stambaugh_liquidity_factor()
        return CSlice(data=df_.values.flatten(), index=df_.index,  name=FACTOR.LIQUIDITY_US_PASTOR_STAMBAUGH.name, ts_type=TimeSeriesType.RETURNS)

if __name__ == "__main__":

    self = cLiquidity.construct_factor(Frequency.BUSINESS_MONTHLY)
