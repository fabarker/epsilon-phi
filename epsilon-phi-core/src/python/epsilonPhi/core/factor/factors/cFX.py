from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.dataModel.dataSources.vendor.PastorStambaugh import Liquidity
from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.factor.Factor import CConstructedFactor
from epsilonPhi.core.dataModel.enums.Factor import FACTOR
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource

ts_type: TimeSeriesType = TimeSeriesType.LEVELS
gds = GlobalDataSource()

sessionMgr = SessionMgr()
session = sessionMgr.getSessionFactory()

class cFX(CConstructedFactor):

    def __init__(self, schema):
        super(CConstructedFactor, self).__init__(schema=schema)
        self.construct_factor(FACTOR.CARRY_GLOBAL_ISG)

    