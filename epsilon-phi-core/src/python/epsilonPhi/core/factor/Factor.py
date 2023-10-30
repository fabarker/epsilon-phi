from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries
from epsilonPhi.core.factor.factorMgr import CFactorMgr

ts_type: TimeSeriesType = TimeSeriesType.LEVELS


class CFactor(CTimeSeries):
    def __init__(self,
                 dataframe=None,
                 schema=None,
                 ts_type=TimeSeriesType.RETURNS
                 ):

        df_ = CFactorMgr._prepare_dataframe_for_factor(schema, dataframe, ts_type)
        super(CFactor, self).__init__(data=df_,
                                     ts_type=ts_type,
                                     )

        self._schema = schema
        self._factorMgr = CFactorMgr(schema)

        self._factor = None
        self._universe = None
        self._provider = None

    def _cast_derived_class(self, klass):
        super(CFactor, self).__init__(klass)

    def get_historical_risk_premia(self):
        pass
    def get_historical_Sharpe(self):
        pass
    def get_historical_volatility(self):
        pass
    def get_Sharpe_ratio(self):
        pass
    def get_risk_premium(self):
        pass


class CConstructedFactor(CFactor):

    def __init__(self, name, schema):
        super(CConstructedFactor, self).__init__(schema=schema)
        self.construct_factor(name)

    def construct_factor(self, name):
        pass




