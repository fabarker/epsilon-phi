from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries
from epsilonPhi.core.estimator.assetReturnEstimator import AssetReturnEstimator
from epsilonPhi.core.factor.factorMgr import CFactorMgr
import numpy as np

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
        self._asset_class = None
        self._universe = None
        self._provider = None

    def create_new_object(self, data=None, attributes=None, ts_type=None, returns_type=None):
        newObj = CFactor(dataframe=data, ts_type=ts_type)
        newObj._schema = self._schema
        return newObj

    def get_historical_risk_premia(self):
        return np.mean(self.values) * self._schema.annualizing_factor

    def get_historical_Sharpe(self):
        return (np.mean(self.values) * self._schema.annualizing_factor) /\
                    np.std(self.values, ddof=1) * np.sqrt(self._schema.annualizing_factor)

    def get_historical_volatility(self):
        return np.std(self.values, ddof=1) * np.sqrt(self._schema.annualizing_factor)

    def get_Sharpe_ratio(self):
        pass

    def get_risk_premium(self):
        pass


class CConstructedFactor(CFactor):

    def __init__(self, name, schema):
        super(CConstructedFactor, self).__init__(schema=schema)
        self.construct_factor(name)

    def _cast_derived_class(self, klass):
        super(CConstructedFactor, self).__init__(klass, self._schema)

    def construct_factor(self, name):
        pass




