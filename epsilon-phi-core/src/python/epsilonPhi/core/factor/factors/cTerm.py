from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.dataModel.enums.Factor import FACTOR
from epsilonPhi.core.dataModel.dataSources.curves.yieldCurve.YieldCurve import YieldCurve
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.factor.Factor import CFactor
import pandas as pd

ts_type: TimeSeriesType = TimeSeriesType.LEVELS

_REGIONS = {}
_REGIONS['France'] = ('BMFR10Y', 7.0)
_REGIONS['United States'] = ('BMUS10Y', 45.3)
_REGIONS['United Kingdom'] = ('BMUK10Y', 6.6)
_REGIONS['Italy'] = ('BMIT10Y', 5.2)
_REGIONS['Japan'] = ('BMJP10Y', 23.9)
_REGIONS['Canada'] = ('BMCN10Y', 3.9)
_REGIONS['Germany'] = ('BMBD10Y', 8.1)

gds = GlobalDataSource()

class cTerm(CFactor):
    def __init__(self,
                 data,
                 ts_type=TimeSeriesType.RETURNS,
                 returns_type=ReturnsType.SIMPLE,
                 **kwargs):

        super(cTerm, self).__init__(data=data,
                                    ts_type=ts_type,
                                    returns_type=returns_type,
                                    **kwargs)

    @staticmethod
    def get_regional_bond_excess_return(region, frequency):
        df_TR = cTerm.get_region_bond_return_series(region, frequency)
        df_rfr = gds.get_risk_free_rate_time_series(region).get_periodic_returns(frequency)
        return df_TR.subtract_over_common_dates(df_rfr)

    @staticmethod
    def get_region_bond_return_series(region, frequency):
        return YieldCurve.get_total_return_time_series(region, 10, frequency)

    @staticmethod
    def get_weight_for_region(region):
        return _REGIONS.get(region)[1] / 100

    @staticmethod
    def construct_factor(frequency):

        df = pd.DataFrame()
        for region in _REGIONS.keys():
            cntr = cTerm.get_regional_bond_excess_return(region, frequency) * cTerm.get_weight_for_region(region)
            df = pd.concat((df, cntr), axis=1)
        df_ = df.sum(axis=1, skipna=False).dropna()
        df_.name = FACTOR.TERM_GLOBAL_ISG.name
        return df_.copy()

if __name__ == "__main__":

    df_ = cTerm.construct_factor(Frequency.BUSINESS_MONTHLY)


