from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.core.factor.Factor import CConstructedFactor
from epsilonPhi.core.dataModel.dataSources.yieldCurve.YieldCurve import YieldCurve
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.Factor import FACTOR
import pandas as pd
import numpy as np

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

class cTerm(CConstructedFactor):
    def __init__(self, schema):
        super(CConstructedFactor, self).__init__(schema=schema)
        self.construct_factor(FACTOR.EQUITY_GLOBAL_ISG)

    def get_regional_bond_excess_return(self, region):
        df_TR = self.get_region_bond_return_series(region)
        df_rfr = gds.get_risk_free_rate_time_series(region).get_periodic_returns(self._schema.frequency)
        return df_TR.subtract_over_common_dates(df_rfr)

    def get_region_bond_return_series(self, region):
        return YieldCurve.get_total_return_time_series(region, 10, self._schema.frequency)

    def get_weight_for_region(self, region):
        return _REGIONS.get(region)[1] / 100

    def construct_factor(self, name):

        df = pd.DataFrame()
        for region in _REGIONS.keys():
            cntr = self.get_regional_bond_excess_return(region) * self.get_weight_for_region(region)
            df = pd.concat((df, cntr), axis=1)

        df_ = df.sum(axis=1, skipna=False).dropna().to_frame('Term')
        self._cast_derived_class(df_)

if __name__ == "__main__":

    from epsilonPhi.core.schema.Schema import ContextCreator
    schema = ContextCreator(currency='GBP',
                            start_date='30-Nov-1983',
                            end_date='31-Dec-2022',
                            frequency='M').create_context()

    self = cTerm(schema)


