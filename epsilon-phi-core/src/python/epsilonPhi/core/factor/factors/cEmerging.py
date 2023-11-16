from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.core.factor.Factor import CConstructedFactor
from epsilonPhi.core.dataModel.enums.Factor import FACTOR
import numpy as np
import pandas as pd

ts_type: TimeSeriesType = TimeSeriesType.LEVELS
gds = GlobalDataSource()

class cEmerging(CConstructedFactor):
    def __init__(self, schema):
        super(CConstructedFactor, self).__init__(schema=schema)
        self.construct_factor(FACTOR.EQUITY_EMERGING_ISG)

    def construct_factor(self, name):

        # Dollar Developed
        EQ_D = gds.get_total_return_series_from_ticker('MSWRLD$').get_periodic_returns(self._schema.frequency)

        # Dollar Emerging
        EM_D = gds.get_total_return_series_from_ticker('MSEMKF$').get_periodic_returns(self._schema.frequency)

        if self._schema.frequency.obs_per_year() <= Frequency.MONTHLY.obs_per_year():
           EM_D_BF = gds.get_total_return_series_from_ticker('EMEQGFD').get_periodic_returns(self._schema.frequency)
           EM_D_BF.columns = EM_D.columns
           EM_D = pd.concat((EM_D, EM_D_BF.loc[np.setdiff1d(EM_D_BF.index, EM_D.index)]), axis=0).sort_index()


        df = EM_D.subtract_over_common_dates(EQ_D)
        df.columns = ['Emerging']
        self._cast_derived_class(df)

if __name__ == "__main__":

    from epsilonPhi.core.schema.Schema import ContextCreator
    schema = ContextCreator(currency='GBP',
                            start_date='31-Dec-1999',
                            end_date='31-Dec-2022').create_context()

    fac = cEmerging(schema)
