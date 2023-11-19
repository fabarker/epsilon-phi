from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.core.factor.Factor import CFactor
from epsilonPhi.core.dataModel.enums.Factor import FACTOR
import numpy as np
import pandas as pd

ts_type: TimeSeriesType = TimeSeriesType.LEVELS
gds = GlobalDataSource()

class cEmerging(CFactor):
    def __init__(self, dataframe, ts_type=TimeSeriesType.RETURNS):
        super(cEmerging, self).__init__(dataframe=dataframe, ts_type=ts_type)

    @staticmethod
    def construct_factor(frequency):

        # Dollar Developed
        EQ_D = gds.get_total_return_series_from_ticker('MSWRLD$').get_periodic_returns(frequency)

        # Dollar Emerging
        EM_D = gds.get_total_return_series_from_ticker('MSEMKF$').get_periodic_returns(frequency)

        if frequency.obs_per_year() <= Frequency.MONTHLY.obs_per_year():
           EM_D_BF = gds.get_total_return_series_from_ticker('EMEQGFD').get_periodic_returns(frequency)
           EM_D_BF.columns = EM_D.columns
           EM_D = pd.concat((EM_D, EM_D_BF.loc[np.setdiff1d(EM_D_BF.index, EM_D.index)]), axis=0).sort_index()


        df = EM_D.subtract_over_common_dates(EQ_D)
        df.columns = [FACTOR.EQUITY_EMERGING_ISG.name]
        return df.copy()

if __name__ == "__main__":

    fac = cEmerging.construct_factor(Frequency.BUSINESS_MONTHLY)
