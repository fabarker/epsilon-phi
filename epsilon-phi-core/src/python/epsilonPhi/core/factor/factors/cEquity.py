from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.core.factor.Factor import CFactor
from epsilonPhi.core.dataModel.enums.Factor import FACTOR
import numpy as np
import pandas as pd

ts_type: TimeSeriesType = TimeSeriesType.LEVELS
gds = GlobalDataSource()


class cEquity(CFactor):
    def __init__(self, dataframe, ts_type=TimeSeriesType.RETURNS):
        super(cEquity, self).__init__(dataframe=dataframe, ts_type=ts_type)


    @staticmethod
    def construct_factor(frequency):

        # Dollar Equity Market
        EQ_D = gds.get_total_return_series_from_ticker('MSWRLD$').get_periodic_returns(frequency)
        rfr_D = gds.get_risk_free_rate_time_series('United States').get_periodic_returns(frequency)
        df_D = EQ_D.subtract_over_common_dates(rfr_D)

        # Local Equity Market
        EQ_L = gds.get_total_return_series_from_ticker('MSWRLDL').get_periodic_returns(frequency)
        rfr_L = gds.get_risk_free_rate_time_series('World').get_periodic_returns(frequency)
        df_L = EQ_L.subtract_over_common_dates(rfr_L)

        df = 0.5 * df_L.addition_over_common_dates(df_D)
        df.columns = [FACTOR.EQUITY_GLOBAL_ISG.name]
        return df.copy()

if __name__ == "__main__":

    fac = cEquity(schema)






