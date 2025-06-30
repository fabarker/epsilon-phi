from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType
from epsilonPhi.core.timeSeries.timeSeriesMain import CSlice
from epsilonPhi.core.factor.Factor import CFactor
from epsilonPhi.core.dataModel.enums.Factor import FACTOR
import numpy as np
import pandas as pd

ts_type: TimeSeriesType = TimeSeriesType.LEVELS
gds = GlobalDataSource()


class CCapm(CFactor):
    def __init__(self,
                 data,
                 ts_type=TimeSeriesType.RETURNS,
                 returns_type=ReturnsType,
                 **kwargs):

        super(CCapm, self).__init__(data=data,
                                      ts_type=ts_type,
                                      returns_type=returns_type,
                                      **kwargs)


    @staticmethod
    def construct_factor(frequency):

        # Dollar Equity Market
        EQ_D = gds.get_total_return_series_from_ticker('MSWRLD$').get_periodic_returns(frequency, ReturnsType.SIMPLE)
        rfr_D = gds.get_risk_free_rate_time_series('United States').get_periodic_returns(frequency, ReturnsType.SIMPLE)
        df_D = EQ_D.subtract_over_common_dates(rfr_D)

        # Global Agg (Hedged USD)
        FI_U = gds.get_total_return_series_from_ticker('LHMGAGH').get_periodic_returns(frequency, ReturnsType.SIMPLE)
        df_F = FI_U.subtract_over_common_dates(df_D)

        df = 0.5 * df_F.addition_over_common_dates(df_D)
        factor = CSlice(data=df.values.flatten(), index=df.index, name=FACTOR.CAPM_BONDS_EQUITY.name,
                    ts_type=TimeSeriesType.RETURNS, returns_type=ReturnsType.SIMPLE)
        return factor.copy()

if __name__ == "__main__":

    fac = CCapm.construct_factor(Frequency.BUSINESS_MONTHLY)