from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.core.timeSeries.timeSeriesMain import CSlice
from epsilonPhi.core.factor.Factor import CFactor
from epsilonPhi.core.dataModel.enums.Factor import FACTOR
import pandas as pd

ts_type: TimeSeriesType = TimeSeriesType.LEVELS
gds = GlobalDataSource()


class cCommodities(CFactor):
    def __init__(self, series, ts_type=TimeSeriesType.RETURNS):
        super(cCommodities, self).__init__(series=series, ts_type=ts_type)

    @staticmethod
    def construct_factor(frequency):

        # Dollar Equity Market
        TOTR_D = gds.get_total_return_series_from_ticker('GSCITOT').get_periodic_returns(frequency)
        rfr_D = gds.get_risk_free_rate_time_series('United States').get_periodic_returns(frequency)
        df = TOTR_D.subtract_over_common_dates(rfr_D)

        factor = CSlice(data=df.values.flatten(), index=df.index, name=FACTOR.COMMODITY_GLOBAL_ISG.name,
                        ts_type=TimeSeriesType.RETURNS)
        return factor.copy()

if __name__ == "__main__":

    fac = cCommodities.construct_factor(Frequency.BUSINESS_MONTHLY)






