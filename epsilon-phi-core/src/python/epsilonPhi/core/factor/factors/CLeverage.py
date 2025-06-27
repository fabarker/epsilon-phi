from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType
from epsilonPhi.core.dataModel.dataSources.vendor.HeKellyManela import Intermediary
from epsilonPhi.core.timeSeries.timeSeriesMain import CSlice
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.dataModel.enums.Factor import FACTOR
from epsilonPhi.core.factor.Factor import CFactor


class CLeverage(CFactor):

    def __init__(self,
                 data,
                 ts_type=TimeSeriesType.RETURNS,
                 returns_type=ReturnsType.SIMPLE,
                 **kwargs):

        super(CLeverage, self).__init__(data=data,
                                         ts_type=ts_type,
                                         returns_type=returns_type,
                                         **kwargs)



    @staticmethod
    def get_He_Kelly_Manela_capital_ratio_factor():
        return Intermediary.get_intermediary_value_weighted_investment_return()

    @staticmethod
    def construct_factor(frequency):

        if frequency.obs_per_year() > Frequency.MONTHLY.obs_per_year():
           raise ValueError('Error - PS Liquidity only available at monthly or higher frequencies')

        df_ = CLeverage.get_He_Kelly_Manela_capital_ratio_factor()
        return CSlice(data=df_.values.flatten(), index=df_.index,  name=FACTOR.LEVERAGE_US_HKM.name, ts_type=TimeSeriesType.RETURNS)

if __name__ == "__main__":

    self = CLeverage.construct_factor(Frequency.BUSINESS_MONTHLY)