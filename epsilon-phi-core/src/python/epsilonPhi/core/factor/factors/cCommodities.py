import pandas as pd

from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.core.factor.Factor import CConstructedFactor
from epsilonPhi.core.factor.factorMgr import CFactorMgr
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.Factor import FACTOR

ts_type: TimeSeriesType = TimeSeriesType.LEVELS
gds = GlobalDataSource()


class cCommodities(CConstructedFactor):
    def __init__(self, schema):
        super(CConstructedFactor, self).__init__(schema=schema)
        self.construct_factor(FACTOR.COMMODITY_GLOBAL_ISG)

    def construct_factor(self, name):

        # Dollar Equity Market
        TOTR_D = gds.get_total_return_series_from_ticker('GSCITOT').get_periodic_returns(self._schema.frequency)
        rfr_D = gds.get_risk_free_rate_time_series('United States').get_periodic_returns(self._schema.frequency)
        df = TOTR_D.subtract_over_common_dates(rfr_D)

        df.columns = ['Commodities']
        self._cast_derived_class(df)

if __name__ == "__main__":

    from epsilonPhi.core.schema.Schema import ContextCreator
    schema = ContextCreator(currency='GBP',
                            start_date='31-Dec-1999',
                            end_date='31-Dec-2022').create_context()

    fac = cCommodities(schema)






