import pandas as pd

from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.core.factor.Factor import CConstructedFactor
from epsilonPhi.core.factor.factorMgr import CFactorMgr
from epsilonPhi.core.dataModel.enums.Factor import FACTOR

ts_type: TimeSeriesType = TimeSeriesType.LEVELS

class CEquity(CConstructedFactor):
    def __init__(self, schema):
        super(CConstructedFactor, self).__init__(schema=schema)
        self.construct_factor(FACTOR.EQUITY_GLOBAL_ISG)

    def construct_factor(self, name):

        # Dollar Equity Market
        EQ_D = self._factorMgr.get_asset_by_name('MSWRLD$')
        df_d = EQ_D.get_excess_return_df()

        # Local Equity Market
        EQ_L = self._factorMgr.get_asset_by_name('MSWRLD$')
        df_l = EQ_L.get_excess_return_df()

        df = 0.5 * (df_l + df_d)
        df.columns = ['Equity']
        self._cast_derived_class(df)

if __name__ == "__main__":

    from epsilonPhi.core.schema.Schema import ContextCreator
    schema = ContextCreator(currency='GBP',
                            start_date='31-Dec-1999',
                            end_date='31-Dec-2022').create_context()

    fac = CEquity(schema)






