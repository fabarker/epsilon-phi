import pandas as pd
import numpy as np
from epsilonPhi.core.dataModel.dataSources.riskFreeRates.RiskFreeRates import CRiskFreeRate
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.core.factor.Factor import CConstructedFactor
from epsilonPhi.core.factor.factorMgr import CFactorMgr
from epsilonPhi.core.dataModel.enums.Factor import FACTOR

ts_type: TimeSeriesType = TimeSeriesType.LEVELS

class CTerm(CConstructedFactor):

    def __init__(self, schema):
        super(CConstructedFactor, self).__init__(schema=schema)
        self.construct_factor(FACTOR.EQUITY_GLOBAL_ISG)

    def construct_factor(self, name):

        _MARKETS = [('BMUS10Y', 'United States', 45.3),
                    ('BMBD10Y', 'Germany', 8.1),
                    ('BMIT10Y', 'Italy', 5.2),
                    ('BMFR10Y', 'France', 7),
                    ('BMUK10Y', 'United Kingdom', 6.6),
                    ('BMJP10Y', 'Japan', 23.9),
                    ('BMCN10Y', 'Canada', 3.9)]

        df = pd.DataFrame()
        for ticker in _MARKETS:
            totr = GlobalDataSource().get_time_series_data_from_ticker(ticker[0], 'RI')
            rfr = CRiskFreeRate.get_risk_free_for_region(ticker[1]).get_levels()

            common_dates = np.intersect1d(totr.get_bmonth_ends().index,
                                          rfr.get_bmonth_ends().index)

            df = pd.concat((df, (totr.loc[common_dates].get_returns() -
                                      rfr.loc[common_dates].get_returns().values) * ticker[-1]/100), axis=1)
        self._cast_derived_class(df.mean(axis=1).to_frame('Term'))

if __name__ == "__main__":

    from epsilonPhi.core.schema.Schema import ContextCreator
    schema = ContextCreator(currency='GBP',
                            start_date=pd.to_datetime('1-Nov-1983') + pd.tseries.offsets.BMonthEnd(1),
                            end_date='31-Dec-2022').create_context()

    fac = CTerm(schema)


