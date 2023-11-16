import numpy as np
import pandas as pd
from scipy.stats import zscore
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries
from epsilonPhi.core.dataModel.dataSources.yieldCurve.YieldCurve import YieldCurve
from epsilonPhi.core.timeSeries.regression import Regression
from epsilonPhi.core.factor.Factor import CConstructedFactor
from epsilonPhi.core.dataModel.enums.Factor import FACTOR
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource

ts_type: TimeSeriesType = TimeSeriesType.LEVELS
gds = GlobalDataSource()

US_FINANCIAL_ASSETS = 'US66XXXAA'
US_FINANCIAL_LIABILITES = 'US66XXXLA'
_CUTOFF = 1968

_REGRESSORS = ['FFS1B1', 'FFS1B2', 'FFS1B3', 'FFS2B1', 'FFS2B2',
               'FFS2B3', 'MOM_US_FF', 'XRUKG10DS', 'XRDEG10DS']



class cFunding(CConstructedFactor):
    def __init__(self, schema):
        super(CConstructedFactor, self).__init__(schema=schema)
        self.construct_factor(FACTOR.FUNDING_US_ISG)

    def get_leverage(self):
        return gds.get_dataframe_from_ticker(US_FINANCIAL_ASSETS) /\
            (gds.get_dataframe_from_ticker(US_FINANCIAL_ASSETS) -
             gds.get_dataframe_from_ticker(US_FINANCIAL_LIABILITES).values)

    def get_log_leverage(self):
        return np.log(self.get_leverage())

    def get_normalized_log_leverage(self):
        return zscore(self.get_log_leverage())

    def get_leverage_growth(self):
        LG = self.get_normalized_log_leverage().diff(periods=1)
        return LG[LG.index.year >= _CUTOFF]

    def get_normalized_leverage_growth(self):
        return zscore(self.get_leverage_growth())

    def get_risk_free_rate(self, ticker):
        if ticker in ['FFS1B1', 'FFS1B2', 'FFS1B3', 'FFS2B1', 'FFS2B2', 'FFS2B3', 'MOM_US_FF','XRUSG10DS']:
            df = gds.get_risk_free_rate_time_series('United States')
        elif ticker in ['XRUKG10DS']:
            df = gds.get_risk_free_rate_time_series('United Kingdom')
        elif ticker in ['XRDEG10DS']:
            df = gds.get_risk_free_rate_time_series('Germany')
        elif ticker in ['XRJPG10DS']:
            df = gds.get_risk_free_rate_time_series('Japan')
        else:
            raise Exception('Error - ticker not supported')

        return CTimeSeries(df, ts_type=TimeSeriesType.RETURNS).get_levels()

    def get_bond_total_return(self, ticker):

        if ticker == 'XRUKG10DS':
            region = 'United Kingdom'
        elif ticker == 'XRDEG10DS':
            region = 'Germany'
        elif ticker == 'XRJPG10DS':
            region = 'Japan'
        elif ticker == 'XRUSG10DS':
            region = 'United States'
        else:
            raise Exception('Error - region {} not supported'.format(ticker))
        return YieldCurve.get_total_return_time_series(region, 10, ts_type=TimeSeriesType.LEVELS)

    def get_regressor(self, ticker):

        if ticker == 'MOM_US_FF':
            return gds.get_dataframe_from_ticker(ticker)
        elif ticker in ['XRUKG10DS', 'XRDEG10DS','XRUSG10DS', 'XRJPG10DS']:
            df = self.get_bond_total_return(ticker)
            rf = self.get_risk_free_rate(ticker)
        else:
            df = gds.get_dataframe_from_ticker(ticker)
            rf = self.get_risk_free_rate(ticker)
        common_dates = np.intersect1d(df.index, rf.index)
        return df.loc[common_dates].pct_change() - rf.loc[common_dates].pct_change().values

    def get_regressor_panel(self):

        panel = pd.DataFrame()
        for regressor in _REGRESSORS:
            df = self.get_regressor(regressor)
            panel = pd.concat((panel, df.resample('B').asfreq().fillna(0)), axis=1)
        levels = CTimeSeries(panel.dropna(), ts_type=TimeSeriesType.RETURNS).get_levels()
        return levels.get_bquarterly_returns(), levels.get_returns()

    def construct_factor(self, name):

        # Get regressor panel
        reg_panel, proj_panel = self.get_regressor_panel()

        # Get leverage growth
        lev = self.get_leverage_growth()

        # Regression end dates
        date_range = pd.date_range('29-12-2000', '31-12-2022', freq='BY')

        factor = pd.DataFrame()
        for date in date_range:

            y = lev[lev.index.min():date]
            X = reg_panel[lev.index.min():date]
            reg = Regression.regress_against_factors(y, X)
            weights = reg[1:] / np.sum(reg[1:])

            proj = proj_panel[:date].dot(weights)
            factor = pd.concat((factor, proj.loc[np.setdiff1d(proj.index, factor.index)]), axis=0)

        factor.columns = ['Funding']
        self._cast_derived_class(factor.get_periodic_returns(self._schema.frequency))



if __name__ == "__main__":

    from epsilonPhi.core.schema.Schema import ContextCreator
    schema = ContextCreator(currency='GBP',
                            start_date='31-Dec-1999',
                            end_date='31-Dec-2022').create_context()





