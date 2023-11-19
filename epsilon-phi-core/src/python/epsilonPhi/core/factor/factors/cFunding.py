import numpy as np
import pandas as pd
from scipy.stats import zscore
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries
from epsilonPhi.core.dataModel.dataSources.yieldCurve.YieldCurve import YieldCurve
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.timeSeries.regression import Regression
from epsilonPhi.core.factor.Factor import CFactor
from epsilonPhi.core.dataModel.enums.Factor import FACTOR
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource

ts_type: TimeSeriesType = TimeSeriesType.LEVELS
gds = GlobalDataSource()

US_FINANCIAL_ASSETS = 'US66XXXAA'
US_FINANCIAL_LIABILITES = 'US66XXXLA'
_CUTOFF = 1968

_REGRESSORS = ['FFS1B1', 'FFS1B2', 'FFS1B3', 'FFS2B1', 'FFS2B2',
               'FFS2B3', 'MOM_US_FF', 'XRUKG10DS', 'XRDEG10DS']



class cFunding(CFactor):
    def __init__(self, dataframe, ts_type=TimeSeriesType.RETURNS):
        super(cFunding, self).__init__(dataframe=dataframe, ts_type=ts_type)

    @staticmethod
    def get_leverage():
        return gds.get_dataframe_from_ticker(US_FINANCIAL_ASSETS) /\
            (gds.get_dataframe_from_ticker(US_FINANCIAL_ASSETS) -
             gds.get_dataframe_from_ticker(US_FINANCIAL_LIABILITES).values)

    @staticmethod
    def get_log_leverage():
        return np.log(cFunding.get_leverage())

    @staticmethod
    def get_normalized_log_leverage():
        return zscore(cFunding.get_log_leverage())

    @staticmethod
    def get_leverage_growth():
        LG = cFunding.get_normalized_log_leverage().diff(periods=1)
        return LG[LG.index.year >= _CUTOFF]

    @staticmethod
    def get_normalized_leverage_growth():
        return zscore(cFunding.get_leverage_growth())

    @staticmethod
    def get_risk_free_rate(ticker):
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

    @staticmethod
    def get_bond_total_return(ticker):

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

    @staticmethod
    def get_regressor(ticker):

        if ticker == 'MOM_US_FF':
            return gds.get_dataframe_from_ticker(ticker)
        elif ticker in ['XRUKG10DS', 'XRDEG10DS','XRUSG10DS', 'XRJPG10DS']:
            df = cFunding.get_bond_total_return(ticker)
            rf = cFunding.get_risk_free_rate(ticker)
        else:
            df = gds.get_dataframe_from_ticker(ticker)
            rf = cFunding.get_risk_free_rate(ticker)
        common_dates = np.intersect1d(df.index, rf.index)
        return df.loc[common_dates].pct_change() - rf.loc[common_dates].pct_change().values

    @staticmethod
    def get_regressor_panel():

        panel = pd.DataFrame()
        for regressor in _REGRESSORS:
            df = cFunding.get_regressor(regressor)
            panel = pd.concat((panel, df.resample('B').asfreq().fillna(0)), axis=1)
        levels = CTimeSeries(panel.dropna(), ts_type=TimeSeriesType.RETURNS).get_levels()
        return levels.get_bquarterly_returns(), levels.get_returns()

    @staticmethod
    def construct_factor(frequency):

        # Get regressor panel
        reg_panel, proj_panel = cFunding.get_regressor_panel()

        # Get leverage growth
        lev = cFunding.get_leverage_growth()

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

        factor.columns = [FACTOR.FUNDING_US_ISG.name]
        return factor.get_periodic_returns(frequency)



if __name__ == "__main__":

    df = cFunding.construct_factor(Frequency.BUSINESS_MONTHLY)





