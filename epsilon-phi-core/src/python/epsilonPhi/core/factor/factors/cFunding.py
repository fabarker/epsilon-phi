import numpy as np
import pandas as pd
from scipy.stats import zscore
import statsmodels.api as sm
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType
from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries
from epsilonPhi.core.dataModel.dataSources.curves.yieldCurve.YieldCurve import YieldCurve
from epsilonPhi.core.timeSeries.timeSeriesMain import CSlice
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.timeSeries.regression import Regression
from epsilonPhi.core.dataModel.dataSources.vendor.FamaFrench import FFDataReader
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

FFFactors = FFDataReader.get_funding_related_factors(Frequency.MONTHLY)

filepath_1 = '/Users/francisbarker/Repositories/Python/epsilon-phi/epsilon-phi-core/src/resources/templates/LEV.xlsx'
filepath_2 = '/Users/francisbarker/repo/epsilon-psi/epsilon-phi-core/src/resources/templates/LEV.xlsx'

try:
    AEM = pd.read_excel(filepath_1, index_col=0)
except:
    AEM = pd.read_excel(filepath_2, index_col=0)





class cFunding(CFactor):
    def __init__(self,
                 data,
                 ts_type=TimeSeriesType.RETURNS,
                 returns_type=ReturnsType.SIMPLE,
                 **kwargs):

        super(cFunding, self).__init__(data=data,
                                       ts_type=ts_type,
                                       returns_type=returns_type,
                                       **kwargs)

    @staticmethod
    def get_broker_dealer_total_assets():
        return gds.get_dataframe_from_ticker(US_FINANCIAL_ASSETS)

    @staticmethod
    def get_broker_dealer_total_liabilities():
        return gds.get_dataframe_from_ticker(US_FINANCIAL_LIABILITES)

    @staticmethod
    def get_broker_dealer_net_equity():
        return cFunding.get_broker_dealer_total_assets() - cFunding.get_broker_dealer_total_liabilities().values

    @staticmethod
    def get_leverage():
        return cFunding.get_broker_dealer_total_assets() / cFunding.get_broker_dealer_net_equity()

    @staticmethod
    def get_log_leverage():
        return np.log(cFunding.get_leverage())

    @staticmethod
    def get_log_leverage_growth():
        return cFunding.get_log_leverage().diff(periods=1).dropna()

    @staticmethod
    def get_seasonaly_adj_log_lev_growth():
        df = cFunding.get_log_leverage_growth()
        df['quarter'] = df.index.quarter

        start = df.index[df.index.get_loc('1968-03-29') - 10]
        tmp = df[df.index.year >= _CUTOFF]

        real_time_residuals = []
        for i, end in enumerate(tmp.index):
            df_subset = df.loc[start:end]
            quarter_dummies = pd.get_dummies(df_subset['quarter'], prefix='Q', drop_first=True).astype(float)
            X = sm.add_constant(quarter_dummies).astype(float)
            y = df_subset.iloc[:,0]
            model = sm.OLS(y, X).fit()
            residual = model.resid.iloc[-1]
            real_time_residuals.append(residual)
        return pd.DataFrame(real_time_residuals, columns=['SA_LLG'], index=tmp.index)


    @staticmethod
    def get_leverage_factor():
        df = cFunding.get_seasonaly_adj_log_lev_growth()
        df.columns = ['BDLEVG']
        return pd.concat((AEM, df[df.index>AEM.index.max()]), axis=0).sort_index()


    @staticmethod
    def get_normalized_leverage_factor():
        return zscore(cFunding.leverage_factor())

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
        tr = YieldCurve.get_total_return_time_series(region, 10, ts_type=TimeSeriesType.LEVELS, frequency='BM')
        tr.columns = [ticker]
        return tr

    @staticmethod
    def get_regressor(ticker):

        if ticker == 'MOM_US_FF':
            return FFFactors.get(ticker)
        elif ticker in ['XRUKG10DS', 'XRDEG10DS']:
            df = cFunding.get_bond_total_return(ticker)
            rf = cFunding.get_risk_free_rate(ticker)
        else:
            return FFFactors.get(ticker)
        common_dates = np.intersect1d(df.index, rf.index)
        return df.loc[common_dates].pct_change() - rf.loc[common_dates].pct_change().values

    @staticmethod
    def get_regressor_panel():

        panel = pd.DataFrame()
        for regressor in _REGRESSORS:
            df = cFunding.get_regressor(regressor)
            panel = pd.concat((panel, df.resample('BM').asfreq().fillna(0)), axis=1)
        levels = CTimeSeries(panel.dropna(), ts_type=TimeSeriesType.RETURNS).get_levels()
        return levels.get_bquarterly_returns(), levels.get_returns()

    @staticmethod
    def construct_factor(frequency):

        # Get regressor panel
        reg_panel, proj_panel = cFunding.get_regressor_panel()

        # Get leverage growth
        lev = cFunding.get_leverage_factor()

        # Regression end dates
        date_range = pd.date_range('29-12-2000', '31-12-2022', freq='BY')

        factor = pd.DataFrame()
        for i, date in enumerate(date_range[:-1]):

            y_period = lev[lev.index.min():date]
            X = reg_panel[lev.index.min():date]
            reg = Regression.regress_against_factors(y_period, X)

            proj = proj_panel[:date_range[i+1]].dot(reg[1:])
            factor = pd.concat((factor, proj.loc[np.setdiff1d(proj.index, factor.index)]), axis=0)

        factor.columns = [FACTOR.FUNDING_US_ISG.name]
        df = CSlice(data=factor.values.flatten(), index=factor.index, name=FACTOR.FUNDING_US_ISG.name,
               ts_type=TimeSeriesType.RETURNS, returns_type=ReturnsType.SIMPLE)
        return df.get_periodic_returns(frequency)



if __name__ == "__main__":

    df = cFunding.construct_factor(Frequency.BUSINESS_MONTHLY)





