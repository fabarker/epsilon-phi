from epsilonPhi.core.dataModel.enums.Composites import CompositeRiskFreeRates
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.Database import FX
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr, TimeSeriesSpec
from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries, TimeSeriesType, ReturnsType
from epsilonPhi.core.utils.PickleUtils import PickleUtils
from epsilonPhi.core.utils.DateUtils import DateUtils
import datetime, os
import pandas as pd
import numpy as np
import platform

f_path_ = os.path.abspath(__file__)
if 'windows' in platform.system().lower():
    DIR_ = os.path.join(f_path_[:f_path_.find('epsilon-phi-core\\') + len('epsilon-phi-core\\')],
                        'src\\resources\\templates\\MSCI Index Construction.xlsx')
else:
    DIR_ = os.path.join(f_path_[:f_path_.find('epsilon-phi-core/') + len('epsilon-phi-core/')],
                        'src/resources/templates/MSCI Index Construction.xlsx')

class CRiskFreeRate(object):
    _cache = dict()
    _rate_cache = dict()
    _sessionMgr = SessionMgr()
    _session = _sessionMgr.getSessionFactory()

    @staticmethod
    def load_composite_risk_free_rate(region):
        CRiskFreeRate._cache[region] = CCompositeRate.get_composite_risk_free_rate(region)

    @staticmethod
    def get_interest_rates_for_region(region, maturities):

        if (region, tuple(maturities)) not in CRiskFreeRate._rate_cache.keys():
            CRiskFreeRate.load_interest_rates_for_region(region, maturities)
        return CRiskFreeRate._rate_cache.get((region, tuple(maturities)))

    @staticmethod
    def load_interest_rates_for_region(region, maturities):

        df_rfrs = CRiskFreeRate._sessionMgr.get_interest_rates_for_region(region, maturities)

        df = pd.DataFrame()
        for col in df_rfrs.columns:
            df_col = df_rfrs.get(col).dropna().to_frame(col) / 100
            df = pd.concat((df, df_col.reindex(pd.date_range(df_col.index.min(),
                                                             df_col.index.max())).ffill()), axis=1)
        CRiskFreeRate._rate_cache[(region, tuple(maturities))] = CTimeSeries(df,
                                                                             ts_type=TimeSeriesType.RETURNS,
                                                                             returns_type=ReturnsType.SIMPLE)

    @staticmethod
    def construct_risk_free_rate(region):

        df = CRiskFreeRate.get_interest_rates_for_region(region, ['ON', '1M', '3M'])
        rfrs = -1 + np.power(1 + df.mean(axis=1).to_frame(region), 1 / DateUtils.days_per_year)
        return CTimeSeries(rfrs, ts_type=TimeSeriesType.RETURNS).get_levels().resample('B').asfreq()


    @staticmethod
    def load_standard_risk_free_rate(region):
        lvls = CRiskFreeRate.construct_risk_free_rate(region)
        CRiskFreeRate._cache[region] = lvls.fillna(1).get_returns()

    @staticmethod
    def load_EUR_risk_free_rate():

        DEM = CRiskFreeRate.construct_risk_free_rate('Germany').get_returns()
        DEM.columns =['Eurozone']
        EUR = CRiskFreeRate.construct_risk_free_rate('Eurozone').get_returns()

        CRiskFreeRate._cache['Eurozone'] = pd.concat((EUR[FX.EUR_START_DATE.value:],
                                       DEM[DEM.index < FX.EUR_START_DATE.value]),
                                       axis=0).sort_index()


    @staticmethod
    def load_risk_free_rate(region):
        if region in CompositeRiskFreeRates.composite_rfr_regions:
           CRiskFreeRate.load_composite_risk_free_rate(region)
        elif region.lower() in ['eurozone', 'emu', 'european union']:
            CRiskFreeRate.load_EUR_risk_free_rate()
        else:
           CRiskFreeRate.load_standard_risk_free_rate(region)

    @staticmethod
    def get_risk_free_for_region(region):

        if region not in CRiskFreeRate._cache.keys():
           CRiskFreeRate.load_risk_free_rate(region)
        return CRiskFreeRate._cache.get(region)

    @staticmethod
    def get_risk_free_rate_from_currency(currency):
        region = CRiskFreeRate._sessionMgr.get_region_from_currency(currency)
        if region:
            return CRiskFreeRate.get_risk_free_for_region(region)
        else:
            raise ValueError('Currency {} not supported'.format(currency))

    @staticmethod
    def get_interest_rate_curve_from_region(region, type=None):
        _mats = CRiskFreeRate._sessionMgr.get_interest_rate_maturities_for_region(region)
        rates = CRiskFreeRate.get_interest_rates_for_region(region, _mats)

        if type is not None:
            _cols = np.isin(rates.columns.get_level_values(3), type)
            rates = rates.iloc[:, _cols].dropna(how='all', axis=1)

        _rates = rates.T.groupby(level=2).mean().T

        mats = DateUtils.Rdate_to_mat(_rates.columns)
        cols = list(zip(mats, _rates.columns))
        _rates.columns = pd.MultiIndex.from_tuples(cols)
        _rates.columns.names = ['maturity', 'tenor']
        return _rates.sort_index(axis=1, level=0).dropna(how='all', axis=1)

class MSCIActivityPanel(object):
    def __init__(self):
        self._panels = pd.DataFrame()
        self.__load_activity_panel()

    def __load_activity_panel(self):

        _df = pd.read_excel(DIR_,
                           sheet_name=None,
                           index_col=0,
                           header=[0, 1])

        self._df = _df.get('Rules')
        self._mapping = _df.get('Mapping')

        if isinstance(self._mapping.columns, pd.MultiIndex):
           self._mapping.columns = self._mapping.columns.get_level_values('MSCI Region')

        indicies = self._df.columns.get_level_values('Index Name')
        for index in indicies:
            self.__construct_activity_panel_single_index(index)
        self._panels.columns.names = ['Index', 'MSCI Region', 'Region', 'Local', 'Dollar', 'Currency']

    def __construct_activity_panel_single_index(self, index_name):

        df = self._df.get(index_name).replace('-', np.nan).dropna()
        regions = df.index
        for region in regions:
            self.__construct_activity_panel_single_index_region(index_name, region)

    def __construct_activity_panel_single_index_region(self, index_name, region):
        val = self._df.get(index_name).loc[region].values[0]

        if isinstance(val, str):
           res = self.__process_period_string(val)
        elif isinstance(val, datetime.date):
           res = self.__process_period_datetime(val)
        else:
           res = pd.DataFrame(columns=[(index_name, region)])

        new_col = [index_name, region] + list(self._mapping.loc[region].values)
        res.columns = pd.MultiIndex.from_tuples([tuple(new_col)])
        self._panels = pd.concat((self._panels, res), axis=1)

    def __process_period_string(self, val):

        res = pd.DataFrame()
        for p in val.split(','):
            res = pd.concat((res,
                             self.__process_period_substring(p)), axis=0)
        return res

    def __process_period_substring(self, substring):

        if 'to' in substring:
            start_str, end_str = substring.split('to')
            start = pd.to_datetime(start_str.replace(" ", ""))
            end = pd.to_datetime(end_str.replace(" ", ""))
        else:
            start = pd.to_datetime(substring.replace(" ", ""))
            end = pd.to_datetime(datetime.date.today())

        assert type(start) == pd.Timestamp, 'Error in start date'
        assert type(end) == pd.Timestamp, 'Error in end date'
        assert start < end, 'Error in dates'
        range = pd.date_range(start + pd.tseries.offsets.BDay(-1), end)
        return pd.DataFrame([True] * range.__len__(), index=range, columns=['X'])


    def __process_period_datetime(self, val):

        range = pd.date_range(val + pd.tseries.offsets.BDay(-1), datetime.date.today())
        return pd.DataFrame([True] * range.__len__(), index=range, columns=['X'])

    @staticmethod
    def get_activity_panel_single_index(index_name):

        _PICKLE_NAME = 'MSCI_CONSTITUENTS'
        if PickleUtils.is_pickled(_PICKLE_NAME):
            panel_df = PickleUtils.load_pickle(_PICKLE_NAME)
        else:
            panel = MSCIActivityPanel()
            panel_df = panel._panels
        return panel_df.get(index_name)

class CCompositeRate(object):

    def __init__(self,
                 index_ticker,
                 ):

        self.ticker = index_ticker
        self._datasource = GlobalDataSource()
        activity_panel = MSCIActivityPanel.get_activity_panel_single_index(self.ticker)

        self.__regions = np.unique(activity_panel.columns.get_level_values('MSCI Region'))
        self._info = pd.DataFrame(activity_panel.columns.to_frame().values,
                                  columns=activity_panel.columns.names).set_index('MSCI Region', drop=True)
        self._info = self._info[~self._info.index.duplicated(keep='first')]

        self._activity_panel = activity_panel.copy()
        self._activity_panel.columns = activity_panel.columns.get_level_values('MSCI Region')
        self.__load_constituent_data()

    @property
    def regions(self):
        return self.__regions

    def get_constituent_region_dollar_ticker(self, region):
        return self._info.loc[region].Dollar

    def get_constituent_region_local_ticker(self, region):
        return self._info.loc[region].Local

    def get_constituent_region_currency(self, region):
        return self._info.loc[region].Currency

    def get_constituent_region_activity(self, region):
        return self._activity_panel.get([region]).dropna(how='all', axis=0).any(axis=1).to_frame('X')

    def get_constituent_region_rate_ticker(self, region):
        return self._datasource.get_interest_rate_tickers(region, maturities=['ON','1m','3m'])

    def get_risk_free_rate_from_constituent_region(self, region):
        rfr = CRiskFreeRate.get_risk_free_for_region(self._info.loc[region].Region)
        rfr.columns = pd.MultiIndex.from_tuples([(region, 'RFR')])
        return rfr.copy()

    def get_constituent_region_MV(self, region):
        ticker = self.get_constituent_region_dollar_ticker(region)
        df_ = self._datasource.get_dataframe_from_ticker(ticker, cols='MV')
        df_.columns = pd.MultiIndex.from_tuples([(region, 'MV')])
        return df_.copy()

    def run_data_test_MV_single_region(self, region):
        dates = self.get_constituent_region_active_dates(region)
        MV = self.get_constituent_region_MV(region).dropna()
        missing_dates = np.setdiff1d(pd.to_datetime(dates),
                                     pd.to_datetime(MV.index))
        df_ = pd.DataFrame(missing_dates, index=[region] * len(missing_dates), columns=['dates'])
        df_['type'] = 'MV'
        return df_.copy()

    def run_data_test_RFR_single_region(self, region):
        dates = self.get_constituent_region_active_dates(region)
        rfr = self.get_risk_free_rate_from_constituent_region(region)
        missing_dates = np.setdiff1d(pd.to_datetime(dates),
                                     pd.to_datetime(rfr.index))
        df_ = pd.DataFrame(missing_dates, index=[region] * len(missing_dates), columns=['dates'])
        df_['type'] = 'RFR'
        return df_.copy()

    def run_data_check(self):

        df_ = pd.DataFrame()
        for region in self.regions:
            print(region)
            df_ = pd.concat((df_, pd.concat((self.run_data_test_RFR_single_region(region),
                             self.run_data_test_MV_single_region(region)), axis=0)), axis=0)

        idx = df_.dates < pd.to_datetime('2023-06-30')
        return df_[idx.values].copy()

    def get_constituent_region_active_dates(self, region):
        dates = self.get_constituent_region_activity(region)
        b_days = np.logical_and(dates.index.dayofweek != 6,
                                dates.index.dayofweek != 5)
        return dates[b_days].dropna().index

    def get_dataframe_for_constituent_region(self, region):
        dates = self.get_constituent_region_active_dates(region)

        MV = self.get_constituent_region_MV(region).reindex(dates)
        rfr = self.get_risk_free_rate_from_constituent_region(region).reindex(dates)

        return pd.concat((MV, rfr), axis=1).dropna(axis=0)

    def __load_constituent_data(self):

        df_ = CTimeSeries()
        for region in self.regions:
            print(region)
            df_ = pd.concat((df_, self.get_dataframe_for_constituent_region(region)), axis=1)

        self._panel = df_.copy()
        self._index_market_value = df_.iloc[:, df_.columns.get_level_values(1) == 'MV'].sum(axis=1)

    def get_constituent_region_weights_and_rates(self, region):
        return ((self._panel.get(region).get('MV') / self._index_market_value).to_frame(region),
                self._panel.get(region).get('RFR').to_frame(region))

    def get_constituent_region_contribution(self, region):
        wts, rts = self.get_constituent_region_weights_and_rates(region)
        return wts.reindex(wts.index[0:-1]).values * rts.reindex(rts.index[1:])

    def construct_history(self):

        panel = pd.DataFrame()
        for region in self.regions:
            panel = pd.concat((panel, self.get_constituent_region_contribution(region)), axis=1)
        self._rfr = panel.sum(axis=1).to_frame(self.ticker)

    @staticmethod
    def get_composite_risk_free_rate(index_name):

        _PICKLE_NAME = index_name.upper().replace(' ', '_').replace('-', '') + '_RFR'
        if PickleUtils.is_pickled(_PICKLE_NAME):
            res = PickleUtils.load_pickle(_PICKLE_NAME)
            return CTimeSeries(res, ts_type=TimeSeriesType.RETURNS, returns_type=ReturnsType.SIMPLE)
        else:
            rfr = CCompositeRate(index_name)
            rfr.construct_history()
            rate = CTimeSeries(rfr._rfr, ts_type=TimeSeriesType.RETURNS, returns_type=ReturnsType.SIMPLE)
            PickleUtils.pickle_it(rate, _PICKLE_NAME)
            return rate



if __name__ == "__main__":

    rates = CRiskFreeRate.get_interest_rate_curve_from_region('United States')

    for region in CompositeRiskFreeRates.composite_rfr_regions:
        rfr = CRiskFreeRate.get_risk_free_for_region(region)