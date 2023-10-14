import datetime
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
import os
import pandas as pd
import numpy as np
import platform

f_path_ = os.path.abspath(__file__)
if 'windows' in platform.system().lower():
    DIR_ = os.path.join(f_path_[:f_path_.find('epsilon-phi-core\\') + len('epsilon-phi-core\\')],
                        'src\\resources\\templates\\MSCI Index Construction.xlsx')
else:
    DIR_ = os.path.join(f_path_[:f_path_.find('epsilon-phi-core/')+ len('epsilon-phi-core/')],
                        'src/resources/templates/MSCI Index Construction.xlsx')

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
        self._panels = self._panels.fillna(False)
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
        range = pd.date_range(start, end)
        return pd.DataFrame([True] * range.__len__(), index=range, columns=['X'])


    def __process_period_datetime(self, val):

        range = pd.date_range(val, datetime.date.today())
        return pd.DataFrame([True] * range.__len__(), index=range, columns=['X'])

    @staticmethod
    def get_activity_panel_single_index(index_name):
        panel = MSCIActivityPanel()
        return panel._panels.get(index_name)

class CRiskFreeRate(object):
    def __init__(self,
                 index_ticker,
                 ):

        self._datasource = GlobalDataSource()
        activity_panel = MSCIActivityPanel.get_activity_panel_single_index(index_ticker)

        self.__regions = np.unique(activity_panel.columns.get_level_values('Region'))
        self._info = pd.DataFrame(activity_panel.columns.to_frame().values,
                                  columns=activity_panel.columns.names).set_index('Region', drop=True)
        self._info = self._info[~self._info.index.duplicated(keep='first')]

        self._activity_panel = activity_panel.copy()
        self._activity_panel.columns = activity_panel.columns.get_level_values('Region')
        self._construct_history()

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
        return self._activity_panel.get([region]).any(axis=1).to_frame((region, 'X'))

    def get_constituent_region_rate_ticker(self, region):
        return self._datasource.get_interest_rate_tickers(region, maturities=['ON','1m','3m'])

    def get_risk_free_rate_from_constituent_region(self, region):

        rfr_tickers = self.get_constituent_region_rate_ticker(region)
        rfr = CTimeSeries()

        for ticker in rfr_tickers.ticker:
            rfr = pd.concat((rfr, self._datasource.get_total_return_series_from_ticker(ticker)), axis=1)

        nan_rows = np.all(rfr.isna(), axis=1)
        rfr = rfr.get_periodic_returns(Frequency.BUSINESS_DAILY).mean(axis=1, skipna=True)

        # be sure to put the nans back in
        rfr[nan_rows] = np.nan
        return rfr.to_frame((region, 'RFR'))

    def get_constituent_region_MV(self, region):
        ticker = self.get_constituent_region_dollar_ticker(region)
        df_ = self._datasource.get_dataframe_from_ticker(ticker, cols='MV')
        df_.columns = pd.MultiIndex.from_tuples([(region, 'MV')])
        return df_.copy()

    def get_dataframe_for_constituent_region(self, region):
        MV = self.get_constituent_region_MV(region)
        rfr = self.get_risk_free_rate_from_constituent_region(region)
        A = self.get_constituent_region_activity(region)
        return pd.concat((MV, rfr, A), axis=1).dropna()

    def _construct_history(self):

        df_ = CTimeSeries()
        for region in self.regions:
            print(region)
            df_ = pd.concat((df_, self.get_dataframe_for_constituent_region(region)), axis=1)

        MVs = df_.get('MV')



if __name__ == "__main__":

    rfr = CRiskFreeRate('ACWI')
