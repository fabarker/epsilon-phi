import datetime
from epsilonPhi.core.lib.Decorators import SingletonDecorator
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
import os
import pandas as pd
import numpy as np

f_path_ = os.path.abspath(__file__)
DIR_ = os.path.join(f_path_[:f_path_.find('epsilon-phi-core/')+ len('epsilon-phi-core/')],
                    'src/resources/templates/MSCI Index Construction.xlsx')

class MSCIActivityPanel(object):
    def __init__(self):
        self._panels = pd.DataFrame()
        self.__load_activity_panel()

    def __load_activity_panel(self):

        self._df = pd.read_excel(DIR_,
                           sheet_name='Rules',
                           index_col=0,
                           header=[0, 1])

        indicies = self._df.columns.get_level_values('Index Name')
        for index in indicies:
            self.__construct_activity_panel_single_index(index)
        self._panels = self._panels.fillna(False)

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

        res.columns = pd.MultiIndex.from_tuples([(index_name, region)])
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
        self._activity_panel = MSCIActivityPanel.get_activity_panel_single_index(index_ticker)
        self.construct_history()

    def construct_history(self):
        pass

    def get_index_constituent_regions(self):
        return self._activity_panel.columns

    def get_constituent_region_ticker(self, region):
        pass

    def get_constituent_region_data(self, region):
        return self._activity_panel.get(region)

    def get_constituent_region_rate(self):
        pass

    def get_constituent_region_rate_ticker(self, region):
        pass

    def get_constituent_region_contribution(self, region):
        pass

    def get_constitient_region_MV(self):
        pass


if __name__ == "__main__":

    panel = MSCIActivityPanel.get_activity_panel_single_index('ACWI')
