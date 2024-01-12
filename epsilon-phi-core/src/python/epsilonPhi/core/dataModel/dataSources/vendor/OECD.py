import pandas as pd
regional_list = 'AUS+AUT+BEL+CAN+CHL+COL+CRI+CZE+DNK+EST+FIN+FRA+DEU+GRC+HUN+ISL+IRL+ISR+ITA+JPN+KOR+LVA+LTU+LUX+MEX+NLD+NZL+NOR+POL+PRT+SVK+SVN+ESP+SWE+CHE+TUR+GBR+USA+ARG+BRA+BGR+CHN+HRV+CYP+IND+IDN+MLT+ROU+RUS+SAU+ZAF'
import requests
class OECDReader(object):
    pass

    @staticmethod
    def get_purchasing_power_parities():

        df = pd.read_csv('https://stats.oecd.org/SDMX-JSON/data/SNA_TABLE4/{}.PPPGDP.CD/OECD?contentType=csv'.format(regional_list))
        cols = ['Country','TRANSACT','Measure','Year','Unit Code','Value']
        df = df.get(cols).set_index('Country', drop=True)
        unique_regions = df.index.unique()
        df_ = pd.concat([df.loc[x].reset_index(drop=True).set_index('Year').get('Value').
                        to_frame((x, 'PPP', df.loc[x].get('Unit Code').unique()[0])) for x in unique_regions],
                        axis=1).sort_index()
        df_.columns.names = ['region', 'field', 'currency']
        return df_.copy()

    @staticmethod
    def get_industrial_production():
        df = pd.read_csv(
            'https://stats.oecd.org/SDMX-JSON/data/KEI/PRINTO01.{}.ST.M?contentType=csv'.format(regional_list))
        cols = ['Country', 'Subject', 'Measure', 'Time', 'Unit Code', 'Value']
        df = df.get(cols).set_index('Country', drop=True)
        unique_regions = df.index.unique()
        df_ = pd.concat([df.loc[x].reset_index(drop=True).set_index('Time').get('Value').
                        to_frame((x, 'PPP', df.loc[x].get('Unit Code').unique()[0])) for x in unique_regions],
                        axis=1)
        df_.index = pd.to_datetime([ '01-' + x for x in df_.index ])
        df_.columns.names = ['region', 'field', 'currency']
        return df_.sort_index()


if __name__ == "__main__":

    df = OECDReader.get_industrial_production()




