from pandas_datareader.famafrench import get_available_datasets
import pandas_datareader.data as web
import datetime as dt


class FFDataReader(object):
    _cache = {}

    @staticmethod
    def load_dataset(dataset):
        FFDataReader._cache[dataset] = web.DataReader(name=dataset,
                                                      data_source='famafrench',
                                                      start = dt.date(year=1920, month=12, day=31),
                                                      )
    @staticmethod
    def get_dataset(dataset):
        if dataset not in FFDataReader._cache.keys():
           FFDataReader.load_dataset(dataset)
        return FFDataReader._cache.get(dataset).get(0)
    @staticmethod
    def get_risk_free_rate():
        dataset = FFDataReader.get_dataset('F-F_Research_Data_Factors_daily')
        return dataset.get('RF').to_frame('rf')/100
    @staticmethod
    def get_SMB():
        dataset = FFDataReader.get_dataset('F-F_Research_Data_Factors_daily')
        return dataset.get('SMB').to_frame('SMB')/100
    @staticmethod
    def get_HML():
        dataset = FFDataReader.get_dataset('F-F_Research_Data_Factors_daily')
        return dataset.get('HML').to_frame('HML')/100
    @staticmethod
    def get_MKT():
        dataset = FFDataReader.get_dataset('F-F_Research_Data_Factors_daily')
        return dataset.get('Mkt-RF').to_frame('MKT')/100
    @staticmethod
    def get_RMW():
        dataset = FFDataReader.get_dataset('F-F_Research_Data_5_Factors_2x3_daily')
        return dataset.get('RMW').to_frame('RMW') / 100
    @staticmethod
    def get_CMA():
        dataset = FFDataReader.get_dataset('F-F_Research_Data_5_Factors_2x3_daily')
        return dataset.get('CMA').to_frame('CMA') / 100
    @staticmethod
    def get_MOM():
        dataset = FFDataReader.get_dataset('F-F_Momentum_Factor_daily')
        return dataset.iloc[:,0].to_frame('MoM') / 100





if __name__ == "__main__":

    dset = FFDataReader.get_MOM()