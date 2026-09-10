from pandas_datareader.famafrench import get_available_datasets
import pandas_datareader.data as web
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
import datetime as dt
import pandas as pd

class FFDataReader(object):
    _cache = {}

    def __init__(self, frequency=Frequency.DAILY):
        self._freq = frequency

    @staticmethod
    def freq_str(frequency):
        if frequency in [Frequency.DAILY, Frequency.BUSINESS_DAILY]:
            return '_daily'
        else:
            return ''

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
    def get_risk_free_rate(frequency=Frequency.DAILY):
        dataset = FFDataReader.get_dataset('F-F_Research_Data_Factors' + FFDataReader.freq_str(frequency))
        return dataset.get('RF').to_frame('rf')/100

    @staticmethod
    def get_SMB(frequency=Frequency.DAILY):
        dataset = FFDataReader.get_dataset('F-F_Research_Data_Factors' + FFDataReader.freq_str(frequency))
        return dataset.get('SMB').to_frame('SMB')/100

    @staticmethod
    def get_HML(frequency=Frequency.DAILY):
        dataset = FFDataReader.get_dataset('F-F_Research_Data_Factors' + FFDataReader.freq_str(frequency))
        return dataset.get('HML').to_frame('HML')/100

    @staticmethod
    def get_MKT(frequency=Frequency.DAILY):
        dataset = FFDataReader.get_dataset('F-F_Research_Data_Factors' + FFDataReader.freq_str(frequency))
        return dataset.get('Mkt-RF').to_frame('MKT')/100

    @staticmethod
    def get_RMW(frequency=Frequency.DAILY):
        dataset = FFDataReader.get_dataset('F-F_Research_Data_5_Factors' + FFDataReader.freq_str(frequency))
        return dataset.get('RMW').to_frame('RMW') / 100

    @staticmethod
    def get_LT_Reversal(frequency=Frequency.DAILY):
        dataset = FFDataReader.get_dataset('F-F_LT_Reversal_Factor' + FFDataReader.freq_str(frequency))
        return dataset.get('LT_Rev').to_frame('LTR') / 100

    @staticmethod
    def get_ST_Reversal(frequency=Frequency.DAILY):
        dataset = FFDataReader.get_dataset('F-F_ST_Reversal_Factor' + FFDataReader.freq_str(frequency))
        return dataset.get('ST_Rev').to_frame('STR') / 100

    @staticmethod
    def get_CMA(frequency=Frequency.DAILY):
        dataset = FFDataReader.get_dataset('F-F_Research_Data_5_Factors_2x3' + FFDataReader.freq_str(frequency))
        return dataset.get('CMA').to_frame('CMA') / 100

    @staticmethod
    def get_industry_portfolios(frequency=Frequency.DAILY):
        return FFDataReader.get_dataset( '10_Industry_Portfolios' + FFDataReader.freq_str(frequency)) / 100

    @staticmethod
    def get_MOM(frequency=Frequency.DAILY):
        dataset = FFDataReader.get_dataset('F-F_Momentum_Factor' + FFDataReader.freq_str(frequency))
        return dataset.iloc[:, 0].to_frame('MOM_US_FF') / 100

    @staticmethod
    def get_FFS1B1(frequency=Frequency.DAILY):
        dataset = FFDataReader.get_dataset('6_Portfolios_2x3' + FFDataReader.freq_str(frequency))
        return dataset.get('SMALL LoBM').to_frame('SL') / 100

    @staticmethod
    def get_FFS1B2(frequency=Frequency.DAILY):
        dataset = FFDataReader.get_dataset('6_Portfolios_2x3' + FFDataReader.freq_str(frequency))
        return dataset.get('ME1 BM2').to_frame('SM') / 100

    @staticmethod
    def get_FFS1B3(frequency=Frequency.DAILY):
        dataset = FFDataReader.get_dataset('6_Portfolios_2x3' + FFDataReader.freq_str(frequency))
        return dataset.get('SMALL HiBM').to_frame('SH') / 100

    @staticmethod
    def get_FFS2B1(frequency=Frequency.DAILY):
        dataset = FFDataReader.get_dataset('6_Portfolios_2x3' + FFDataReader.freq_str(frequency))
        return dataset.get('BIG LoBM').to_frame('BL') / 100

    @staticmethod
    def get_FFS2B2(frequency=Frequency.DAILY):
        dataset = FFDataReader.get_dataset('6_Portfolios_2x3' + FFDataReader.freq_str(frequency))
        return dataset.get('ME2 BM2').to_frame('BM') / 100

    @staticmethod
    def get_FFS2B3(frequency=Frequency.DAILY):
        dataset = FFDataReader.get_dataset('6_Portfolios_2x3' + FFDataReader.freq_str(frequency))
        return dataset.get('BIG HiBM').to_frame('BH') / 100

    @staticmethod
    def get_double_sorts_size_value(frequency=Frequency.DAILY):
        dataset = FFDataReader.get_dataset('6_Portfolios_2x3' + FFDataReader.freq_str(frequency))
        dataset.columns = ['SL', 'SM', 'SH', 'BL', 'BM', 'BH']
        return dataset / 100

    @staticmethod
    def get_funding_related_factors(frequency=Frequency.MONTHLY):
        df = pd.concat((
            FFDataReader.get_double_sorts_size_value(frequency),
            FFDataReader.get_MOM(frequency)
        ), axis=1 )
        df.index = df.index.to_timestamp() + pd.tseries.offsets.BMonthEnd(0)
        return df.copy()


if __name__ == "__main__":

    import pandas as pd
    M = FFDataReader.get_double_sorts_size_value(Frequency.MONTHLY).add(1).cumprod()
    M.index = M.index.to_timestamp() + pd.tseries.offsets.BMonthEnd(0)
    D = FFDataReader.get_double_sorts_size_value(Frequency.DAILY).add(1).cumprod().resample('B').asfreq().ffill()

    pd.concat((M, D.loc[M.index]), axis=1).to_clipboard()