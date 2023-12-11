import pandas
import pandas as pd
import numpy as np
import datetime

path = r'/Users/francisbarker/Desktop/MSCI Index Construction.xlsx'
df = pd.read_excel(path, sheet_name='Rules', index_col=0)

df_ = pd.DataFrame()
for col in df.columns:
    idx = df.get(col)
    regions = idx.index

    for region in regions:

            info = idx.loc[region]

            if isinstance(info, datetime.datetime):
               range = pd.date_range(info, datetime.date.today())
               df_region = pd.DataFrame([True] * range.__len__(), index=range, columns=[region])
            elif isinstance(info, str) and info != '-':

                df_region = pd.DataFrame()
                for p in info.split(','):

                    if 'to' in p:
                        start_str, end_str = p.split('to')
                        start = pd.to_datetime(start_str.replace(" ",""))
                        end = pd.to_datetime(end_str.replace(" ",""))
                    else:
                        start = pd.to_datetime(p.replace(" ", ""))
                        end = pd.to_datetime(datetime.date.today())

                    assert type(start) == pd.Timestamp, 'Error in start date'
                    assert type(end) == pd.Timestamp, 'Error in end date'
                    assert start < end, 'Error in dates'
                    range = pd.date_range(start, end)
                    df_p = pd.DataFrame([True] * range.__len__(), index=range, columns=[region])
                    df_region = pd.concat((df_region, df_p), axis=0)
            else:
                df_region = pd.DataFrame(columns=[region])

            new_col = pd.MultiIndex.from_tuples([(col, region.capitalize())])
            df_region.columns = new_col
            df_ = pd.concat((df_, df_region), axis=1)

dataframe = df_.fillna(False)

