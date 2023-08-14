import os, sys

import pandas as pd

old_path = r'C:\Users\fabar\Documents\Data\futures\Data Repository Old'
new_path = r'C:\Users\fabar\Documents\Data\futures\Data Repository'

list_dir = os.listdir(old_path)
ticker_list = list()
for dir in list_dir:
    _df = pd.read_csv(os.path.join(old_path, dir), index_col=0)
    _df.to_csv(os.path.join(new_path, dir + '.csv'))
    ticker_list.extend([_df.loc['MNEM'].unique()[0]])

