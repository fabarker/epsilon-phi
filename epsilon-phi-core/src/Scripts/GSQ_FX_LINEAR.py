import os, sys

import pandas as pd

old_path = r'C:\Users\fabar\Documents\Data\gsquant\fx_spot_fwds\Data Repository'

list_dir = os.listdir(old_path)
ticker_list = list()
for dir in list_dir:
    ticker_list.extend([dir.replace('.csv','')])