import os, sys
import pandas as pd

folder_path = r'C:\Users\fabar\Documents\Data\equities\MSCI\ACWI\Data Repository'
contents = os.listdir(folder_path)

tickers = list()
for content in contents:
    ticker = content.replace('_', ':').replace('.csv','')
    tickers.extend([ticker])


df_tickers = pd.DataFrame(tickers)