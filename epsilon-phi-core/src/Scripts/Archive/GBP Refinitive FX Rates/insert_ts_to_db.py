import numpy as np
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.utils.FrameUtils import FrameUtils
from epsilonPhi.core.dataModel.dataSources.vendor.Bloomberg import Bloomberg
from epsilonPhi.core.dataModel.dataSources.vendor.Datastream import pyDatastream
import pandas as pd
import datetime as dt

sessionMgr = SessionMgr()
session = SessionMgr().getSessionFactory()

data = pd.read_csv('USDx rates from GBP.csv', index_col=0, header=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
data.index = pd.to_datetime(data.index, format="%d/%m/%Y")
unique_tickers = data.columns.get_level_values('ticker').unique()

for ticker in unique_tickers:
    subset = data.get(ticker).dropna(how='all')
    keep_dates_before = pd.to_datetime(subset.columns.get_level_values('start').unique()[0], format="%d/%m/%Y %H:%M")

    if not Bloomberg.is_ticker_in_database(ticker):

        try:
            spec = FXRateSpec()
            spec.category = 'FX'
            spec.datasource = 'Datastream'
            spec.domestic_currency = subset.columns.get_level_values('domestic_currency')[0].upper()
            spec.foreign_currency = subset.columns.get_level_values('foreign_currency')[0].upper()
            spec.bbid = spec.foreign_currency + spec.domestic_currency
            spec.frequency = 'B'
            spec.maturity = subset.columns.get_level_values('maturity')[0].lower()
            spec.name = subset.columns.get_level_values('long_name')[0]
            spec.provider = subset.columns.get_level_values('provider')[0]
            spec.region = subset.columns.get_level_values('region')[0]

            spec.ticker = ticker
            spec.type = 'Outright'
            spec.uid = int(Bloomberg.get_max_uid() + 1)

            session.add_all([spec])
            session.commit()
        except:
            session.rollback()
            print('Error - could not add time series spec info for ticker {}'.format(ticker))
        finally:
            session.close()

        df_db = subset.iloc[subset.index < keep_dates_before, :]
        df_db.columns = list(subset.columns.get_level_values('dataType'))
        df_db['pricing_location'] = 'LDN'
        df_db.index.name = 'date'
        df_db = df_db.reset_index(drop=False)
        df_db = df_db.dropna(axis=1, how='all')
        df_db['uid'] = Bloomberg.get_uid_from_ticker(ticker)

        df_db.to_sql(name='fx_rates',
                      con=SessionMgr().getEngine(),
                      if_exists='append',
                      index=False)
        print('Data added for {}'.format(ticker))


