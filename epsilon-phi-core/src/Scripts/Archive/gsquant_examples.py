import os
import datetime, os
import numpy as np
import pandas as pd

from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.dataModel.alchemist.DataModel import *

mgr = SessionMgr()
session = mgr.getSessionFactory()
q = session.query(FXRateSpec).filter(FXRateSpec.provider == 'GS',
                                     FXRateSpec.category == 'FX')
specs = mgr.query_format_df(q)
unique_bbids = specs.bbid.unique()
unique_ccys = ['ARS', 'AUD', 'BRL', 'CAD', 'CHF', 'CLP', 'CNH', 'COP', 'CZK',
               'EUR', 'GBP', 'HKD', 'HUF', 'IDR', 'ILS', 'INR', 'JPY',
               'KRW', 'MXN', 'MYR', 'NOK', 'NZD', 'PEN', 'PHP', 'PLN', 'RUB',
               'SEK', 'TRY', 'TWD', 'ZAR']

from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource

ds = GlobalDataSource()

all_data = pd.DataFrame()
for ccy in unique_ccys:
    print(ccy)

    locs = specs[specs.bbid.isin(['USD' + ccy, ccy + 'USD'])]
    for _, row in locs.iterrows():
        df_ = ds.get_dataframe_from_uid(row.uid)
        tmp = df_.stack(level=0)[['pricing_location', 'ER']].reset_index()
        tmp['bbid'] = row.bbid
        tmp['tenor'] = row.maturity

        dta = tmp.set_index('date').pivot(columns=['pricing_location', 'uid', 'bbid']).swaplevel(3, 0, axis=1)
        all_data = pd.concat((all_data, dta), axis=1)

for sheet in np.unique(all_data.columns.get_level_values(0)):
    print(sheet)
    s = all_data.get([sheet]).sort_index(level=2, axis=1)

    # use to_excel function and specify the sheet_name and index
    # to store the dataframe in specified sheet
    s.to_excel('/Users/francisbarker/Desktop/ivol cache/FX Rates/' + sheet + '.xlsx', sheet_name=sheet)

gsq = GSQuantManager()
ds = Dataset('FX_FORWARD_POINTS_STORE')

triplets = list()
for spec in specs:
    t = spec.ticker

    q = session.query(FXRateSpec).filter(FXRateSpec.uid == spec.uid)
    fx_spec = mgr.query_format_df(q).T

    db_bid = fx_spec.loc['bbid'].values[0]

    asset = SecurityMaster.get_asset(t, id_type=AssetIdentifier.MARQUEE_ID)
    gs_bbid = asset.entity.get('parameters').get('pair')

    triplets.extend([(t, asset.name.split()[2].replace('/', ''), gs_bbid, db_bid)])
info = pd.DataFrame(triplets, columns=['ticker', 'from_name', 'GS', 'db']).to_clipboard()

_cov = ds.get_coverage()
_info = [(x.split(' ')[2].replace('/', ''), x.split(' ')[3]) for x in _cov.name]
pairs = pd.concat((_cov, pd.DataFrame(_info, columns=['bbid', 'tenor'])), axis=1)
mat_pairs = pairs[pairs.tenor.isin(['1d', '1w', '2w', '1m', '2m', '3m', '6m', '9m', '1y'])]
ts = mat_pairs[np.array(['USD' in x for x in mat_pairs.bbid.values])]

target_pairs = ['AUDUSD', 'EURUSD', 'GBPUSD', 'NZDUSD', 'USDBRL', 'USDCAD', 'USDCHF', 'USDJPY', 'USDMXN', 'USDNOK',
                'USDSEK', 'USDZAR']
pull_pairs = ts[np.array([x in target_pairs for x in ts.bbid.values])]

failed = list()
for id, row in pull_pairs.iterrows():
    try:
        t = row.assetId
        res = ds.get_data(start=date(year=1980, month=12, day=31), end=datetime.date.today(),
                          assetId=t, pricing_location=['LDN', 'NYC'])
        res.to_csv(_SAVE_PATH + '/' + t + '.csv')
    except:
        failed.extend([t])
con_pd = res.reset_index(drop=False).set_index('assetId')

asset = SecurityMaster.get_asset(t, id_type=AssetIdentifier.MARQUEE_ID)

_SD = date(year=1997, month=10, day=27)
_ED = date(year=1997, month=10, day=27)

df = pd.read_excel('/Users/francisbarker/Desktop/GS Tickers.xlsx', sheet_name='Tickers')

failed = list()
for id, row in df.iterrows():
    try:
        t = row.ticker
        d = mgr.get_latest_observation_date_from_tickers(t) + pd.to_timedelta(1, 'D')

        res = ds.get_data(start=date(year=d.year, month=d.month, day=d.day), end=datetime.date.today(),
                          assetId=t, pricing_location=['LDN', 'NYC'])
        res.to_csv(_SAVE_PATH + '/' + t + '.csv')
    except:
        failed.extend([t])
con_pd = res.reset_index(drop=False).set_index('assetId')

####

cov = _DATASET.get_coverage()
coverage = pd.DataFrame([x.split() for x in cov.name])
coverage.columns = ['type', 'currency', 'tenor', 'delta', 'putcall']
coverage['assetID'] = cov.assetId
coverage.drop(columns='type')
coverage = coverage.set_index('assetID')
coverage['foreign'] = coverage['currency'].apply(lambda x: x[0:3])
coverage['domestic'] = coverage['currency'].apply(lambda x: x[3:])

_D = coverage[coverage.delta.isin(['10D', '25D', 'DN', 'ATMF', 'Spot', '25C', '10C'])]
_DFX = _D[_D.currency == 'GBPUSD']
_DFXM = _DFX[_DFX.tenor.isin(['1m'])]

tickers = _DFXM.index
df = [pd.DataFrame()]
for ticker in tickers:
    tmp = _DATASET.get_data(_SD, _ED, assetId=ticker, pricingLocation='NYC')

    if coverage.loc[ticker].putcall == 'Call':
        _del = coverage.loc[ticker].delta
    elif coverage.loc[ticker].putcall == 'Put':
        _del = '-' + coverage.loc[ticker].delta
    else:
        _del = 'DN' + coverage.loc[ticker].putcall[0]
    tmp['relative_strike'] = _del
    tmp['tenor'] = coverage.loc[ticker].tenor
    df.extend([tmp])

vols = pd.concat(df, axis=0)
subset = vols[['impliedVolatility', 'relative_strike']]
subset.reset_index(drop=False).drop_duplicates(subset=['date', 'relative_strike']).set_index('date').pivot(
    columns='relative_strike').to_clipboard()

spx = gsq.get_security('SPX')

start = datetime.date(year=1980, month=1, day=1)
end = datetime.date.today()

O = spx.get_data_series(DataMeasure.ADJUSTED_OPEN_PRICE, frequency=DataFrequency.DAILY, start=start, end=end)
H = spx.get_data_series(DataMeasure.ADJUSTED_HIGH_PRICE, frequency=DataFrequency.DAILY, start=start, end=end)
L = spx.get_data_series(DataMeasure.ADJUSTED_LOW_PRICE, frequency=DataFrequency.DAILY, start=start, end=end)
C = spx.get_data_series(DataMeasure.ADJUSTED_CLOSE_PRICE, frequency=DataFrequency.DAILY, start=start, end=end)

dfs = (O.to_frame('open'),
       H.to_frame('high'),
       L.to_frame('low'),
       C.to_frame('close'))

df = pd.concat(dfs, axis=1)

start = datetime.date(year=1999, month=1, day=1)
OHLC = spx.get_hloc_prices(start=start)

strikes = np.array(range(40, 180, 5))
tenors = ['2m']

df_ = pd.DataFrame()
for tenor in tenors:
    for k in strikes:
        print(str(k) + ' ' + tenor)
        try:
            res = gsq.get_ivol('SPX', '31-Dec-2004', tenor=tenor, vol_reference='normalized', relative_strike=k)
            res.to_csv(os.path.join(_SAVE_PATH, str(k) + '_' + tenor + '.csv'))
        except:
            pass

tenors = ['1m', '2m', '3m', '6m', '1y']
# relative_strike = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

ds = Dataset('EDRVOL_PERCENT_INTERNAL')

_all_vols = pd.DataFrame()
for tenor in tenors:
    print(tenor)
    res_1 = ds.get_data(start=datetime.date(year=1999, month=1, day=1), end=datetime.date(year=2009, month=12, day=31),
                        assetId='MA4B66MW5E27U8P32SB',
                        strikeReference='forward',
                        tenor=tenor)
    res_2 = ds.get_data(start=datetime.date(year=2010, month=1, day=1), end=datetime.date.today(),
                        assetId='MA4B66MW5E27U8P32SB',
                        strikeReference='spot',
                        tenor=tenor)

    res = pd.concat((res_1, res_2), axis=0)
    _all_vols = pd.concat((_all_vols, res), axis=0)

cov = GSQuantManager()._get_coverage(ds)
cov = cov.rename(columns={'name': 'names'})

save_dir = r'C:\Users\fabar\Documents\Data\gsquant\fx_spot_fwds'

USD_list = list()
for idx, row in cov.iterrows():
    if 'USD' in row.names:
        USD_list.extend([row.assetId])

chunks = _nest_list(USD_list, nested_size=20)
frames = [pd.DataFrame()]
print("Reading Data:", end=" ")
for ch in chunks:
    print("#", end="")
    res = ds.get_data(start=datetime.date(year=1980, month=12, day=31), end=datetime.date.today(), assetId=ch,
                      pricing_location=['LDN', 'NYC'])
    con_pd = res.reset_index(drop=False).set_index('assetId')
    uniqueIDs = np.unique(con_pd.index)

    for id in uniqueIDs:
        save_path = os.path.join(save_dir, id + '.csv')
        con_pd.loc[id].reset_index(drop=False).set_index('date').to_csv(save_path)

df = pd.read_csv('/Users/francisbarker/Desktop/SPX Vols by Moneyness.csv')
df.date = pd.to_datetime(df.date)
df = df.set_index('date', drop=True)

spx = gsq.get_security('SPX')
prices = spx.get_data_series(DataMeasure.CLOSE_PRICE,
                             frequency=DataFrequency.DAILY,
                             start='1990-12-31',
                             end=df.index.max().strftime("%Y-%m-%d"))
prices = prices.resample('D').asfreq().ffill()
prices = prices.loc[df.index].to_frame('spot')
vols = pd.concat((df, prices), axis=1)

df = vols.to_csv('/Users/francisbarker/Desktop/SPX Vols by Moneyness.csv')



