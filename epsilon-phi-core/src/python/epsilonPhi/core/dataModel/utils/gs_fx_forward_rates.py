import datetime

from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.dataModel.dataSources.vendor.GSQuant import GSQuantManager, Dataset, Datasets
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource

sessionMgr = SessionMgr()
session = sessionMgr.getSessionFactory()
gsq = GSQuantManager()
gds = GlobalDataSource()

# Get all tickers
q = session.query(FXRateSpec).filter(FXRateSpec.provider == 'GS')
_spec = sessionMgr.query_format_df(q)

fwd_specs = _spec[~_spec.maturity.isin(['0m'])]

#ds = Dataset('FXSPOT_V2_PREMIUM')
ds = Dataset('FX_FORWARD_POINTS_STORE')

ctr = 0
for id, row in fwd_specs.iterrows():
    print(fwd_specs.shape[0] - ctr)
    ctr += 1

    symbol = row.ticker

    from_date = sessionMgr.get_latest_observation_date_from_tickers(symbol) - pd.to_timedelta(61, 'D')
    to_date = datetime.datetime.today()

    res = ds.get_data(start=from_date.date(), end=to_date.date(),
                      assetId=symbol, pricing_location=['LDN', 'NYC'])
    res.index = pd.to_datetime(res.index)

    df_ = gds.get_dataframe_from_uid(row.uid).get(row.uid)
    df_.index = pd.to_datetime(df_.index)
    idx = df_.reset_index().set_index(['date','pricing_location']).index

    res_ = res.reset_index().set_index(['date', 'pricingLocation'])
    keep_rows = res_.index.difference(idx)

    db_data = res_.loc[keep_rows].reset_index()
    unduped = db_data.drop_duplicates(subset=['date', 'pricingLocation'])

    for_db = unduped[['date', 'pricingLocation', 'spot']]
    frame = for_db.rename(columns={'pricingLocation': 'pricing_location', 'spot': 'ER'})
    frame['uid'] = row.uid
    frame['X'] = frame['ER']

    if frame.size > 0:
        frame.to_sql(name='fx_rates',
                       con=SessionMgr().getEngine(),
                       if_exists='append',
                       index=False)
        print('Data appended to table ' + 'fx_rates' + ' for time series with ticker {}'.format(
            symbol))
