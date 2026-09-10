import pandas as pd

from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries
from epsilonPhi.core.timeSeries.regression import Regression
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.core.factor.factors.cEquity import cEquity
from epsilonPhi.core.factor.factors.cTerm import cTerm
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.dataModel.alchemist.DataModel import *
import datetime as dt

session = SessionMgr()
sessionFac = session.getSessionFactory()
gds = GlobalDataSource()

# 0. Load a schema
from epsilonPhi.core.schema.Schema import ContextCreator
schema = ContextCreator(currency='USD',
                        start_date='30-Nov-1983',
                        end_date='31-Dec-2022').create_context()

START_DATE = dt.date(year=1989, month=12, day=31)

# 1. Get the factor data for equity and term, construct X panel
EQ_D = gds.get_total_return_series_from_ticker('MSWRLD$').get_periodic_returns(schema.frequency)
rfr_D = gds.get_risk_free_rate_time_series('United States').get_periodic_returns(schema.frequency)
X1 = EQ_D.subtract_over_common_dates(rfr_D).get_levels()

X2 = cTerm(schema).get_levels()
X = X1.concat(X2).dropna()

tickers = ['LHHYCOR',
            'LHPEHCO',
            'MLIE00L',
            'MLF6C4L',
            'MLAC3BL',
            'MLUR41L',
            'LHHYCPE',
            'LHHYCOR',
            'MLU3BBL',
            'MLFVC4L',
            'MLEC8GL',
            'MLU3BAL',
            'MLU3BDL',
            'MLCC3BL',
            'MLJC3BL',
            'MLU3BH',
            'MLEC8DL',
            'MLEC8EL',
            'MLUR45L',
            'MLXP3BL',
            'MLU3BTL',
            'MLUR40L',
            'LHIGBAA']

# 2. Get all corporate bond total return data from database
tickers = sessionFac.query(BondIndexSpec.uid, BondIndexSpec.region, BondIndexSpec.ticker).filter(BondIndexSpec.ticker.in_(tickers)).all()

uids = [x[0] for x in tickers]
q = sessionFac.query(BondIndex.uid, BondIndex.date, BondIndex.RI, BondIndex.IN).filter(BondIndex.uid.in_(uids))
data = session.query_format_df(q)
data = data.set_index('uid', drop=True)

panel = CTimeSeries(ts_type=TimeSeriesType.RETURNS)
ctr = 0
for uid, region, ticker in tickers:

    print(len(tickers) - ctr)
    df_ = data.loc[uid].reset_index(drop=True).set_index('date')
    df_['IN'] = df_['IN'] + 100
    ts = CTimeSeries(df_.dropna(axis=1), ts_type=TimeSeriesType.LEVELS)
    ts = ts[START_DATE:].dropna()
    try:
            rfr = gds.get_risk_free_rate_time_series(region).get_levels()

            common_dates = np.intersect1d(ts.index, rfr.index)
            ts_prime = ts.loc[common_dates].get_returns()
            rfr_prime = rfr.loc[common_dates].get_returns()
            xr = ts_prime.subtract_over_common_dates(rfr_prime).get_levels()

            common_dates_XY = np.intersect1d(xr.index, X.index)
            Y = xr.reindex(common_dates_XY).get_returns()
            X_prime = X.reindex(common_dates_XY).get_returns()

            res = Regression.get_residuals_from_OLS_fit(X_prime.values, Y.values, False)
            new_df = pd.DataFrame(res, index=common_dates_XY[1:], columns=[ticker])

            spec = sessionFac.query(BondIndexSpec).filter(BondIndexSpec.uid == uid).first()
            new_df.columns = pd.MultiIndex.from_tuples([(spec.name, spec.rating, spec.maturity_band, spec.maturity, ticker)])

            new_ts = CTimeSeries(new_df, ts_type=TimeSeriesType.RETURNS)
            panel = panel.concat(new_ts)

    except:
            pass
    ctr += 1


_POST_90 = panel.dropna(axis=1)

# Run PCA on the residuals
X_centred = _POST_90 - _POST_90.mean()
cov_mat = np.cov(X_centred, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)

# Sort eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Project the data onto the principal components
pca_result = np.dot(_POST_90, eigenvectors / eigenvectors.sum(axis=0))

# Create a new DataFrame to store the PCA results
pca_df = pd.DataFrame(data=pca_result, columns=[f'PCA_Component_{i+1}' for i in range(len(_POST_90.columns))])

