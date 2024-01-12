import pandas as pd
import numpy as np
from scipy.io import loadmat
from epsilonPhi.research.option_attribution.Fundailydivstrip import Fundailydivstrip
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.alchemist.DataModel import ImpliedVolatility, TimeSeriesSpec
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.utils.DateUtils import DateUtils
from sqlalchemy import distinct

gds = GlobalDataSource()
mgr = SessionMgr()
session = mgr.getSessionFactory()


uids_tups = session.query(TimeSeriesSpec).filter(
    TimeSeriesSpec.ticker.startswith('SPX'),
    TimeSeriesSpec.category == 'Implied Volatility',
    TimeSeriesSpec.provider == 'GS').all()

mats = uids_tups[0].ticker.split(':')[1]


df = pd.DataFrame()
for spec in uids_tups:
    mat = spec.ticker.split(':')[1]

    if DateUtils.Rdate_to_mat(mat) <= 1 and DateUtils.Rdate_to_mat(mat) >= (1/12)-(7/365):
        df_ = mgr.get_dataframe_from_uid(spec.uid)
        if df_.size > 0:
           print(spec)
           df_ts = df_[['date', 'mid']].set_index('date')
           info = session.query(ImpliedVolatility.tenor, ImpliedVolatility.relative_strike, ImpliedVolatility.strike_reference).filter(ImpliedVolatility.uid == spec.uid).first()
           df_ts.columns = pd.MultiIndex.from_tuples([info])
           df_ts.columns.names = ['T', 'K', 'ref']
           df = pd.concat((df, df_ts), axis=1)


_df = df.dropna()

mat = _df.cov()
L, V = np.linalg.eigh(mat)
loadings = pd.DataFrame(V[:, -5:], index=_df.columns)
loadings = loadings[range(4,0,-1)].T.reset_index(drop=True)

one_month = loadings.iloc[:, loadings.columns.get_level_values('T') == '1m']
one_month.columns = one_month.columns.get_level_values('K')
one_month.columns = [int(x.split('.')[0]) for x in one_month.columns]
one_month = one_month[one_month.columns.sort_values()]



mnth = _df.get('1m')
mnth.columns = [ int(x.split('.')[0]) for x in mnth.columns.get_level_values('K') ]

T, N  = mnth.shape

from scipy.stats import norm
A = np.array([norm.ppf(x/100) for x in mnth.columns]).reshape(1,-1).repeat(T,0)
sig = mnth.values
tau = np.sqrt(1/12)

d1 = A * sig * tau + ( 0.5 * np.power(sig, 2) ) * (1/12)
d1_ = np.exp(-d1)
K = S.reindex(_df.index).values.repeat(N, 1) * d1_

k_ = np.log(K / S.reindex(_df.index))

import numpy as np
from scipy.stats import norm

S = gds.get_dataframe_from_uid(18405, cols='PI')

# Define the known parameters
S = 100.0  # Current stock price
T = 1.0    # Time to maturity (in years)
r = 0.05   # Risk-free interest rate
Delta = 0.6  # Delta of the option (e.g., 0.6 for 60% Delta)
sigma = 0.2  # Implied volatility (annualized)

# Calculate moneyness (K/S) using Black-Scholes formula
def calculate_moneyness(S, T, r, Delta, sigma):
    d1 = norm.ppf(Delta) * sigma * np.sqrt(T) + (r + 0.5 * sigma**2) * T
    K = S * np.exp(-d1)
    return K / S

moneyness = calculate_moneyness(S, T, r, Delta, sigma)
print("Moneyness:", moneyness)