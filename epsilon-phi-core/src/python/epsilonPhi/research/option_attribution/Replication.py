import pandas as pd
import numpy as np
import os, sys
import statsmodels.api as sm
from epsilonPhi.research.option_attribution.callopt import *


ds = pd.read_csv('/Users/francisbarker/Desktop/SPX Options 1.csv')
ds = ds.set_index('date', drop=True)
ds.index = pd.to_datetime(ds.index)
ds = ds[['Z', 'T', 'dsig', 'sig', 's', 'rx']].sort_index()
unique_dates = ds.index.unique()
unique_mats = ds.get('T').unique()
T = len(unique_dates)

# Get the Vol Changes

dsig = ds[['Z', 'T', 'dsig']].reset_index(drop=False).set_index(['date', 'Z', 'T'])
dsig = dsig.unstack(level=[1, 2]).get('dsig')
dsig = dsig.sort_index(axis=1, level=1)

# Principle Component Analysis
cov_mat = dsig.cov()
L, V = np.linalg.eig(cov_mat)

pca = pd.DataFrame(V, columns=L, index=dsig.columns).sort_index(axis=1, ascending=False)

plot_pc = 4
pca.get(pca.columns[plot_pc-1]).unstack().plot()

# ATM Analysis - Table 2 : Summary of ATM Implied Vol and Changes

sig = ds[['Z', 'T', 'sig']].reset_index(drop=False).set_index(['date', 'Z', 'T'])
sig = sig.unstack(level=[1, 2]).get('sig')
sig = sig.sort_index(axis=1, level=1)

At = sig.get(0)
At_sq = np.power(At, 2)
TA = At_sq * At_sq.columns.values

mu = (0.5 * At_sq.diff(axis=1)/TA.diff(axis=1)).dropna(axis=1)
amh = 0.5 * (TA.columns.values[0:-1] + TA.columns.values[1:])
mu.columns = amh
mu[[At_sq.columns]] = np.nan
mu_ = mu.sort_index(axis=1).interpolate(axis=1)
mu_[mu_.columns[0]] = mu[mu.columns[0]]
mu_ = mu_.get(At.columns)

# Summary Stats for ATM Implied Vol Level (At)
print(At.describe().reindex(['mean','std','min','max']))

# Summary Stats for Change in ATM Implied Vol (diff(log(A))
print(np.log(At).diff().describe().reindex(['mean','std','min','max']) * np.array([252, np.sqrt(252), 252, 252]).reshape(-1, 1))

# Summary Stats for Change in ATM Floating
print(dsig.get(0).describe().reindex(['mean','std','min','max']) * np.array([252, np.sqrt(252), 252, 252]).reshape(-1, 1))

print(mu_.describe().reindex(['mean','std','min','max']))

# Check to see if We can Forecast Future Vol from Drift

# X is the implied vol drift term
# Y is the change in implied vol oevr the next time step
# Therefore these are prediction regressions over 1 day

Xs = sm.add_constant(mu_)
y = dsig.get(0) * 252

for col in y.columns:
  X_ = Xs[['const', col]]
  y_ = y.get(col)
  reg = np.linalg.lstsq(X_, y_)

  betas = reg[0]
  res = y_ - X_ @ reg[0]
  r_sq = 1 - np.var(res) / np.var(y_)

# Wing Analysis - Table 2 : Summary of ATM Implied Vol and Changes

LL = 21 # Rolling Window Lookback
fh = 20 # Holding Period for Trading Strategies

dI = dsig.copy()
dIsq = np.power(dI, 2)
dA = dI.get(0)
dAsq = dIsq.get(0)

# rx is the spot return
rx = ds.get('rx').drop_duplicates().reindex(dA.index).fillna(0)
spt = ds.get('s').drop_duplicates().reindex(dA.index).ffill()

dIdS = dI * rx.values.reshape(-1, 1)
dAdS = dA * rx.values.reshape(-1, 1)

omega = dIsq.rolling(window=LL).mean() * 252
gamma = dIdS.rolling(window=LL).mean() * 252

# Lag the Omega and Gammas so we can run regressions
omega_f = omega.shift(-LL).get(0)
gamma_f = gamma.shift(-LL).get(0)

# S is the spread vols
sig_sq = np.power(sig, 2)
S = sig_sq - sig_sq.get(0)[sig.columns.get_level_values('T')].values

X = np.array(S.columns.get_level_values(0)).reshape(1, -1).repeat(T, 0)
M = np.array(S.columns.get_level_values(1)).reshape(1, -1).repeat(T, 0)

z_plus = X * sig * np.sqrt(M)
z_minus = z_plus - sig_sq * M
k = z_plus - 0.5 * sig_sq * M

strikes = np.exp(k) * 100
call_prices, call_delta, call_vega = call_price(spt * 0 + 100, strikes, 0, M, np.sqrt(M), sig)
put_prices, put_delta, put_vega = put_price(spt * 0 + 100, strikes, 0, M, np.sqrt(M), sig)

_paths = np.array(range(0, T - 21)).reshape(1, T-21).repeat(21, 0) +\
            np.array(range(21)).reshape(-1, 1).repeat(T - 21, 1)

sim_paths = pd.DataFrame(_paths.T, index=put_prices.index[:T-21])
ttms = pd.DataFrame((unique_mats.reshape(-1, 1).repeat(21, 1) -
             ((1/365.25) * np.array(sim_paths.columns).reshape(1, -1).repeat(len(unique_mats), 0))), index=unique_mats)

spt_paths = pd.DataFrame(spt.values[sim_paths.values], index=sim_paths.index)
spt_paths = 100 * spt_paths / spt_paths.values[:,0].reshape(-1, 1)

_vols_paths = pd.concat([sig.unstack()] * 21, axis=1)
_strike_paths = pd.concat([strikes.unstack()] * 21, axis=1)
_spt_paths = spt_paths.reindex(_vols_paths.index.get_level_values(2))
_spt_paths.index = _vols_paths.index

_mat_paths = ttms.loc[_vols_paths.index.get_level_values(1)]
_mat_paths.index = _vols_paths.index

call_prices_, _, _ = call_price(_spt_paths, _strike_paths, 0, _mat_paths, np.sqrt(_mat_paths), _vols_paths)
put_prices_, _, _ = call_price(_spt_paths, _strike_paths, 0, _mat_paths, np.sqrt(_mat_paths), _vols_paths)
