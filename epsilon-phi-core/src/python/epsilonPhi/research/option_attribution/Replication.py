import pandas as pd
import numpy as np
import os, sys
import statsmodels.api as sm



ds = pd.read_csv('/Users/francisbarker/Desktop/SPX Options 1.csv')
ds = ds.set_index('date', drop=True)
ds.index = pd.to_datetime(ds.index)
ds = ds[['Z', 'T', 'dsig', 'sig', 's', 'rx']].sort_index()


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
# Y is the change in implied vol

Xs = sm.add_constant(mu_)
y = dsig.get(0) * 252

for col in y.columns:
  X_ = Xs[['const', col]]
  y_ = y.get(col)

  model = sm.OLS(y_, X_)
  results = model.fit()

