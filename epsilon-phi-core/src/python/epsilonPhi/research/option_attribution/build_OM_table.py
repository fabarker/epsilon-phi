import pandas as pd
import numpy as np
from epsilonPhi.core.utils.DateUtils import DateUtils

df = pd.read_csv('/Users/francisbarker/Desktop/SPX Normalized Vols.csv')

df = df.drop_duplicates()
df.date = pd.to_datetime(df.date, format='%Y-%m-%d')
df = df.set_index('date', drop=True)

_DAYS_PER_YEAR = ((3*365) + 366)/4
Days = DateUtils.Rdate_to_mat(df.tenor.values) * _DAYS_PER_YEAR
time_delta = pd.to_timedelta(Days, unit='D')
expiry = df.index + time_delta
F = df.get('spot').values
S = df.get('spot').values
R = df.get('spot').values * 0
Div = df.get('spot').values * 0

CP = np.full(S.shape, np.nan)
CP[df.get('relativeStrike') < 0] = 2
CP[df.get('relativeStrike') >= 0] = 1

IV = df.get('impliedVolatility').values

Bid = df.get('spot').values * 0 + 9999
Ask = df.get('spot').values * 0 + 9999

Volume = df.get('spot').values * 0
OpenInterests = df.get('spot').values * 0

ttm = Days / _DAYS_PER_YEAR

Z_plus = df.get('relativeStrike').values

atm = df[df.relativeStrike == 0].reset_index(drop=False).set_index(['date','tenor']).get('impliedVolatility')
df_ = df.reset_index(drop=False).set_index(['date','tenor'])
atms = atm.loc[df_.index]

lnm = Z_plus * atms.values * np.sqrt(ttm) - 0.5 * np.power(atms.values, 2) * ttm

K = S * np.exp(lnm)
relativeStrike = K / S
ID = [ str(Z_plus[x]).replace('.','') + str(np.round(Days[x])).replace('.','')  for x in range(df.shape[0]) ]
date = df.index

vals = [date, expiry, ID, Days, F, S, R, Div, K, CP, IV, Bid, Ask, Volume, OpenInterests, Z_plus, ttm, atms.values]

df__ = pd.DataFrame(vals).T
df__.columns = ['date', 'expiry', 'ID','Days','F','S','R','Div','K','CP','IV','Bid','Ask','Volume','OpenInterests','relativeStrike','ttm', 'atms']
df__.to_csv('/Users/francisbarker/Documents/MATLAB/Option Attribution/OptionMetrics 2.csv')