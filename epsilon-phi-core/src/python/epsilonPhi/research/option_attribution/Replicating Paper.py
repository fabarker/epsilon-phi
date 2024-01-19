import pandas as pd
import numpy as np
from scipy.io import loadmat
from epsilonPhi.research.option_attribution.Fundailydivstrip import Fundailydivstrip
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.alchemist.DataModel import ImpliedVolatility, TimeSeriesSpec
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.utils.DateUtils import DateUtils
from sqlalchemy import distinct
from scipy.stats import norm

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

           _strike = session.query(distinct(ImpliedVolatility.relative_strike)).filter(ImpliedVolatility.uid == spec.uid).scalar()
           _cols = (int(_strike.split('.')[0])/100, DateUtils.Rdate_to_mat(mat))

           df_ts.columns = pd.MultiIndex.from_tuples([_cols])
           df_ts.columns.names = ['K', 'T']
           df = pd.concat((df, df_ts), axis=1)

sig = df.dropna()
S = gds.get_dataframe_from_uid(18405, cols='PI')
S_ = S.reindex(sig.index)

def tau(sig):
    T, N = sig.shape
    return sig.columns.get_level_values('T').values.reshape(1, -1).repeat(T, 0)

def delta(sig):
    T, N = sig.shape
    return sig.columns.get_level_values('K').values.reshape(1, -1).repeat(T, 0)
#%%
def log_moneyness(S_, sig):
    from scipy.stats import norm

    T, N = sig.shape

    K_hat = sig.columns.get_level_values('K').values.reshape(1, -1).repeat(T, 0)
    T_hat = sig.columns.get_level_values('T').values.reshape(1, -1).repeat(T, 0)

    sqrt_T = np.sqrt(T_hat)
    K_hat_ = norm.ppf(K_hat)

    d1 = K_hat_ * sig.values * sqrt_T + (0.5 * np.power(sig.values, 2)) * T_hat
    d1_ = np.exp(-d1)
    K = S_.values.repeat(N, 1) * d1_

    k_ = np.log(K / S_.values.repeat(N, 1))
    return k_

def standardized_moneyness(S_, sig):
    T, N = sig.shape
    z_plus = z_plus_minus(S_, sig, plus_minus=1)
    T_hat = sig.columns.get_level_values('T').values.reshape(1, -1).repeat(T, 0)
    return z_plus / (sig.values * np.sqrt(T_hat))


def z_plus_minus(S_, sig, plus_minus=1):
    from scipy.stats import norm

    T, N = sig.shape

    K_hat = sig.columns.get_level_values('K').values.reshape(1, -1).repeat(T, 0)
    T_hat = sig.columns.get_level_values('T').values.reshape(1, -1).repeat(T, 0)

    sqrt_T = np.sqrt(T_hat)
    K_hat_= norm.ppf(K_hat)

    d1 = K_hat_ * sig.values * sqrt_T + (0.5 * np.power(sig.values, 2)) * T_hat
    d1_ = np.exp(-d1)
    K = S_.values.repeat(N, 1) * d1_

    k_ = np.log(K/S_.values.repeat(N, 1))
    return k_ + plus_minus * 0.5 * np.power(sig, 2) * T_hat

z_plus = z_plus_minus(S_, sig)
z_minus = z_plus_minus(S_, sig, -1)

# log moneyness
k = log_moneyness(S_, sig)
# standardized moneyness
x = standardized_moneyness(S_, sig)

# deltas
del_ = delta(sig)
abs_del = np.abs(del_)
ind = abs_del < 0.8

wpc = norm.cdf(z_minus)

# Times to maturity
tau_ = tau(sig)
sqrt_tau = np.sqrt(tau)
log_tau = np.log(tau)

# calucalute bandwidth 1, h(x)
sigmaz = np.std(z_plus)
hz = 1 * (4 / 3) ** (1 / 5) * sigmaz / T ** (1 / 5)

# calculate bandwidth 2, h(tau)
sigmam = np.nanstd(k, axis=0)
hm = 2 * (4 / 3) ** (1 / 5) * sigmam / T ** (1 / 5)


# Interpolant Points
_x = np.arange(-2, 2.5, 0.5)
_T = np.array([30, 60, 91, 182, 365])/365

dIV_I = np.log(sig / sig.shift(1))
I_t = sig







