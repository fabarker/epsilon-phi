import numpy as np
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.alchemist.DataModel import InterestRateSpec, TimeSeriesSpec
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.utils.DateUtils import DateUtils
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm
import scipy.optimize as optimize

ds = pd.read_csv('/Users/francisbarker/Documents/MATLAB/Option Attribution/OptionMetrics.csv')
ds = ds.set_index('date', drop=True)
ds.index = pd.to_datetime(ds.index, format='%d/%m/%Y')

# SPX
spx = pd.read_csv('/Users/francisbarker/Desktop/SPX.csv', index_col=0)
spx.index = pd.to_datetime(spx.index, format='%d/%m/%Y')
spx = spx.reindex(np.unique(ds.index)).ffill()

# Pricing Dates
unique_Ts = np.unique(ds.index)
T = len(unique_Ts)

mh = DateUtils.Rdate_to_mat(['1m', '2m', '3m', '6m', '12m'])
lmh = np.log(mh)
nm = len(mh)

xv = np.arange(-2, 2.5, 0.5)
nx = len(xv)

res = pd.DataFrame()
for t in range(T-1):

    print(T-1-t)
    tmp_t = ds.loc[unique_Ts[t]].reset_index(drop=True).set_index(['ttm','relativeStrike'])
    tmp_T = ds.loc[unique_Ts[t+1]].reset_index(drop=True).set_index(['ttm','relativeStrike'])

    tmp_t = tmp_t[~tmp_t.index.duplicated(keep='first')]
    tmp_T = tmp_T[~tmp_T.index.duplicated(keep='first')]

    ivol_t = tmp_t.get('IV')
    ivol_T = tmp_T.get('IV')

    dy = np.log(ivol_T.loc[ivol_t.index] / ivol_t)
    y = ivol_t.copy()

    tau = np.array(tmp_t.index.get_level_values(0))
    lm = np.log(tau).reshape(-1, 1)

    k = np.log(tmp_t.get('K') / spx.loc[unique_Ts[t]].values)
    zp = (k + 0.5 * np.power(y, 2) * tau) / (y * np.sqrt(tau))
    zm = (k - 0.5 * np.power(y, 2) * tau) / (y * np.sqrt(tau))

    wpc = norm.cdf(zm) * 0 + 1
    #wpc[wpc < 0.2] = 0

    sig_zp = np.std(zp, ddof=1)
    sig_lm = np.std(lm, ddof=1)

    hz = 1 * np.power(4/3, 1/5) * sig_zp / np.power(tmp_t.shape[0], 1/5)
    hm = 2 * np.power(4/3, 1/5) * sig_lm / np.power(tmp_t.shape[0], 1/5) * 0.1

    xm = np.abs((lm.repeat(nm, 1) - lmh.reshape(1,-1).repeat(tmp_t.shape[0], 0)))/hm

    # Maturity Weights
    wm = wpc.reshape(-1, 1) * np.exp(-1 * np.power(xm, 2)/2)
    mat_wts = pd.DataFrame(wm, columns=mh)
    mat_wts.index = pd.MultiIndex.from_tuples(list(zip(tau, zp.values)))

    xk = (np.abs((pd.concat([zp] * nx, axis=1) - xv.reshape(1, -1).repeat(tmp_t.shape[0], 0))))/hz
    xk.columns = xv
    x_wts = np.exp(-1 * np.power(xk, 2)/2)
    x_wts.index = pd.MultiIndex.from_tuples(list(zip(tau, zp.values)))

    x_wts__ = pd.concat([x_wts] * mat_wts.shape[1], axis=1)
    m_wts__ = pd.concat([mat_wts] * x_wts.shape[1], axis=1)

    wts_ = x_wts__ * m_wts__.reindex(x_wts__.index).values
    wts_ = wts_ / wts_.sum()
    wts_.columns = pd.MultiIndex.from_tuples(list(zip(x_wts__.columns, m_wts__.columns)))

    dy_ = np.sum(wts_ * dy.values.reshape(-1, 1), axis=0)
    y_ = np.sum(wts_ * y.values.reshape(-1, 1), axis=0)

    df_ = pd.concat((dy_, y_), axis=1).reset_index(drop=False)
    df_.columns = ['Z', 'T', 'dsig', 'sig']
    df_['s'] = spx.loc[unique_Ts[t]].values[0]
    df_['rx'] = np.log(spx.loc[unique_Ts[t+1]].values / spx.loc[unique_Ts[t]].values)[0]
    df_['date'] = unique_Ts[t]

    res = pd.concat((res, df_), axis=0)

res.to_csv('/Users/francisbarker/Desktop/SPX Options.csv')





