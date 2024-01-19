import numpy as np
from scipy.interpolate import interp1d
import numpy as np
from scipy.linalg import inv
import statsmodels.api as sm

# Load data from a MATLAB .mat file
import scipy.io



data = scipy.io.loadmat('../data/spxfloatingfixedchanges4.mat')
DIV = data['DIV']
LIV = data['LIV']
PPV = data['PPV']
DPV = data['DPV']
DCV = data['DCV']
PCV = data['PCV']
Rv = data['Rv']
Sv = data['Sv']
ud = data['ud']
xv = data['xv']
mh = data['mh']

nx = len(xv)
inda = xv == 0
indm = np.arange(5)
mh = mh[indm]
nm = len(mh)
indf = np.isfinite(np.sum(np.sum(DIV, axis=2), axis=1))
T = np.sum(indf)
ud = ud[indf]
Rv = Rv[indf]
sigma = np.sqrt(np.mean(Rv**2) * 252)

dA = DIV[indf][:, inda][:, :, indm]
A = LIV[indf][:, inda][:, :, indm]
A2 = A**2
TA = A**2 * mh
mu = 0.5 * np.diff(A2, axis=1) / np.diff(TA, axis=1)
amh = (mh[:-1] + mh[1:]) / 2
mT = np.tile(mh, (T, 1))
sl = np.diff(A, axis=1) / np.diff(mT, axis=1)
asl = np.empty((T, nm))
mmu = np.empty((T, nm))
LV = [21]  # You may need to provide appropriate values for LV and nl
nl = len(LV)
theta = np.empty((T, nm, nl))


# Assuming mmu, asl, theta, Ad, cAd, cdA, t2a, t2b, t2c, and t3 are defined previously

for t in range(T):
    mmu[t, :] = interp1d(amh, mu[t, :], kind='linear', fill_value='extrapolate')(mh)
    asl[t, :] = interp1d(amh, sl[t, :], kind='linear', fill_value='extrapolate')(mh)
    for j in range(nl):
        t0 = t - LV[j] + 1
        if t0 > 0:
            tt = np.arange(t0, t + 1)
            theta[t, :, j] = np.nanmean(A2[tt, :], axis=0)

Ad = np.diff(np.log(A), axis=1)

cAd = np.corrcoef(Rv[:-1], Ad, rowvar=False)[0, 1]
cdA = np.corrcoef(Rv, dA, rowvar=False)[0, 1]

t2a = [np.mean(A, axis=0), np.std(A, axis=0), np.min(A, axis=0), np.max(A, axis=0), np.correlate(A, A, mode='full')[len(A)-1:]]

t2b = [252 * np.mean(Ad, axis=0), np.sqrt(252) * np.std(Ad, axis=0), 252 * np.min(Ad, axis=0),
       252 * np.max(Ad, axis=0), np.correlate(Ad, Ad, mode='full')[len(Ad)-1:], cAd]

t2c = [252 * np.mean(dA, axis=0), np.sqrt(252) * np.std(dA, axis=0), 252 * np.min(dA, axis=0),
       252 * np.max(dA, axis=0), np.correlate(dA, dA, mode='full')[len(dA)-1:], cdA]

print("Table 2. Summary stats of atm implied vol levels and changes")
stats = ['Mean', 'Stdev', 'Minimum', 'Maximum', 'Auto']
ns = len(stats)
print("A. Implied vol level")
for j in range(ns):
    print(f"{stats[j]:<10s}", end='  ')
    print("  &   ".join([f"{t2a[j, i]:8.3f}" for i in range(nm)]) + "  \\\\ ")

stats = ['Mean', 'Stdev', 'Minimum', 'Maximum', 'Auto', 'Corr']
ns = len(stats)
print("B. Floating Implied vol change")
for j in range(ns):
    print(f"{stats[j]:<10s}", end='  ')
    print("  &   ".join([f"{t2b[j, i]:8.3f}" for i in range(nm)]) + "  \\\\ ")

print("C. Change in fixed contract implied vol change")
for j in range(ns):
    print(f"{stats[j]:<10s}", end='  ')
    print("  &   ".join([f"{t2c[j, i]:8.3f}" for i in range(nm)]) + "  \\\\ ")

t3 = [np.mean(mmu, axis=0), np.std(mmu, axis=0), np.min(mmu, axis=0), np.max(mmu, axis=0),
      np.correlate(mmu, mmu, mode='full')[len(mmu)-1:]]

print("Table 3. Summary stats for extracted rate of change from the term structure")
stats = ['Mean', 'Stdev', 'Minimum', 'Maximum', 'Auto']
ns = len(stats)
for j in range(ns):
    print(f"{stats[j]:<10s}", end='  ')
    print("  &   ".join([f"{t3[j, i]:8.3f}" for i in range(nm)]) + "  \\\\ ")


# Assuming mmu, dA, nm, T, and mh are defined previously
nlag = 21
dt = 1 / 252

R2v = np.empty(nm)
BV = np.empty((2, nm))
stdv = np.empty((2, nm))

for j in range(nm):
    X = np.column_stack((np.ones(T), mmu[:, j]))
    y = dA[:, j] / dt
    indf = np.isfinite(y + np.sum(X, axis=1))
    X = X[indf, :]
    y = y[indf]
    Tf = np.sum(indf)

    model = sm.OLS(y, X).fit()
    B = model.params
    e = model.resid
    R2v[j] = 1 - np.var(e) / np.var(y)

    data = np.column_stack((X, y))
    ja = sm.OLS(np.ones(Tf), X).fit().params
    W = sm.OLS(B, data).fit(cov_type='HAC', cov_kwds={'maxlags': nlag}).cov_params()
    vcov = inv(ja.dot(W).dot(ja.T)) / (Tf - 1)

    BV[:, j] = B
    stdv[:, j] = np.sqrt(np.diag(vcov))

print('Table 4. Predict implied vol changes')
nuv = np.zeros_like(BV)
nuv[1, :] = 1
tv1 = np.abs(BV / stdv)
tv2 = np.abs((BV - nuv) / stdv)
mmh = np.round(mh * 12)
t4 = np.empty((nm, 6))

for j in range(nm):
    t4[j, :] = [BV[0, j], tv2[0, j], BV[1, j], tv1[1, j], tv2[1, j], 100 * R2v[j]]

for i in range(nm):
    print(
        f'{mmh[i]:7.0f} & {t4[i, 0]:7.3f} & ({t4[i, 1]:7.2f}) && {t4[i, 2]:7.3f} & ({t4[i, 3]:7.2f}) & ({t4[i, 4]:7.2f}) && {t4[i, 5]:7.2f} \\\\')
