import numpy as np
import pandas as pd

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

T, nx, nm = DIV.shape
x = DIV.reshape(T, nx * nm)
indf = np.isfinite(np.sum(x, axis=1))
Tf = np.sum(indf)
xf = x[indf, :]
df = ud[indf]
Rf = Rv[indf]

cc = np.corrcoef(xf, rowvar=False)

xm = np.tile(xv, (1, nm)).reshape(-1)
mm = np.tile(mh, (nx, 1)).T.reshape(-1)
nxm = nx * nm
dx = np.empty((nxm, nxm))
dm = np.empty((nxm, nxm))
dlm = np.empty((nxm, nxm))

for i in range(nxm):
    for j in range(nxm):
        dx[i, j] = xm[i] - xm[j]
        dm[i, j] = mm[i] - mm[j]
        dlm[i, j] = np.log(mm[i]) - np.log(mm[j])

adx = np.abs(dx)
adm = np.abs(dm)
uadm = np.unique(adm)
num = len(uadm)

ind0 = adm == 0
x0 = adx[ind0]
c0 = cc[ind0]

ind1 = adm == uadm[-1]
x1 = adx[ind1]
c1 = cc[ind1]

avol = np.std(Rf) * np.sqrt(252)
Rff = np.tile(Rf, (1, nx * nm))

L = 21
LL = 63

omg = np.empty((Tf, nx * nm))
gmg = np.empty((Tf, nx * nm))
mug = np.empty((Tf, nx * nm))

for t in range(L, Tf):
    tt = np.arange(t - L + 1, t + 1)
    omg[t, :] = np.mean(xf[tt, :]**2, axis=0) * 252
    gmg[t, :] = np.mean(xf[tt, :] * Rff[tt, :], axis=0) * 252
    mug[t, :] = np.mean(xf[tt, :], axis=0)

# Now you can use omg, gmg, and mug for further analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assuming cc0, dlm, adx, xm, mm, DIV, and mh are defined previously

cc0 = cc - np.tril(np.full((45, 45), np.nan))
y = cc0.flatten()
X = np.column_stack((np.ones(len(y)), np.abs(dlm.flatten()), adx.flatten()))
indf1 = np.isfinite(y)
y = y[indf1]
X = X[indf1, :]
B = np.linalg.lstsq(X, y, rcond=None)[0]
e = y - X.dot(B)
R2 = 1 - np.var(e) / np.var(y)
sB = np.sqrt(np.diag(np.linalg.inv(X.T.dot(X)).dot(np.var(e))))
print('Table 1 Local commonality regression')
print(f'Estimates   &  {B[0]:7.3f}  &  {B[1]:7.3f}  &  {B[2]:7.3f} &  {R2:7.3f}      ')
print(f'Std Error   &  {sB[0]:7.3f}  &  {sB[1]:7.3f}  &  {sB[2]:7.3f} &     ---       ')

print('correlation stats: mean, median, min, max std')
print([np.mean(y), np.median(y), np.min(y), np.max(y), np.std(y)])

idz = np.where((xm == 0) & (mm == 91 / 365))
DF = DIV[indf][:, 5, 3]
cp = 100 * np.corrcoef(DF, xf, rowvar=False)

plt.figure(1)
plt.clf()
plt.plot(xv, cp[:, 0], '-.', xv, cp[:, 1], '--', xv, cp[:, 2], '-', xv, cp[:, 3], '--', xv, cp[:, 4], '-.', linewidth=5)
plt.grid(True)
plt.xlabel('Moneyness, $x$', fontsize=36)
plt.ylabel('Correlation, %', fontsize=16)
plt.legend([str(round(12 * m)) for m in mh])
plt.gca().set_box_on(True)
plt.gca().set_linewidth(5)
plt.xticks(fontsize=36)
plt.yticks(fontsize=36)

cv = np.cov(xf, rowvar=False)
D, C = np.linalg.eig(cv)
dD = D / np.sum(D)
dD = dD[::-1]
plt.figure(2)
plt.clf()
plt.bar(range(1, 11), 100 * dD[:10], linewidth=5)
plt.xlabel('Principle components', fontsize=36)
plt.ylabel('Explained variation, %', fontsize=36)
plt.gca().set_box_on(True)
plt.gca().set_linewidth(5)
plt.xticks(fontsize=36)
plt.yticks(fontsize=36)

C = C[:, ::-1]
C1 = C[:, 0].reshape(nx, nm)
plt.figure(3)
plt.clf()
plt.plot(xv, C1[:, 0], xv, C1[:, 1:3], '--', xv, C1[:, 3:5], '-.', linewidth=5)
plt.xlabel('Moneyness, $x$', fontsize=36)
plt.ylabel('1st principle loading', fontsize=36)
plt.legend([str(round(12 * m)) for m in mh], loc='NW')
plt.gca().set_box_on(True)
plt.gca().set_linewidth(5)
plt.xticks(fontsize=36)
plt.yticks(fontsize=36)

C2 = C[:, 1].reshape(nx, nm)
plt.figure(4)
plt.clf()
plt.plot(xv, C2[:, 0], xv, C2[:, 1:3], '--', xv, C2[:, 3:5], '-.', linewidth=5)
plt.xlabel('Moneyness, $x$', fontsize=36)
plt.ylabel('2nd principle loading', fontsize=36)
plt.legend([str(round(12 * m)) for m in mh], loc='NE')
plt.gca().set_box_on(True)
plt.gca().set_linewidth(5)
plt.xticks(fontsize=36)
plt.yticks(fontsize=36)

C3 = C[:, 2].reshape(nx, nm)
plt.figure(5)
plt.clf()
plt.plot(xv, C3[:, 0], xv, C3[:, 1:3], '--', xv, C3[:, 3:5], '-.', linewidth=5)
plt.xlabel('Moneyness, $x$', fontsize=36)
plt.ylabel('3rd principle loading', fontsize=36)
plt.legend([str(round(12 * m)) for m in mh], loc='NW')
plt.gca().set_box_on(True)
plt.gca().set_linewidth(5)
plt.xticks(fontsize=36)
plt.yticks(fontsize=36)

plt.show()
