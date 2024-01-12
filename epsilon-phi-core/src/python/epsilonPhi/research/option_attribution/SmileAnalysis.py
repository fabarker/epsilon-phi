import numpy as np
import scipy.optimize as optimize
import scipy.io as sio
from epsilonPhi.research.option_attribution.nanacf import nanacf
from epsilonPhi.research.option_attribution.fundeltahedgesputpl import fundeltahedgesputpl
from epsilonPhi.research.option_attribution.jac import jac
from epsilonPhi.research.option_attribution.newey2 import newey2
from epsilonPhi.research.option_attribution.callopt import callopt


# Set the global variable
moments = 'momlap'

# Load data from the .mat file
data = sio.loadmat('../data/spxfloatingfixedchanges4.mat')

# Extract the necessary variables
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

nm = len(mh)
nk = len(xv)
inda = np.where(xv == 0)[0]
xm = np.tile(xv, (1, nm)).reshape(-1)
indam = np.where(xm == 0)[0]
indf = np.isfinite(np.sum(np.sum(DIV, axis=2), axis=1))
T = np.sum(indf)
ud = ud[indf]
Rv = Rv[indf]
Sv = Sv[indf]
dA = DIV[indf, 0:nk, :].reshape(T, nk * nm)
A = LIV[indf, 0:nk, :].reshape(T, nk * nm)
dAI = DIV[indf, inda, :].reshape(T, nm)
IVVf = LIV[indf, :, :]

indk = np.where(np.abs(xv) <= 1)[0]
ndk = len(indk)
indi = np.where(np.abs(xv) <= 1 & (xv != 0))[0]
ndi = len(indi)  # investment strike choice

LL = 21  # Historical TS estimate horizon
fh = 20  # Trading holding horizon
dt = 1 / 365
estimates = 0
trading = 0
summarystats = 0
forecastingregresson = 0
tradingstats = 1

if estimates:
    # All strike
    omegaa = np.empty((T, nm * nk))
    gammaa = np.empty((T, nm * nk))

    # ATM
    Z = np.empty((T, nm))
    gammam = Z
    omegam = Z  # historical ts 1m
    gammacs = Z
    omegacs = Z
    R2v = Z  # cs
    gammaf = Z
    omegaf = Z  # future realized

    options = optimize.LinprogOptions(Display='off')

    Rva = np.tile(Rv, (1, nm))
    Rvm = np.tile(Rv, (1, nk * nm))

    PLS = np.empty((T, nk, nm))
    for t in range(T):
        if t - LL + 1 > 0:
            # Historical TS estimates
            tv = np.arange(t - LL + 1, t + 1)
            omegam[t, :] = np.mean(dAI[tv, :]**2, axis=0) * 252
            gammam[t, :] = np.mean(dAI[tv, :] * Rva[tv, :], axis=0) * 252

            omegaa[t, :] = np.mean(dA[tv, :]**2, axis=0) * 252
            gammaa[t, :] = np.mean(dA[tv, :] * Rvm[tv, :], axis=0) * 252

        if t + LL <= T:
            # Future realized (co)variance estimates
            tv = np.arange(t + 1, t + LL + 1)
            omegaf[t, :] = np.mean(dAI[tv, :]**2, axis=0) * 252
            gammaf[t, :] = np.mean(dAI[tv, :] * Rva[tv, :], axis=0) * 252

        # Current CS regression estimates
        for j in range(nm):
            A2 = IVVf[t, inda, j]**2
            It = IVVf[t, indk, j]
            I2 = It**2
            S = I2 - A2
            zp = xv[indk] * It * np.sqrt(mh[j])
            zm = zp - I2 * mh[j]
            k = zp - 0.5 * I2 * mh[j]
            X = np.column_stack((2 * zp, zp * zm))
            B = optimize.lsq_linear(X, S, bounds=([-np.inf, 0], [np.inf, np.inf]), method='trf')
            gammacs[t, j] = B[0]
            omegacs[t, j] = B[1]
            e = S - X.dot(B)
            R2v[t, j] = 1 - np.mean(e**2) / np.var(S)

            if t + fh <= T:
                # Investment PL over horizon fh
                It = IVVf[t, :, j]
                I2 = It**2
                zp = xv * It * np.sqrt(mh[j])
                k = zp - 0.5 * I2 * mh[j]
                FV = 100 * Sv[t:t + fh] / Sv[t]
                KV = 100 * np.exp(k)
                tv = (mh[j] * 365 - ud[t:t + fh] + ud[t]) / 365
                stv = np.sqrt(tv)
                Ka = 100 * np.exp(-0.5 * A2 * mh[j])
                Pa, _, Vga = callopt(100, Ka, 0, mh[j], np.sqrt(mh[j]), np.sqrt(A2))
                Pk, _, Vgk = callopt(100, KV, 0, mh[j], np.sqrt(mh[j]), np.sqrt(I2))
                PL = np.empty(nk)
                for kk in range(nk):
                    PL[kk] = fundeltahedgesputpl(np.sqrt(I2[kk]), FV, KV[kk], tv, stv, 0, 1)
                PLA = fundeltahedgesputpl(np.sqrt(A2), FV, Ka, tv, stv, 0, 1)
                PLS[t, :, j] = (PL * Vga / Vgk - PLA)

    # Save the results
    np.savez('Smilemoments.npz', gammacs=gammacs, omegacs=omegacs, gammam=gammam, omegam=omegam, omegaf=omegaf,
             gammaf=gammaf, omegaa=omegaa, gammaa=gammaa, R2v=R2v, PLS=PLS)
else:
    # Load previously saved results
    data = np.load('Smilemoments.npz')
    gammacs = data['gammacs']
    omegacs = data['omegacs']
    gammam = data['gammam']
    omegam = data['omegam']
    omegaf = data['omegaf']
    gammaf = data['gammaf']
    omegaa = data['omegaa']
    gammaa = data['gammaa']
    R2v = data['R2v']
    PLS = data['PLS']

import numpy as np

if summarystats:
    mmh = np.round(mh * 12)
    mA = np.reshape(np.nanmean(A), (nk, nm))
    sA = np.reshape(np.nanstd(A), (nk, nm))

    mo = np.reshape(np.nanmean(omegaa), (nk, nm))
    so = np.reshape(np.nanstd(omegaa), (nk, nm))
    mg = np.reshape(np.nanmean(gammaa), (nk, nm))
    sg = np.reshape(np.nanstd(gammaa), (nk, nm))

    # Select strikes to report
    sxv = [-2, -1, 0, 1, 2]
    nsx = len(sxv)
    inds = np.isin(xv, sxv)

    print("Table 5:")
    print("A. Mean implied vol smile")
    print(f"{'Maturity':<10s}", end='')
    for x_value in sxv:
        print(f"{x_value:<10d}", end='')
    print("\n", end='')

    for i in range(len(mmh)):
        print(f"{mmh[i]:<10d}", end='')
        for j in range(nk):
            if inds[j]:
                print(f"{mA[j][i]:<10.3f}", end='')
        print("\n", end='')

    print("\nB. Historical Covariance estimates")
    print(f"{'Maturity':<10s}", end='')
    for x_value in sxv:
        print(f"{x_value:<10d}", end='')
    print("\n", end='')

    for i in range(len(mmh)):
        print(f"{mmh[i]:<10d}", end='')
        for j in range(nk):
            if inds[j]:
                print(f"{mg[j][i]:<10.3f}", end='')
        print("\n", end='')

    print("\nC. Historical Variance estimates")
    print(f"{'Maturity':<10s}", end='')
    for x_value in sxv:
        print(f"{x_value:<10d}", end='')
    print("\n", end='')

    for i in range(len(mmh)):
        print(f"{mmh[i]:<10d}", end='')
        for j in range(nk):
            if inds[j]:
                print(f"{mo[j][i]:<10.3f}", end='')
        print("\n", end='')

    print("Table 6:")
    x = gammacs
    y = gammam
    cxy = np.diag(np.corrcoef(x, y, rowvar=False))  # Using np.corrcoef to compute correlation
    t6a = [np.mean(x), np.std(x), np.min(x), np.max(x), np.nanacf(x, 1), *cxy]
    stats = ['Mean', 'Stdev', 'Minimum', 'Maximum', 'Auto', 'Corr']
    ns = len(stats)
    print("A. Covariance")
    for j in range(ns):
        print(f"{stats[j]:<10s}", end='')
        for _ in range(nm):
            print(f"{t6a[j]:<10.3f}", end='')
        print("\n", end='')

    x = omegacs
    y = omegam
    cxy = np.diag(np.corrcoef(x, y, rowvar=False))  # Using np.corrcoef to compute correlation
    t6b = [np.mean(x), np.std(x), np.min(x), np.max(x), np.nanacf(x, 1), *cxy]
    print("B. Variance")
    for j in range(ns):
        print(f"{stats[j]:<10s}", end='')
        for _ in range(nm):
            print(f"{t6b[j]:<10.3f}", end='')
        print("\n", end='')

    x = R2v
    t6c = [np.mean(x), np.std(x), np.min(x), np.max(x), np.nanacf(x, 1)]
    print("C. R-squared")
    for j in range(ns - 1):
        print(f"{stats[j]:<10s}", end='')
        for _ in range(nm):
            print(f"{t6c[j]:<10.3f}", end='')
        print("\n", end='')

if forecastingregresson:
    BV = np.empty((3, nm))
    stdv = BV.copy()
    BV2 = BV.copy()
    stdv2 = BV.copy()
    R2f = np.empty(nm)
    R2f2 = R2v.copy()
    tvp = np.arange(1, T + 1)
    nlag = 21

    for j in range(nm):
        yo = omegaf[tvp - 1, j]
        xo = np.column_stack([omegacs[tvp - 1, j], omegam[tvp - 1, j]])
        indf = np.isfinite(yo + np.sum(xo, axis=1))
        Tf = np.sum(indf)
        X = np.column_stack([np.ones(Tf), xo[indf]])
        y = yo[indf]

        B = np.linalg.lstsq(X, y, rcond=None)[0]
        yh = np.dot(X, B)
        e = y - yh
        R2f[j] = 1 - np.var(e) / np.var(y)

        data = np.column_stack([X, y])
        W = newey2(B, data, nlag)
        ja = jac(B, data)
        vcov = np.linalg.inv(np.dot(ja, np.dot(W, ja.T))) / (Tf - 1)
        BV[:, j] = B
        stdv[:, j] = np.sqrt(np.diag(vcov))

        yg = gammaf[tvp - 1, j]
        xg = np.column_stack([gammacs[tvp - 1, j], gammam[tvp - 1, j]])
        indf = np.isfinite(yg + np.sum(xg, axis=1))
        Tf = np.sum(indf)
        X = np.column_stack([np.ones(Tf), xg[indf]])
        y = yg[indf]

        B = np.linalg.lstsq(X, y, rcond=None)[0]
        yh = np.dot(X, B)
        e = y - yh
        R2f2[j] = 1 - np.var(e) / np.var(y)

        data = np.column_stack([X, y])
        W = newey2(B, data, nlag)
        ja = jac(B, data)
        vcov = np.linalg.inv(np.dot(ja, np.dot(W, ja.T))) / (Tf - 1)
        BV2[:, j] = B
        stdv2[:, j] = np.sqrt(np.diag(vcov))

    print("Table 7: Predict realized (co)variance with CS and TS estimates")
    BTV2 = np.abs(BV2 / stdv2)
    BTV = np.abs(BV / stdv)
    for j in range(nm):
        table2 = [BV2[0, j], BTV2[0, j], BV2[1, j], BTV2[1, j], BV2[2, j], BTV2[2, j], 100 * R2f2[j]]
        print("   &  {:.3f}  &  ( {:.2f} )  &&  {:.3f}  & ( {:.2f} ) &&  {:.3f}  & ( {:.2f} ) &&  {:.2f}      \\\\".format(*table2))

    print()
    for j in range(nm):
        table2 = [BV[0, j], BTV[0, j], BV[1, j], BTV[1, j], BV[2, j], BTV[2, j], 100 * R2f[j]]
        print("   &  {:.3f}  &  ( {:.2f} )  &&  {:.3f}  & ( {:.2f} ) &&  {:.3f}  & ( {:.2f} ) &&  {:.2f}      \\\\".format(*table2))

if trading:
    T0 = 253 * 4  # starting point
    L = 252 * 4  # rolling window
    PLRR = np.empty((T, ndi + 1, nm))
    PLSA = PLRR.copy()
    WeightRR = PLRR.copy()
    WeightSA = PLRR.copy()

    for j in range(nm):
        for t in range(T0, T - LL):

            # rolling window forecasts
            tvp = np.maximum(1, t - L - LL)
            tv = tvp + LL

            yo = omegaf[tvp, j]
            xo = np.column_stack([omegam[tvp, j], omegacs[tvp, j]])
            indf = np.isfinite(yo + np.sum(xo, axis=1))
            Nf = np.sum(indf)
            X = np.column_stack([np.ones(Nf), xo[indf, :]])
            Bo = np.linalg.lstsq(X.T @ X + np.eye(3), X.T @ yo[indf], rcond=None)[0]
            yoh = np.array([1, omegam[t, j], omegacs[t, j]]) @ Bo

            yg = gammaf[tvp, j]
            xg = np.column_stack([gammam[tvp, j], gammacs[tvp, j]])
            indf = np.isfinite(yg + np.sum(xg, axis=1))
            Nf = np.sum(indf)
            X = np.column_stack([np.ones(Nf), xg[indf, :]])
            Bg = np.linalg.lstsq(X.T @ X + np.eye(3), X.T @ yg[indf], rcond=None)[0]
            ygh = np.array([1, gammam[t, j], gammacs[t, j]]) @ Bg

            # Current moneyness levels
            A2 = IVVf[t, inda, j] ** 2
            It = IVVf[t, indi, j]
            I2 = It ** 2
            zp = xv[indi] * It * np.sqrt(mh[j])
            zm = zp - I2 * mh[j]
            k = zp - 0.5 * I2 * mh[j]
            X = np.column_stack([zp * 2, zp * zm])

            S = I2 - A2  # Observed spread
            PLst = PLS[t, indi, j]  # PL from each spread

            # Risk-return strategy
            BP = np.column_stack([ygh, yoh])
            SH = X @ BP  # out-of-sample forecasts of historical moments
            wt = (S - SH) / A2
            PLt = wt * PLst
            PLRR[t, :, j] = np.column_stack([PLt, np.nansum(PLt)])
            WeightRR[t, :, j] = np.column_stack([wt, np.sum(wt)])

            # Static arbitrage strategy
            BP = np.column_stack([gammacs[t, j], omegacs[t, j]])
            SH = X @ BP  # CS fitting
            wt = 10 * (S - SH) / A2
            PLt = wt * PLst
            PLSA[t, :, j] = np.column_stack([PLt, np.nansum(PLt)])
            WeightSA[t, :, j] = np.column_stack([wt, np.sum(wt)])

    np.savez('Smiletrading.npz', PLRR=PLRR, PLSA=PLSA, WeightRR=WeightRR, WeightSA=WeightSA)
else:
    data = np.load('Smiletrading.npz')
    PLRR = data['PLRR']
    PLSA = data['PLSA']
    WeightRR = data['WeightRR']
    WeightSA = data['WeightSA']

if tradingstats:
    print('Table 8, summary stats on risk-return strategy weight')
    mz = np.empty((nm, ndi))
    sz = mz.copy()
    minz = mz.copy()
    maxz = mz.copy()
    autz = mz.copy()

    for j in range(nm):
        z = WeightRR[:, :ndi, j]
        mz[j, :] = np.nanmean(z, axis=0)
        sz[j, :] = np.nanstd(z, axis=0)
        minz[j, :] = np.nanmin(z, axis=0)
        maxz[j, :] = np.nanmax(z, axis=0)
        autz[j, :] = nanacf(z, 1)

    print('Sample Average')
    print(' %8.0f     &   %8.3f  &  %8.3f  &  %8.3f  &  %8.3f' % (np.column_stack([round(mh * 12), mz]).ravel()))

    print('Standard Deviation')
    print(' %8.0f     &   %8.3f  &  %8.3f  &  %8.3f  &  %8.3f' % (np.column_stack([round(mh * 12), sz]).ravel()))

    print('Minimum')
    print(' %8.0f     &   %8.3f  &  %8.3f  &  %8.3f  &  %8.3f' % (np.column_stack([round(mh * 12), minz]).ravel()))

    print('Maximum')
    print(' %8.0f     &   %8.3f  &  %8.3f  &  %8.3f  &  %8.3f' % (np.column_stack([round(mh * 12), maxz]).ravel()))

    print('Daily Autocorrelation')
    print(' %8.0f     &   %8.3f  &  %8.3f  &  %8.3f  &  %8.3f' % (np.column_stack([round(mh * 12), autz]).ravel()))

    print('Table 9, Summary stats on risk-return strategy investment PL')
    ttm = np.empty((nm, ndi + 1))
    tts = ttm.copy()
    ttir = ttm.copy()
    cc = np.empty((nm, 2))

    for j in range(nm):
        z = PLRR[:, :, j]
        ttm[j, :] = [12 * np.nanmean(z)]
        tts[j, :] = [np.sqrt(12) * np.nanstd(z)]
        ttir[j, :] = [np.sqrt(12) * np.nanmean(z) / np.nanstd(z)]
        c = (np.corrcoef(z[:, :4], rowvar=False) - np.diag(np.full(4, np.nan))).ravel()
        cc[j, 0] = np.nanmean(c)

    print('Annualized Return')
    print(' %8.0f     &   %8.3f  &  %8.3f  &  %8.3f  &  %8.3f  &  %8.3f' % (np.column_stack([round(mh * 12), ttm]).ravel()))

    print('Annualized Standard Deviation')
    print(' %8.0f     &   %8.3f  &  %8.3f  &  %8.3f  &  %8.3f  &  %8.3f' % (np.column_stack([round(mh * 12), tts]).ravel()))

    print('Annualized Information Ratio')
    print(' %8.0f     &   %8.2f  &  %8.2f  &  %8.2f  &  %8.2f  &  %8.2f' % (np.column_stack([round(mh * 12), ttir]).ravel()))

    print('Table 10, summary stats on stat-arb strategy weight')
    mz = np.empty((nm, ndi))
    sz = mz.copy()
    minz = mz.copy()
    maxz = mz.copy()
    autz = mz.copy()

    for j in range(nm):
        z = WeightSA[:, :ndi, j]
        mz[j, :] = np.nanmean(z, axis=0)
        sz[j, :] = np.nanstd(z, axis=0)
        minz[j, :] = np.nanmin(z, axis=0)
        maxz[j, :] = np.nanmax(z, axis=0)
        autz[j, :] = nanacf(z, 1)

    print('Sample Average')
    print(' %8.0f     &   %8.3f  &  %8.3f  &  %8.3f  &  %8.3f' % (np.column_stack([round(mh * 12), mz]).ravel()))

    print('Standard Deviation')
    print(' %8.0f     &   %8.3f  &  %8.3f  &  %8.3f  &  %8.3f' % (np.column_stack([round(mh * 12), sz]).ravel()))

    print('Minimum')
    print(' %8.0f     &   %8.3f  &  %8.3f  &  %8.3f  &  %8.3f' % (np.column_stack([round(mh * 12), minz]).ravel()))

    print('Maximum')
    print(' %8.0f     &   %8.3f  &  %8.3f  &  %8.3f  &  %8.3f' % (np.column_stack([round(mh * 12), maxz]).ravel()))

    print('Daily Autocorrelation')
    print(' %8.0f     &   %8.3f  &  %8.3f  &  %8.3f  &  %8.3f' % (np.column_stack([round(mh * 12), autz]).ravel()))

    print('Table 11, Summary stats on stat-arb strategy investment PL')
    ttm = np.empty((nm, ndi + 1))
    tts = ttm.copy()
    ttir = ttm.copy()

    for j in range(nm):
        z = PLSA[:, :, j]
        ttm[j, :] = [12 * np.nanmean(z)]
        tts[j, :] = [np.sqrt(12) * np.nanstd(z)]
        ttir[j, :] = [np.sqrt(12) * np.nanmean(z) / np.nanstd(z)]
        c = (np.corrcoef(z[:, :4], rowvar=False) - np.diag(np.full(4, np.nan))).ravel()
        cc[j, 1] = np.nanmean(c)

    print('Annualized Return')
    print(' %8.0f     &   %8.3f  &  %8.3f  &  %8.3f  &  %8.3f  &  %8.3f' % (
        np.column_stack([round(mh * 12), ttm]).ravel()))

    print('Annualized Standard Deviation')
    print(' %8.0f     &   %8.3f  &  %8.3f  &  %8.3f  &  %8.3f  &  %8.3f' % (
        np.column_stack([round(mh * 12), tts]).ravel()))

    print('Annualized Information Ratio')
    print(' %8.0f     &   %8.2f  &  %8.2f  &  %8.2f  &  %8.2f  &  %8.2f' % (
        np.column_stack([round(mh * 12), ttir]).ravel()))
