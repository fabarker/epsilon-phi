import numpy as np
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.alchemist.DataModel import InterestRateSpec, TimeSeriesSpec
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.research.option_attribution.DataManager import getDataManager
from epsilonPhi.core.utils.DateUtils import DateUtils
import pandas as pd
from tqdm import tqdm
import scipy.optimize as optimize

tqdm.pandas()

gds = GlobalDataSource()
mgr = SessionMgr()
session = mgr.getSessionFactory()
df = getDataManager()

# Implied Vols
sig_ = df[['date', 'relativeStrike', 'impliedVolatility', 'ttm']]
sig_dedup = sig_.drop_duplicates(subset=['date', 'relativeStrike', 'ttm'])
sig = sig_dedup.pivot(index='date', columns=['relativeStrike', 'ttm'], values='impliedVolatility')

# Spot Rates
spt = df[['date','spot']].set_index('date').drop_duplicates().reindex(sig.index)

## Reshape the Data Into Time Series Matrix Array

LL = 21  # Historical TS estimate horizon
fh = 20  # Trading holding horizon
dt = 1 / 365

DIV = np.log(sig).diff()
LIV = sig.copy()

Rv = np.log(spt.reindex(sig.index)).diff() # Log spot change
Sv = spt.reindex(sig.index) # Spot price
ud = sig.index # pricing date
xv = sig.columns.get_level_values(0).to_numpy().reshape(1,-1)
mh = sig.columns.get_level_values(1).to_numpy().reshape(1,-1) # maturities

atm_locs = (xv == 0).flatten()

unique_mats = np.unique(mh)
unique_strikes = np.unique(xv)


## Figure 2: Factor Loading of Principle Components

cv = np.cov(sig)
D, C = np.linalg.eig(cv)
C_ = pd.DataFrame(C, index=sig.columns)

idx = np.argsort(D)
C_.index = C_.index.reorder_levels([1,0])


## Table 2: Summary Statistics for ATM Implied Volatility Levels and Daily Changes




## Table 3: Extracted Rate of Change from ATM Implied Variance Term Structure





## Table 4; Predict Impled Volatility Changes with Term Structure Slope




## Table 5: Mean Implied Volatility Smile and Time Seriex Var/Covar Estimates




## Table 6: Cross-Sectional Regression Estimates of Variance and Covariance Rates



## Table 7: Predict Realized Variance/Covariance with Cross-Sectional and Time Series Estimators



## Table 8: Summary Statistics on the Risk-Return Strategy Investment Weights




## Table 9: Out of Sample Option Investment P&L SStatistics from the Risk-Return Trade-Off Strategy




## Table 10: Summary Statistics for Stat Arb Strategy Investment Weights



## Table 11: Out-of-sample Option Investment P&L Statistics from Stat Arb Strategy








# squared change in log IV
# omega = (( dI / I ) ^ 2 ) / dt
omegam = (DIV ** 2) * 252

# change in log IV of ATM across maturities multiplied by the spot price
# ((ds/s) * (dI/I))/dt
gammam = DIV * Rv.values.repeat(DIV.shape[1], 1)

# Run regression of historical realized performance on future realized
# That is, does historical realzied moments predict future?

# IVVf is the level of implied vol at t
# A2 = A^2 which is ATM
A2 = LIV.iloc[:,atm_locs]**2
A2.columns = A2.columns.droplevel(0)

# It is in the wings
It = LIV.iloc[:,~atm_locs]**2
# I^2
I2 = It**2

# Spread between ATM and Wing Points, Our LHS of regression
S = I2 - A2[I2.columns.get_level_values('ttm')].values

# z-plus and z-minus, which are the factors on the RHS of regression equation
# CHECK IN PAPER TO SEE IF IT SHOULD BE STANDARDIZED STRIKE OR RELATIVE STRIKE

zp = It.columns.get_level_values('relativeStrike').to_numpy().reshape(1,-1).repeat(It.shape[0], 0) * It * np.sqrt(It.columns.get_level_values('ttm').to_numpy().reshape(1,-1).repeat(It.shape[0], 0))
zm = zp - I2 * It.columns.get_level_values('ttm').to_numpy().reshape(1,-1).repeat(It.shape[0], 0)

k = zp - 0.5 * I2 * It.columns.get_level_values('ttm').to_numpy().reshape(1,-1).repeat(It.shape[0], 0)

for mat in unique_mats:

    # stack together our 2 factors
    zp_ = zp.iloc[:, zp.columns.get_level_values(1) == mat]
    zm_ = zm.iloc[:, zm.columns.get_level_values(1) == mat]
    S_ = S.iloc[:, S.columns.get_level_values(1) == mat]

    X = np.column_stack((2 * zp, zp * zm))

    # Run the regreession and extract the coefficients, which are, gamma and omega^2
    # S is the spread between the ATM vol and the point on the wing we care about
    # This is a cross sectional regression for a given t

    # We regress implied vol spreads on the coefficients in the pricing relation
    # I2(t) − A2(t) = γ(t)(2z+) + ω2(t)(z+z−)+et.
    # In the paper, they use the trailing 21 day average
    # for I2 and A2

    B = optimize.lsq_linear(X, S_, bounds=([-np.inf, 0], [np.inf, np.inf]), method='trf')

    gammacs[t, j] = B[0]
    omegacs[t, j] = B[1]

    # compute the erros
    e = S - X.dot(B)
    # compute the RSQ
    R2v[t, j] = 1 - np.mean(e**2) / np.var(S)