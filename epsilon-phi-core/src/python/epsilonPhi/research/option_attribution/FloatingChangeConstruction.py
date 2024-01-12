import pandas as pd
import numpy as np
from scipy.io import loadmat
from epsilonPhi.research.option_attribution.Fundailydivstrip import Fundailydivstrip

# Load data (assuming the .mat file has a similar structure to a Pandas DataFrame)
data = loadmat('SPXChain.mat')  # Adjust the path as needed
# Suppose 'IVS' is the key in the .mat file for the data we are interested in
IVS = data['IVS']

# Assigning column names
pos = {
    'date': 0, 'expiry': 1, 'id': 2, 'days': 3, 'F': 4,
    'S': 5, 'R': 6, 'd': 7, 'K': 8, 'CP': 9, 'iv': 10,
    'bid': 11, 'ask': 12
}


# Filtering data - only keep rows that meet the expiry and liquidity conditions
IVS_filtered = IVS[(IVS.iloc[:, pos['days']] > 7) & (IVS.iloc[:, pos['iv']] > 0.01) &
                   (IVS.iloc[:, pos['bid']] > 0) & (IVS.iloc[:, pos['ask']] > 0)]

# Calculating forward mid-price
fm = np.nanmean(IVS_filtered.iloc[:, [pos['bid'], pos['ask']]], axis=1) * np.exp(IVS_filtered.iloc[:, pos['days']] / 365)
IVS_filtered['fm'] = fm  # Adding as a new column
pos['fm'] = 14  # Assuming fm is the 14th column

# Unique dates and setup

# get all dates
dd = IVS_filtered.iloc[:, pos['date']]
# get all unique dates
ud, indi = pd.unique(dd, return_index=True)
# get the underlying spot prixe corresponding to each unique
# pricing date
uS = IVS_filtered.iloc[indi, pos['S']]

# Time to maturity and moneyness levels
# get the target maturity/expiries
mh = np.array([30, 60, 91, 182, 365]) / 365
# number of expiries to include
nm = len(mh)
# log expiry
lmh = np.log(mh)
# create moneyness ranges
xv = np.arange(-2, 2.5, 0.5)
# nx is the number of moneyness points on the skew
nx = len(xv)

# Preparing matrices
mm = np.tile(lmh, (1, nx))
xm = np.tile(xv, (nm, 1))
zzm = np.column_stack((mm.flatten(), xm.flatten()))
nz = len(zzm)

# Initializing matrices for storage
T = len(ud) # T is the number of unique dates
# Z is the 3D surface across T, strikes and maturities
Z = np.full((T, nx, nm), np.nan)
LIV, DIV, PPV, DPV, PCV, DCV = [Z.copy() for _ in range(6)]
Rv = np.full((T, 1), np.nan)
Sv = Rv.copy()

import multiprocessing
import scipy.io

def process_date(t):
    # The implementation will depend on the specifics of the Fundailydivstriptest function
    Rvt, Svt, LIt, DIt, LCt, DCt, LPt, DPt = Fundailydivstrip(t, ud, dd, IVS_filtered, lmh, xv, nm, nx, pos)
    return Rvt, Svt, LIt, DIt, LCt, DCt, LPt, DPt

# Prepare for parallel processing
pool = multiprocessing.Pool()

# Run the process_date function in parallel for the required range of dates
results = pool.map(process_date, range(T-1, T))

# Store the results in the appropriate matrices
for idx, (t) in enumerate(range(T-1, T)):
    Rvt, Svt, LIt, DIt, LCt, DCt, LPt, DPt = results[idx]
    DIV[t, :, :] = DIt
    LIV[t, :, :] = LIt
    PCV[t, :, :] = LCt
    DCV[t, :, :] = DCt
    PPV[t, :, :] = LPt
    DPV[t, :, :] = DPt
    Sv[t] = Svt
    Rv[t] = Rvt

# Save the results in a .mat file
output = {'DIV': DIV, 'LIV': LIV, 'PPV': PPV, 'DPV': DPV, 'DCV': DCV, 'PCV': PCV, 'Rv': Rv, 'Sv': Sv, 'ud': ud, 'xv': xv, 'mh': mh}
scipy.io.savemat('../data/spxfloatingfixedchanges4.mat', output)
