import pandas as pd
import numpy as np
from tqdm import tqdm
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
import datetime as datetime
import os
tqdm.pandas()
from epsilonPhi.core.dataModel.alchemist.DataModel import *
import os
import pandas as pd

# Replace 'path/to/your/file' with the actual file path
file_path = r'C:\Users\fabar\OneDrive\Desktop\data\WRDS_MMN_Corrected_Data.csv.gzip'

# Read the gzipped CSV file
df_db = pd.read_csv(file_path, compression='gzip')
db_cusips = df_db.get('cusip').unique()
db_cusips = [ str(x) for x in db_cusips ]

df_ds = pd.read_excel(r'C:\Users\fabar\OneDrive\Desktop\data\Bond Data.xlsx', sheet_name='Info')
df_cusips = df_ds.get('Cusips').unique()
df_cusips = [ str(x) for x in df_cusips ]

common_cusips = np.intersect1d(db_cusips, df_cusips)
print(len(common_cusips))

file_dir = '/Users/francisbarker/Downloads/WRDS_DMR_Replication/'
sample_type      = 'bbw'
weighting_scheme = 'vw'
return_type      = 'duration_adj'#excess_rf'
factor_type      = 'v0'


df = pd.read_hdf(os.path.join(file_dir, 'wrds_2002_2022.h5')).reset_index(drop=False)
export_path = 'all_factors_wrds'
base_dir = file_dir

# Weighting Scheme to use
W = 'bond_amount_out'

# Remove bonds with less than 1-Year to maturity
df = df[df.tmt > 1.0]

# Return Type
if return_type == 'excess_rf':
    yy = 'exretn_t+1'
    df = df[~df[yy].isnull()]
    R  = 'bond_ret'
elif return_type == 'duration_adj':
    yy = 'exretnc_dur_t+1'
    df = df[~df[yy].isnull()]
    R  = 'exretnc_dur'
elif return_type == 'maturity_adj':
    yy = 'exretnc_t+1'
    df = df[~df[yy].isnull()]
    R  = 'exretnc'

# ratings to use
RAT = 'spr_mr_fill'


# Construct the Market Factor
df['value-weights'] = df.groupby(by='date')[W].progress_apply(lambda x: x / np.nansum(x)).reset_index(level=0, drop=True)
MKTx = df.groupby('date')[[yy, 'value-weights']].apply(lambda x: np.nansum(x[yy] * x['value-weights'])).to_frame()
MKTx.columns = ['MKTx']
MKTx.index = MKTx.index + MonthEnd(1)

# Construct the Credit Risk Factor - Equal Weighted
# Credit Downside Risk Factor
# Credit Reversal Factor
# Credit Liquidity Factor

# Downside Risk Factor

# Remove any bonds that dont have a rating
df = df[~df[RAT].isnull()]

# Remove any bonds that dont have a Value at Risk Measure
VaR = 'var5br'
dfVAR = df[~df[VaR].isnull()]

# Group by date and split data into quintiles based on the bond rating
dfVAR['ratingQ2'] = dfVAR.groupby(by='date')[RAT].progress_apply(lambda x: pd.qcut(x, 5, labels=False, duplicates='drop') + 1).reset_index(level=0, drop=True)
# Group by date and split data into quintiles based on the bond tail risk
dfVAR['drfQ1'] = dfVAR.groupby(by='date')[VaR].progress_apply(lambda x: pd.qcut(x, 5, labels=False, duplicates='drop') + 1).reset_index(level=0, drop=True)

# Group by Date, Tail Risk Quintiles, Credit Rating Quinitles and get in quintile asset weights
dfVAR['value-weights'] = dfVAR.groupby(['date', 'drfQ1', 'ratingQ2'])[W].progress_apply(lambda x: x / np.nansum(x)).reset_index(level=[0, 1, 2], drop=True)

sorts = dfVAR.groupby(['date', 'drfQ1', 'ratingQ2'])[[yy, 'value-weights']].progress_apply(lambda x: np.nansum(x[yy] * x['value-weights'])).to_frame()
sorts.columns = ['retn']
sorts = sorts.pivot_table(index=['date'], columns=['drfQ1', 'ratingQ2'], values="retn")

# Low Credit Rating - High Credit Rating
n = 1
sub1 = (sorts.loc[:, (1, 5)] - (1*sorts.loc[:, (1, 1)]))/n
sub2 = (sorts.loc[:, (2, 5)] - (1*sorts.loc[:, (2, 1)]))/n
sub3 = (sorts.loc[:, (3, 5)] - (1*sorts.loc[:, (3, 1)]))/n
sub4 = (sorts.loc[:, (4, 5)] - (1*sorts.loc[:, (4, 1)]))/n
sub5 = (sorts.loc[:, (5, 5)] - (1*sorts.loc[:, (5, 1)]))/n

CRF_drf = pd.concat([sub1,sub2,sub3,sub4,sub5], axis = 1)
CRE_drf_mat = CRF_drf.unstack(level=0)

CRF_drf = CRF_drf.mean(axis = 1).to_frame()
CRF_drf.columns = ['CRF_drf']

# REVERSAL FACTOR

dfREV = df[~df[R].isnull()]
dfREV['revQ1'] = dfREV.groupby(by = ['date'])[R].progress_apply(lambda x: pd.qcut(x,5,labels=False,duplicates='drop')+1).reset_index(level=[0], drop=True)
dfREV['ratingQ2'] = dfREV.groupby(by = ['date'])[RAT].progress_apply(lambda x: pd.qcut(x,5,labels=False,duplicates='drop')+1).reset_index(level=[0], drop=True)
dfREV['value-weights'] = dfREV.groupby(['date', 'revQ1', 'ratingQ2'])[W].progress_apply( lambda x: x/np.nansum(x)).reset_index(level=[0,1,2], drop=True)

sorts = dfREV.groupby(['date', 'revQ1', 'ratingQ2'])[[yy, 'value-weights']].progress_apply(lambda x: np.nansum(x[yy] * x['value-weights'])).to_frame()
sorts.columns = ['retn']
sorts = sorts.pivot_table(index=['date'], columns=['revQ1', 'ratingQ2'], values="retn")
credit_reversal_portfolios = sorts.mean().unstack(level=0) * 12

# CRF_rev #
n = 1
sub1 = (sorts.loc[:,(1, 5)] - (1*sorts.loc[:,(1, 1)]))/n
sub2 = (sorts.loc[:,(2, 5)] - (1*sorts.loc[:,(2, 1)]))/n
sub3 = (sorts.loc[:,(3, 5)] - (1*sorts.loc[:,(3, 1)]))/n
sub4 = (sorts.loc[:,(4, 5)] - (1*sorts.loc[:,(4, 1)]))/n
sub5 = (sorts.loc[:,(5, 5)] - (1*sorts.loc[:,(5, 1)]))/n

CRF_rev = pd.concat([sub1,sub2,sub3,sub4,sub5], axis = 1)
CRF_rev = CRF_rev.mean(axis = 1).to_frame()
CRF_rev.columns = ['CRF_rev']

# CREDIT LIQUIDITY FACTOR

dfLIQ = df[~df.illiq.isnull()]
dfLIQ = dfLIQ[dfLIQ.n >= 5]

dfLIQ['liqQ1'] = dfLIQ.groupby(by=['date'])['illiq'].progress_apply(lambda x: pd.qcut(x, 5, labels=False, duplicates='drop') + 1).reset_index(level=[0, 1], drop=True)
dfLIQ['ratingQ2'] = dfLIQ.groupby(by=['date'])[RAT].progress_apply(lambda x: pd.qcut(x, 5, labels=False, duplicates='drop') + 1).reset_index(level=[0, 1], drop=True)

# Value-weights #
dfLIQ['value-weights'] = dfLIQ.groupby(['date', 'liqQ1', 'ratingQ2'])[W].progress_apply(lambda x: x / np.nansum(x)).reset_index(level=[0, 1, 2], drop=True)
sorts = dfLIQ.groupby(['date', 'liqQ1', 'ratingQ2'])[[yy, 'value-weights']].progress_apply(lambda x: np.nansum(x[yy] * x['value-weights'])).to_frame()

sorts.columns = ['retn']
sorts = sorts.pivot_table(index=['date'], columns=['liqQ1', 'ratingQ2'], values="retn")

# CRF_lrf #
n = 1
sub1 = (sorts.loc[:, (1, 5)] - (1 * sorts.loc[:, (1, 1)])) / n
sub2 = (sorts.loc[:, (2, 5)] - (1 * sorts.loc[:, (2, 1)])) / n
sub3 = (sorts.loc[:, (3, 5)] - (1 * sorts.loc[:, (3, 1)])) / n
sub4 = (sorts.loc[:, (4, 5)] - (1 * sorts.loc[:, (4, 1)])) / n
sub5 = (sorts.loc[:, (5, 5)] - (1 * sorts.loc[:, (5, 1)])) / n

CRF_lrf = pd.concat([sub1, sub2, sub3, sub4, sub5], axis=1)
CRF_lrf = CRF_lrf.mean(axis=1).to_frame()
CRF_lrf.columns = ['CRF_lrf']


# Create the CRF Factor #
CRF_Merge = CRF_rev.merge(CRF_drf, how = "inner",
                              left_index = True,
                              right_index = True)
CRF_Merge = CRF_Merge.merge(CRF_lrf, how = "inner",
                                left_index = True,
                                right_index = True)

CRFx = CRF_Merge.mean(axis = 1).to_frame()
CRFx.columns = ['CRFx']
CRFx.index =CRFx.index + MonthEnd(1)




