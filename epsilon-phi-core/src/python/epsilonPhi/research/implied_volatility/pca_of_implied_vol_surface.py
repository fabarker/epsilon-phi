from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.alchemist.DataModel import *
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from sqlalchemy import distinct
from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries, TimeSeriesType

sessionMgr = SessionMgr()
session = sessionMgr.getSessionFactory()
gds = GlobalDataSource()

# 1 . Define the underlier security
security = 'USDCHF'

# 2. Get all of the uids associated with this security
uids = [ int(x[0]) for x in session.query(distinct(ImpliedVolatility.uid)).filter(ImpliedVolatility.security == security).all() ]

# 3. Get the time series data for each maturity and delta quote

fxivols = CTimeSeries(ts_type=TimeSeriesType.LEVELS)
for uid in uids:
    print(uid)
    new_ts = gds.get_time_series_data_from_uid(uid, cols='mid', ts_type=TimeSeriesType.LEVELS)
    df_no_duplicates = new_ts[~new_ts.index.duplicated(keep='first')]
    fxivols = fxivols.concat(df_no_duplicates)

X = fxivols.dropna(axis=0).get_returns(return_type='log')

# 4. PCA Analysis of Implied Vol Surface
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaling = StandardScaler()

# Use fit and transform method
scaling.fit(X)
Scaled_data = scaling.transform(X)

principal = PCA(n_components=3)
principal.fit(Scaled_data)
x=principal.transform(Scaled_data)


#pca_result = pca.fit_transform(X)

# 5.
X_centred = X - X.mean()
cov_mat = np.cov(X_centred, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)

# Sort eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Project the data onto the principal components
pca_result = np.dot(X_centred, eigenvectors)

# Create a new DataFrame to store the PCA results
pca_df = pd.DataFrame(data=pca_result, columns=[f'PCA_Component_{i+1}' for i in range(len(X.columns))])
