import pandas as pd

df = pd.read_csv('/Users/francisbarker/Desktop/SPX Vols.csv', index_col=0, header=[0,1,2,3])
df.index = pd.to_datetime(df.index)
df = df / 100
df_ = df.dropna(axis=1)
cov_mat = df_.cov()

import numpy as np
eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)

# Sort eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Project the data onto the principal components
pca_result = np.dot(df_, eigenvectors)

# Create a new DataFrame to store the PCA results
pca_df = pd.DataFrame(data=pca_result, columns=[f'PCA_Component_{i+1}' for i in range(len(X.columns))])