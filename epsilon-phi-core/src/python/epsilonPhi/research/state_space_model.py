import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
gds = GlobalDataSource()

rfr = gds.get_risk_free_rate_for_currency_region('USD')
cpi = gds.get_consumer_price_index_for_currency('USD')
cpi.index = cpi.index + pd.tseries.offsets.BMonthEnd(0)
shocks = np.log(cpi).diff().dropna()

# Define the Markov regime-switching model with two states
model = MarkovRegression(shocks, k_regimes=3, switching_variance=True)
results = model.fit()
print(results.summary())

X = pd.concat((rfr, cpi.get_returns()), axis=1).dropna().diff().dropna()
X.columns = ['Cash', 'Inflation']
# Prepare data: 2D array of observations (n_samples, n_features)

YoY_inflation = cpi.get_returns()
YoY_inflation.columns = ['cpi']


# Ensure returns are reshaped and standardized
X = cpi_returns.values.reshape(-1, 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit 2-state HMM
model = GaussianHMM(
    n_components=2,
    covariance_type="full",
    n_iter=1000,
    random_state=42
)

model.fit(X_scaled)

# Predict regimes
hidden_states = model.predict(X_scaled)

# Attach to DataFrame
cpi_returns = cpi_returns.to_frame(name="cpi_return")
cpi_returns["regime"] = hidden_states