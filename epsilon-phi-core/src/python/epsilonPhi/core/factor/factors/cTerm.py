import pandas as pd

from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.core.factor.Factor import CConstructedFactor
from epsilonPhi.core.factor.factorMgr import CFactorMgr
from epsilonPhi.core.dataModel.enums.Factor import FACTOR

ts_type: TimeSeriesType = TimeSeriesType.LEVELS

class CTerm(CConstructedFactor):
    def __init__(self, schema):
        super(CConstructedFactor, self).__init__(schema=schema)
        self.construct_factor(FACTOR.EQUITY_GLOBAL_ISG)

    def construct_factor(self, name):

        _TICKERS = ['BMUS10Y', 'BMBD10Y', 'BMIT10Y', 'BMFR10Y', 'BMUK10Y', 'BMJP10Y', 'BMCN10Y']

        df = pd.DataFrame()
        for ticker in _TICKERS:
            asset = self._factorMgr.get_asset_by_name(ticker)
            df = pd.concat((df, asset.get_excess_return_df()), axis=1)

        X_centred = df - df.mean()

        import numpy as np
        cov_mat = np.cov(X_centred, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Project the data onto the principal components
        pca_result = np.dot(X_centred, eigenvectors)

        # Create a new DataFrame to store the PCA results
        pca_df = pd.DataFrame(data=pca_result, columns=[f'PCA_Component_{i + 1}' for i in range(len(df.columns))])

        self._cast_derived_class(df)

if __name__ == "__main__":

    from epsilonPhi.core.schema.Schema import ContextCreator
    schema = ContextCreator(currency='GBP',
                            start_date='31-Dec-1999',
                            end_date='31-Dec-2022').create_context()

    fac = CTerm(schema)