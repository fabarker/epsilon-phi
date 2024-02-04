import pandas as pd
import numpy as np
from epsilonPhi.core.simulation.Bootstrap import AbstractBootstrapper as strapper
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.utils.DateUtils import DateUtils
from scipy import stats

_DATA_PATH = '/Users/francisbarker/Desktop/SPX Vols by Moneyness.csv'
gds = GlobalDataSource()


class VolSurface(object):

    _nbstraps = 100
    _DAYS_PER_YEAR = 365.25
    _TRADING_DAYS_PER_YEAR = 252
    _UNDERLIER = 'S&PCOMP'
    _NUMBER_OF_PCs = 4

    def __init__(self, nbstraps=10, sim_horizon=1, frequency=Frequency.BUSINESS_MONTHLY):

        if isinstance(frequency, str):
            self._frequency = Frequency(frequency)
        elif isinstance(frequency, Frequency):
            self._frequency = frequency
        else:
            raise ValueError('Error - frequency type {} not supported'.format(type(frequency)))

        # Simulation
        self._sim_horizon = sim_horizon * self._frequency.obs_per_year()
        self._simulated = {}

        # Bootstrpping
        self._nbstraps = nbstraps
        self._bootstrap_indicies = None
        self._q = 1/self._frequency.obs_per_year()

        # Surface Estimates
        self._estimated = False
        self._fits = {}
        self._prin_comps = {}
        self._coeffs = {}

        # Data
        self._ivols = None
        self._moneyness = None
        self._maturities = None
        self._spot = None
        self._rfr = None
        self.load_data()



    @property
    def dates(self):
        return self._dates

    @property
    def obs_per_year(self):
        return self._frequency.obs_per_year()

    @property
    def T(self):
        return len(self.dates)

    def set_frequency(self, frequency):
        self._frequency = frequency

    def get_bootstrap_indicies(self):
        if self._bootstrap_indicies is None:
            self.load_bootstrap_indicies()
        return self._bootstrap_indicies

    def load_bootstrap_indicies(self):
        if self._bootstrap_indicies is None:
            _bootstrap_indicies = strapper.stationary_block_bootstrap(self.T,
                                                                       self._nbstraps,
                                                                       self._q)

            self._bootstrap_indicies = _bootstrap_indicies[:self._sim_horizon, :]


    def load_data(self):

        # 1. Load ivol data
        df_ = pd.read_csv(_DATA_PATH, index_col=0)
        df_.index = pd.to_datetime(df_.index, format='%d/%m/%Y')
        df_['maturity'] = DateUtils.Rdate_to_mat(df_.get('tenor').values)
        df_ = df_[['relativeStrike', 'maturity', 'impliedVolatility']].reset_index(drop=False).set_index(
            ['date', 'relativeStrike', 'maturity'])
        df_.index.names = ['date', 'strike', 'mat']

        self._ivols = df_.unstack(level=[1, 2]).resample('B').ffill().droplevel(0, axis=1).reorder_levels([1, 0], axis=1)
        self._maturities = np.unique(self._ivols.columns.get_level_values('mat'))
        self._moneyness = np.unique(self._ivols.columns.get_level_values('strike'))

        # 2. Load Underlier Data
        self._spot = gds.get_dataframe_from_tickers(VolSurface._UNDERLIER).droplevel('ticker', axis=1)
        self._spot = self._spot.resample('B').ffill()

        # 3. Load Risk Free Rates
        T = DateUtils.mat_to_Rdate(self._maturities)
        T[T.index('1y')] = '12m' if '1y' in T else T

        rfrs = gds.get_interest_rates_for_region('United States', DateUtils.mat_to_Rdate(self._maturities))
        rfrs = rfrs.T.groupby(level=2).mean().T.resample('B').ffill().sort_index(axis=1)
        rfrs.columns = DateUtils.Rdate_to_mat(rfrs.columns)
        self._rfr = rfrs.sort_index(axis=1).interpolate()

        common_dates = np.intersect1d(pd.to_datetime(np.intersect1d(self._ivols.index, self._spot.index)), self._rfr.index)
        self._dates = pd.date_range(min(common_dates), max(common_dates), freq=self._frequency.value)

    def get_spot_price(self):
        return self._spot.get('PI')

    def get_risk_free_rate(self, maturity):
        return self._rfr.get(maturity)

    def get_ivols(self, strike=None, maturity=None):
        return self._ivols.get((strike, maturity), self._ivols).loc[self.dates]

    def get_dividend_yield(self):
        return self._spot.get('DY') / 100

    def get_dS(self):
        return np.log(self.get_spot_price()).diff()

    def get_dI(self):
        return np.log(self.get_realized_vol()).diff()

    def get_dSdI(self):
        return self.get_dS().reshape(-1, 1) * self.get_dI()

    def get_dIdI(self):
        return np.power(self.get_dI(), 2)

    def get_dSdS(self):
        return np.power(self.get_dS(), 2)

    def get_realized_vol(self, mat):
        vol = self._spot.get('PI').pct_change().rolling(round(mat*self._TRADING_DAYS_PER_YEAR), min_periods=1).std()
        return vol.loc[self.dates] * np.sqrt(self._TRADING_DAYS_PER_YEAR)

    def get_realized_moments(self, mat):
        return self.get_realized_vol(mat)

    def get_principle_components(self, series):
        L, V = np.linalg.eig(series.iloc[1:, :].cov())
        loadings = pd.DataFrame(V, index=series.columns)
        PCs = series @ loadings
        PCs.columns = L
        return PCs, loadings

    def get_estimated_vol_surface(self, mat=None):
        if self._estimated is False:
           self.estimate_vol_surface()
        return self._fits.get(mat, self._fits)

    def estimate_vol_surface(self):

        ivols = self.get_ivols().loc[self.dates]
        for mat in self._maturities:
            imp_vols = ivols.get(mat)
            rvols = self.get_realized_moments(mat)

            log_ivols = np.log(imp_vols)
            log_rvols = np.log(rvols)

            # Extract Principle Componenta
            pcs, coeffs = self.get_principle_components(log_ivols)

            regs = list()
            for t in range(coeffs.shape[1]):
                reg = stats.linregress(pcs.iloc[:, t], log_rvols)
                reg.sigma = np.std(log_rvols - reg.slope * pcs.iloc[:, t] + reg.intercept, ddof=1)
                regs.extend([reg])

            self._fits[mat] = regs
            self._prin_comps[mat] = pcs.copy()
            self._coeffs[mat] = coeffs.copy()

    def get_bootstrapped_underlier_moments(self, mat):
        idx = self.get_bootstrap_indicies()
        realized_vol = self.get_realized_moments(mat)
        return realized_vol.values[idx]

    def get_simulated_principle_components(self):
        for mat in self._maturities:
            self._simulated[mat] = self.get_simulated_principle_components_single_maturity(mat)
        return self._simulated

    def get_simulated_principle_components_single_maturity(self, mat):

        vol_0 = self.get_realized_vol(mat).values[-1] * np.ones((1, self._nbstraps))

        sim_underlier_vol = self.get_bootstrapped_underlier_moments(mat)
        log_underlier_vol = np.log(np.vstack((vol_0, sim_underlier_vol)))
        VS = self.get_estimated_vol_surface(mat)

        sim_pc = np.full((self._sim_horizon+1, self._nbstraps, self._NUMBER_OF_PCs), np.nan)
        for k in range(self._NUMBER_OF_PCs):

            beta_0 = VS[k].intercept
            beta_1 = VS[k].slope
            sigma = VS[k].sigma

            # We use the last value of the principle component to seed from
            sim_pc[0, :, k] = self._prin_comps.get(mat).values[-1, k]
            # We simulate from the end to the begining to capture the relationship between impled and realized vol
            sim_pc[-1, :, k] = sigma * np.random.normal(size=self._nbstraps) + beta_1 * log_underlier_vol[self._sim_horizon, :] + beta_0
            for tm in range(1, self._sim_horizon):
                t = self._sim_horizon - tm + 1
                sys_sim = beta_0 + beta_1 * log_underlier_vol[t, :]
                ido_sim = sigma * np.random.normal(size=self._nbstraps)
                sim_pc[t, :, k] = sys_sim + ido_sim
        return sim_pc

    @staticmethod
    def Rdate_to_mat(rdate):
        return DateUtils.Rdate_to_mat(rdate)

    @staticmethod
    def mat_to_Rdate(mat):
        return DateUtils.mat_to_Rdate(mat)

    def get_simulated_ivols(self):

        princomps = self.get_simulated_principle_components()
        for mat in princomps.keys():

            _cross_section = princomps.get(mat)
            _coeffs = self._coeffs.get(mat)

            IV = np.full((_coeffs.shape[0], _cross_section.shape[1], _cross_section.shape[0]), np.nan)
            for t in range(_cross_section.shape[0]):
                IV[:, :, t] = np.exp(_coeffs.iloc[:, :self._NUMBER_OF_PCs] @ _cross_section[t, :, :].T).values





if __name__ == "__main__":

        self = VolSurface()
        vols = self.get_simulated_ivols()

