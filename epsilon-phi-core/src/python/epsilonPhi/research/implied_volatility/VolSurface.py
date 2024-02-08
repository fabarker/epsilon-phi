import pandas as pd
import numpy as np
from epsilonPhi.core.simulation.Bootstrap import AbstractBootstrapper as strapper
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.utils.DateUtils import DateUtils
from sklearn.linear_model import LinearRegression
from scipy import stats

_DATA_PATH = '/Users/francisbarker/Desktop/SPX Vols by Moneyness.csv'


_KEY_SERIES = '/Users/francisbarker/Documents/MATLAB/Sim Vol Surface/Key Series.xlsx'
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
        self._var_estimated = False
        self._var_fits = {}
        self._var_prin_comps = {}
        self._var_coeffs = {}

        self._vol_estimated = False
        self._vol_fits = {}
        self._vol_prin_comps = {}
        self._vol_coeffs = {}

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

        _series_df = pd.read_excel(_KEY_SERIES, index_col=0, sheet_name=None)

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
        self._spot = _series_df.get('Key Series')[['open','high','low','close','DivYield','PriceReturns','TotalReturns']]
        self._spot = self._spot.rename(columns={'PriceReturns':'PI'})
        self._spot = self._spot.rename(columns={'DivYield': 'DY'})

        # 3. Load Risk Free Rates
        T = DateUtils.mat_to_Rdate(self._maturities)
        T[T.index('1y')] = '12m' if '1y' in T else T

        rfrs = gds.get_interest_rates_for_region('United States', DateUtils.mat_to_Rdate(self._maturities))
        rfrs = rfrs.T.groupby(level=2).mean().T.resample('B').ffill().sort_index(axis=1)
        rfrs.columns = DateUtils.Rdate_to_mat(rfrs.columns)
        self._rfr = rfrs.sort_index(axis=1).interpolate()

        common_dates = pd.to_datetime(np.intersect1d(self._ivols.index, self._spot.index))
        M_Year = common_dates.year * 100 + common_dates.month
        unique_MY = np.unique(M_Year)
        self._dates = pd.to_datetime([ np.max(common_dates[M_Year == x]) for x in unique_MY ])

    @staticmethod
    def _simulate_atm_vol(mat):

        # 1. Regress Implied ATM on Realized Vol
        y = np.log(self.get_ATM(mat))
        X = np.log(self.get_realized_vol(mat))
        common_dates = np.intersect1d(y.index,
                                      X.index)

        T = len(common_dates)
        N = self._nbstraps
        y_hat = y.loc[common_dates].values.reshape(-1,1)
        x_hat = X.loc[common_dates].values.reshape(-1,1)
        reg = LinearRegression().fit(x_hat, y_hat)

        dS = self.get_dS().dropna()
        # 2. Bootstrap the realized variance and re-build ATM variances
        idxs = strapper.stationary_block_bootstrap(T, N, 1/252)
        _strapped = dS.values[idxs]
        sig = pd.DataFrame(_strapped).rolling(window=21,
                                              center=True,
                                              min_periods=2).std() * np.sqrt(252)


        sys = reg.intercept_.item() + np.log(sig.values) * reg.coef_.item()
        idio = np.random.normal(size=(T, N)) * (np.std(y_hat) - np.std(sys))
        sim = np.exp(sys + idio)

    @staticmethod
    def _extract_shocks(df_):

        V = np.power(df_, 2)
        Y = V.diff().dropna()
        X = V.shift(1).dropna().values.reshape(-1, 1)

        reg = LinearRegression().fit(X, Y)
        shocks = Y - reg.predict(X)
        kappa = -reg.coef_.item()
        theta = reg.intercept_ / kappa
        return kappa, theta, shocks

    def get_spot_price(self):
        return self._spot.get('PI')

    def get_risk_free_rate(self, maturity):
        return self._rfr.get(maturity)

    def get_ivols(self, strike=None, maturity=None):
        idx = ((self._ivols.columns.get_level_values('strike') >= 0.7) &
               (self._ivols.columns.get_level_values('strike') <= 1.3))
        return self._ivols.iloc[:, idx].loc[self.dates]

    def get_dividend_yield(self):
        return self._spot.get('DY') / 100

    def get_ATM(self, mat):
        return self._ivols.reorder_levels([1, 0], axis=1).get(1).get(mat)

    def get_dS(self):
        return np.log(self.get_spot_price()).diff()

    def get_dI(self, mat):
        return np.log(self.get_ATM(mat)).diff()

    def get_dSdI(self, mat):
        _ds = self.get_dS()
        _dI = self.get_dI(mat)

        _common_dates = np.intersect1d(_ds.index, _dI.index)
        return _ds.loc[_common_dates] * _dI.loc[_common_dates]

    def get_dIdI(self, mat):
        return np.power(self.get_dI(mat), 2)

    def get_dSdS(self):
        return np.power(self.get_dS(), 2)

    def get_moving_function(self, df, window, function, **kwargs):
        return df.rolling(window, min_periods=2, center=True).apply(lambda x: function(x, **kwargs))

    def get_trading_days_per_period(self, freq):
        if isinstance(freq, str):
            freq = DateUtils.Rdate_to_mat(freq)
        return round(freq * self._TRADING_DAYS_PER_YEAR)

    def get_realized_dSdI(self, mat):
        dSdI = self.get_dSdI(mat)

        _per_period = self.get_trading_days_per_period(mat)
        res =  self.get_moving_function(dSdI, window=_per_period, function=np.mean) * self._TRADING_DAYS_PER_YEAR
        return res.to_frame('dSdI')

    def get_realized_dI(self, mat):
        dI = self.get_dI(mat)

        _per_period = self.get_trading_days_per_period(mat)
        res = self.get_moving_function(dI, window=_per_period, function=np.mean) * self._TRADING_DAYS_PER_YEAR
        return res.to_frame('dI')

    def get_realized_dIdI(self, mat):
        dIdI = self.get_dIdI(mat)

        _per_period = self.get_trading_days_per_period(mat)
        res = self.get_moving_function(dIdI, window=_per_period, function=np.sum)
        res = res.to_frame('dIdI')
        return self._TRADING_DAYS_PER_YEAR * (res / (_per_period - 1))

    def get_realized_dSdS(self, mat):

        dSdS = self.get_dSdS()

        _per_period = self.get_trading_days_per_period(mat)
        res = self.get_moving_function(dSdS, window=_per_period, function=np.sum)
        res = res.to_frame('dSdS')
        return self._TRADING_DAYS_PER_YEAR * (res / (_per_period-1))

    def get_realized_vol(self, mat):
        ln_rtns = np.log(self._spot.get('PI')).diff()

        _per_period = self.get_trading_days_per_period(mat)
        sig = self.get_moving_function(ln_rtns, _per_period, np.std, ddof=1)
        return sig * np.sqrt(self._TRADING_DAYS_PER_YEAR)

    def get_realized_moments(self, mat):
        if not hasattr(self, '_moments'):
            _moments = (self.get_realized_dI(mat),
                        self.get_realized_dSdS(mat),
                        self.get_realized_dIdI(mat),
                        self.get_realized_dSdI(mat))
            self._moments = pd.concat(_moments, axis=1).dropna()
        return self._moments.loc[self.dates]

    def get_principle_components(self, series):
        L, coeff = np.linalg.eig(series.cov())

        p, d = coeff.shape
        maxind = np.argmax(np.abs(coeff), axis=0)
        colsign = np.sign(coeff[maxind, np.arange(d)])
        coeff = coeff * colsign

        prin_comps = pd.DataFrame(series @ coeff)
        prin_comps.columns = L
        return prin_comps, pd.DataFrame(coeff, index=series.columns)

    def get_estimated_var_surface(self, mat=None):
        if self._var_estimated is False:
           self.estimate_variance_surface()
        return self._var_fits.get(mat, self._var_fits)

    def get_estimated_vol_surface(self, mat=None):
        if self._vol_estimated is False:
            self.estimate_vol_surface()
        return self._vol_fits.get(mat, self._vol_fits)

    def estimate_variance_surface(self):

        for mat in self._maturities:

            # Get Implied Vol Surface
            ivars = np.power(self.get_ivols().loc[self.dates].get(mat), 2)

            # Extract Principle Components
            pcs, coeffs = self.get_principle_components(ivars)

            # Get Realized Moments
            rvars = self.get_realized_moments(mat=mat)

            regs = list()
            reg = {}
            for t in range(coeffs.shape[1]):
                tmp = LinearRegression().fit(rvars.values, pcs.iloc[:, t].values.reshape(-1, 1))
                reg['beta'] = tmp.coef_
                reg['alpha'] = tmp.intercept_
                reg['sigma'] = np.std(pcs.iloc[:, t].values.reshape(-1, 1) - tmp.predict(rvars.values), ddof=1)
                regs.extend([reg.copy()])

            self._var_fits[mat] = regs
            self._var_prin_comps[mat] = pcs.copy()
            self._var_coeffs[mat] = coeffs.copy()


    def estimate_vol_surface(self):

        ivols = self.get_ivols().loc[self.dates]
        for mat in self._maturities:
            imp_vols = ivols.get(mat)
            rvols = self.get_realized_vol(mat)

            log_ivols = np.log(imp_vols)
            log_rvols = np.log(rvols)

            # Extract Principle Componenta
            pcs, coeffs = self.get_principle_components(log_ivols)

            regs = list()
            for t in range(coeffs.shape[1]):
                reg = stats.linregress(log_rvols, pcs.iloc[:, t])
                res = pcs.iloc[:, t] - reg.slope * log_rvols - reg.intercept
                reg.sigma = np.std(res, ddof=1)
                regs.extend([reg])

            self._vol_fits[mat] = regs
            self._vol_prin_comps[mat] = pcs.copy()
            self._vol_coeffs[mat] = coeffs.copy()

    def get_bootstrapped_underlier_moments(self, mat):
        idx = self.get_bootstrap_indicies()
        realized = self.get_realized_moments(mat)

        strapped = np.stack([realized.values[idx, x] for x in range(realized.shape[1])], 2)
        return strapped.transpose((0,2,1))

    def get_simulated_principle_components(self):
        for mat in self._maturities:
            self._simulated[mat] = self.get_simulated_principle_components_single_maturity(mat)
        return self._simulated

    def get_simulated_principle_components_single_maturity(self, mat):

        vol_0 = self.get_realized_moments(mat).values[-1,:].reshape(1, -1)
        vol_0 = vol_0.repeat(self._nbstraps, 0).reshape((1, vol_0.size, self._nbstraps))

        sim_underlier_vol = self.get_bootstrapped_underlier_moments(mat)
        log_underlier_vol = np.vstack((vol_0, sim_underlier_vol))
        VS = self.get_estimated_var_surface(mat)

        sim_pc = np.full((self._sim_horizon+1, self._nbstraps, self._NUMBER_OF_PCs), np.nan)
        for k in range(self._NUMBER_OF_PCs):

            beta_0 = VS[k].get('alpha').item()
            beta_1 = VS[k].get('beta')
            sigma = VS[k].get('sigma')

            # We use the last value of the principle component to seed from
            sim_pc[0, :, k] = self._var_prin_comps.get(mat).values[-1, k]
            # We simulate from the end to the begining to capture the relationship between impled and realized vol
            sim_pc[-1, :, k] = sigma * np.random.normal(size=self._nbstraps) + beta_1 @ log_underlier_vol[self._sim_horizon, :, :] + beta_0
            for tm in range(1, self._sim_horizon+1):
                t = self._sim_horizon - tm + 1
                sys_sim = beta_0 + beta_1 @ log_underlier_vol[t, :, :]
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
        _bootstrapped_VS = dict()
        for mat in princomps.keys():

            _cross_section = princomps.get(mat)
            _coeffs = self._var_coeffs.get(mat)

            IV = np.full((_coeffs.shape[0], _cross_section.shape[1], _cross_section.shape[0]), np.nan)
            for t in range(_cross_section.shape[0]):
                IV[:, :, t] = np.sqrt((_coeffs.iloc[:, :self._NUMBER_OF_PCs] @ _cross_section[t, :, :].T).values)






if __name__ == "__main__":

        self = VolSurface()
        self._simulate_atm_vol(self._maturities[0])
        vols = self.get_simulated_ivols()

