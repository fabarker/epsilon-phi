import warnings

import numpy as np
import pandas as pd
import typing as tp
import numpy.lib.scimath as SC
from epsilonPhi.core.config.appConfig import CAppConfig
from epsilonPhi.core.dataModel.enums.Factor import FactorType
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.dataModel.enums.Rates import RateType
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType
from epsilonPhi.core.schema.Schema import CContext
from epsilonPhi.core.simulation.SimStructs import *
from epsilonPhi.core.timeSeries.regression import *

def logicalValues(vec: np.ndarray):
    """
    Returns a boolean array of the same shape as `vec`, filled with True.

    Parameters
    ----------
    vec : np.ndarray
        Input array whose shape will be used to generate the output.

    Returns
    -------
    np.ndarray
        Boolean array of the same shape as `vec`, filled with True.
    """
    return np.ones_like(vec, dtype=bool)

def nans(vec: np.ndarray):
    """
    Returns an array of the same shape as `vec`, filled with np.nan.

    Parameters
    ----------
    vec : np.ndarray
        Input array whose shape will be used for the output.

    Returns
    -------
    np.ndarray
        A NumPy array of the same shape as `vec`, filled with NaN values.
    """
    return np.full_like(vec, np.nan, dtype=float)


class AbstractBootstrapper(object):
    _state = 5489
    _R = np.random.RandomState(5489)

    def __init__(self):
        pass

    @staticmethod
    def optimal_block_length(data: np.ndarray):

        """
        This is a function to select the optimal (in the sense of minimising the MSE
        of the estimator of the long-run variance) block length for the stationary
        bootstrap or circular bootstrap.
        The code follows Politis and White, 2001,
        "Automatic Block-Length Selection for the Dependent Bootstrap".


        NOTE: The optimal average block length for the stationary bootstrap,
        and it does not need to be an integer.
        The optimal block length for the circular bootstrap should be an
        integer. Politis and White suggest rounding the output UP to the
        nearest integer.

        Args:
          data, an nxk matrix

          Returns: a 2xk NumPy array of optimal bootstrap block lengths,
                      [[b_star_sb], [b_star_cb]], where
                      b_star_sb: optimal block length for stationary bootstrap
                      b_star_cb: optimal block length for circular bootstrap
        """

        def lam(kk: np.ndarray) -> np.ndarray:
            """
            Helper function, calculates the flattop kernel weights.

            Adapted for Python August 12, 2018 by Michael C. Nowotny
            """
            return (np.abs(kk) >= 0) * (np.abs(kk) < 0.5) \
                + 2 * (1.0 - np.abs(kk)) * (np.abs(kk) >= 0.5) * (np.abs(kk) <= 1)

        def mlag(x: np.ndarray,
                 n: tp.Optional[int] = 1,
                 init: tp.Optional[float] = 0.0) -> np.ndarray:

            nobs, nvar = x.shape

            xlag = np.ones((nobs, nvar * n), dtype=x.dtype) * init
            icnt = 0
            for i in range(nvar):
                for j in range(n):
                    xlag[j + 1:, icnt + j] = x[0:-j - 1, i]
                icnt += n

            return xlag

        if data.ndim == 1:
            data = data.reshape((-1, 1))
        elif data.ndim > 2:
            raise ValueError(
                'data must be a two dimensional NumPy array'
                '(number of observations x number of variables)')
        n, k = data.shape

        # these are optional in the original Matlab implementation
        # opt_block_length_full.m, but fixed at default values here
        kn = int(max(5, np.sqrt(np.log10(n))))

        # adding kn extra lags to employ Politis' (2002) suggestion
        # for finding largest significant m
        m_max = int(np.ceil(np.sqrt(n)) + kn)

        # maximum value of b_star_sb to consider.
        # dec07: new idea for rule-of-thumb to put an upper bound on estimated
        # optimal block length
        b_max = np.ceil(min(3 * np.sqrt(n), n / 3))

        c = 2
        original_data = data
        # b_star_final = np.zeros((2, k), dtype=np.float64)
        b_star_final = []

        for i in range(k):
            data = original_data[:, i].reshape((-1, 1))

            # FIRST STEP: finding m_hat-> the largest lag for which the
            # auto-correlation is still significant.
            temp = mlag(data, m_max)

            # dropping the first m_max rows, as they are filled with zeros
            temp = temp[m_max:, :]
            temp = np.corrcoef(np.hstack((data[m_max:], temp)), rowvar=False)
            temp = temp[1:, 0].reshape((-1, 1))

            # We follow the empirical rule suggested in
            # Politis, 2002, "Adaptive Bandwidth Choice".
            # as suggested in Remark 2.3, setting c=2, kn=5

            # looking at vectors of auto-correlations,
            # from lag m_hat to lag m_hat+kn
            temp2 = np.hstack((np.transpose(mlag(temp, kn)), temp[-kn:]))

            # dropping the first kn-1, as the vectors have empty cells
            temp2 = temp2[:, kn:]

            # checking which are less than the critical value
            temp2 = np.abs(temp2) < (c * np.sqrt(np.log10(n) / n)
                                     * np.ones((kn, m_max - kn + 1)))

            # this counts the number of insignificant autocorrelations
            temp2 = np.sum(temp2, axis=0).reshape((1, -1))
            temp3 = np.hstack((np.arange(1, temp2.shape[1] + 1).reshape((-1, 1)),
                               temp2.transpose()))

            # selecting all rows where ALL kn auto-correlations are not significant
            temp3 = temp3[np.squeeze(temp2 == kn), :]

            if temp3.size == 0:
                # this means that NO collection of kn auto-correlations were all
                # insignificant, so pick largest significant lag
                m_hat = max(
                    np.flatnonzero(np.abs(temp) > (c * np.sqrt(np.log10(n) / n))))
            else:
                # if more than one collection is possible, choose the smallest m
                m_hat = temp3[0, 0]

            if 2 * m_hat > m_max:
                m = m_max
            else:
                m = 2 * m_hat

            del temp, temp2, temp3

            # SECOND STEP: computing the inputs to the function for b_star_sb
            kk = np.arange(-m, m + 1).reshape((-1, 1))
            if m > 0:
                temp = mlag(data, m)

                # dropping the first m_max rows, as they're filled with zeros
                temp = temp[m:, :]
                temp = np.cov(np.hstack((data[m:], temp)).transpose())

                # auto-covariances
                acv = temp[:, 0].reshape((-1, 1))
                acv2 = np.hstack(
                    (-np.arange(1, m + 1).reshape((-1, 1)), acv[1:, :]))
                if acv2.shape[0] > 1:
                    acv2 = acv2[acv2[:, 0].argsort(),]

                # auto-covariances from -m to m
                acv = np.vstack((acv2[:, 1].reshape((-1, 1)), acv))
                del acv2

                g_hat = np.sum(lam(kk / m) * np.abs(kk) * acv)
                dcb_hat = (4.0 / 3.0) * np.sum(lam(kk / m) * acv) ** 2

                # first part of dsb_hat (note cos(0)=1)
                dsb_hat = 2 * (np.sum(lam(kk / m) * acv) ** 2)

                # FINAL STEP: constructing the optimal block length estimator

                # optimal block lenght for stationary bootstrap
                b_star_sb = ((2 * (g_hat ** 2) / dsb_hat) ** (1.0 / 3.0)) \
                            * (n ** (1.0 / 3.0))
                if b_star_sb > b_max:
                    b_star_sb = b_max

                # optimal block length for circular bootstrap
                b_star_cb = ((2 * (g_hat ** 2) / dcb_hat) ** (1.0 / 3.0)) \
                            * (n ** (1.0 / 3.0))
                if b_star_cb > b_max:
                    b_star_cb = b_max

                # b_star = (b_star_sb, b_star_cb)
                return b_star_sb,b_star_cb
            else:
                return 1, 1
        return None

    @staticmethod
    def stationary_block_bootstrap(T, N, q):

        if q > 1:
           q = 1/q

        nRands = T * N
        R = np.random.RandomState(AbstractBootstrapper._state)
        blockLocations = R.choice(range(0, T), size=(nRands, 1), replace=True)
        blockLengths = np.minimum(R.geometric(q, (nRands, 1)), T)

        block_locs_list = list()
        L = 0
        b = 0

        while L < nRands:
            block_size = int(blockLengths[b].item())
            block_loc = int(blockLocations[b].item())

            locs = np.mod(block_loc + np.array(range(0, block_size, 1)), T)
            block_locs_list.extend(locs.tolist())

            L = L + block_size
            if np.mod(L, T) < block_size:
                L = L - np.mod(L, T)
            b = b + 1

        del blockLengths
        del blockLocations

        return np.reshape(np.array(block_locs_list)[:nRands], (N, T)).T

    @staticmethod
    def random_generator(output_row, output_col, output_range, use_replacement=False):

        low = 0
        high = 0
        if isinstance(output_range, range):
            low = output_range[0]
            high = len(output_range) + 1
        elif isinstance(output_range, int):
            high = output_range

        return AbstractBootstrapper._R.choice(range(low, high), size=(output_row, output_col), replace=use_replacement)

    @staticmethod
    def block_bootstrap(T, N, q):

        """
        Generate block bootstrap indices using a stationary bootstrap method.

        Parameters
        ----------
        T : int
            The number of time periods in the original time series (length of each bootstrap sample).

        N : int
            The number of bootstrap replications (i.e., how many separate samples to generate).

        q : float
            The probability of starting a new block at any time step (block break probability).
            Smaller values produce longer average block lengths (expected block length = 1/q).

        Returns
        -------
        theta : np.ndarray of shape (T, N)
            An integer array where each column represents a bootstrap sample of time indices.
            Each entry is an index from the original time series [0, T-1], constructed using
            the stationary block bootstrap algorithm.

        Notes
        -----
        This implementation uses a version of the stationary bootstrap where each new time index
        has a probability `q` of starting a new randomly drawn block and a probability `1 - q`
        of continuing from the previous index + 1. If the continuation index exceeds T-1,
        it wraps around to 0.

        This method preserves short-term dependence within blocks and is useful for bootstrapping
        time series data where independence cannot be assumed.
        """

        if q > 1:
           q = 1/q

        theta = np.zeros((T, N), dtype=int)
        for strap in range(N):
            t = 0
            theta[t, strap] = AbstractBootstrapper.random_generator(1, 1, range(0, T-1)).item()
            while t < T-1:
                t = t + 1
                U = AbstractBootstrapper._R.random()
                if U < q:
                    theta[t, strap] = AbstractBootstrapper.random_generator(1, 1, range(0, T-1)).item()
                else:
                    if theta[t-1, strap] + 1 > T-1:
                        theta[t, strap] = 0
                    else:
                        theta[t, strap] = theta[t-1, strap] + 1
        return theta


    def circular_bootstrap(self):
        pass

    def bootstrap(self):
        pass

class TimeSeriesBootstrapper(AbstractBootstrapper):
    def __init__(self, df, N, q):
        super(TimeSeriesBootstrapper, self).__init__()

        self._df = df.copy()
        self._bootstrap_df()

        self._T = df.shape[0]
        self._N = N
        self._q = q

    def _bootstrap_df(self):
        pass

    def _bootstrap_indicies(self, N, q):
        pass


class _Bootstrapper(AbstractBootstrapper):
    def __init__(self, function_apply, sample):
        super(_Bootstrapper, self).__init__()

        self._function_apply = function_apply
        self._sample = sample



class SAABootstrapper(AbstractBootstrapper):

    NUM_MONTHS = 12
    RANDOM_SEED = 5489
    R1 = np.random.RandomState(RANDOM_SEED)

    def __init__(
            self,
            context: CContext
    ):
        super(SAABootstrapper, self).__init__()

        self._schema = context
        self._curr_env_ind = None
        self._bootstrap_indicies = None

    @property
    def currency_config(self):
        return self._schema.get_currency_config()

    @property
    def simulation_config(self):
        return self._schema.get_simulation_config()

    @property
    def sim_start_date(self):
        return self.simulation_config.simStartDate if self.simulation_config.simStartDate else self._schema.start_date

    @property
    def sim_end_date(self):
        return self.simulation_config.simEndDate if self.simulation_config.simEndDate else self._schema.end_date

    @property
    def long_horizon(self):
        return self.simulation_config.simLongHorizon

    @property
    def short_horizon(self):
        return self.simulation_config.simShortHorizon

    @property
    def medium_horizon(self):
        return self.simulation_config.simMidHorizon

    @property
    def horizon_mult(self):
        return self.simulation_config.simHorizonMult

    @property
    def nbstraps(self):
        return self.simulation_config.simNumBoostraps

    @property
    def hold_start_states(self):
        return int(self.simulation_config.simHoldStartStates)

    @staticmethod
    def extract_shocks(df):

        vals = df.dropna().values.reshape(-1, 1)
        Y = vals[1:]
        X = vals[:-1]

        T, N = X.shape

        X_prime = np.append(X, np.ones([T, 1]), axis=1)
        res, _, _, _ = np.linalg.lstsq(X_prime, Y, rcond=None)

        shocks = vals[1:] - X_prime @ res
        shocks = np.insert(shocks, 0, 0)
        return res[-1], res[:N], shocks

    def load_bootstrap_indicies(self):

        hor_mult = self.horizon_mult
        medium_horizon = self.medium_horizon
        long_horizon = self.long_horizon

        factor_panel = self._schema.get_return_factors_panel().loc[self.sim_start_date:self.sim_end_date]
        data_length = hor_mult * len(factor_panel.index)
        current_ind = self.get_current_environment_indicator()
        data_length_curr = hor_mult * np.sum(current_ind.values, axis=0, dtype=int).item()

        short_mult = int(round(data_length / data_length_curr))
        data_length_blend = short_mult * data_length_curr + data_length

        # Check that we have enough data
        if SAABootstrapper.NUM_MONTHS * medium_horizon > data_length_curr or \
           SAABootstrapper.NUM_MONTHS * long_horizon > data_length or \
           SAABootstrapper.NUM_MONTHS * medium_horizon > data_length_blend:
            raise Exception("Increase horizon multiplier in schema pars")

        block_length = self.simulation_config.simBlockLen

        # generate indicies
        self._bootstrap_indicies = self.generate_bootstrap_indicies(
            block_length,
            self.nbstraps,
            data_length_curr,
            data_length_blend,
            data_length,
        )

    def get_bootstrap_indicies(self):
        if self._bootstrap_indicies is None:
            self.load_bootstrap_indicies()
        return self._bootstrap_indicies


    def generate_bootstrap_indicies(
            self,
            q,
            nbstraps=None,
            curr_count=None,
            med_count=None,
            long_count=None
    ):
        bs_indicies = BootstrapIndicies()

        # Current Environemnt
        bs_indicies.set_short_term_indicies(
            self.stationary_block_bootstrap(curr_count, nbstraps, q)
        )

        # Medium Environment
        bs_indicies.set_medium_term_indicies(
            self.stationary_block_bootstrap(med_count, nbstraps, q)
        )

        # Long Environemnt
        bs_indicies.set_long_term_indicies(
            self.stationary_block_bootstrap(long_count, nbstraps, q)
        )

        return bs_indicies

    def prepare_inflation_paths(
            self,
            bs_indicies
    ):
        asset = self._schema.get_inflation_rate_asset()
        return self.prepare_cash_and_inflation_paths(
            asset,
            bs_indicies,
            0,
            RateType.INFLATION)

    def prepare_cash_paths(self, bs_indicies, long_term_shocks=False):
        asset = self._schema.get_risk_free_rate_asset()
        rfr_floor = 0
        return self.prepare_cash_and_inflation_paths(
            asset, bs_indicies, long_term_shocks, RateType.CASH, rfr_floor)

    #def prepare_lending_paths(
    #        self,
    #        lending_rate,
    #        bs_indicies,
    #        long_term_shocks,
    #        risk_free_floor=None,
    #):
    #return self.prepare_cash_and_inflation_paths(
    #    lending_rate,
    #    bs_indices,
    #    long_term_shocks,
    #    RateType.Lending,
    #    risk_free_floor)


    def prepare_cash_and_inflation_paths(
            self,
            asset,
            bs_indices,
            long_term_shocks,
            rate_type,
            risk_free_floor=None
    ) -> PathsPanel:

        frequency = self._schema.frequency

        #curr_rate = self.currency_config.medium_risk_free_rate
        curr_rate = 0.003
        #curr_rate_name = self.currency_config.risk_free_mapping
        #equi_rate_name = self.currency_config.risk_free_mapping

        alpha, beta, shocks = self.extract_shocks(asset)

        medium_horizon = self.medium_horizon
        long_horizon = self.long_horizon
        hold_start_states = self.hold_start_states

        if rate_type == RateType.CASH:
            eq_R = self.currency_config.risk_free_rate / SAABootstrapper.NUM_MONTHS
        elif rate_type == RateType.INFLATION:
            eq_R = self.currency_config.inflation_rate / SAABootstrapper.NUM_MONTHS
            betas = beta.item()
        elif rate_type == RateType.LENDING:
            eq_R = self.currency_config.risk_free_rate / SAABootstrapper.NUM_MONTHS
            eq_R += (asset.get_spread() + asset.get_borrowing_cost()) / SAABootstrapper.NUM_MONTHS
        else:
            raise Exception("Rate type not supported")

        new_methodology_list = ['USD', 'GBP']
        if self._schema.currency in new_methodology_list and rate_type != RateType.INFLATION:
            beta, shocks = self.get_beta_and_shocks(
                frequency=frequency.obs_per_year(),
                currency=self._schema.currency,
                asset_ts=asset,
                ts_values=asset.values,
                rate_ts=asset.deepcopy(),
                new_beta=beta.item()
            )


        current_indicator = self.get_current_environment_indicator()
        demeaned_shocks = self.demean_values_in_blocks(
            shocks,
            current_indicator
        )

        paths_panel = self.prepare_boostrap_blocks(
            bs_indices,
            long_term_shocks
        )
        blocks = paths_panel.get_paths()
        curr_env_idx = paths_panel.get_short_block_indicator().astype(bool)

        shocks_panel = shocks[blocks]
        shocks_panel[curr_env_idx] = demeaned_shocks[blocks[curr_env_idx]]

        # pre-allocate paths array
        paths = np.full((long_horizon * SAABootstrapper.NUM_MONTHS, self.nbstraps), np.nan)
        T = SAABootstrapper.NUM_MONTHS * max(hold_start_states, max(medium_horizon, long_horizon))
        trend = np.full((long_horizon * SAABootstrapper.NUM_MONTHS, 1), np.nan)

        AR1_process = 'trend' if self.simulation_config.simAR1Process == 1 else 'old'
        rates = [eq_R] * T
        betas = [beta] * T

        for t in range(T):

            if rate_type == RateType.INFLATION:
               if t == 0:
                   paths[t] = rates[t]
               elif t < round(SAABootstrapper.NUM_MONTHS * long_horizon):
                   paths[t] = rates[t] * (1-beta) + beta * paths[t-1] + shocks_panel[t]
            else:
                if t == 0:
                    paths[t] = rates[t]
                    trend[t] = rates[t]
                elif t < round(SAABootstrapper.NUM_MONTHS * hold_start_states):
                    trend[t] = rates[t]
                    paths[t] = rates[t] + shocks_panel[t]
                elif t < round(SAABootstrapper.NUM_MONTHS * long_horizon):
                    trend[t] = rates[t] * (1 - betas[t]) + betas[t] * trend[t - 1]
                    paths[t] = rates[t] * (1 - betas[t]) + betas[t] * paths[t - 1] + shocks_panel[t]

                if AR1_process == 'old' and not RateType.INFLATION == rate_type:
                    paths[t] = np.maximum(risk_free_floor / SAABootstrapper.NUM_MONTHS, paths[t])
                elif AR1_process == 'trend' and not RateType.INFLATION == rate_type:
                    excess_level = np.mean(np.maximum(risk_free_floor / SAABootstrapper.NUM_MONTHS, paths[t])) - trend[t]
                    paths[t] = np.maximum(risk_free_floor / SAABootstrapper.NUM_MONTHS, paths[t] - excess_level)

        return PathsPanel(paths)

    @property
    def curr_env_ind_name(self):
        return CAppConfig.get_config_util().get_current_environment_indicator_name()

    def get_current_environment_indicator(self):
        if self._curr_env_ind is None:
           curr_env_ind = GlobalDataSource().get_time_series_data_from_ticker(self._schema.risk_free_rate_ticker)
           curr_env_ind = curr_env_ind[self._schema.start_date: self._schema.end_date] / 100
           self._curr_env_ind = curr_env_ind[self.sim_start_date:self.sim_end_date]
           self._curr_env_ind = (self._curr_env_ind > self._curr_env_ind.quantile(0.75)).astype(int)
        return self._curr_env_ind


    def demean_values_in_blocks(
            self,
            shocks: np.array,
            current_indicator,
    ) -> np.array:

        """
        Demeans a time series of shocks within contiguous blocks marked as "current" periods.

        Parameters
        ----------
        shocks : np.ndarray
            A 1D or 2D NumPy array of time series shocks, typically residuals from an AR(1) model.
            Shape is (T,) or (T, N), where T is time and N is number of series.

        current_indicator : object
            An object with a `.get_dataframe().values` method that returns a 1D binary array of shape (T,),
            indicating whether each time step is considered part of a "current" period (1 = current, 0 = not current).

        Returns
        -------
        np.ndarray
            A copy of the shocks array where each contiguous block of current periods (1s) has been demeaned separately.
            All other values are left unchanged.

        Raises
        ------
        Exception
            If the length of `shocks` does not match the length of `current_indicator`.

        Notes
        -----
        This function performs block-wise demeaning rather than global demeaning of all time steps
        labeled as "current" (1) in order to preserve the structure and internal consistency of
        contiguous time regimes.

        Block-level demeaning is particularly important in simulation or bootstrapping contexts,
        where these segments may be resampled as self-contained windows. Global demeaning across
        all current periods would introduce cross-regime distortions and contaminate the time-local
        statistical properties of the shocks, potentially undermining the realism and stationarity
        of the resulting simulated paths.
        """


        if not len(shocks) == len(current_indicator):
            raise Exception('Shocks dataframe does not match current indicator')

        vals_no_mean = np.copy(shocks)
        ind_vals = np.copy(current_indicator.values)

        i = 0
        len_vals_no_mean = len(vals_no_mean) - 1

        while i < len_vals_no_mean:

            while i < len_vals_no_mean and ind_vals[i] == 0:
                i += 1

            if i > len_vals_no_mean:
                i -= 1

            if i < len(shocks):

                block_start = i
                while i < len_vals_no_mean and ind_vals[i] == 1:
                    i += 1

                if i > len_vals_no_mean:
                    i -= 1

                block_end = i
                if ind_vals[i] == 0:
                    block_end -= 1

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    mu = np.mean(vals_no_mean[block_start:block_end + 1])

                vals_no_mean[block_start:block_end + 1] = vals_no_mean[block_start:block_end + 1] - mu
                if i == len_vals_no_mean:
                    i += 1
        return vals_no_mean

    def prepare_paths(self, asset, bs_indicies, long_term_shocks) -> PathsPanel:

        paths_panels = self.prepare_boostrap_blocks(bs_indicies, long_term_shocks)
        blocks = paths_panels.get_paths()

        df = asset[self.sim_start_date:self.sim_end_date]
        return PathsPanel(df.values[blocks])

    def prepare_boostrap_blocks(self,
            bs_indicies: BootstrapIndicies,
            long_term_shocks: bool,
    ):

        T = SAABootstrapper.NUM_MONTHS * max(self.medium_horizon, self.long_horizon)

        # Get the current environemnt indicator
        curr_env_ind = self.get_current_environment_indicator()

        # Long blocks capture all timesteps (Unconditional)
        long_blocks = np.array([np.arange(0, len(curr_env_ind.values))]).T
        long_blocks_ext = np.copy(long_blocks)

        # Short blocks capute timesteps where curr_env_ind.values is equl to 1
        short_blocks = np.array([np.flatnonzero(curr_env_ind.values)]).T
        short_blocks_ext = np.copy(short_blocks)

        for i in range(0, self.horizon_mult - 1):
            long_blocks_ext = np.vstack([long_blocks_ext, long_blocks])
            short_blocks_ext = np.vstack([short_blocks_ext, short_blocks])

        # Medium blocks capture both short and long
        med_blocks_ext = np.copy(long_blocks_ext)
        short_mult = round(long_blocks.size / short_blocks.size)
        # Medium term indicator signals if a strapped timestep falls in the short block/current environment
        mt_indicator = np.zeros(long_blocks_ext.shape, dtype=int)

        # loop through the iterations equal to short_mult and append the short blocks
        for i in range(0, short_mult):
            med_blocks_ext = np.vstack([med_blocks_ext, short_blocks_ext])
            mt_indicator = np.vstack(
                (mt_indicator, np.ones(short_blocks_ext.shape, dtype=int))
            )

        paths = np.full((T, self.nbstraps), np.nan)
        short_block_indicator = np.zeros((T, self.nbstraps), dtype=int)

        for t in range(0, T):
            if long_term_shocks:
                paths[t] = long_blocks_ext[bs_indicies.get_long_term_indicies()[t]].T
            else:
                if t < SAABootstrapper.NUM_MONTHS * self.short_horizon:
                    paths[t] = short_blocks_ext[bs_indicies.get_short_term_indicies()[t]].T
                    short_block_indicator[t] = np.ones(self.nbstraps, dtype=int)
                elif t < SAABootstrapper.NUM_MONTHS * self.medium_horizon:
                    paths[t] = med_blocks_ext[bs_indicies.get_medium_term_indicies()[t]].T
                    short_block_indicator[t] = mt_indicator[bs_indicies.get_medium_term_indicies()[t]].T
                else:
                    # Long-Horizon
                    paths[t] = long_blocks_ext[bs_indicies.get_long_term_indicies()[t]].T

        return PathsShortBlockPanel(paths.astype(int), short_block_indicator)

    def prepare_portfolio_paths(self,
            ptf,
            bs_indicies,
            long_term_shocks
    ):

        long_horizon = 20
        num_bstraps = self.nbstraps
        scaling_factor = np.sqrt(SAABootstrapper.NUM_MONTHS)

        curr_env_ind = self.get_current_environment_indicator()

        if long_term_shocks:
            # This is used for risk analysis, such as VaR and PoL
            returns_panel = ptf.get_stressed_returns_panel()
            betas = returns_panel.betas
        else:
            returns_panel = ptf.get_stressed_returns_panel()

            curr_inds = curr_env_ind.loc[returns_panel.dates].values.astype(bool)


            # Set Medium Term Returns Panel
            returns_panel.factor_panel_normalized_MT = returns_panel.factor_panel_normalized[curr_inds[:, 0]]
            returns_panel.stress_coeff_panel_MT = returns_panel.stress_coeff_panels[curr_inds[:, 0]]

            ########  If any factors are normalized to LT (i.e. we dont use the MT Sharpes), do it now
            #idxs = [CAppConfig.get_config_util().get_factor_config(x).normalize_to_LT
            #        if hasattr(CAppConfig.get_config_util().get_factor_config(x), 'normalize_to_LT') else False
            #        for x in CAppConfig.get_config_util().get_return_factors().keys()]


            #returns_panel._factor_panel_MT_normalized[:, idxs] = (returns_panel._factor_panel_MT_normalized[:, idxs] -
            #                                                      np.mean(returns_panel._factor_panel_MT_normalized[:,
            #                                                              idxs], axis=0))
            #returns_panel._factor_panel_MT_normalized[:, idxs] = (returns_panel._factor_panel_MT_normalized[:, idxs] /
            #                                                      np.std(returns_panel._factor_panel_MT_normalized[:,
            #                                                              idxs], axis=0, ddof=1))
            #returns_panel._factor_panel_MT_normalized[:, idxs] = (returns_panel._factor_panel_MT_normalized[:, idxs] +
            #                                                      np.mean(returns_panel._factor_panel_normalized[:,
            #                                                              idxs], axis=0))

            ####### Cut any Sharpe Ratios ######
            #return_factors = CAppConfig.get_config_util().get_return_factors().values()

            #cap_factors = [
            #    factor for factor in return_factors
            #    if factor.factorType == FactorType.Return
            #       and factor.factorConfig.cap_medium_sharpe not in (None, 0.0)
            #]

            #if cap_factors:
            #    for cap_factor in cap_factors:
            #        loc = [(idx, factor_name) for idx, factor_name in enumerate(context.get_return_factors_panel().factor_names) \
            #               if factor_name == cap_factor.factorName]
            #        if len(loc) == 1:
            #            returns_panel._factor_panel_MT_normalized[:, loc[0][0]] = \
            #                self.cap_mean_by(returns_panel._factor_panel_MT_normalized[:, loc[0][0]],
            #                                  cap_factor.factorConfig.cap_medium_sharpe / np.sqrt(SAABootstrapper.NUM_MONTHS))
            #        else:
            #            raise Exception("Unable to find factor or multiple factors with identical names for factor " + cap_factor.factor.symbol + " in context.getReturnFactorList()")

            normalized_factors = returns_panel.factor_panel_normalized_MT

            returns_panel.factor_sharpes_MT = np.mean(
                normalized_factors, axis=0
            ) * scaling_factor

            stress_coeffs = returns_panel.stress_coeff_panel_MT
            sharpe_ratios = returns_panel.factor_sharpes_MT.conj().T  # Transpose and conjugate
            betas = returns_panel.betas

            # Tile and transpose beta vector to match shape
            tiled_betas = np.tile(betas.conj().T, (normalized_factors.shape[0], 1))

            # Element-wise multiply stress coefficients by betas
            stress_weights = stress_coeffs * tiled_betas

            # Final factor contributions
            returns_panel.factor_contributions_panel_MT = normalized_factors * stress_weights
            returns_panel.medium_term_risk_premia = sharpe_ratios * betas * scaling_factor

            #returns_panel._factor_contributions_panel_MT = returns_panel._factor_panel_MT_normalized * (returns_panel._stress_coeff_panel_MT * np.tile((returns_panel._betas).conj().T, (returns_panel._factor_panel_MT_normalized.shape[0], 1)))
            #returns_panel._medium_term_risk_premia = returns_panel._factor_sharpe_MT.conj().T * returns_panel._betas * np.math.sqrt(SAABootstrapper.NUM_MONTHS)

        # Take the average over the first 60 months for a 5 year average return estimate
        returns_panel.medium_term_risk_premia_5Y = np.mean(
            returns_panel.factor_contributions_panel[0:60], axis=0
        ) * SAABootstrapper.NUM_MONTHS

        paths_panels = self.prepare_boostrap_blocks(
            bs_indicies,
            long_term_shocks
        )

        # Resample the systematic component
        blocks = paths_panels.get_paths()
        ptf_returns = returns_panel.get_risk_premia()
        systematic_panel = ptf_returns[blocks]
        portfolio_paths = PortfolioPaths(systematic_panel)

        # If we are not using LT shocks, replace LT shocks with ST/MT Shocks
        if not long_term_shocks:

            curr_env_ind = paths_panels.get_short_block_indicator().astype(bool)

            mt_ptf_returns = np.sum(
                returns_panel.factor_contributions_panel_MT, axis=1, keepdims=True
            )
            current_env_map = np.cumsum(curr_inds, 0) * curr_inds
            current_env_map[np.flatnonzero(current_env_map)] -= 1
            portfolio_paths.get_systematic_panel()[curr_env_ind] = mt_ptf_returns[
                current_env_map[blocks[curr_env_ind]]
            ][:, 0, 0]

        # Get the factor covariance matrix
        cov = np.cov(
            returns_panel.factor_panel_normalized.conj().T
        )

        # compute the portfolio idiosyncratic variance
        idio_var = max(
            0, np.power(ptf.get_risk(), 2) / SAABootstrapper.NUM_MONTHS -
               np.matmul(
                   np.matmul(
                       betas.conj().T, cov), betas
               )
        )

        # Add the idio noise at the end to preserve total vol
        idio_panel = np.full((SAABootstrapper.NUM_MONTHS * long_horizon, num_bstraps), np.nan)
        for t in range(0, long_horizon * SAABootstrapper.NUM_MONTHS):
            idio_panel[t] = SC.sqrt(idio_var) * SAABootstrapper.R1.randn(1, num_bstraps)

        # set variables in portfolio paths struct
        portfolio_paths.set_idio_panel(idio_panel)
        portfolio_paths.set_alpha_total_monthly(returns_panel.alpha_total_monthly)
        portfolio_paths.set_returns_panel(returns_panel)

        return portfolio_paths

    def cap_mean_by(
            self,
            values,
            max_mean
    ):
        if np.mean(values, axis=0) > max_mean:
            new_vals = values - np.nanmean(values, axis=0) + max_mean
        else:
            new_vals = values
        return new_vals

    def get_beta_and_shocks(
            self,
            frequency=None,
            currency=None,
            asset_ts=None,
            rate_ts=None,
            ts_values=None,
            new_beta=-1
    ):

        if isinstance(frequency, Frequency):
           frequency = frequency.obs_per_year()

        temp_rate = rate_ts.loc[self.sim_start_date:self.sim_end_date]
        temp_vals = temp_rate.values.reshape(-1, 1)
        temp_trend = np.column_stack([list(range(1, len(temp_vals) + 1))])
        temp_trend = temp_trend - np.mean(temp_trend)

        _, betas = Regression.simple_regression_OLS_with_array(
            temp_trend,
            temp_vals,
        )

        temp_vals_detrended = temp_vals - betas.item() * temp_trend

        temp_beta = [None] * frequency
        temp_beta_detrended = [None] * frequency

        for m in range(frequency):

            if m == 0:
                start = frequency - 1
                stop = temp_vals.shape[0]
            else:
                start = m - 1
                stop = temp_vals_detrended.shape[0]

            tmp_idx = range(start, stop, frequency)

            tmp_data = np.asarray([temp_vals[i][0] for i in tmp_idx])
            tmp_data_detrended = np.asarray([temp_vals_detrended[i][0] for i in tmp_idx])

            a, b, s = self.extract_shocks(
                pd.DataFrame(tmp_data, index=tmp_idx, columns=['values'])
            )
            temp_beta[m] = b.item()

            a_prime, b_prime, s_prime = self.extract_shocks(
                pd.DataFrame(tmp_data_detrended, index=tmp_idx, columns=['values'])
            )
            temp_beta_detrended[m] = b_prime.item()

        if new_beta > -1:
            beta_M = new_beta
        else:
            raise ValueError('Unknown Beta')

        #idx = rate_ts.index.get_loc(self.sim_start_date)
        shocks_ann = ts_values[1:] - beta_M * ts_values[:-1]
        shocks_ann = np.insert(shocks_ann, 0, 0)
        shocks_ann = shocks_ann - np.mean(shocks_ann)
        shocks = np.reshape(shocks_ann, (shocks_ann.shape[0],))
        beta = beta_M

        return beta, shocks


if __name__ == "__main__":


    from epsilonPhi.core.schema.Schema import ContextCreator
    from epsilonPhi.core.portfolio.SAAPortfolio import SAAPortfolio

    schema = ContextCreator(currency='USD',
                            start_date='30-Nov-1983',
                            end_date='31-Dec-2022').create_context()

    ptf = SAAPortfolio('portfolio', schema)
    ptf.add_asset_by_name('MSUSAML', 0.5, 0)
    ptf.add_asset_by_name('LHAGGBD', 0.5, 0)
    ptf.setup()

    bootstrap = SAABootstrapper(schema)
    bs_indicies = bootstrap.get_bootstrap_indicies()
    b = bootstrap.prepare_portfolio_paths(ptf, bs_indicies, 0)
















