import os

import pandas as pd
from epsilonPhi.core.portfolio.SAAPortfolio import SAAPortfolio
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.schema.Schema import ContextCreator
import numpy as np

schema = ContextCreator(
    currency="USD",
    start_date="30-Nov-1983",
    end_date='31-Dec-2022'
).create_context()

UNDERLYING = "S&PCOMP"

class AutoCallable(object):

    def __init__(self, schema, underlying):

        self._underlying = underlying
        self._schema = schema
        self._coupon_level = None
        self._coupon = None
        self._buffer_level = None
        self._call_level = None
        self._frequency = Frequency.QUARTERLY
        self._maturity = None
        self._first_call_period = None

    @property
    def frequency(self):
        return self._frequency

    @property
    def num_obs_per_year(self):
        return self.frequency.obs_per_year()

    @property
    def call_dates_per_year(self):
        return self.num_obs_per_year

    @property
    def obs_per_period(self):
        return int(12 / self.call_dates_per_year)

    @property
    def long_horizon(self):
        return self._maturity * 12

    @property
    def first_call_period(self):
        return self._first_call_period

    @property
    def coupon_level(self):
        return self._coupon_level

    def set_first_call_period(self, period):
        self._first_call_period = period

    def set_coupon_level(self, level):
        self._coupon_level = level

    def set_buffer_level(self, level):
        self._buffer_level = level

    def set_call_level(self, level):
        self._call_level = level

    def set_coupon(self, coupon):
        self._coupon = coupon

    def set_frequency(self, frequency: Frequency.QUARTERLY):
        self._frequency = frequency

    def set_maturity(self, maturity):
        self._maturity = maturity

    def get_underlying(self):
        return self._schema.get_asset_from_name(self._underlying)

    def simulate_underlying(self):
        if not hasattr(self, '_paths'):
            _, lvls = self.get_underlying().simulate(long_term_shocks=True)
            self._paths = np.concatenate((np.ones((1, lvls.shape[1])), lvls))
        return self._paths

    def get_call_date_locs(self):
        return range(self.obs_per_period,  self.long_horizon+1, self.obs_per_period)

    def get_prices_on_call_dates(self):
        lvls = self.simulate_underlying()
        return lvls[self.get_call_date_locs(), :]

    def get_underlying_quantiles(self):

        # 1. Quantiles to test
        q = [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1]

        # 2. Get underlying cumulative returns
        rtns = pd.DataFrame(-1 + self.get_prices_on_call_dates()[-1])

        return rtns.quantile(q)

    def get_payoff_quantiles(self):

        # 1. Quantiles to test
        q = [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1]

        # 2. Get underlying cumulative returns
        rtns = pd.DataFrame(self.simulate())

        return rtns.quantile(q)

    def get_quantiles(self):
        df = pd.concat((
            self.get_underlying_quantiles(),
            self.get_payoff_quantiles()
        ), axis=1)

        df.columns = [self._underlying, "AC Note"]
        return df

    def simulate(self):

        # get the prices on the call dates
        lvls = self.get_prices_on_call_dates()

        # Period coupon
        period_coupon = self._coupon / self.call_dates_per_year

        # pre-allocate array of couponds
        arr = np.zeros(lvls.shape)
        arr[:self.first_call_period-1] = period_coupon

        # track if the not has been called
        is_called = np.full(arr.shape[1], False)

        T = lvls.shape[0]
        for t in range(self.first_call_period-1, T):

            if t < self.first_call_period:
               arr[t] = period_coupon
            else:

                # find the index of all the coupons that are above coupon level
                idx = np.logical_and(
                    ~is_called,
                    lvls[t, :] > self._coupon_level)

                # add the coupon to the valid paths
                arr[t, idx] = period_coupon

                # Set any notes called to True
            is_called[lvls[t, :] > 1] = True

        mask = (lvls[t] < self._buffer_level) & (~is_called)
        arr[t, mask] += lvls[t, mask] - self._buffer_level
        return np.nansum(arr, axis=0)

self = AutoCallable(schema, UNDERLYING)
self.set_coupon(0.09)
self.set_coupon_level(0.85)
self.set_call_level(1)
self.set_buffer_level(0.85)
self.set_maturity(2)
self.set_first_call_period(2)
note_rtns = self.simulate()

# path to portfolio weights
path = '/Users/francisbarker/Repositories/Python/epsilon-phi/epsilon-phi-core/src/Scripts/Case Studies/4. Structured Notes/Autocallable/Structured Notes.xlsx'


# 2. Load Portfolios
ptfs = SAAPortfolio.get_portfolios_from_template(
        currency="USD",
        template_path=path,
        schema=schema)

_, ws_ptf = ptfs[0].get_portfolio_simulated_returns_panel(long_term_shocks=True, frequency=Frequency.YEARLY)
ptf_rtns = -1 + ( ws_ptf[2] / 1 )

sim_rtns = []
for p in range(1, len(ptfs)):

    tmp = ptfs[p].deepcopy(name='tmp')

    wt = tmp.get_asset("S&PCOMP").weight
    tmp.remove_asset_by_name("S&PCOMP", rebalance=True)
    _, tmp_ws = tmp.get_portfolio_simulated_returns_panel(long_term_shocks=True, frequency=Frequency.YEARLY)
    tmp_sim_rtns = -1 + (tmp_ws[2] / 1)
    tmp_total_returns = wt * note_rtns + (1-wt) * tmp_sim_rtns
    sim_rtns.append(tmp_total_returns)

sim_rtns.insert(0, ptf_rtns)
df = pd.DataFrame(sim_rtns).T


q = [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1]
quants = df.quantile(q)

sim_rtns

sim = self.simulate_underlying()
rtns = np.diff(np.log(sim), axis=0)

sub = rtns[:24]

vols = np.sqrt(12) * np.std(sub, ddof=1, axis=0)
rtns = np.sum(sub, axis=0)


rtns_df = pd.DataFrame([rtns, note_rtns], index=["U", "N"]).T
rtns_df['quantile_bin'] = pd.qcut(rtns_df['U'], q=5, labels=False)

group_stats = rtns_df.groupby('quantile_bin').agg(['mean', 'std'])

# Step 1: Compute mean and std of returns_1
mean_r1 = rtns_df['U'].mean()
std_r1 = rtns_df['U'].std()

def bucket(row):
    r = row['U']
    if r > std_r1:
        return '> +1σ'
    elif 0 < r <= std_r1:
        return '0 to +1σ'
    elif -std_r1 <= r < 0:
        return '0 to –1σ'
    elif r < -std_r1:
        return '< –1σ'
    else:
        return '0'  # optional: for exactly zero


rtns_df['std_bucket'] = rtns_df.apply(bucket, axis=1)
rtns_df['out'] = rtns_df["U"] < rtns_df["N"]

# Step 3: Group by the bucket and compute mean and std
grouped_stats = rtns_df.groupby('std_bucket').agg(['mean', 'std'])