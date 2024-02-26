import datetime as dt

import pandas as pd
from matplotlib import pyplot as plt
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.dataModel.enums.Database import PriceQuote
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries
from epsilonPhi.core.utils.DateUtils import DateUtils
from epsilonPhi.ep_strategies.fx.linear.signals import Signals
import numpy as np



class Factor(object):

    _EURO_LEGACY = ['FRF', 'DEM', 'NLG', 'BEF', 'PTE', 'ESP', 'FIM', 'IEP', 'GRD', 'ATS', 'CYP', 'EEK', 'LUF', 'MCF', 'MTL', 'SIT', 'SKK', 'SNL', 'VAL']
    _EURO_CUTOFF_DATE = dt.date(year=1998, month=12, day=31)
    _G_10_CURRENCIES = ['AUD', 'CAD', 'DKK', 'JPY', 'NZD', 'NOK', 'SEK', 'CHF', 'GBP', 'DEM', 'FRF', 'ITL', 'NLG', 'BEF', 'EUR']
    _BEF_EXCLUSION = (dt.date(year=1989, month=12, day=15), dt.date(year=1994, month=2, day=3))
    _TRY_START_DATE = dt.date(year=2001, month=12, day=20)
    _PRICE_QUOTE_TYPES = [PriceQuote.MID,
                          PriceQuote.ASK,
                          PriceQuote.BID]

    _datasource = GlobalDataSource()

    def __init__(self, start_date, end_date, frequency=Frequency.BUSINESS_MONTHLY):

        self._start_date = start_date
        self._end_date = end_date
        self._frequency = frequency
        self._price_quote_type = PriceQuote.MID

        self._number_of_portfolios = 5

        self._foreign_currencies = None
        self._domestic_currency = None
        self._signal = None

    def reset_properties(self):
        self._signal = None

    @property
    def domestic_currency(self):
        return self._domestic_currency

    @domestic_currency.setter
    def domestic_currency(self, value):
        pass

    @property
    def foreign_currencies(self):
        return self._foreign_currencies

    @foreign_currencies.setter
    def foreign_currencies(self, value):
        pass

    @property
    def currency_pairs(self):
        if self.domestic_currency and self.foreign_currencies:
            return np.array([X + self.domestic_currency for X in self.foreign_currencies])

    @property
    def signal(self):
        return self._signal

    @signal.setter
    def signal(self, value: CTimeSeries):
        self._signal = value

    def set_signal(self, value: CTimeSeries):

        self._foreign_currencies = [x[0:3] for x in value.columns]
        dom_ccy = np.unique([x[3:] for x in value.columns])
        assert len(dom_ccy) == 1, 'Error - can only have 1 base/domestic currency'

        if isinstance(dom_ccy, str):
            self._domestic_currency = dom_ccy
        else:
            self._domestic_currency = dom_ccy[0]

        self.signal = value.copy()

    @property
    def number_of_portfolios(self):
        return self._number_of_portfolios

    @number_of_portfolios.setter
    def number_of_portfolios(self, value):
        self._number_of_portfolios = value

    @property
    def price_quote_type(self):
        return self._price_quote_type

    @price_quote_type.setter
    def price_quote_type(self, value):
        self._price_quote_type = value

    @property
    def rebalancing_frequency(self):
        return self._rebalancing_frequency

    @rebalancing_frequency.setter
    def rebalancing_frequency(self, value):

        self.reset_properties()
        self._rebalancing_frequency = value
        rebal_dates = DateUtils.get_date_range(self._start_date,
                                               self._end_date,
                                               value)

        self._rebalancing_dates = rebal_dates.insert(0, DateUtils.shift_date(rebal_dates[-1], value, 1)).sort_values()

    @property
    def pricing_dates(self):
        if not hasattr(self, '_pricing_dates'):
            self._pricing_dates = DateUtils.get_date_range(self._start_date,
                                                           self._end_date,
                                                           self._frequency)
        return self._pricing_dates

    @property
    def rebalancing_dates(self):
        if not hasattr(self, '_rebalancing_dates'):
            rebal = DateUtils.get_date_range(self._start_date,
                                                           self._end_date,
                                                           self.rebalancing_frequency)
            self._rebalancing_dates = rebal.insert(0, DateUtils.shift_date(rebal[-1], self.rebalancing_frequency, 1)).sort_values()
        return self._rebalancing_dates

    @property
    def rebalancing_bool(self):
        return np.isin(self.pricing_dates,
                       self.rebalancing_dates)

    @property
    def fwd_maturities(self):
        return self.rebalancing_dates[1:][np.cumsum(self.rebalancing_bool)-1]

    def __load_data(self):
        self.__load_forward_prices()

    def plol_PnLCurve(self, strategy=None):
        self.PnLCurve.get(strategy, self.PnLCurve).dropna().plot()
        plt.show()

    def __load_forward_prices(self):

        rebal_idx = np.isin(self.pricing_dates, self.rebalancing_dates)
        maturity_dates = self.rebalancing_dates[np.cumsum(rebal_idx)]
        self._forward_prices = self._datasource.get_fx_forward_prices(self.currency_pairs,
                                                                      self.pricing_dates.append(self.pricing_dates),
                                                                      maturity_dates.append(self.pricing_dates),
                                                                      ['mid','ask','bid'])


    def get_reverse_currency_pair(self, currency_pair):
        return currency_pair[3:] + currency_pair[0:3]

    def get_returns_panel(self, pricing_date, maturity_date):

        if pricing_date.date() > self._EURO_CUTOFF_DATE:
           EUR_LEGACIES = [self.domestic_currency + X for X in self._EURO_LEGACY]
           EUR_LEGACIES = EUR_LEGACIES + [self.get_reverse_currency_pair(x) for x in EUR_LEGACIES]
           tradable_currencies = np.setdiff1d(self.currency_pairs, EUR_LEGACIES)
        else:
           tradable_currencies = np.setdiff1d(self.currency_pairs,
                                               ['EUR' + self.domestic_currency, self.domestic_currency + 'EUR'])


        if np.logical_and(pricing_date.date() < self._BEF_EXCLUSION[1], pricing_date.date() > self._BEF_EXCLUSION[1]):
            tradable_currencies = np.setdiff1d(tradable_currencies,
                                          [self.domestic_currency + 'BEF',
                                               'BEF' + self.domestic_currency])

        if pricing_date.date() < self._TRY_START_DATE:
            tradable_currencies = np.setdiff1d(tradable_currencies,
                                          [self.domestic_currency + 'TRY',
                                              'TRY' + self.domestic_currency])

        dates = self.pricing_dates[(self.pricing_dates <= maturity_date) &
                                   (self.pricing_dates >= pricing_date)]

        mid_dates = self.pricing_dates[(self.pricing_dates < maturity_date) &
                                   (self.pricing_dates > pricing_date)]

        # Get the signal that we sort on
        signal = self.signal[tradable_currencies].loc[[pricing_date]].dropna(axis=1)

        if self.price_quote_type == PriceQuote.MID:
            pqt = ('mid', 'mid')
        else:
            pqt = ('bid', 'ask')

        # Get all prices corresponding to this rebalancing period

        #### Forward Prices - Long Leg ####
        # On the long side - we buy at the ask and sell at the bid

        fwd_prices_period = self._forward_prices.loc[maturity_date]

        long_fwd_asks = fwd_prices_period.loc[[pricing_date], pd.IndexSlice[:, pqt[1]]].droplevel(level=1, axis=1)
        long_fwd_bids = fwd_prices_period.loc[[maturity_date], pd.IndexSlice[:, pqt[0]]].droplevel(level=1, axis=1)
        long_fwd_mids = fwd_prices_period.loc[mid_dates, pd.IndexSlice[:, 'mid']].droplevel(level=1, axis=1)
        long_fwds = pd.concat([long_fwd_asks, long_fwd_mids, long_fwd_bids], axis=0).sort_index()

        long_rx_panel = long_fwds.get(signal.columns).dropna(axis=1)
        long_rx_returns = (np.log(long_rx_panel) - np.log(long_rx_panel.shift(1))) # Bid - Ask


        #### Spot Prices - Long Leg ####

        spt_prices_period = self._forward_prices.loc[list(zip(dates, dates))].droplevel(level=0, axis=0)

        long_spt_asks = spt_prices_period.loc[[pricing_date], pd.IndexSlice[:, pqt[1]]].droplevel(level=1, axis=1)
        long_spt_mids = spt_prices_period.loc[mid_dates, pd.IndexSlice[:, 'mid']].droplevel(level=1, axis=1)
        long_spt_bids = spt_prices_period.loc[[maturity_date], pd.IndexSlice[:, pqt[0]]].droplevel(level=1, axis=1)
        long_spts = pd.concat([long_spt_asks, long_spt_mids, long_spt_bids], axis=0).sort_index()

        long_spt_panel = long_spts.get(signal.columns).dropna(axis=1)
        long_spt_returns = (np.log(long_spt_panel) - np.log(long_spt_panel.shift(1)))

        #### Forward Prices - Short Leg ####
        # On the long side - we buy at the ask and sell at the bid

        short_fwd_bids = fwd_prices_period.loc[[pricing_date], pd.IndexSlice[:, pqt[0]]].droplevel(level=1, axis=1)
        short_fwd_asks = fwd_prices_period.loc[[maturity_date], pd.IndexSlice[:, pqt[1]]].droplevel(level=1, axis=1)
        short_fwd_mids = fwd_prices_period.loc[mid_dates, pd.IndexSlice[:, 'mid']].droplevel(level=1, axis=1)
        shot_fwds = pd.concat([short_fwd_asks, short_fwd_mids, short_fwd_bids], axis=0).sort_index()

        short_rx_panel = shot_fwds.get(signal.columns).dropna(axis=1)
        short_rx_returns = (np.log(short_rx_panel) - np.log(short_rx_panel.shift(1)))

        #### Spot Prices - Short Leg ####

        short_spt_bids = spt_prices_period.loc[[pricing_date], pd.IndexSlice[:, pqt[0]]].droplevel(level=1, axis=1)
        short_spt_asks = spt_prices_period.loc[[maturity_date], pd.IndexSlice[:, pqt[1]]].droplevel(level=1, axis=1)
        short_spt_mids = spt_prices_period.loc[mid_dates, pd.IndexSlice[:, 'mid']].droplevel(level=1, axis=1)
        shot_spts = pd.concat([short_spt_asks, short_spt_mids, short_spt_bids], axis=0).sort_index()

        short_spt_panel = shot_spts.get(signal.columns).dropna(axis=1)
        short_spt_returns = (np.log(short_spt_panel) - np.log(short_spt_panel.shift(1)))

        return long_rx_returns, long_spt_returns, short_rx_returns, short_spt_returns

    def get_currency_portfolio_locs(self, N):

        ccys_per_portfolio = np.tile(int(np.floor(N / self.number_of_portfolios)), self.number_of_portfolios)
        remaining_currencies = np.mod(N, self.number_of_portfolios)
        ccys_per_portfolio[0:remaining_currencies] = ccys_per_portfolio[0:remaining_currencies] + 1

        locs = np.zeros((1, N), dtype=int)
        locs[0, np.hstack((0, ccys_per_portfolio[:, -1].cumsum()))] = 1
        return np.cumsum(locs)

    def run_strategy(self):

        rebal_dates = self.rebalancing_dates
        fwd_mats = self.fwd_maturities.unique()
        self.__load_data()

        str_df = pd.DataFrame()
        T = len(rebal_dates)
        PV = 1

        for t in range(1, T-1):
            print('{}'.format(rebal_dates[t-1]))

            long_excess, long_spt, short_excess, short_spt = self.get_returns_panel(rebal_dates[t-1], fwd_mats[t-1])
            ranked_signal = self.signal[long_excess.columns].loc[rebal_dates[t-1]].sort_values()

            if ranked_signal.size > self.number_of_portfolios:

                N = int(np.ceil(ranked_signal.size / self.number_of_portfolios))

                # High Minus Low Portfolio
                wt = np.repeat(1/N, N)

                H = np.sum(wt * np.exp(long_excess.get(ranked_signal.index[-N:]).fillna(0).cumsum()), axis=1)
                L = np.sum(wt * np.exp(short_excess.get(ranked_signal.index[0:N]).fillna(0).cumsum()), axis=1)
                HML = np.exp((H - L).diff()) - 1
                H_rtns = np.exp(H.diff())-1
                L_rtns = np.exp(L.diff())-1

                bmk = np.mean(np.exp(long_spt.fillna(0).cumsum()), axis=1).diff()
                HML_spt = np.exp((np.sum(wt * np.exp(long_spt.get(ranked_signal.index[-N:]).fillna(0).cumsum()), axis=1) -
                          np.sum(wt * np.exp(short_spt.get(ranked_signal.index[0:N]).fillna(0).cumsum()), axis=1)).diff()) - 1

                # Linear in signal size
                W_sig = (2/np.sum(np.abs(ranked_signal - np.mean(ranked_signal)))) * (ranked_signal - np.mean(ranked_signal))
                lin_sig = np.exp((np.sum(W_sig[W_sig > 0].values * np.exp(long_excess.get(W_sig[W_sig > 0].index).fillna(0).cumsum()), axis=1) +
                              np.sum(W_sig[W_sig < 0].values * np.exp(short_excess.get(W_sig[W_sig < 0].index).fillna(0).cumsum()), axis=1)).diff()) - 1

                lin_sig_spt = np.exp((np.sum(W_sig[W_sig > 0].values * np.exp(long_spt.get(W_sig[W_sig > 0].index).fillna(0).cumsum()), axis=1) +
                                  np.sum(W_sig[W_sig < 0].values * np.exp(short_spt.get(W_sig[W_sig < 0].index).fillna(0).cumsum()), axis=1)).diff()) - 1

                # Build Portfolio linear in rank
                W_rank = 2 * ((ranked_signal.rank() - ranked_signal.rank().mean()) /\
                         ((ranked_signal.rank() - ranked_signal.rank().mean()).abs().sum()))

                lin_rank = np.exp((np.sum(
                    W_rank[W_rank > 0].values * np.exp(long_excess.get(W_rank[W_rank > 0].index).fillna(0).cumsum()), axis=1) +
                                  np.sum(W_rank[W_rank < 0].values * np.exp(
                                      short_excess.get(W_rank[W_rank < 0].index).fillna(0).cumsum()), axis=1)).diff()) - 1

                lin_rank_spt = np.exp((np.sum(
                    W_rank[W_rank > 0].values * np.exp(long_spt.get(W_rank[W_rank > 0].index).fillna(0).cumsum()), axis=1) +
                                      np.sum(W_rank[W_rank < 0].values * np.exp(
                                          short_spt.get(W_rank[W_rank < 0].index).fillna(0).cumsum()),
                                             axis=1)).diff()) - 1

                df = pd.concat([HML, L_rtns, H_rtns, HML_spt, lin_sig, lin_sig_spt, lin_rank, lin_rank_spt, bmk], axis=1).dropna()
                str_df = pd.concat((str_df, df), axis=0)

        str_df.columns = ['HML', 'L', 'H', 'HML_spt', 'SIGNAL_WEIGHTED', 'SIGNAL_WEIGHTED_SPT', 'RANK_WEIGHTED', 'RANK_WEIGHTS_SPT', 'Benchmark']
        self.strategies = CTimeSeries(str_df.reindex(self.pricing_dates), returns_type=ReturnsType.SIMPLE, ts_type=TimeSeriesType.RETURNS)
        self.PnLs = self.strategies.get(['HML', 'SIGNAL_WEIGHTED', 'RANK_WEIGHTED', 'Benchmark'])
        self.PnLCurve = self.PnLs.get_levels()


if __name__ == '__main__':

    G10 = Factor._G_10_CURRENCIES

    start_date = dt.date(year=1983, month=10, day=31)
    end_date = dt.date(year=2023, month=10, day=31)
    frequency = Frequency.BUSINESS_MONTHLY

    base_currency = 'USD'
    currency_pairs = [x + '/' + base_currency for x in G10]

    sig_df = Signals.get_CAR(currency_pairs, '1m')

    CAR = Factor(start_date,
                 end_date,
                 frequency)

    CAR.rebalancing_frequency = Frequency.BUSINESS_MONTHLY
    CAR.set_signal(sig_df)
    CAR.price_quote_type = PriceQuote.MID
    CAR.number_of_portfolios = 5
    CAR.run_strategy()
    df = CAR.PnLCurve



























