import datetime as dt

import pandas as pd
from matplotlib import pyplot as plt
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.dataModel.enums.Database import PriceQuote
from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries
from epsilonPhi.core.utils.DateUtils import DateUtils
import numpy as np



class Factor(object):

    _EURO_LEGACY = ['FRF', 'DEM', 'NLG', 'BEF', 'PTE', 'ESP', 'FIM', 'IEP', 'GRD', 'ATS', 'CYP', 'EEK', 'LUF', 'MCF', 'MTL', 'SIT', 'SKK', 'SNL', 'VAL']
    _EURO_CUTOFF_DATE = dt.date(year=1998, month=12, day=31)
    _G_10_CURRENCIES = ['AUD', 'CAD', 'DKK', 'JPY', 'NZD', 'NOK', 'SEK', 'CHF', 'GBP', 'DEM', 'FRF', 'ITL', 'NLG', 'BEF']
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
            return np.array([X + '/' + self.domestic_currency for X in self.foreign_currencies])

    @property
    def signal(self):
        return self._signal

    @signal.setter
    def signal(self, value: CTimeSeries):
        self._signal = value

    def set_signal(self, value: CTimeSeries):

        self._foreign_currencies = [x[0:3] for x in value.columns]
        dom_ccy = np.unique([x[4:] for x in value.columns])
        assert len(dom_ccy) == 1, 'Error - can only have 1 base/domestic currency'

        if isinstance(dom_ccy, str):
            self._domestic_currency = dom_ccy
        else:
            self._domestic_currency = dom_ccy[0]

        self.signal = value.reindex(self.pricing_dates)

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
        self._rebalaning_frequency

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

        prices =CTimeSeries()
        for pqt in self._PRICE_QUOTE_TYPES:
            fwds = pd.concat([self._datasource.get_forward_rates(x, self.pricing_dates, maturity_dates, pqt).T for x in self.currency_pairs]).T
            spts = pd.concat([self._datasource.get_forward_rates(x, self.pricing_dates, self.pricing_dates, pqt).T for x in self.currency_pairs]).T
            prices = pd.concat((prices, pd.concat((fwds, spts), axis=0)), axis=0)
        self._forward_prices = prices.copy()

    def get_returns_panel(self, pricing_date, maturity_date):

        if pricing_date > self._EURO_CUTOFF_DATE:
           EUR_LEGACIES = [self.domestic_currency + '/' + X for X in self._EURO_LEGACY]
           EUR_LEGACIES = EUR_LEGACIES + [self._datasource.get_reverse_currency_pair(x) for x in EUR_LEGACIES]
           tradable_currencies = np.setdiff1d(self.currency_pairs, EUR_LEGACIES)
        else:
            tradable_currencies = np.setdiff1d(self.currency_pairs,
                                               ['EUR/' + self.domestic_currency, self.domestic_currency + '/EUR'])

        if np.logical_and(pricing_date < self._BEF_EXCLUSION[1], pricing_date > self._BEF_EXCLUSION[1]):
            tradable_currencies = np.setdiff1d(tradable_currencies,
                                               [self.domestic_currency + '/BEF',
                                                    'BEF/' + self.domestic_currency])

        if pricing_date < self._TRY_START_DATE:
            tradable_currencies = np.setdiff1d(tradable_currencies,
                                               [self.domestic_currency + '/TRY',
                                                    'TRY/' + self.domestic_currency])

        dates = self.pricing_dates[np.logical_and(self.pricing_dates <= maturity_date,
                                                  self.pricing_dates >= pricing_date)]

        signal = self.signal[tradable_currencies].loc[pricing_date].dropna().to_frame().T

        if 'm' in self.price_quote_type.lower():
            pqt = ('mid','mid')
        else:
            pqt = ('bid','ask')

        long_rx = self._datasource.get_fwd_price_keys(dates, maturity_date, 'mid')
        long_rx[0] = self._datasource.get_fwd_price_keys(dates[0], maturity_date, pqt[1])[0]
        long_rx[-1] = self._datasource.get_fwd_price_keys(dates[-1], maturity_date, pqt[0])[0]

        long_spot = self._datasource.get_fwd_price_keys(dates, dates, 'mid')
        long_spot[0] = self._datasource.get_fwd_price_keys(dates[0], dates[0], pqt[1])[0]
        long_spot[-1] = self._datasource.get_fwd_price_keys(dates[-1], dates[-1], pqt[0])[0]

        long_rx_panel = self._forward_prices.loc[long_rx].get(signal.columns).dropna(axis=1)
        long_rx_returns = (np.log(long_rx_panel) - np.log(long_rx_panel.shift(1)))
        long_rx_returns.index = dates

        long_spt_panel = self._forward_prices.loc[long_spot].get(signal.columns).dropna(axis=1)
        long_spt_returns = (np.log(long_spt_panel) - np.log(long_spt_panel.shift(1)))
        long_spt_returns.index = dates



        short_rx = self._datasource.get_fwd_price_keys(dates, maturity_date, 'mid')
        short_rx[0] = self._datasource.get_fwd_price_keys(dates[0], maturity_date, pqt[0])[0]
        short_rx[-1] = self._datasource.get_fwd_price_keys(dates[-1], maturity_date, pqt[1])[0]

        short_spot = self._datasource.get_fwd_price_keys(dates, dates, 'mid')
        short_spot[0] = self._datasource.get_fwd_price_keys(dates[0], dates[0], pqt[0])[0]
        short_spot[-1] = self._datasource.get_fwd_price_keys(dates[-1], dates[-1], pqt[1])[0]

        short_rx_panel = self._forward_prices.loc[short_rx].get(signal.columns).dropna(axis=1)
        short_rx_returns = (np.log(short_rx_panel) - np.log(short_rx_panel.shift(1)))
        short_rx_returns.index = dates

        short_spt_panel = self._forward_prices.loc[short_spot].get(signal.columns).dropna(axis=1)
        short_spt_returns = (np.log(short_spt_panel) - np.log(short_spt_panel.shift(1)))
        short_spt_returns.index = dates

        return long_rx_returns, long_spt_returns, short_rx_returns, short_spt_returns

    def get_currency_portfolio_locs(self, N):

        ccys_per_portfolio = np.tile(int(np.floor(N / self.number_of_portfolios)), self.number_of_portfolios)
        remaining_currencies = np.mod(N, self.number_of_portfolios)
        ccys_per_portfolio[0:remaining_currencies] = ccys_per_portfolio[0:remaining_currencies] + 1

        locs = np.zeros((1, N), dtype=int)
        locs[0, np.hstack((0, ccys_per_portfolio[:,-1].cumsum()))] = 1
        return np.cumsum(locs)

    def run_strategy(self):

        rebal_dates = self.rebalancing_dates
        fwd_mats = self.fwd_maturities.unique()
        self.__load_data()

        str_df = pd.DataFrame()
        T = len(rebal_dates)

        for t in range(1, T):

            long_excess, long_spt, short_excess, short_spt = self.get_returns_panel(rebal_dates[t-1], fwd_mats[t-1])
            ranked_signal = self.signal[long_excess.columns].loc[rebal_dates[t-1]].sort_values()

            if ranked_signal.size > self.number_of_portfolios:

                N = int(np.ceil(ranked_signal.size / self.number_of_portfolios))

                # High Minus Low Portfolio
                wt = np.repeat(1/N, N)
                H = np.exp(long_excess.get(ranked_signal.index[-N:]) @ wt) - 1
                L = np.exp(short_excess.get(ranked_signal.index[0:N]) @ wt) - 1
                HML = H-L

                HML_spt = np.exp(long_spt.get(ranked_signal.index[-N:]) @ wt) - \
                          np.exp(short_spt.get(ranked_signal.index[0:N]) @ wt)

                # Linear in signal size
                W_sig = (2/np.sum(np.abs(ranked_signal - np.mean(ranked_signal)))) + (ranked_signal - np.mean(ranked_signal))
                lin_sig = np.exp(short_excess.get(W_sig[W_sig < 0].index)) @ W_sig[W_sig < 0] +\
                          np.exp(long_excess.get(W_sig[W_sig > 0].index)) @ W_sig[W_sig > 0]

                lin_sig_spt = np.exp(short_spt.get(W_sig[W_sig < 0].index)) @ W_sig[W_sig < 0] + \
                            np.exp(long_spt.get(W_sig[W_sig > 0].index)) @ W_sig[W_sig > 0]

                # Build Portfolio linear in rank
                W_rank = 2 * ((ranked_signal.rank() - ranked_signal.rank().mean()) /\
                         ((ranked_signal.rank() - ranked_signal.rank().mean()).abs().sum()))

                lin_rank = np.exp(short_excess.get(W_rank[W_rank < 0].index)) @ W_rank[W_rank < 0] +\
                          np.exp(long_excess.get(W_rank[W_rank > 0].index)) @ W_rank[W_rank > 0]

                lin_rank_spt = np.exp(short_spt.get(W_rank[W_rank < 0].index)) @ W_rank[W_rank < 0] + \
                               np.exp(long_spt.get(W_rank[W_rank > 0].index)) @ W_rank[W_rank > 0]

                df = pd.concat([HML, L, H, HML_spt, lin_sig, lin_sig_spt, lin_rank, lin_rank_spt], axis=1).dropna()
                str_df = pd.concat((str_df, df), axis=0)

            str_df.columns = ['HML', 'L', 'H', 'HML_spt', 'SIGNAL_WEIGHTED', 'SIGNAL_WEIGHTED_SPT', 'RANK_WEIGHTED', 'RANK_WEIGHTS_SPT']
            self.strategies = str_df.reindex(self.pricing_dates).copy()
            self.PnLs = self.strategies.get(['HML', 'SIGNAL_WEIGHTED', 'RANK_WEIGHTED'])
            self.PnLCurve = (1+self.PnLs).cumprod()


if __name__ == '__main__':

    G10 = Factor._G_10_CURRENCIES

    start_date = dt.date(year=1982, month=12 , day=31)
    end_date = dt.date(year=2022, month=12, day=31)
    frequency = Frequency.BUSINESS_MONTHLY

    base_currency = 'USD'
    currency_pairs = [X + '/' + base_currency for x in G10]

    sig_df = Signal.get_CAR(start_date,
                            end_date,
                            Frequency)

    CAR = Factor(start_date,
                 end_date,
                 frequency)

    CAR.rebalancing_frequency = Frequency.BUSINESS_MONTHLY
    CAR.set_signal(sig_df)
    CAR.price_quote_type = PriceQuote.BID
    CAR.number_of_portfolios = 5
    CAR.run_strategy()



























