from epsilonPhi.core.dataModel.dataSources.impliedVolatility.VolSurface import AbstractVolSurface
from epsilonPhi.core.dataModel.enums.ImpliedVolatility import *
from epsilonPhi.core.utils.FrameUtils import FrameUtils
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
from epsilonPhi.core.utils.DateUtils import DateUtils
from epsilonPhi.core.utils.OptionUtils import *
import numpy as np


class StrategyDriver(object):
    def __init__(self, underlier, start_date, end_date):

        # Set start and end date for strategy backtest
        self._start_date = pd.to_datetime(start_date)
        self._end_date = pd.to_datetime(end_date)

        # Set the underlier in the object
        self._underlier = underlier

        # Initialize some default values for the strategy pars
        self._strike_reference = StrikeReference.DELTA
        self._relative_strikes = 0.25
        self._maturity = '1m'
        self._open_frequency = Frequency.BUSINESS_DAILY
        self._close_frequency = '1m'

        self._instrument = Instruments.EUROPEAN_VANILLA_PUT

        self._position = -1
        self._notional = 100
        self._bid_ask_vol_spread = 0.3

        self._delta_hedge = True
        self._mkt_data = None
        self._weights = None
        self._surface_interpolator = Interpolator.CUBIC_SPLINE

    def set_weights(self, weights_df):
        self._weights = weights_df.copy()

    def get_contract_weights(self):
        if self._weights is None:
           contract_open = self.get_contract_price_pnl() / self.get_contract_price_pnl()
           return contract_open / contract_open.sum(axis=1, skipna=True).values.reshape(-1, 1)
        return self._weights

    def get_pricing_dates(self):
        if hasattr(self, '_pricing_dates'):
           return pd.to_datetime(self._pricing_dates)

    def set_open_frequency(self, frequency: Frequency.DAILY):
        assert isinstance(frequency, Frequency)
        self._open_frequency = frequency
        self._mkt_data = None

    def set_surface_interpolator(self, interpolator):
        self._surface_interpolator = interpolator
        self._mkt_data = None

    def set_close_frequency(self, frequency):
        self._close_frequency = frequency
        self._mkt_data = None

    def set_maturity(self, maturity):
        self._maturity = maturity
        self._mkt_data = None

    def get_open_dates(self):
        return self._open_frequency.get_period_ends(self.get_pricing_dates())

    def get_close_dates(self, open_dates):
        return DateUtils.expiry_from_settlement(open_dates,
                                                self._close_frequency,
                                                self.get_holidays())
    def get_expiry_dates(self):
        return DateUtils.expiry_from_settlement(self.get_open_dates(),
                                                self._maturity,
                                                self.get_holidays())

    def get_holidays(self):
        return pd.to_datetime(np.setdiff1d(pd.date_range(self._vs.common_dates[0], self._vs.common_dates[-1]),
                     self._vs.common_dates))

    def _load_market_data(self):

        # Load the volatility surface
        self._vs = AbstractVolSurface.get_volatility_surface(self._underlier,
                                                             interpolation_method=self._surface_interpolator)

        # Set pricing dates, which is intercept of spot and vol observations
        self._pricing_dates = self._vs.common_dates[((self._vs.common_dates >= self._start_date) &
                                                     (self._vs.common_dates <= self._end_date))]

        # Set the bid ask vol spreads in the vol surface
        self._vs.set_bid_ask_vol_spreads(self._bid_ask_vol_spread/100)

        # Get the market data -
        market_data = self._vs.get_option_prices(self.get_open_dates(),
                                                 self._strike_reference,
                                                 self._instrument.value * np.abs(self._relative_strike),
                                                 MaturityType.YEARFRAC,
                                                 DateUtils.Rdate_to_mat(self._maturity),
                                                 self._instrument.value,
                                                 self.get_close_dates(self.get_open_dates()))

        # Drop any prices beyond the target close date
        close_dates = self.get_close_dates(market_data.index.get_level_values('open'))
        self._mkt_data = market_data.set_index(pd.Series(close_dates, name='close'), append=True)
        _keep_locs = ((self._mkt_data.index.get_level_values('date') <= self._mkt_data.index.get_level_values('close')) &
                      (self._mkt_data.index.get_level_values('date') <= self._pricing_dates.max()))
        self._mkt_data = self._mkt_data[_keep_locs].sort_index('date')

    def set_market_data(self, market_data):
        self._mkt_data = market_data.reset_index().set_index(['date','k','open','expiry'])

    def get_market_data(self):
        if self._mkt_data is None:
           self._load_market_data()
        return self._mkt_data.copy()

    @property
    def open_bool(self):
        if self._mkt_data is not None:
            return (self.get_market_data().index.get_level_values('open') ==
                    self.get_market_data().index.get_level_values('date'))
    @property
    def close_bool(self):
        if self._mkt_data is not None:
            close_bool = (self.get_market_data().index.get_level_values('close') ==
                    self.get_market_data().index.get_level_values('date'))
            return close_bool != self.expiry_bool

    @property
    def expiry_bool(self):
        if self._mkt_data is not None:
            return (self.get_market_data().index.get_level_values('expiry') ==
                    self.get_market_data().index.get_level_values('date'))

    @property
    def delta_hedge(self):
        return self._delta_hedge

    @delta_hedge.setter
    def delta_hedge(self, true_false):
        if not isinstance(true_false, bool):
            raise ValueError("Value must be bool")
        self._delta_hedge = true_false

    @property
    def reporting_currency(self):
        return self._vs.premium_currency

    def set_bid_ask_vol_spread(self, spread):
        self._bid_ask_vol_spread = spread
        self._mkt_data = None

    def set_strike_reference(self, strike_reference, relative_strike):
        self._strike_reference = strike_reference
        self._relative_strike = relative_strike
        self._mkt_data = None

    def set_position(self, position):
        assert position in [1, -1], 'Error - position must be (1, -1) for (long, short)'
        self._position = position

    def set_instrument(self, instrument):
        self._instrument = instrument
        self._mkt_data = None

    def get_deposit_rates(self):
        curve = self._vs.get_risk_free_rate_curve(self.reporting_currency)
        ttm = DateUtils.get_date_delta(self._mkt_data.index.get_level_values('open'),
                                       self._mkt_data.index.get_level_values('close'),
                                       True)
        return curve.get_stacked_curve(self._mkt_data.index.get_level_values('open'), ttm)

    def get_accrued_interest(self):

        # Get Option Premiums..
        premiums =  -1 * self._position * self.get_market_data().get('p')
        p = premiums[self.open_bool].droplevel(level=0)

        r = self.get_deposit_rates()[self.open_bool]
        t = np.array((p.index.get_level_values('close') - p.index.get_level_values('open'))
                     / np.timedelta64(365, 'D'), np.float64)

        periods = np.maximum(0, premiums.groupby('open').count() - 1)
        ai_pnl = (p * (-1 + np.exp(r.values * t))) / periods
        resampled_pnl = ai_pnl.loc[self.get_market_data().index.droplevel(0)]
        resampled_pnl.index = premiums.index
        resampled_pnl.loc[self.open_bool] = np.nan
        return resampled_pnl.unstack(level=[1, 2, 3, 4]).sort_index(level='open', axis=1)

    def get_open_vols(self):
        return self.get_market_data().iloc[self.open_bool, :][['i']]

    def get_contract_ttm(self):
        return self.get_market_data().get('t').unstack(level=[1, 2, 3, 4]).sort_index(level='open', axis=1)

    def get_contract_strikes(self):
        return self.get_market_data().get('k').unstack(level=[1, 2, 3,4]).sort_index(level='open', axis=1)

    def get_contract_prices(self):
        return self.get_market_data().get('p').unstack(level=[1, 2, 3,4]).sort_index(level='open', axis=1)

    def get_get_ask_prices(self):
        return self.get_market_data().get('a').unstack(level=[1, 2, 3,4]).sort_index(level='open', axis=1)

    def get_bid_prices(self):
        return self.get_market_data().get('b').unstack(level=[1, 2, 3,4]).sort_index(level='open', axis=1)

    def get_contract_premiums(self):
        return self.get_market_data().get('p').unstack(level=[1, 2, 3,4]).sort_index(level='open', axis=1)

    def get_contract_price_pnl(self):
        return self.get_contract_prices().diff()

    def get_contract_premium_pnl(self):
        return self.get_contract_premiums().diff()

    def get_contract_delta(self):
        return self.get_market_data().get('d').unstack(level=[1,2,3,4]).sort_index(level='open', axis=1)

    def get_delta_pnl(self):
        return (self.get_hedging_instrument_pnl() *
                self.get_contract_delta().shift(1).mul(self._position))

    def get_contract_gamma(self):
        return self.get_market_data().get('g').unstack(level=[1, 2, 3, 4]).sort_index(level='open', axis=1)

    def get_gamma_pnl(self):
        return (0.5 * self.get_contract_gamma().shift(1).mul(self._position) *
                (self.get_spot_prices().diff() ** 2))

    def get_contract_vol(self):
        return self.get_market_data().get('i').unstack(level=[1, 2, 3,4]).sort_index(level='open', axis=1)

    def get_contract_vega(self):
        return self.get_market_data().get('v').unstack(level=[1, 2, 3,4]).sort_index(level='open', axis=1)

    def get_vega_pnl(self):
        return (self.get_contract_vega().shift(1).mul(self._position)  *
                self.get_contract_vol().diff())

    def get_contract_volga(self):
        return self.get_market_data().get('vo').unstack(level=[1, 2, 3,4]).sort_index(level='open', axis=1)

    def get_volga_pnl(self):
        return (0.5 * self.get_contract_volga().shift(1).mul(self._position) *
                (self.get_contract_vol().diff() ** 2))

    def get_contract_vanna(self):
        return self.get_market_data().get('va').unstack(level=[1, 2, 3, 4]).sort_index(level='open', axis=1)

    def get_vanna_pnl(self):
        return (self.get_contract_vanna().shift(1).mul(self._position) *
                self.get_hedging_instrument_pnl() * self.get_contract_vol().diff())

    def get_contract_theta(self):
        return self.get_market_data().get('theta').unstack(level=[1, 2, 3, 4]).sort_index(level='open', axis=1)

    def get_theta_pnl(self):
        return (self.get_contract_theta().shift(1).mul(self._position) *
                self.get_contract_ttm().diff().abs())


    def get_spot_prices(self):
        return self.get_market_data().get('s').unstack(level=[1, 2, 3,4]).sort_index(level='open', axis=1)

    def get_spot_price_pnl(self):
        return self.get_spot_prices().diff()

    def get_spot_return(self):
        return self.get_spot_price_pnl() / self.get_spot_prices().shift(1)

    def get_forward_prices(self):
        return self.get_market_data().get('f').unstack(level=[1, 2, 3, 4]).sort_index(level='open', axis=1)

    def get_forward_pnl(self):
        return self.get_forward_prices().diff()

    def get_hedging_instrument_prices(self):
        return self.get_market_data().get('h').unstack(level=[1, 2, 3, 4]).sort_index(level='open', axis=1)

    def get_hedging_instrument_pnl(self):
        return self.get_hedging_instrument_prices().diff()

    def get_hedging_instrument_return(self):

        hdg_prices = self.get_hedging_instrument_prices().diff()
        if self.reporting_currency == self._underlier[0:3]:
            return hdg_prices / self.get_spot_prices().shift(1)
        else:
            return hdg_prices / self.get_contract_strikes().shift(1)

    def get_forward_return(self):
        return self.get_forward_pnl() / self.get_forward_prices().shift(1)

    def get_option_position_return(self):
        return self.get_contract_premium_pnl().mul(self._position)

    def get_hedge_position_return(self):
        return (self.get_hedging_instrument_return() *
                self.get_contract_delta().shift(1).mul(-1).mul(self._position))

    def get_strategy_returns(self):
        rtns = self.get_hedged_contract_returns()
        wts = self.get_contract_weights()
        return (rtns * wts).sum(axis=1)

    def get_hedged_contract_returns(self):
        opts = self.get_option_position_return()
        hdg = self.get_hedge_position_return()
        aci = self.get_accrued_interest()
        return opts + hdg + aci

    def get_strategy_cumulative_return(self):
        return self.get_strategy_returns().add(1).cumprod() - 1

    def get_option_transaction_costs(self):

        # Pay the spread to open
        open_pct = (self.get_market_data().get('a') - self.get_market_data().get('b')) * 0.5
        open_pct[~self.open_bool] = np.nan
        open_tcs = open_pct.unstack(level=[1, 2, 3, 4]).sort_index(level='open', axis=1)

        # Pay the spread to close
        close_pct = (self.get_market_data().get('a') - self.get_market_data().get('b')) * 0.5
        close_pct[~self.close_bool] = np.nan
        close_tcs = close_pct.unstack(level=[1, 2, 3, 4]).sort_index(level='open', axis=1)
        return (open_tcs.fillna(0).shift(1).values + close_tcs.fillna(0)).replace(0, np.nan)

    def get_delta_hedge_transaction_costs(self):

        # Assume 2 bips transaction cost on the underlying, bips is % of notional
        spot_tc_bp = 2 / (2 * 100 * 100)
        agg_delta = self.get_contract_delta().mean(axis=1)
        rebal = agg_delta.diff()
        rebal.iloc[0] = agg_delta.values[0]
        return (rebal.abs() * spot_tc_bp).shift(1)

    def get_option_position_return_net_transaction_costs(self):
        return self.get_option_position_return() - self.get_option_transaction_costs().fillna(0)

    def get_strategy_return_net_transaction_costs(self):

        option_pnl_net = self.get_option_position_return_net_transaction_costs()

        hdg = self.get_hedge_position_return()
        aci = self.get_accrued_interest()
        pnl = (option_pnl_net + hdg + aci)

        # We hedge the aggregate delta instead of contract by contract
        delta_hedge_pnl = self.get_delta_hedge_transaction_costs()
        return pnl.mean(axis=1) - delta_hedge_pnl

    def get_strategy_cumulative_return_net_transaction_costs(self):
        return self.get_strategy_return_net_transaction_costs().add(1).cumprod() - 1

    def get_single_option_trade_data(self, trade_date):
        market_data = self.get_market_data()
        idx = market_data.index.get_level_values('open') == trade_date
        return market_data[idx].unstack(level=[1,2,3])

    def get_strategy_return_decomposition(self):

        d = self.get_delta_pnl().mean(axis=1)
        g = self.get_gamma_pnl().mean(axis=1)
        v = self.get_vega_pnl().mean(axis=1)
        vo = self.get_volga_pnl().mean(axis=1)
        va = self.get_vanna_pnl().mean(axis=1)
        t = self.get_theta_pnl().mean(axis=1)

        df = pd.concat((t, d, g, v, va, vo), axis=1)
        df.columns =['theta','delta','gamma','vega','vanna','volga']
        return df




if __name__ == "__main__":

    import pandas as pd
    import numpy as np
    from epsilonPhi.core.utils.DateUtils import DateUtils

    SD = '24-Jan-1996'
    ED = '13-Oct-2022'
    self = StrategyDriver('GBPUSD', SD, ED)

    self.set_instrument(Instruments.EUROPEAN_VANILLA_PUT)
    self.set_strike_reference(StrikeReference.DELTA, 0.10)
    self.set_surface_interpolator(Interpolator.CUBIC_SPLINE)

    self.set_open_frequency(Frequency.BUSINESS_DAILY)
    self.set_close_frequency('1m')
    self.set_maturity('1m')

    self.set_position(-1)
    self.set_bid_ask_vol_spread(0.3) # Typically in the range of 0.2-0.7 vol points depending on delta and maturity (0.3 for 1M 25 Delta)
    straddle = self.get_strategy_cumulative_return()

















