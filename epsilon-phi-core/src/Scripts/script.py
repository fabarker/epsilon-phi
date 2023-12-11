from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.ep_strategies.fx.factor import Factor, Signals, PriceQuote
from epsilonPhi.core.reporting import quantstats as report
import datetime as dt
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency


# 1. EXAMPLE OF CONSTRUCTING FX STRATEGY AND STRATEGY PERFORMANCE METRICS

# Set the asset universe
G10 = Factor._G_10_CURRENCIES + ['BRL','KRW','ZAR','CZK','HUF','TWD','INR','IDR','COP','CLP','MXN','PHP','MYR','ILS','KWD','SGD']

# Start and end date for backtesting and the frequency that we want to price the strategies
start_date = dt.date(year=2013, month=10, day=31)
end_date = dt.date(year=2023, month=10, day=31)
frequency = Frequency.BUSINESS_DAILY

base_currency = 'USD'
currency_pairs = [x + '/' + base_currency for x in G10]

######### Construct some generic FX strategy - Here, FX Carry ########

# CONSTRUCT CARRY SIGNAL
sig_df = Signals.get_CAR(currency_pairs, '1m')

# Throw the signal panel into the strategy constructor
CAR = Factor(start_date,  end_date, frequency)
CAR.rebalancing_frequency = Frequency.BUSINESS_MONTHLY
CAR.set_signal(sig_df)
CAR.price_quote_type = PriceQuote.BID
CAR.number_of_portfolios = 5
CAR.run_strategy()
df = CAR.PnLCurve

# Construct a Spot Reversal Portfolio
sig_df_RVS = Signals.get_RVS(currency_pairs, '1m')

# Throw the signal panel into the strategy constructor
RVS = Factor(start_date,  end_date, frequency)
RVS.rebalancing_frequency = Frequency.BUSINESS_MONTHLY
RVS.set_signal(sig_df_RVS)
RVS.price_quote_type = PriceQuote.BID
RVS.number_of_portfolios = 5
RVS.run_strategy()
df_RVS = RVS.PnLCurve

# Examine the strategy PnL characteristics by creating a tearsheet
output = '/Users/francisbarker/Repositories/Python/epsilon-phi/epsilon-phi-core/src/Scripts/tearsheet.html'
report.reports.html(df.get('SIGNAL_WEIGHTED').pct_change(),
                    df.get('Benchmark').pct_change(),
                    output=output,
                    rf=0.013)


