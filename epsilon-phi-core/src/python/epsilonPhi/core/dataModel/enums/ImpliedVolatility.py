from enum import Enum

class Interpolator(Enum):

    VANNA_VOLGA = 1
    KERNEL_SMOOTHING_1D = 2
    GAUSSIAN_KERNEL_SMOOTHING = 3
    POLYNOMIAL_REGRESSION_1D = 4
    POLYNOMIAL_REGRESSION_2D = 5
    CLAMPED_CUBIC_SPLINE = 6
    CUBIC_SPLINE = 7
    ROLLOOS = 8

class Instruments(Enum):

    EUROPEAN_VANILLA_PUT = -1
    EUROPEAN_VANILLA_CALL = 1


class StrikeReference(Enum):

    DELTA = 'delta'
    MONEYNESS = 'moneyness'
    Z_SCORE = 'zscore'
    CONVEXITY_MN = 'convextiy_moneyness'
    LOG_MONEYNESS = 'log_moneyness'
    STRIKE_PRICE = 'strike_price'

class MaturityType(Enum):

    EXPIRY_DATE = 0
    YEARFRAC = 1
    MATURITY_STRING = 2


class AtTheMoneyType(Enum):
    SPOT = 1  # Spot
    FWD = 2  # Forward
    FWD_DELTA_NEUTRAL = 3  # K = F*exp(0.5*sigma*sigma*T)
    FWD_DELTA_NEUTRAL_PREM_ADJ = 4  # K = F*exp(-0.5*sigma*sigma*T)

class DeltaType(Enum):
    SPOT_DELTA = 1
    FORWARD_DELTA = 2
    SPOT_DELTA_PREM_ADJ = 3
    FORWARD_DELTA_PREM_ADJ = 4

###############################################################################

prem_currency = {'EURUSD': 'USD',
                 'USDJPY': 'USD',
                 'EURJPY': 'EUR',
                 'USDCHF': 'USD',
                 'EURCHF': 'EUR',
                 'GBPUSD': 'USD',
                 'EURGBP': 'EUR',
                 'AUDUSD': 'USD',
                 'AUDJPY': 'AUD',
                 'USDCAD': 'USD',
                 'USDBRL': 'USD',
                 'USDMXN': 'USD',
                 'NZDUSD': 'USD'}

deltaConvention = {'EURUSD': DeltaType.FORWARD_DELTA,
                   'USDJPY': DeltaType.SPOT_DELTA_PREM_ADJ,
                   'EURJPY': DeltaType.SPOT_DELTA_PREM_ADJ,
                   'USDCHF': DeltaType.SPOT_DELTA_PREM_ADJ,
                   'EURCHF': DeltaType.SPOT_DELTA_PREM_ADJ,
                   'GBPUSD': DeltaType.FORWARD_DELTA,
                   'EURGBP': DeltaType.SPOT_DELTA_PREM_ADJ,
                   'AUDUSD': DeltaType.FORWARD_DELTA,
                   'AUDJPY': DeltaType.SPOT_DELTA_PREM_ADJ,
                   'USDCAD': DeltaType.SPOT_DELTA_PREM_ADJ,
                   'USDBRL': DeltaType.SPOT_DELTA_PREM_ADJ,
                   'USDMXN': DeltaType.SPOT_DELTA_PREM_ADJ,
                   'NZDUSD': DeltaType.FORWARD_DELTA}
