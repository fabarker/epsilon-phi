from enum import Enum

class Interpolator(Enum):

    VANNA_VOLGA = 1
    KERNEL_SMOOTHING_1D = 2
    KERNEL_SMOOTHING_2D = 3
    POLYNOMIAL_REGRESSION_1D = 4
    POLYNOMIAL_REGRESSION_2D = 5
    CLAMPED_CUBIC_SPLINE = 6


class StrikeReference(Enum):

    SPOT_DELTA = 1
    FORWARD_DELTA = 2
    MONEYNESS = 5
    Z_SCORE = 6
    CONVEXITY_MN = 7
    SPOT_DELTA_PREM_ADJ = 3
    FORWARD_DELTA_PREM_ADJ = 4

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
                 'USDMXN': 'USD'}

deltaConvention = {'EURUSD': DeltaType.SPOT_DELTA,
                   'USDJPY': DeltaType.SPOT_DELTA_PREM_ADJ,
                   'EURJPY': DeltaType.SPOT_DELTA_PREM_ADJ,
                   'USDCHF': DeltaType.SPOT_DELTA_PREM_ADJ,
                   'EURCHF': DeltaType.SPOT_DELTA_PREM_ADJ,
                   'GBPUSD': DeltaType.SPOT_DELTA,
                   'EURGBP': DeltaType.SPOT_DELTA_PREM_ADJ,
                   'AUDUSD': DeltaType.SPOT_DELTA,
                   'AUDJPY': DeltaType.SPOT_DELTA_PREM_ADJ,
                   'USDCAD': DeltaType.SPOT_DELTA_PREM_ADJ,
                   'USDBRL': DeltaType.SPOT_DELTA_PREM_ADJ,
                   'USDMXN': DeltaType.SPOT_DELTA_PREM_ADJ}
