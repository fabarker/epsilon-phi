from enum import Enum

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
