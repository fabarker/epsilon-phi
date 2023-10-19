from enum import Enum

class PrivateAsset(Enum):

    BUYOUT = 'Buyout'
    GROWTH = 'Growth'
    VENTURE = 'Venture'
    SECONDARIES = 'Secondaries'
    MULTI_STRATEGY = 'Multi-Strategy'
    DIVERSIFIED = 'Diversified'
    EMERGING = 'Emerging'
    PRIVATE_CREDIT = 'Private Credit'
    DISTRESSED = 'Distressed'
    REAL_ESTATE = 'Real Estate'
    OPPORTUNISTIC_REAL_ESTATE = 'Opportunistic Real Estate'
    INFRASTRUCTURE = 'Infrastructure'
    ENERGY = 'Energy'

class RiskFreeMap(Enum):

    MSCI_WORLD = 'MSWRLDL'

    AUD = 'ADBR090,AUPRIMF'


