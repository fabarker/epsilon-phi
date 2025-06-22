from enum import Enum, auto

class CustomAssets(Enum):

    LHYIELD_GE20 = auto()
    CSFBMTT = auto()
    INFRA_EQUITY = auto()
    DISTRESSED_PME = auto()
    VENTURE_PME = auto()
    REAL_ESTATE_PME = auto()
    PE_BUYOUT = auto()
    PE_DISTRESSED = auto()
    PRIVATE_CREDIT = auto()
    PA_REAL_ESTATE = auto()
    PE_VENTURE = auto()
    PA_INFRA = auto()

    @classmethod
    def is_custom_asset(cls, asset_name):
        return asset_name in cls.__members__


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
    pass



if __name__ == "__main__":

    CustomAssets(CustomAssets.LHYIELD_GE20)

