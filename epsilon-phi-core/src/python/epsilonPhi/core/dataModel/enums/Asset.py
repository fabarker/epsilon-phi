from enum import Enum, auto


class CustomAssets(Enum):
    LHYIELD_GE20 = auto()
    CSFBMTT = auto()
    INFRA_EQUITY = auto()
    DISTRESSED_PME = auto()
    VENTURE_PME = auto()
    REAL_ESTATE_PME = auto()
    PRIVATE_CREDIT_PME = auto()
    GLOBAL_REITS = auto()
    GROWTH_PME = auto()
    PE_BUYOUT = auto()
    PE_DISTRESSED = auto()
    PRIVATE_CREDIT = auto()
    PA_REAL_ESTATE = auto()
    PE_VENTURE = auto()
    PA_INFRA = auto()
    PE_GROWTH = auto()

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

    @property
    def asset_name(self):
        if self == PrivateAsset.BUYOUT:
            return "PE_BUYOUT"
        elif self == PrivateAsset.GROWTH:
            return "PE_GROWTH"
        elif self == PrivateAsset.VENTURE:
            return "PE_VENTURE"
        elif self == PrivateAsset.SECONDARIES:
            return "PE_SECONDARIES"
        elif self == PrivateAsset.MULTI_STRATEGY:
            return "PE_MULTI_STRATEGY"
        elif self == PrivateAsset.DIVERSIFIED:
            return "PE_DIVERSIFIED"
        elif self == PrivateAsset.EMERGING:
            return "PE_EMERGING"
        elif self == PrivateAsset.PRIVATE_CREDIT:
            return "PRIVATE_CREDIT"
        elif self == PrivateAsset.DISTRESSED:
            return "PE_DISTRESSED"
        elif self == PrivateAsset.REAL_ESTATE:
            return "REAL_ESTATE"
        elif self == PrivateAsset.OPPORTUNISTIC_REAL_ESTATE:
            return "OPPORTUNISTIC_REAL_ESTATE"
        elif self == PrivateAsset.INFRASTRUCTURE:
            return "INFRASTRUCTURE"
        elif self == PrivateAsset.ENERGY:
            return "ENERGY"
        else:
            raise ValueError("Error - stype not supported")




class RiskFreeMap(Enum):
    pass


if __name__ == "__main__":
    CustomAssets(CustomAssets.LHYIELD_GE20)
