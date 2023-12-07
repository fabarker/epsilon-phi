from enum import Enum

class FX(Enum):
    EUR_START_DATE = '1-Jan-1999'

class Datatype(Enum):

    RETURN_INDEX = 'RI'
    TOTAL_RETURN = 'TOTR'
    PRICE_INDEX = 'PI'


class Provider(Enum):
    WMR = 'WM/Refinitiv'
    REFINITIV = 'Refinitiv'
    BBI = 'Barclays Bank PLC'
    GTIS = 'GTIS - FTID/TR'
    GS = 'GS'
    MSCI = 'MSCI'

class PricingLocation(Enum):
    LONDON = 'LDN'
    NEW_YORK = 'NYC'
    HONG_KONG = 'HKD'

class Category(Enum):
    FX = 1
    INTEREST_RATE = 2
    EQUITY_INDEX = 3
    EQUITY = 4
    IMPLIED_VOLATILITY = 5
    YIELD = 6
    YIELD_CURVE = 7
    BOND_INDEX = 8
    FIXED_INCOME = 11
    COMMODITY = 9
    HEDGE_FUND = 10

class PriceQuote(Enum):
    BID = 'bid'
    MID = 'mid'
    ASK = 'ask'

    @staticmethod
    def get_inverse_quote(QUOTE):

        if QUOTE == PriceQuote.ASK:
           return PriceQuote.BID

        if QUOTE == PriceQuote.MID:
           return PriceQuote.MID

        if QUOTE == PriceQuote.BID:
            return PriceQuote.ASK

        if isinstance(QUOTE, str) and \
                QUOTE.lower() == PriceQuote.ASK.value:
           return PriceQuote.BID.value

        if isinstance(QUOTE, str) and \
                QUOTE.lower() == PriceQuote.BID.value:
           return PriceQuote.ASK.value

        if isinstance(QUOTE, str) and \
                QUOTE.lower() == PriceQuote.MID.value:
           return PriceQuote.MID.value

        raise ValueError('Error - {} not recognized'.format(QUOTE))