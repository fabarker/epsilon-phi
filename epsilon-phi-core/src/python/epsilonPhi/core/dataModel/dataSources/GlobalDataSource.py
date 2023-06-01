from epsilonPhi.core.lib.Decorators import SingletonDecorator
from epsilonPhi.core.dataModel.dataSources.FXCurve import FXCurve


@SingletonDecorator
class GlobalDataSource(object):
    _cache = dict()

    def __init__(self):
        self.initalize()

    def initialize(self):
        self._fx_curve = FXCurve()

    # Methods associated with currencies / FX

    def get_fx_forward_prices(self, currency_pairs, pricing_dates, maturity_dates, price_quotes):
        return self._fx_curve.get_forward_prices(currency_pairs, pricing_dates, maturity_dates, price_quotes)

    def get_fx_forward_rates(self, currency_pairs, maturities, price_quotes):
        return self._fx_curve.get_forward_rates(currency_pairs, maturities, price_quotes)

    def get_fx_spot_rates(self, currency_pairs, price_quotes):
        return self._fx_curve.get_spot_rates(currency_pairs, price_quotes)

    def get_fx_carry(self, currency_pairs, maturities, price_quotes):
        return self._fx_curve.get_forward_prices(currency_pairs, maturities, price_quotes)


    # Methods associated with interest rates




    # Methods associateg with implied volatilties




    # Methods associated with querying datastream




    # Methods associated with querying GSQuant







