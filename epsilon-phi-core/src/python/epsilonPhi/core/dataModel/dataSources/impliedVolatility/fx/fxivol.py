from epsilonPhi.core.dataModel.dataSources.fxCurve.FXCurve import FXCurve
class FXVols(object):

    def __init__(self, currency_pair):

        self._bbid = currency_pair.replace('/', '')
        self._currency_pair = self._bbid[0:2] + '/' + self._bbid[3:]
        self._foreign_currency = self._bbid[0:2]
        self._domestic_currency = self._bbid[3:]
        self._fx_curve = FXCurve()

    def load_data(self):
        pass

    def get_option_contract_prices(self, from_date, to_date, strike_reference, tenor):
        pass

    def get_option_contract_price(self, date, strike_reference, tenor):
        pass

    def get_vol_surface(self, date):
        pass

