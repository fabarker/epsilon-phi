from epsilonPhi.core.dataModel.dataSources.impliedVolatility.VolSurfaceMgr import VolSurfaceMgr

class AbstractVolSurface(object):
    _cache = {}

    def __init__(self, underlier, pricing_location=None):
        self._underlier = underlier
        self._mgr = VolSurfaceMgr(underlier, pricing_location)

    def get_vol_surface_mgr(self):
        return self._mgr

    def get_spot_prices(self):
        return self._mgr.get_spot_prices()

    def get_interest_rate_curve(self):
        return self._mgr.get_interest_rate_curve()

    def get_ivol_data(self):
        pass

    def get_ivol_surface(self):
        pass

    def get_option_prices(self):
        pass

    def get_option_price(self):
        pass

    def get_bsdelta(self):
        pass

    def get_bsgamma(self):
        pass

    def get_bsvega(self):
        pass

    def get_bsvolga(self):
        pass

    def get_bsvanna(self):
        pass

    def get_bstheta(self):
        pass

    def get_omega(self):
        pass

    def get_lambda(self):
        pass

    def simulate_vol_surface(self):
        pass

    def simulate_option_prices(self):
        pass

    def get_straddle_prices(self):
        pass

    def get_strangle_prices(self):
        pass

    def get_risk_reversal_prices(self):
        pass

    def get_butterfly_prices(self):
        pass





if __name__ == "__main__":

    self = AbstractVolSurface('SPX')



