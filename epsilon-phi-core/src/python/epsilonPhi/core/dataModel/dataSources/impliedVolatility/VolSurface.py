from epsilonPhi.core.dataModel.dataSources.impliedVolatility.VolSurfaceMgr import VolSurfaceMgr

class AbstractVolSurface(object):
    _cache = {}

    def __init__(self,
                 underlier,
                 pricing_location=None,
                 cross_section=None):

        self._underlier = underlier
        self._mgr = VolSurfaceMgr(underlier,
                                  pricing_location,
                                  cross_section)

    def get_vol_surface_mgr(self):
        return self._mgr

    def get_spot_prices(self):
        return self._mgr.get_spot_prices()

    def get_interest_rate_curve(self, maturities=None):
        return self._mgr.get_interest_rate_curve(maturities=maturities)

    def get_funding_rate_curve(self, maturities=None):
        return self._mgr.get_funding_rate_curve(maturities=maturities)

    def get_ivol_panel(self):
        return self._mgr.get_ivols()

    def get_ivols(self):
        return self.get_ivol_panel().get('mid')

    def get_ivol_tseries(self):
        ivols = self.get_ivols()
        return ivols[~ivols.index.duplicated(keep='first')].unstack(level=[1,2])

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

    self = AbstractVolSurface('SPX', )
    spt = self.get_ivol_panel()



