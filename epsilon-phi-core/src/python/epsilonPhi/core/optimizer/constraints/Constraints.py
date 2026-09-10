from scipy.interpolate import interp1d
import numpy as np

class Constraints:

    map = {
        3.1: 75.4,
        4.9: 58.8,
        6.2: 47.4,
        7.4: 36.0,
        8.9: 25.5,
        10.3: 15.0,
    }

    def __init__(self, currency, target_vol):
        self._currency = currency
        self._target_vol = target_vol

    @staticmethod
    def get(currency, target_vol):

        combined = {
            'assets': [],
            'proportional': ''
        }

        for name, method in Constraints.__dict__.items():
            # Check if it's a static method and callable
            if isinstance(method, staticmethod) and "category" in name:
                if name == 'get_fixed_income_category':
                    result = method.__func__(currency, target_vol)
                else:
                    result = method.__func__()  # Call the static method
                combined['assets'].extend(result.get('assets', []))

                prop = result.get('proportional', '')
                if prop:
                    if combined['proportional']:
                        combined['proportional'] += ';' + prop
                    else:
                        combined['proportional'] = prop
        return combined


    @staticmethod
    def get_sleep_well_from_vol(vol):

        # Sort the keys and values
        x = np.array(sorted(Constraints.map.keys()))
        y = np.array([Constraints.map[k] for k in x])

        # Create linear interpolation function
        f = interp1d(x, y, kind='linear', fill_value='extrapolate')  # can extrapolate outside range

        sw = f(vol * 100).item() / 100
        return np.clip(round(sw / 0.005) * 0.005, 0.1, 1) * 100

    @staticmethod
    def get_fixed_income_category(currency, target_vol):

        ret = {}
        ret['assets'] = [Constraints.get_fixed_income_ticker(currency)]
        ret['proportional'] = Constraints.get_fixed_income_ticker(currency) + ">" + str(Constraints.get_sleep_well_from_vol(target_vol))
        return ret

    @staticmethod
    def get_fixed_income_ticker(currency):

        if currency == "USD":
            return "LHTRYIN"
        elif currency == "GBP":
            return "MLUK9YL"
        else:
            raise ValueError("Error - currency {} not supported".format(currency))


    @staticmethod
    def get_equity_category():

        US_CONSTRAINTS = "(-45)*FRUS1GR+40*FRUS1VA=0;(-45)*FRUSS2L+15*FRUS1VA=0;"
        EAFE_CONSTRAINTS = "MSEXUKL+(-3.42)*MSUTDKL=0;MSEXUKL+(-2.17)*MSJPANL=0;MSEXUKL+(-4.32)*MSPXJPL=0;"
        INCOME_CONSTRAINTS = "(-1)*GLOBAL_REITS+INFRA_EQUITY=0"

        ret = {}
        ret['proportional'] = US_CONSTRAINTS + EAFE_CONSTRAINTS + INCOME_CONSTRAINTS
        ret['assets'] = [
            "FRUS1GR",
            "FRUS1VA",
            "FRUSS2L",
            "MSEXUKL",
            "MSUTDKL",
            "MSJPANL",
            "MSPXJPL",
            "MSEMKF$",
            "GLOBAL_REITS",
            "INFRA_EQUITY",
        ]

        return ret

    @staticmethod
    def get_other_fixed_income_category():

        ret = {}
        ret['assets'] = ["LHYIELD"]
        ret['proportional'] = ''
        return ret

    @staticmethod
    def get_hedge_funds_category():

        HF_CONSTRAINTS = "CSFBMTT+(-2)*CSTEVDH=0;CSFBMTT+(-1)*CSTLNSH=0"

        ret = {}
        ret['proportional'] = HF_CONSTRAINTS
        ret['assets'] = [
            "CSFBMTT",
            "CSTEVDH",
            "CSTLNSH",
        ]
        return ret

    @staticmethod
    def get_other_private_assets_category():

        ret = {}
        ret["assets"] = ["PRIVATE_CREDIT", "PA_REAL_ESTATE", "PA_INFRA"]
        ret['proportional'] = "PA_REAL_ESTATE+(-3)*PA_INFRA=0"
        return ret

    @staticmethod
    def get_private_equity_category():

        ret = {}
        ret['assets'] = ["PE_BUYOUT", "PE_GROWTH", "PE_VENTURE"]
        ret['proportional'] = "PE_BUYOUT+(-3)*PE_GROWTH=0;PE_GROWTH+(3.5)*PE_VENTURE=0"
        return ret

    @staticmethod
    def get_proportional_constraints():

        combined = {
            'assets': [],
            'proportional': ''
        }

        for name, method in Constraints.__dict__.items():
            # Check if it's a static method and callable
            if isinstance(method, staticmethod) and name != "get_proportional_constraints":
                result = method.__func__()  # Call the static method
                combined['assets'].extend(result.get('assets', []))

                prop = result.get('proportional', '')
                if prop:
                    if combined['proportional']:
                        combined['proportional'] += ';' + prop
                    else:
                        combined['proportional'] = prop
        return combined


if __name__ == "__main__":

    res = Constraints.get("USD", 0.078)

