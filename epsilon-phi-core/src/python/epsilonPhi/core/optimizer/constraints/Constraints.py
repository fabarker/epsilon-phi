
class Constraints:

    def get_equity_category(self):

        US_CONSTRAINTS = "(-45)*RUS1G+40*RUS1V;(-40)*RUS2+15*RUS1V=0"
        EAFE_CONSTRAINTS = "MSEXUKL+(-3.42)*MSUKL=0;MSEXUKL+(-2.17)*MSUKL=0;MSEXUKL+(-4.32)*MSPXJPL=0"
        INCOME_CONSTRAINTS = "(-1)*SPGLREITH+MSWDIF=0"

        ret = {}
        ret['proportional'] = US_CONSTRAINTS + EAFE_CONSTRAINTS + INCOME_CONSTRAINTS
        ret['assets'] = [
            "RUS1G",
            "RUS1V",
            "RUS2",
            "MSEXUKL",
            "MSUKL",
            "MSPXJPL",
            "MSEM",
            "SPGLREITH",
            "MSWDIF",
        ]

        return ret

    def get_other_fixed_income(self):

        ret = {}
        ret['assets'] = "IUSHY_GE0"
        ret['proportional'] = ''
        return ret


    def get_hedge_funds(self):

        HF_CONSTRAINTS = "CSFBMA+(-2)*CSFBED_GE25=0;CSFBMA+(-1)*CSFBLSE=0"

        ret = {}
        ret['proportional'] = HF_CONSTRAINTS
        ret['assets'] = [
            "CSFBMA",
            "CSFBED_GE25",
            "CSFBLSE",
        ]
        return ret

    def get_other_private_assets(self):

        ret = {}
        ret["assets"] = ["PACREDIT", "PARE", "PAINFRA"]
        ret['proportional'] = "PARE+(-3)*PAINFRA=0"
        return ret

    def private_equity(self):

        ret = {}
        ret['assets'] = ["PEBUYOUT", "PEGROWTH", "PEVENTURE"]
        ret['proportional'] = "PEBUYOUT+(-3)*PEGROWTH=0;PEGROWTH+(3.5)*PEVENTURE=0"
        return ret

