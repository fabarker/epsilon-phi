from epsilonPhi.core.schema.Schema import ContextCreator
from epsilonPhi.core.portfolio.SAAPortfolio import SAAPortfolio
from epsilonPhi.core.reporting.Reporting import Reporting
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
import os
import pandas as pd
import numpy as np
from epsilonPhi.core.optimizer.constraints.Constraints import Constraints
from scipy.optimize import root_scalar


# path to portfolio weights
path = '/Users/francisbarker/Repositories/Python/epsilon-phi/epsilon-phi-core/src/Scripts/Case Studies/2. Optimizations/Simple Analysis.xlsx'

# Set the schema currency
currency = "USD"

# 1. Create Context
schema = ContextCreator(
    currency='USD',
    frequency=Frequency.BUSINESS_MONTHLY,
    start_date='30-Nov-1983',
    end_date='31-Dec-2022'
).create_context()

# 2. Load Portfolios
ptf = SAAPortfolio.get_portfolios_from_template(
        currency="USD",
        template_path=path,
        schema=schema)[0]

# Override uncertainties
ptf.get_asset("LHCCRIN").set_uncertainty(ptf.get_asset("LHCCRIN").get_uncertainty() * 2)
ptf.get_asset("GLOBAL_REITS").set_uncertainty(ptf.get_asset("GLOBAL_REITS").get_uncertainty() * 3)
ptf.get_asset("LHYIELD").set_uncertainty(ptf.get_asset("LHYIELD").get_uncertainty() * 1.5)
ptf.get_asset("JPMGCOC").set_uncertainty(ptf.get_asset("JPMGCOC").get_uncertainty() * 1.5)
ptf.get_asset("MLG5QIL").set_uncertainty(ptf.get_asset("JPMGCOC").get_uncertainty() * 1.2)

report = Reporting(os.getcwd(),"Manager Alpha")
report.add_portfolios(ptf.deepcopy())

# Set the constraints string
constraints_str = 'PA_REAL_ESTATE+(-3)*PA_INFRA=0;LHTRYIN>16.5;(-45)*FRUS1GR+40*FRUS1VA=0;(-45)*FRUSS2L+15*FRUS1VA=0;(-1)*GLOBAL_REITS+INFRA_EQUITY=0' + ';MSACWFL=0;HFRIFFD=0;'

alpha_levels = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
assets = ['PE_BUYOUT', 'PE_GROWTH', 'PE_VENTURE']

for a in alpha_levels:
    print(str(round(a * 100, 1)))

    ptf_name = str(round(a * 100, 1)) + "% Alpha"
    tmp = ptf.deepcopy(name=ptf_name)

    for asset_name in assets:
        tmp.get_asset(asset_name).set_alpha(a)

    tmp.optimize(
        target_vol=ptf.get_risk(),
        contstraints=constraints_str
    )

    report.add_portfolios(tmp.deepcopy())


report.generate_report()

