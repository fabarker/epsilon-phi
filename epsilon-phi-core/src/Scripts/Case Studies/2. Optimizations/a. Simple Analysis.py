from epsilonPhi.core.schema.Schema import ContextCreator
from epsilonPhi.core.portfolio.SAAPortfolio import SAAPortfolio
from epsilonPhi.core.reporting.Reporting import Reporting
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
import os
import pandas as pd
import numpy as np
from epsilonPhi.core.optimizer.constraints.Constraints import Constraints


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


path = '/Users/francisbarker/repo/epsilon-psi/epsilon-phi-core/src/Scripts/Case Studies/2. Optimizations/Simple Analysis.xlsx'

ptf = SAAPortfolio.get_portfolios_from_template(
        currency="USD",
        template_path=path,
        schema=schema
)[0]

# Run a vol matched optimization
target_vol = ptf.get_risk()
constraints = Constraints.get(currency, target_vol)

vol_match = ptf.deepcopy(name="Volatility Matched")

constr_str = constraints.get("proportional")
vol_match.optimize(
    target_vol=target_vol,
    contstraints='PA_REAL_ESTATE+(-3)*PA_INFRA=0;LHTRYIN>16.5;(-45)*FRUS1GR+40*FRUS1VA=0;(-45)*FRUSS2L+15*FRUS1VA=0;(-1)*GLOBAL_REITS+INFRA_EQUITY=0' + ';MSACWFL=0;HFRIFFD=0;'
)

print(target_vol)
print(vol_match.get_risk())

# Run a return matched optimization
ptfs = [ptf, vol_match]
report = Reporting(
    os.getcwd(),
    "vol_matched"
)
report.add_portfolios(ptfs)
report.generate_report()

# Run a CVaR matched optimization



# Run a Financial Crisis matched optimization



# Run an asset risk parity optimization, with leverage



# Run a return factor risk parity optimization



# Run a tracking error optimization

