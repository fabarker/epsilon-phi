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
path = '/Users/francisbarker/Repositories/Python/epsilon-phi/epsilon-phi-core/src/Scripts/Case Studies/2. Optimizations/Benchmarks.xlsx'

# Set the schema currency
currency = "USD"

# 1. Create Context
schema = ContextCreator(
    currency='USD',
    frequency=Frequency.BUSINESS_MONTHLY,
    start_date='30-Nov-1983',
    end_date='31-Dec-2022'
).create_context()

schema = schema.get_extended_schema()

# 2. Load Portfolios
ptfs = SAAPortfolio.get_portfolios_from_template(
        currency="USD",
        template_path=path,
        schema=schema)

# Build Outflows Panel
from epsilonPhi.core.simulation.SimStructs import WealthFlows

outflows = [WealthFlows(percent=0.02) if x < 6 else WealthFlows(percent=0.03) for x in range(20)]

# for each benchmark portfolio, get the real return after 20 years

probs = []
risk = []
for ptf in ptfs:
    ws = ptf.get_portfolio_wealth_projection(
        ws_outflows=outflows
    )

    p = ws.get_prob_of_beating_inflation_plus(0.02)

    risk.extend([ptf.get_risk()])
    probs.extend([p])

df = pd.DataFrame(probs, index=risk).T

# Get the assets we can optimize over
ptf = SAAPortfolio.get_portfolios_from_template(
        currency="USD",
        template_path='/Users/francisbarker/Repositories/Python/epsilon-phi/epsilon-phi-core/src/Scripts/Case Studies/2. Optimizations/Simple Analysis.xlsx',
        schema=schema)[0]

# drop assets we dont want
ptf = ptf.remove_assets(['LHUT1T3', 'JPMGCOC', 'MLG5QIL', 'MSACWFL', 'HFRIFFD'], rebalance=True)

# Override uncertainties
ptf.get_asset("LHCCRIN").set_uncertainty(ptf.get_asset("LHCCRIN").get_uncertainty() * 2)
ptf.get_asset("GLOBAL_REITS").set_uncertainty(ptf.get_asset("GLOBAL_REITS").get_uncertainty() * 3)
ptf.get_asset("LHYIELD").set_uncertainty(ptf.get_asset("LHYIELD").get_uncertainty() * 1.5)

# we need to find the volatility so that the median simulated value in real terms equals todays value

# Set the constraints string
constraints_str = 'PA_REAL_ESTATE+(-3)*PA_INFRA=0;LHTRYIN>12;(-45)*FRUS1GR+40*FRUS1VA=0;(-45)*FRUSS2L+15*FRUS1VA=0;(-1)*GLOBAL_REITS+INFRA_EQUITY=0;'

# copy the current portfolio
opt_ptf = ptf.deepcopy(name="Optimized Portfolio")

opt_ptf.optimize(
    target_vol=0.0784,
    contstraints=constraints_str
)

ws = opt_ptf.get_portfolio_wealth_projection(ws_outflows=outflows)
print(ws.get_prob_of_beating_inflation_plus(0.02))

ptfs = [opt_ptf]
report = Reporting(
    os.getcwd(),
    "Real Return with Spending"
)
report.add_portfolios(ptfs)
report.generate_report()

