from epsilonPhi.core.schema.Schema import ContextCreator
from epsilonPhi.core.portfolio.SAAPortfolio import SAAPortfolio
from epsilonPhi.core.reporting.Reporting import Reporting
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
import os
import pandas as pd
import numpy as np

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
path = '/Users/francisbarker/Repositories/Python/epsilon-phi/epsilon-phi-core/src/Scripts/Case Studies/1. Portfolio Analysis/Simple Analysis.xlsx'

ptfs = SAAPortfolio.get_portfolios_from_template(
        currency="USD",
        template_path=path,
        schema=schema
)

saa = ptfs[-1]
sim_ws = saa.get_portfolio_wealth_projection()

# Get Asset Risk Decomposition
#risk_decomp_asset = pd.DataFrame(saa.get_risk_decomposition(), index=[ saa.get_asset(x).reporting_name for x in saa.get_asset_names() ])

# Get Factor Risk Decomposition
#asset, fx = saa.get_fx_risk_decomposition()

# Get factor return decomposition
#premias = pd.DataFrame(saa.get_risk_premias(), index=schema.BaseModel.return_factor_list)

# Get factor return decomposition
#risk_factors = pd.DataFrame(saa.get_risk_decomposition_factor(), columns=schema.BaseModel.factor_list)

# Run Wealth Simulation with 4% Spend


simPtfs = []
outflows = np.linspace(0.01, 0.06, 11)
for outflow in outflows:
    ptf_new = saa.deepcopy(name="SAA_" + str(outflow * 100) + "%")
    flows = [outflow * 100] * 20
    saa.set_ws_outflows(flows, 'real')
    wp = saa.get_portfolio_wealth_projection()
    df_new = pd.DataFrame(wp.get_prob_of_capital_exhaustion(), columns=[ptf_new.name])

    simPtfs.extend([df_new.copy()])



# 3. Create Reporting Object
report = Reporting(
    os.getcwd(),
    report_name='Simple Portfolio Report Spending'
)

report.add_portfolios(simPtfs)
report.write_wealth_simulations()


