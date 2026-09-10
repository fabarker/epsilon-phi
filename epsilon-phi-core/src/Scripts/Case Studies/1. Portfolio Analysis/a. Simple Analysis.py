from epsilonPhi.core.dataModel.enums.TimeSeries import ReturnsType, TimeSeriesType
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
path = '/Users/francisbarker/Repositories/Python/epsilon-phi/epsilon-phi-core/src/Scripts/Case Studies/1. Portfolio Analysis/Core Portfolios.xlsx'

ptfs = SAAPortfolio.get_portfolios_from_template(
        currency="USD",
        template_path=path,
        schema=schema
)

# Get Asset Risk Decomposition
#risk_decomp_asset = pd.DataFrame(saa.get_risk_decomposition(), index=[ saa.get_asset(x).reporting_name for x in saa.get_asset_names() ])

# Get Factor Risk Decomposition
#asset, fx = saa.get_fx_risk_decomposition()

# Get factor return decomposition
#premias = pd.DataFrame(saa.get_risk_premias(), index=schema.BaseModel.return_factor_list)

# Get factor return decomposition
#risk_factors = pd.DataFrame(saa.get_risk_decomposition_factor(), columns=schema.BaseModel.factor_list)

# Run Wealth Simulation with 4% Spend

# Loop across risk level, spending levels, starting value



outflows = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
start_values = [100, 125, 150, 175, 200]

simPtfs = []
for ptf in ptfs:
    for outflow in outflows:
        for s in start_values:
            ptf_new = ptf.deepcopy(name="SAA_" + str(outflow) + "_" + str(s))
            ptf_new.set_current_value(s)
            flows = [outflow] * 20
            ptf_new.set_ws_outflows(flows, 'real')
            wp = ptf_new.get_portfolio_wealth_projection()
            df_new = pd.DataFrame(wp.get_prob_of_real_capital_preservation(), columns=[(ptf.name, outflow, s)])

            simPtfs.extend([df_new.copy()])

df = pd.concat(simPtfs, axis=1)

