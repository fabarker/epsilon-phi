from epsilonPhi.core.schema.Schema import ContextCreator
from epsilonPhi.core.portfolio.SAAPortfolio import SAAPortfolio
from epsilonPhi.core.simulation.privateAssets.PrivateUtils import CPrivateUtils
from epsilonPhi.core.simulation.SAASimulation import SAASimulation

path = '/Users/francisbarker/repo/epsilon-psi/epsilon-phi-core/src/Scripts/Case Studies/Private Assets/Simple Analysis.xlsx'

schema = ContextCreator(
    currency='USD',
    start_date='30-Nov-1983',
    end_date='31-Dec-2022'
).create_context()

ptf = SAAPortfolio.get_portfolios_from_template(
        currency="USD",
        template_path=path,
        schema=schema
)[0]

# Set current value
ptf.set_current_value(100)

# Construct the Intial Commitments
initial_vintage = {
        "PE_BUYOUT" : [
            {
                  'commitment_size': 50,
            },
        ],
        "PRIVATE_CREDIT": [
            {
                'commitment_size': 50,
            }
        ],
}

sim = SAASimulation(ptf.portfolio_mgr)
res = sim.get_current_private_markets_portfolio_projection(initial_vintage)

res.set_liquid_weights(ptf.get_weights())

import numpy as np
import pandas as pd

market_vals = np.concatenate((res.get_liquid_market_values(), res.pe_MV)).T
wts = market_vals / np.sum(market_vals, axis=1, keepdims=True)
wts_df = pd.DataFrame(wts, columns=["LHTRYIN","MSACWFL","PE_BUYOUT","PRIVATE_CREDIT"])

tmp = SAAPortfolio.create_equal_weighted_portfolio(wts_df.columns, context=schema)
sig = tmp.get_sigma()

stats = []
for idx, row in wts_df.iterrows():
    tmp = SAAPortfolio.create_equal_weighted_portfolio(wts_df.columns, context=schema)
    tmp.set_weights(row.values.flatten())

    risk = tmp.get_portfolio_var_pol()
    year = [ tmp.get_total_return(), tmp.get_risk(), tmp.get_sharpe_ratio(), risk.get_CVaR(11)[0] ]
    stats.extend([year])

stats_df = pd.DataFrame(stats)

import os
from epsilonPhi.core.reporting.Reporting import Reporting
reporting = Reporting(
    os.getcwd(),
    "Single Commitment"
)

reporting.add_portfolios(ptf)
reporting.generate_report()



