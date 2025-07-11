from epsilonPhi.core.schema.Schema import ContextCreator
from epsilonPhi.core.portfolio.SAAPortfolio import SAAPortfolio
from epsilonPhi.core.simulation.privateAssets.PrivateUtils import CPrivateUtils
from epsilonPhi.core.simulation.SAASimulation import SAASimulation

path = '/Users/francisbarker/Repositories/Python/epsilon-phi/epsilon-phi-core/src/Scripts/Case Studies/Private Assets/Commitment Pacing.xlsx'

schema = ContextCreator(
    currency='USD',
    start_date='30-Nov-1983',
    end_date='31-Dec-2022'
).create_context()


ptfs = SAAPortfolio.get_portfolios_from_template(
        currency="USD",
        template_path=path,
        schema=schema
)

# Set current value
ptfs[1].set_current_value(100)

sim = SAASimulation(ptfs[1].portfolio_mgr)
res = sim.get_private_assets_distributions()
res.set_liquid_weights(ptfs[0].get_weights()[0:2])

import numpy as np
import pandas as pd

market_vals = np.concatenate((res.get_liquid_market_values(), res.pe_MV)).T
wts = market_vals / np.sum(market_vals, axis=1, keepdims=True)
wts_df = pd.DataFrame(wts, columns=ptfs[0].asset_names)

all_ptfs = []
for idx, row in wts_df.iterrows():
    tmp = SAAPortfolio.create_equal_weighted_portfolio(wts_df.columns, context=schema, portfolio_name=str(idx))
    tmp.set_weights(row.values.flatten())
    all_ptfs.extend([tmp.deepcopy()])

ws = ptfs[0].get_portfolio_wealth_projection(ptf_list=all_ptfs, ptf_sim_order=list(np.array(range(20))))
ws_1 = ptfs[0].get_portfolio_wealth_projection()
ws_2 = ptfs[-1].get_portfolio_wealth_projection()




