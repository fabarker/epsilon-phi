from epsilonPhi.core.schema.Schema import ContextCreator
from epsilonPhi.core.portfolio.SAAPortfolio import SAAPortfolio
from epsilonPhi.core.simulation.privateAssets.PrivateUtils import CPrivateUtils
from epsilonPhi.core.simulation.SAASimulation import SAASimulation

path = '/Users/francisbarker/Repositories/Python/epsilon-phi/epsilon-phi-core/src/Scripts/Case Studies/Private Assets/Simple Analysis.xlsx'

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
        "PE_GROWTH": [
            {
                'commitment_size': 50,
            }
        ],
        "PRIVATE_CREDIT": [
            {
                'commitment_size': 50,
            }
        ],
}

sim = SAASimulation(ptf.portfolio_mgr)
res = sim.get_current_private_markets_portfolio_projection(initial_vintage)



