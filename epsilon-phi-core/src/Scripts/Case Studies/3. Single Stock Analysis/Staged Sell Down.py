from epsilonPhi.core.schema.Schema import ContextCreator
from epsilonPhi.core.portfolio.SAAPortfolio import SAAPortfolio
from epsilonPhi.core.reporting.Reporting import Reporting
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
import os
import pandas as pd
import numpy as np
from epsilonPhi.core.optimizer.constraints.Constraints import Constraints
from scipy.optimize import root_scalar
from epsilonPhi.core.asset.proxies.SingleStock import CSingleStock

from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
gds = GlobalDataSource()


# path to portfolio weights
path = '/Users/francisbarker/Repositories/Python/epsilon-phi/epsilon-phi-core/src/Scripts/Case Studies/3. Single Stock Analysis/SS Analysis 2.xlsx'

# Set the schema currency
currency = "USD"

# 1. Create Context
schema = ContextCreator(
    currency='USD',
    frequency=Frequency.BUSINESS_MONTHLY,
    start_date='30-Nov-1983',
    end_date='31-Dec-2022'
).create_context()

# Instantiate a single stock object
df = gds.get_time_series_data_from_ticker('@INTC', cols='RI').get_returns()
ss = CSingleStock(df, schema, "USD", "USD", 0)
ss.set_market_equivalent('SP5EINT')
ss.cache_asset()

# 2. Load Portfolios
ptfs = SAAPortfolio.get_portfolios_from_template(
        currency="USD",
        template_path=path,
        schema=schema)


from epsilonPhi.core.reporting.Reporting import Reporting
report = Reporting(os.getcwd(), "Staged Sell Down")
report.add_portfolios(ptfs)
#report.generate_report()


#ra = ptfs[-1].get_implied_risk_aversion()
#ir = [ x.get_mean_variance_implied_returns(ra)[-1] for x in ptfs ]

# Simulate portfolios

ptf_sim_order = list(np.arange(6)) + [len(ptfs)-1] * (20-6)
ws = ptfs[0].get_portfolio_wealth_projection(
    ptf_sim_order=ptf_sim_order,
    ptf_list=ptfs,
)


