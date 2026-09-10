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
path = '/Users/francisbarker/Repositories/Python/epsilon-phi/epsilon-phi-core/src/Scripts/Case Studies/3. Single Stock Analysis/SS Analysis.xlsx'

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

# Override uncertainties
#ptf.get_asset("LHCCRIN").set_uncertainty(ptf.get_asset("LHCCRIN").get_uncertainty() * 2)
#ptf.get_asset("GLOBAL_REITS").set_uncertainty(ptf.get_asset("GLOBAL_REITS").get_uncertainty() * 3)
#ptf.get_asset("LHYIELD").set_uncertainty(ptf.get_asset("LHYIELD").get_uncertainty() * 1.5)

ptf_ss = ptfs[1]

ss_assets = ptf_ss.get_asset_names() + ['PE_BUYOUT',
                                        'PE_GROWTH',
                                        'PE_VENTURE',
                                        'PRIVATE_CREDIT',
                                        'PA_REAL_ESTATE',
                                        'PA_INFRA',
                                        'JPMGCOC',
                                        'HFRIFFD',
                                        'MLG5QIL']

ss_ptf = SAAPortfolio.create_equal_weighted_portfolio(
    ss_assets,
    context=ptf_ss.schema
)

stock_name = "@INTC"
idx = ss_ptf.get_asset_names().index(stock_name)

asset_betas = ss_ptf.get_assets_risk_betas()
asset_idio = ss_ptf.get_assets_idio_variances()

ptf_betas = ptfs[3].get_risk_betas()
ptf_idio  = ptfs[3].get_idio_variance()

betas = np.concatenate((asset_betas, ptf_betas))
idios = np.concatenate((asset_idio, [ptf_idio * 2]))

betas_mult_cov = np.matmul(betas, schema.get_risk_factor_covariance().values)
sig = np.matmul(betas_mult_cov, betas.transpose()) + np.diag(idios)

stddev = np.sqrt(np.diag(sig))
corr = sig / np.outer(stddev, stddev)
corr[sig == 0] = 0

reporting_names = ss_ptf.get_asset_reporting_names() + ["Moderate Portfolio"]

out = pd.DataFrame([corr[idx], stddev], columns=reporting_names, index=['corr', 'vol'])

from epsilonPhi.core.reporting.Reporting import Reporting
report = Reporting(os.getcwd(), "Single Stock Analysis 2")
report.add_portfolios(ptfs)
report.generate_report()

#
