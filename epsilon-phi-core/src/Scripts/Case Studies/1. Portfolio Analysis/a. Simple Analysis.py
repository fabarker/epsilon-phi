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
path = '/Users/francisbarker/Repositories/Python/epsilon-phi/epsilon-phi-core/src/Scripts/Case Studies/1. Portfolio Analysis/Simple Analysis.xlsx'

ptfs = SAAPortfolio.get_portfolios_from_template(
        currency="USD",
        template_path=path,
        schema=schema
)

saa = ptfs[-1]

from epsilonPhi.core.dataModel.enums.Factor import FACTOR
from epsilonPhi.core.asset.Asset import CAsset
from epsilonPhi.core.asset.AssetMgr import CAssetMgr

assetMgr = CAssetMgr(schema=schema)

def get_factor_asset(factor=FACTOR.EQUITY_GLOBAL_ISG):

    df = schema.get_risk_factors_panel().get([factor.name])
    rfr = schema.get_risk_free_rate_asset()
    tmp = rfr.addition_over_common_dates(df)
    tmp.name = factor.name

    asset = CAsset(
        tmp,
        schema,
        "USD",
        "USD",
        0,
        ReturnsType.SIMPLE,
        TimeSeriesType.RETURNS
    )

    try:
        assetMgr.add_asset_to_cache(asset)
    except:
        pass
    return assetMgr.get_asset_by_name(tmp.name)

factor_list = [
    FACTOR.EQUITY_GLOBAL_ISG,
    FACTOR.TERM_GLOBAL_ISG,
    FACTOR.FUNDING_US_ISG,
    FACTOR.LIQUIDITY_US_PASTOR_STAMBAUGH,
    FACTOR.CARRY_GLOBAL_ISG,
    FACTOR.EQUITY_EMERGING_ISG,
]

res = {}
for factor_enum in factor_list:
    factor_asset = get_factor_asset(factor_enum)

    sim_fac, _ = factor_asset.simulate(long_term_shocks=True)
    sim_ptf, _ = saa.get_portfolio_simulated_returns_panel(long_term_shocks=True)

    # simulate factor one year ahead
    sim_fac_T = np.prod((1 + sim_fac[:3, :]), axis=0) - 1
    sim_ptf_T = np.prod((1 + sim_ptf[:3, :]), axis=0) - 1

    mu_fac = np.mean(sim_fac_T)
    std_fac = np.std(sim_fac_T)

    # Find paths with a -1σ factor shock (from mean)
    mask = sim_fac_T < (mu_fac - 2 * std_fac)

    mu_factor_in_shock = np.mean(sim_fac_T[mask])
    mu_asset_in_shock = np.mean(sim_ptf_T[mask])
    res[factor_enum.name] = (mu_factor_in_shock, mu_asset_in_shock)





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


