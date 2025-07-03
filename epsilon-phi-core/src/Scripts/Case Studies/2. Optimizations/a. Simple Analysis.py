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
path = '/Users/francisbarker/Repositories/Python/epsilon-phi/epsilon-phi-core/src/Scripts/Case Studies/2. Optimizations/Simple Analysis.xlsx'

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
ptf = SAAPortfolio.get_portfolios_from_template(
        currency="USD",
        template_path=path,
        schema=schema)[0]

# Override uncertainties
ptf.get_asset("LHCCRIN").set_uncertainty(ptf.get_asset("LHCCRIN").get_uncertainty() * 2)
ptf.get_asset("GLOBAL_REITS").set_uncertainty(ptf.get_asset("GLOBAL_REITS").get_uncertainty() * 3)
ptf.get_asset("LHYIELD").set_uncertainty(ptf.get_asset("LHYIELD").get_uncertainty() * 1.5)
ptf.get_asset("JPMGCOC").set_uncertainty(ptf.get_asset("JPMGCOC").get_uncertainty() * 1.5)


# Run a vol matched optimization
vol_match = ptf.deepcopy(name="Volatility Matched")
vol_match.optimize(
    target_vol=ptf.get_risk(),
    contstraints='MLG5QIL>5;PA_REAL_ESTATE+(-3)*PA_INFRA=0;LHTRYIN>16.5;(-45)*FRUS1GR+40*FRUS1VA=0;(-45)*FRUSS2L+15*FRUS1VA=0;(-1)*GLOBAL_REITS+INFRA_EQUITY=0' + ';MSACWFL=0;HFRIFFD=0;'
)
vol_match.add_asset_by_name("USD_RFR", 0, 0)
vol_match.get_asset("USD_RFR").set_reporting_info('Leverage (SOFR + 1%)', 'Cash, Money Markets and Deposits')
vol_match.get_asset("USD_RFR").set_alpha(1.25/100)

# Run a CVaR matched optimization
risk = ptf.get_portfolio_var_pol()
cvar_ref, _ = risk.get_VaR(11)
cvar_port = ptf.deepcopy(name="CVaR Matched")
def f(v):
    cvar_port.optimize(
        target_vol=v,
        contstraints='MLG5QIL>5;PA_REAL_ESTATE+(-3)*PA_INFRA=0;LHTRYIN>16.5;(-45)*FRUS1GR+40*FRUS1VA=0;(-45)*FRUSS2L+15*FRUS1VA=0;(-1)*GLOBAL_REITS+INFRA_EQUITY=0' + ';MSACWFL=0;HFRIFFD=0;'
    )
    cvar, _ = cvar_port.get_portfolio_var_pol().get_VaR(11)
    return cvar - cvar_ref

v_star = 0.0727
#v_star = root_scalar(f, bracket=[0.07, 0.073], method="brentq").root
f(v_star) # 0.07479929994278148
cvar_port.add_asset_by_name("USD_RFR", 0, 0)
cvar_port.get_asset("USD_RFR").set_reporting_info('Leverage (SOFR + 1%)', 'Cash, Money Markets and Deposits')
cvar_port.get_asset("USD_RFR").set_alpha(1.25/100)

# Return Matched Optimization
rtn_target = ptf.get_total_return()
rtn_port = ptf.deepcopy(name="Return Matched")
def f_rtn(v):
    rtn_port.optimize(
        target_vol=v,
        contstraints='MLG5QIL>5;PA_REAL_ESTATE+(-3)*PA_INFRA=0;LHTRYIN>16.5;(-45)*FRUS1GR+40*FRUS1VA=0;(-45)*FRUSS2L+15*FRUS1VA=0;(-1)*GLOBAL_REITS+INFRA_EQUITY=0' + ';MSACWFL=0;HFRIFFD=0;'
    )
    return rtn_port.get_total_return() - rtn_target
v_rtn = 0.05413783515306369
#v_rtn = root_scalar(f_rtn, bracket=[0.052, 0.056], method="brentq").root
f_rtn(v_rtn)
rtn_port.add_asset_by_name("USD_RFR", 0, 0)
rtn_port.get_asset("USD_RFR").set_reporting_info('Leverage (SOFR + 1%)', 'Cash, Money Markets and Deposits')
rtn_port.get_asset("USD_RFR").set_alpha(1.25/100)
# Financial Crisis Matched
loss_target = ptf.get_factor_stress_tests().get('Covid-19').total
loss_port = ptf.deepcopy(name="Max Drawdown Matched")

def f_loss(v):
    loss_port.optimize(
        target_vol=v,
        contstraints='MLG5QIL>5;PA_REAL_ESTATE+(-3)*PA_INFRA=0;LHTRYIN>16.5;(-45)*FRUS1GR+40*FRUS1VA=0;(-45)*FRUSS2L+15*FRUS1VA=0;(-1)*GLOBAL_REITS+INFRA_EQUITY=0' + ';MSACWFL=0;HFRIFFD=0;'
    )
    l = loss_port.get_factor_stress_tests().get('Covid-19').total
    return np.abs(l) - np.abs(loss_target)

v_loss =  0.06996156630373873
#v_loss = root_scalar(f_loss, bracket=[0.069, 0.07], method="brentq").root
f_loss(v_loss)
loss_port.add_asset_by_name("USD_RFR", 0, 0)
loss_port.get_asset("USD_RFR").set_reporting_info('Leverage (SOFR + 1%)', 'Cash, Money Markets and Deposits')
loss_port.get_asset("USD_RFR").set_alpha(1.25/100)

from epsilonPhi.core.optimizer.riskBudgeting.allocation import EqualRiskContributionWithRiskTarget
rb = EqualRiskContributionWithRiskTarget(ptf.get_sigma(), ptf.get_risk())
rb.solve()
rp_port = ptf.deepcopy(name="Volatility Parity")
rp_port.add_asset_by_name("USD_RFR", 0, 0)
rp_port.get_asset("USD_RFR").set_reporting_info('Leverage (SOFR + 1%)', 'Cash, Money Markets and Deposits')
rp_port.get_asset("USD_RFR").set_alpha(1.25/100)
wts = np.concatenate([rb._x, np.array([1 - sum(rb._x)])])
rp_port.set_weights(wts)
hrs = np.concatenate([ptf.get_hedging_ratios(), np.array([0])])
rp_port.set_hedging_ratios(hrs)

ptf.add_asset_by_name("USD_RFR", 0, 0)
ptf.get_asset("USD_RFR").set_reporting_info('Leverage (SOFR + 1%)', 'Cash, Money Markets and Deposits')
ptf.get_asset("USD_RFR").set_alpha(1.25/100)

ptfs = [ptf, vol_match, cvar_port, rtn_port, loss_port, rp_port]
report = Reporting(
    os.getcwd(),
    "Optimizations"
)
report.add_portfolios(ptfs)
report.generate_report()
