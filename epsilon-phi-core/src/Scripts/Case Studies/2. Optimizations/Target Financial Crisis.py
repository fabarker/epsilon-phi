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
path = '/Users/francisbarker/repo/epsilon-psi/epsilon-phi-core/src/Scripts/Case Studies/2. Optimizations/Simple Analysis.xlsx'

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
ptf.get_asset("MLG5QIL").set_uncertainty(ptf.get_asset("JPMGCOC").get_uncertainty() * 1.2)

# Set the return CVaR
tgt_loss = 0.3

# Set the constraints string
constraints_str = 'PA_REAL_ESTATE+(-3)*PA_INFRA=0;LHTRYIN>10;(-45)*FRUS1GR+40*FRUS1VA=0;(-45)*FRUSS2L+15*FRUS1VA=0;(-1)*GLOBAL_REITS+INFRA_EQUITY=0' + ';MSACWFL=0;HFRIFFD=0;'

# copy the current portfolio
loss_port = ptf.deepcopy(name="Drawdown Constrained")

def f_loss(v):
    loss_port.optimize(
        target_vol=v,
        contstraints=constraints_str)

    l = loss_port.get_factor_stress_tests().get('Financial Crisis').total
    return np.abs(l) - np.abs(tgt_loss)

v_loss = root_scalar(f_loss, bracket=[0.08, 0.082], method="brentq").root
f_loss(v_loss)

# copy the current portfolio
loss_ptf_2 = ptf.deepcopy(name="Drawdown Constrained (No Private Assets)")
pe_const = "PA_REAL_ESTATE=0;PA_INFRA=0;PE_BUYOUT=0;PE_GROWTH=0;PE_VENTURE=0;PRIVATE_CREDIT=0"

def f_loss_2(v):
    loss_ptf_2.optimize(
        target_vol=v,
        contstraints=constraints_str + pe_const
    )

    l = loss_ptf_2.get_factor_stress_tests().get('Financial Crisis').total
    return np.abs(l) - np.abs(tgt_loss)

v_loss_2 = root_scalar(f_loss_2, bracket=[0.075, 0.085], method="brentq").root
f_loss_2(v_loss_2)

ptfs = [ptf, loss_port, loss_ptf_2]
report = Reporting(
    os.getcwd(),
    "Drawdown Matched"
)
report.add_portfolios(ptfs)
report.generate_report()

