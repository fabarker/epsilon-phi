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

# we need to find the volatility so that the median simulated value in real terms equals todays value
