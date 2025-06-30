from epsilonPhi.core.schema.Schema import ContextCreator
from epsilonPhi.core.portfolio.SAAPortfolio import SAAPortfolio
from epsilonPhi.core.reporting.Reporting import Reporting
from epsilonPhi.core.dataModel.enums.FrequencyType import Frequency
import os


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
path = '/Users/francisbarker/repo/epsilon-psi/epsilon-phi-core/src/Scripts/Case Studies/1. Portfolio Analysis/Simple Analysis.xlsx'

ptfs = SAAPortfolio.get_portfolios_from_template(
        currency="USD",
        template_path=path,
        schema=schema
)

# 3. Create Reporting Object
report = Reporting(
    os.getcwd(),
    report_name='Simple Portfolio Report_new_pe'
)

report.add_portfolios(ptfs)
report.generate_report()


