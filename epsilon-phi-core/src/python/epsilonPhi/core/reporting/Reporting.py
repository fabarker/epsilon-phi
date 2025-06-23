from typing import Union
from epsilonPhi.core.portfolio.SAAPortfolio import SAAPortfolio

class Reporting(object):

    def __init__(self,
                 report_directory: str,
                 report_name: str
                 ):

        self._report_dir = report_directory
        self._name = report_name
        self._portfolios = []

    def add_portfolios(self, portfolios: Union[SAAPortfolio, Iterable[SAAPortfolio]]) -> None:
        if not isinstance(portfolios, Iterable) or isinstance(portfolios, (str, bytes)):
            portfolios = [portfolios]

        self._portfolios.extend(portfolios)

    def generate_report(self):
        pass

    def get_assumptions(self):
        pass

    def get_portfolio_table(self):
        pass

    def get_single_stock_analysis(self):
        pass

    def get_risk_metrics(self):
        pass

    def get_wealth_simulation(self):
        pass

    def get_private_assets_commitments(self):
        pass


from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, NamedStyle

wb = Workbook()
ws = wb.active

# Write header
ws['A1'] = 'Name'
ws['B1'] = 'Score'

# Apply styles to header
header_style = NamedStyle(name="header_style")
header_style.font = Font(bold=True, color="FFFFFF")
header_style.fill = PatternFill("solid", fgColor="4F81BD")
header_style.alignment = Alignment(horizontal="center")

ws['A1'].style = header_style
ws['B1'].style = header_style

# Write data
data = [("Alice", 92), ("Bob", 85), ("Charlie", 78)]
for row in data:
    ws.append(row)

# Format number column
for cell in ws["B"][1:]:
    cell.number_format = '0.00'

# Auto-fit column widths
for col in ws.columns:
    max_length = max(len(str(cell.value)) if cell.value else 0 for cell in col)
    ws.column_dimensions[col[0].column_letter].width = max_length + 2

wb.save("formatted_excel.xlsx")