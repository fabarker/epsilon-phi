from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource
from epsilonPhi.core.dataModel.alchemist.SessionManager import SessionMgr
from epsilonPhi.core.timeSeries.timeSeriesMain import CTimeSeries
from epsilonPhi.core.dataModel.enums.TimeSeries import TimeSeriesType, ReturnsType
import pandas as pd

mgr = SessionMgr()
gds = GlobalDataSource()

info = pd.read_excel('/Users/francisbarker/Desktop/Implied Currencies.xlsx','Implied Currencies', index_col='ticker')

spts = CTimeSeries(ts_type=TimeSeriesType.LEVELS, returns_type=ReturnsType.SIMPLE)
for L in info.index:
    D = L[0:-1] + '$'
    Local = gds.get_time_series_data_from_ticker(L, cols='PI')
    Denom_L, _, _ = mgr.get_time_series_currency(L)

    Foreign = gds.get_time_series_data_from_ticker(D, cols='PI')
    Denom_F, _, _ = mgr.get_time_series_currency(D)

    fx = Foreign.division_over_common_dates(Local)
    fx.columns = [Denom_F + Denom_L]
    spts = spts.concat(fx)


#############

from epsilonPhi.core.dataModel.dataSources.riskFreeRates.RiskFreeRates import MSCIActivityPanel
panel = MSCIActivityPanel.get_activity_panel_single_index('Europe ex-UK')
currencies = panel.columns.get_level_values('Currency')

xrates = ['USD' + x for x in currencies]
xrate_spt = gds.get_fx_spot_rates(xrates, 'mid')

