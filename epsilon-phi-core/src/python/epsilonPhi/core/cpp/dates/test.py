import pandas as pd
from epsilonPhi.core.cpp.dates import cDates

dates = pd.date_range('31-Dec-2012', '31-Dec-2013')
term = ['1m'] * len(dates)

weekdays = pd.date_range('31-Dec-2012', '31-Dec-2013', freq='B')
holidays = dates.difference(weekdays)

res = cDates.shiftdates(dates.to_pydatetime(), term)