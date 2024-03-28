import pandas as pd
from epsilonPhi.core.cpp.dates import cDates

dates = pd.date_range('31-Dec-2012', '31-Dec-2013')
res = cDates.to_ordinals(dates)