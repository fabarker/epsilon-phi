from epsilonPhi.core.cpp.dates import cDates
from datetime import datetime
import numpy as np

test_date = datetime(year=2024, month=3, day=31)
res = cDates.get_expiry_date(test_date, '1y')

import pandas as pd
_dr = pd.date_range('31-Mar-2024', '31-Mar-2025')
dts = _dr.to_pydatetime()
mats = ['1y'] * len(dts)
res = cDates.shiftdates(dts, mats)