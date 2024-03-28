from epsilonPhi.core.cpp.smootiing.gaussian_kernel.kse import *
import numpy as np
from epsilonPhi.core.utils.DateUtils import DateUtils
import pandas as pd

_X = np.random.randint(0, 100, (10, 10)).flatten()
_Y = np.random.randint(0, 100, (10, 10)).flatten()
_Z = np.random.randint(0, 100, (10, 10)).flatten()
_D = DateUtils.to_ordinals(pd.date_range('31-Dec-2021', freq='B', periods=10)).reshape(-1, 1).repeat(10, 1).flatten()

A = np.column_stack((_Y, _Z, _X))

res = kse(A, _D)