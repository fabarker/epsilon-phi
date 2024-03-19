
from epsilonPhi.core.cpp.fastfind.find_1st import *
import numpy as np

input = np.random.randint(1, 100, (100, 1)).repeat(2, 1)
limit = np.random.randint(1, 100, (1000000, 1)).repeat(2, 1)

idx = find_1st(input, limit)