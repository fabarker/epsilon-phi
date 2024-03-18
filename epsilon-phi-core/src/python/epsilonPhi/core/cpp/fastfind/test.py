
from epsilonPhi.core.cpp.fastfind.find_1st import *
import numpy as np

input = np.random.randint(1, 20, (1000000, 1)).repeat(2, 1)
limit = np.random.randint(1, 10, (1000000, 1))

idx = find_1st(input, limit)