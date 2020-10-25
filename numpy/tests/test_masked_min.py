import sys
import numpy as np
sys.path.insert(1, 'numpy/numpy/ma')

import core
a = np.array([1, 2, 3], dtype=np.object)
a_masked = core.masked_array(a)
print(a_masked.min())
