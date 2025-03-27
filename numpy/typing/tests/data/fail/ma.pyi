from typing import Any

import numpy as np
import numpy.ma

m: np.ma.MaskedArray[tuple[int], np.dtype[np.float64]]

m.shape = (3, 1)  # E: Incompatible types in assignment
m.dtype = np.bool  # E: Incompatible types in assignment

np.amin(m, axis=1.0)  # E: No overload variant
np.amin(m, keepdims=1.0)  # E: No overload variant
np.amin(m, out=1.0)  # E: No overload variant
np.amin(m, fill_value=lambda x: 27)  # E: No overload variant
