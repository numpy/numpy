from typing import Any

import numpy as np
import numpy.ma

m: np.ma.MaskedArray[tuple[int], np.dtype[np.float64]]

m.shape = (3, 1)  # E: Incompatible types in assignment
m.dtype = np.bool  # E: Incompatible types in assignment

np.ma.min(m, axis=1.0)  # E: No overload variant
np.ma.min(m, keepdims=1.0)  # E: No overload variant
np.ma.min(m, out=1.0)  # E: No overload variant
np.ma.min(m, fill_value=lambda x: 27)  # E: No overload variant

np.ma.max(m, axis=1.0)  # E: No overload variant
np.ma.max(m, keepdims=1.0)  # E: No overload variant
np.ma.max(m, out=1.0)  # E: No overload variant
np.ma.max(m, fill_value=lambda x: 27)  # E: No overload variant

np.ma.ptp(m, axis=1.0)  # E: No overload variant
np.ma.ptp(m, keepdims=1.0)  # E: No overload variant
np.ma.ptp(m, out=1.0)  # E: No overload variant
np.ma.ptp(m, fill_value=lambda x: 27)  # E: No overload variant
