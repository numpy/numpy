from typing import Any

import numpy as np
import numpy.ma

m: np.ma.MaskedArray[tuple[int], np.dtype[np.float64]]

m.shape = (3, 1)  # E: Incompatible types in assignment
m.dtype = np.bool  # E: Incompatible types in assignment
