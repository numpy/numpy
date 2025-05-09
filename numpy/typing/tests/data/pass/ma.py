from typing import Any

import numpy as np
import numpy.ma
import numpy.typing as npt

ar_b: npt.NDArray[np.bool] = np.array([True, False, True])
m: np.ma.MaskedArray[Any, np.dtype[np.float64]] = np.ma.masked_array([1.5, 2, 3], mask=[True, False, True])

m.mask = ar_b
m.mask = np.False_

