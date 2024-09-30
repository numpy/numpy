from typing import Any

import numpy as np

from typing_extensions import assert_type

arr: np.ma.MaskedArray[Any, np.dtype[np.float64]] = np.ma.masked_array([[1.0, 2.0]], mask=[True, False])
result = np.ma.min(arr, axis=None)
assert_type(result, np.float64)

tgt: np.ma.MaskedArray[Any, np.dtype[np.float64]] = np.ma.empty(shape=())
result = np.ma.min(arr, axis=None, out=tgt)
assert_type(result, np.ma.MaskedArray)
