import numpy as np
from typing_extensions import assert_type
m: np.ma.MaskedArray[tuple[int], np.dtype[np.float64]]

assert_type(m.shape, tuple[int])

assert_type(m.dtype, np.dtype[np.float64])

assert_type(int(m), int)
assert_type(float(m), float)
