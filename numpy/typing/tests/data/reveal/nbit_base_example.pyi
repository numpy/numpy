from typing import assert_type

import numpy as np
import numpy.typing as npt
from numpy._typing import _32Bit, _64Bit

def add[T1: npt.NBitBase, T2: npt.NBitBase](a: np.floating[T1], b: np.integer[T2]) -> np.floating[T1 | T2]: ...  # type: ignore[deprecated]

i8: np.int64
i4: np.int32
f8: np.float64
f4: np.float32

assert_type(add(f8, i8), np.floating[_64Bit])
assert_type(add(f4, i8), np.floating[_32Bit | _64Bit])
assert_type(add(f8, i4), np.floating[_32Bit | _64Bit])
assert_type(add(f4, i4), np.floating[_32Bit])
