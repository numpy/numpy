from typing import Any

import numpy as np
import numpy.ma


m: np.ma.MaskedArray[Any, np.dtype[np.float64]] = np.ma.masked_array([1.5, 2, 3], mask=[True, False, True])

A = np.array(True, ndmin=2, dtype=bool)
B = np.array(1.0, ndmin=2, dtype=np.float32)
C: np.ma.MaskedArray[Any, np.dtype[np.uint8]] = np.ma.masked_array(1, ndmin=2, dtype=np.uint8)
D: np.ma.MaskedArray[Any, np.dtype[np.float64]] = np.ma.masked_array([1.0, 2.0], mask=[True, False])
A.setflags(write=False)
B.setflags(write=False)
C.setflags(write=False)

a = np.bool(True)
b = np.float32(1.0)
c = 1.0

np.ma.min(a)
np.ma.min(b)
np.ma.min(c)
np.ma.min(A)
np.ma.min(B)
np.ma.min(A, axis=0)
np.ma.min(B, axis=0)
np.ma.min(C, axis=0)
np.ma.min(D, axis=0)
np.ma.min(A, keepdims=True)
np.ma.min(B, keepdims=True)
np.ma.min(C, keepdims=True)
np.ma.min(D, keepdims=True)
np.ma.min(A, fill_value=1)
np.ma.min(B, fill_value=1)
np.ma.min(C, fill_value=1)
np.ma.min(D, fill_value=1)
