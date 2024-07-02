from __future__ import annotations

from typing import NewType, TypeVar

import numpy as np
import numpy.typing as npt
from typing_extensions import assert_type

# if sys.version_info >= (3, 11):
DType = TypeVar("DType", bound=np.generic)

# Check that typevartuple in alias is packed correctly
Length = NewType("Length", int)
Width = NewType("Width", int)
arr: npt.Array[Length, Width, np.int8] = np.array([[0]])
assert_type(arr, np.ndarray[tuple[Length, Width], np.dtype[np.int8]])

# Check that typevartuple in alias is unpacked correctly
M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)
T = TypeVar("T", bound=np.generic)


def mult(vec: npt.Array[N, T], mat: npt.Array[M, N, T]) -> npt.Array[M, T]:
    return mat @ vec  # type: ignore


arr2: np.ndarray[tuple[Width], np.dtype[np.int8]] = np.array([0])
assert_type(mult(arr2, arr), np.ndarray[tuple[Length], np.dtype[np.int8]])


# Check that shape works
def return_shp(a: "npt.Array[M, N, DType]") -> tuple[M, N]:
    return a.shape

shp = return_shp(arr)
assert_type(shp, tuple[Length, Width])
