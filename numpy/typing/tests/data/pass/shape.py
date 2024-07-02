from typing import Any, cast, NewType, TypeVar

import numpy as np
from typing_extensions import assert_type

Time = NewType("Time", int)
Series = NewType("Series", int)
Ax1 = TypeVar("Ax1", bound=int)
Ax2 = TypeVar("Ax2", bound=int)

arr: np.ndarray[tuple[Time, Series], Any] = np.arange(4.0).reshape((2, 2))

def check_shape(a: np.ndarray[tuple[int, ...], Any]) -> bool: return True

check_shape(arr)

def transpose(
    a: np.ndarray[tuple[Ax1, Ax2], Any]
) -> np.ndarray[tuple[Ax2, Ax1], Any]:
    return cast(np.ndarray[tuple[Ax2, Ax1], Any], np.transpose(a))

assert_type(arr.shape, tuple[Time, Series])
assert_type(transpose(arr).shape, tuple[Series, Time])
