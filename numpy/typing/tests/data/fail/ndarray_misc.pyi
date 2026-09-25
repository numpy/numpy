"""
Tests for miscellaneous (non-magic) ``np.ndarray``/``np.generic`` methods.

More extensive tests are performed for the methods'
function-based counterpart in `../from_numeric.py`.

"""
from typing import Any, Never

import numpy as np
import numpy.typing as npt

f8: np.float64

_b_nd: npt.NDArray[np.bool]
_i8_nd: npt.NDArray[np.int64]
_i8_1d: np.ndarray[tuple[int], np.dtype[np.int64]]
_f8_nd: npt.NDArray[np.float64]
_c16_nd: npt.NDArray[np.complex128]
_M_nd: npt.NDArray[np.datetime64]
_m_nd: npt.NDArray[np.timedelta64]
_U_nd: npt.NDArray[np.str_]
_T_nd: np.ndarray[tuple[Any, ...], np.dtypes.StringDType]

###

ctypes_obj = _f8_nd.ctypes

f8.argpartition(0)  # type: ignore[attr-defined]
f8.partition(0)  # type: ignore[attr-defined]
f8.dot(1)  # type: ignore[attr-defined]

# NOTE: The following functions return `Never`, causing mypy to stop analysis at that
# point, which we circumvent by wrapping them in a function.

def f8_diagonal(x: np.float64) -> Never:
    return x.diagonal()  # type: ignore[misc]

def f8_nonzero(x: np.float64) -> Never:
    return x.nonzero()  # type: ignore[misc]

def f8_setfield(x: np.float64) -> Never:
    return x.setfield(2, np.float64)  # type: ignore[misc]

def f8_sort(x: np.float64) -> Never:
    return x.sort()  # type: ignore[misc]

def f8_trace(x: np.float64) -> Never:
    return x.trace()  # type: ignore[misc]

_f8_nd.__array_finalize__(object())  # type: ignore[arg-type]

_i8_1d.__bool__()  # type: ignore[misc]

_i8_1d.__int__()  # type: ignore[misc]
_c16_nd.__int__()  # type: ignore[misc]
_M_nd.__int__()  # type: ignore[misc]
_m_nd.__int__()  # type: ignore[misc]

_i8_1d.__float__()  # type: ignore[misc]
_c16_nd.__float__()  # type: ignore[misc]
_M_nd.__float__()  # type: ignore[misc]
_m_nd.__float__()  # type: ignore[misc]

_i8_1d.__complex__()  # type: ignore[misc]
_M_nd.__complex__()  # type: ignore[misc]
_m_nd.__complex__()  # type: ignore[misc]
_U_nd.__complex__()  # type: ignore[misc]
_T_nd.__complex__()  # type: ignore[misc]

_i8_1d.__index__()  # type: ignore[misc]
_b_nd.__index__()  # type: ignore[misc]
_f8_nd.__index__()  # type: ignore[misc]

_f8_nd[1.5]  # type: ignore[call-overload]
_f8_nd["field_a"]  # type: ignore[call-overload]
_f8_nd[["field_a", "field_b"]]  # type: ignore[index]
