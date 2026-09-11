from collections.abc import Mapping
from typing import Any, SupportsIndex, assert_type

import numpy as np
import numpy.typing as npt

###

def mode_func(
    ar: npt.NDArray[np.number],
    width: tuple[int, int],
    iaxis: SupportsIndex,
    kwargs: Mapping[str, Any],
) -> None: ...

AR_i8: npt.NDArray[np.int64]
AR_f8: npt.NDArray[np.float64]
AR_f8_1d: np.ndarray[tuple[int], np.dtype[np.float64]]
AR_f8_2d: np.ndarray[tuple[int, int], np.dtype[np.float64]]
AR_LIKE: list[int]

_py_b_1d: list[bool]
_py_i_1d: list[int]
_py_f_1d: list[float]
_py_c_1d: list[complex]
_py_b_2d: list[list[bool]]
_py_i_2d: list[list[int]]
_py_f_2d: list[list[float]]
_py_c_2d: list[list[complex]]
_py_s_1d: list[str]

###

assert_type(np.pad(AR_i8, (2, 3), "constant", constant_values=1), npt.NDArray[np.int64])
assert_type(np.pad(AR_i8, (2, 3), "linear_ramp", end_values=1), npt.NDArray[np.int64])
assert_type(np.pad(AR_i8, (2, 3), "mean", stat_length=2), npt.NDArray[np.int64])
assert_type(np.pad(AR_i8, (2, 3), "reflect", reflect_type="odd"), npt.NDArray[np.int64])
assert_type(np.pad(AR_i8, (2, 3), "edge"), npt.NDArray[np.int64])

assert_type(np.pad(AR_f8, (2, 3), mode_func), npt.NDArray[np.float64])

assert_type(np.pad(AR_i8, {-1: (2, 3)}), npt.NDArray[np.int64])
assert_type(np.pad(AR_i8, {-2: 4}), npt.NDArray[np.int64])
pad_width: dict[int, int | tuple[int, int]] = {-1: (2, 3), -2: 4}
assert_type(np.pad(AR_i8, pad_width), npt.NDArray[np.int64])

assert_type(np.pad(AR_f8_1d, (2, 3)), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(np.pad(AR_f8_2d, (2, 3)), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(np.pad(_py_b_1d, 1), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.pad(_py_i_1d, 1), np.ndarray[tuple[int], np.dtype[np.int_]])
assert_type(np.pad(_py_f_1d, 1), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(np.pad(_py_c_1d, 1), np.ndarray[tuple[int], np.dtype[np.complex128]])
assert_type(np.pad(_py_b_2d, 1), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.pad(_py_i_2d, 1), np.ndarray[tuple[int, int], np.dtype[np.int_]])
assert_type(np.pad(_py_f_2d, 1), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(np.pad(_py_c_2d, 1), np.ndarray[tuple[int, int], np.dtype[np.complex128]])
assert_type(np.pad(_py_s_1d, 1), npt.NDArray[Any])
