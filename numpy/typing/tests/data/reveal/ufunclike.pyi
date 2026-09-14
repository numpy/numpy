from typing import Any, assert_type

import numpy as np
import numpy.typing as npt
from numpy._typing import _SupportsArray

AR_LIKE_b: list[bool]
AR_LIKE_u: list[np.uint32]
AR_LIKE_i: list[int]
AR_LIKE_f: list[float]
AR_LIKE_f_2d: list[list[float]]
AR_LIKE_O: list[np.object_]

AR_U: npt.NDArray[np.str_]

AR_f8_1d: np.ndarray[tuple[int], np.dtype[np.float64]]
AR_f8_2d: np.ndarray[tuple[int, int], np.dtype[np.float64]]

_to_f8: _SupportsArray[np.dtype[np.float64]]

###

assert_type(np.fix(AR_LIKE_b), npt.NDArray[np.floating])  # type: ignore[deprecated]
assert_type(np.fix(AR_LIKE_u), npt.NDArray[np.floating])  # type: ignore[deprecated]
assert_type(np.fix(AR_LIKE_i), npt.NDArray[np.floating])  # type: ignore[deprecated]
assert_type(np.fix(AR_LIKE_f), npt.NDArray[np.floating])  # type: ignore[deprecated]
assert_type(np.fix(AR_LIKE_O), npt.NDArray[np.object_])  # type: ignore[deprecated]
assert_type(np.fix(AR_LIKE_f, out=AR_U), npt.NDArray[np.str_])  # type: ignore[deprecated]

assert_type(np.isposinf(0.0), np.bool)
assert_type(np.isposinf(AR_f8_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.isposinf(AR_LIKE_f), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.isposinf(AR_LIKE_f_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.isposinf(_to_f8), npt.NDArray[np.bool] | Any)
assert_type(np.isposinf(AR_LIKE_f, out=AR_U), npt.NDArray[np.str_])

assert_type(np.isneginf(AR_LIKE_b), npt.NDArray[np.bool])
assert_type(np.isneginf(AR_LIKE_u), npt.NDArray[np.bool])
assert_type(np.isneginf(AR_LIKE_i), npt.NDArray[np.bool])
assert_type(np.isneginf(AR_LIKE_f), npt.NDArray[np.bool])
assert_type(np.isneginf(AR_LIKE_f, out=AR_U), npt.NDArray[np.str_])
assert_type(np.isneginf(AR_f8_1d), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(np.isneginf(AR_f8_2d), np.ndarray[tuple[int, int], np.dtype[np.bool]])
