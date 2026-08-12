from typing import Any, assert_type

import numpy as np
import numpy.typing as npt
from numpy.lib._arraysetops_impl import (
    UniqueAllResult,
    UniqueCountsResult,
    UniqueInverseResult,
)

type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Int1D = _Array1D[np.intp]
type _IntND = npt.NDArray[np.intp]

AR_b: npt.NDArray[np.bool]
AR_i8: npt.NDArray[np.int64]
AR_f8: npt.NDArray[np.float64]
AR_f8_1d: _Array1D[np.float64]
AR_M: npt.NDArray[np.datetime64]
AR_O: npt.NDArray[np.object_]

AR_LIKE_f8: list[float]

assert_type(np.ediff1d(AR_b), _Array1D[np.int8])
assert_type(np.ediff1d(AR_i8, to_end=[1, 2, 3]), _Array1D[np.int64])
assert_type(np.ediff1d(AR_M), _Array1D[np.timedelta64])
assert_type(np.ediff1d(AR_O), _Array1D[np.object_])
assert_type(np.ediff1d(AR_LIKE_f8, to_begin=[1, 1.5]), _Array1D[Any])

assert_type(np.intersect1d(AR_i8, AR_i8), _Array1D[np.int64])
# NOTE: Mypy incorrectly infers `ndarray[Any, Any]`, but pyright behaves correctly.
assert_type(np.intersect1d(AR_M, AR_M, assume_unique=True), _Array1D[np.datetime64])  # type: ignore[assert-type]
assert_type(np.intersect1d(AR_f8, AR_i8), _Array1D[Any])
assert_type(
    np.intersect1d(AR_f8, AR_f8, return_indices=True),
    tuple[_Array1D[np.float64], _Array1D[np.intp], _Array1D[np.intp]],
)

assert_type(np.setxor1d(AR_i8, AR_i8), _Array1D[np.int64])
# NOTE: Mypy incorrectly infers `ndarray[Any, Any]`, but pyright behaves correctly.
assert_type(np.setxor1d(AR_M, AR_M, assume_unique=True), _Array1D[np.datetime64])  # type: ignore[assert-type]
assert_type(np.setxor1d(AR_f8, AR_i8), _Array1D[Any])

assert_type(np.isin(AR_i8, AR_i8), npt.NDArray[np.bool])
assert_type(np.isin(AR_M, AR_M, assume_unique=True), npt.NDArray[np.bool])
assert_type(np.isin(AR_f8, AR_i8), npt.NDArray[np.bool])
assert_type(np.isin(AR_f8, AR_LIKE_f8, invert=True), npt.NDArray[np.bool])

assert_type(np.union1d(AR_i8, AR_i8), _Array1D[np.int64])
# NOTE: Mypy incorrectly infers `ndarray[Any, Any]`, but pyright behaves correctly.
assert_type(np.union1d(AR_M, AR_M), _Array1D[np.datetime64])  # type: ignore[assert-type]
assert_type(np.union1d(AR_f8, AR_i8), _Array1D[Any])

assert_type(np.setdiff1d(AR_i8, AR_i8), _Array1D[np.int64])
# NOTE: Mypy incorrectly infers `ndarray[Any, Any]`, but pyright behaves correctly.
assert_type(np.setdiff1d(AR_M, AR_M, assume_unique=True), npt.NDArray[np.datetime64])  # type: ignore[assert-type]
assert_type(np.setdiff1d(AR_f8, AR_i8), _Array1D[Any])

###

assert_type(np.unique(AR_f8), _Array1D[np.float64])
assert_type(np.unique(AR_f8_1d), _Array1D[np.float64])
assert_type(np.unique(AR_LIKE_f8), _Array1D[Any])
assert_type(np.unique(AR_f8, axis=0), npt.NDArray[np.float64])
assert_type(np.unique(AR_f8_1d, axis=0), _Array1D[np.float64])
assert_type(np.unique(AR_LIKE_f8, axis=0), npt.NDArray[Any])

assert_type(np.unique(AR_f8, return_index=True), tuple[_Array1D[np.float64], _Int1D])
assert_type(np.unique(AR_f8_1d, return_index=True), tuple[_Array1D[np.float64], _Int1D])
assert_type(np.unique(AR_LIKE_f8, return_index=True), tuple[_Array1D[Any], _Int1D])
assert_type(np.unique(AR_f8, return_index=True, axis=0), tuple[npt.NDArray[np.float64], _Int1D])
assert_type(np.unique(AR_f8_1d, return_index=True, axis=0), tuple[_Array1D[np.float64], _Int1D])
assert_type(np.unique(AR_LIKE_f8, return_index=True, axis=0), tuple[npt.NDArray[Any], _Int1D])

assert_type(np.unique(AR_f8, False, True), tuple[_Array1D[np.float64], _IntND])
assert_type(np.unique(AR_f8_1d, False, True), tuple[_Array1D[np.float64], _Int1D])
assert_type(np.unique(AR_LIKE_f8, False, True), tuple[_Array1D[Any], _IntND])
assert_type(np.unique(AR_f8, False, True, axis=0), tuple[npt.NDArray[np.float64], _Int1D])
assert_type(np.unique(AR_f8_1d, False, True, axis=0), tuple[_Array1D[np.float64], _Int1D])
assert_type(np.unique(AR_LIKE_f8, False, True, axis=0), tuple[npt.NDArray[Any], _Int1D])

assert_type(np.unique(AR_f8, return_inverse=True), tuple[_Array1D[np.float64], _IntND])
assert_type(np.unique(AR_f8_1d, return_inverse=True), tuple[_Array1D[np.float64], _Int1D])
assert_type(np.unique(AR_LIKE_f8, return_inverse=True), tuple[_Array1D[Any], _IntND])
assert_type(np.unique(AR_f8, return_inverse=True, axis=0), tuple[npt.NDArray[np.float64], _Int1D])
assert_type(np.unique(AR_f8_1d, return_inverse=True, axis=0), tuple[_Array1D[np.float64], _Int1D])
assert_type(np.unique(AR_LIKE_f8, return_inverse=True, axis=0), tuple[npt.NDArray[Any], _Int1D])

assert_type(np.unique(AR_f8, False, False, True), tuple[_Array1D[np.float64], _Int1D])
assert_type(np.unique(AR_f8_1d, False, False, True), tuple[_Array1D[np.float64], _Int1D])
assert_type(np.unique(AR_LIKE_f8, False, False, True), tuple[_Array1D[Any], _Int1D])
assert_type(np.unique(AR_f8, False, False, True, axis=0), tuple[npt.NDArray[np.float64], _Int1D])
assert_type(np.unique(AR_f8_1d, False, False, True, axis=0), tuple[_Array1D[np.float64], _Int1D])
assert_type(np.unique(AR_LIKE_f8, False, False, True, axis=0), tuple[npt.NDArray[Any], _Int1D])

assert_type(np.unique(AR_f8, return_counts=True), tuple[_Array1D[np.float64], _Int1D])
assert_type(np.unique(AR_f8_1d, return_counts=True), tuple[_Array1D[np.float64], _Int1D])
assert_type(np.unique(AR_LIKE_f8, return_counts=True), tuple[_Array1D[Any], _Int1D])
assert_type(np.unique(AR_f8, return_counts=True, axis=0), tuple[npt.NDArray[np.float64], _Int1D])
assert_type(np.unique(AR_f8_1d, return_counts=True, axis=0), tuple[_Array1D[np.float64], _Int1D])
assert_type(np.unique(AR_LIKE_f8, return_counts=True, axis=0), tuple[npt.NDArray[Any], _Int1D])

assert_type(np.unique(AR_f8, return_index=True, return_inverse=True), tuple[_Array1D[np.float64], _Int1D, _IntND])
assert_type(np.unique(AR_f8_1d, return_index=True, return_inverse=True), tuple[_Array1D[np.float64], _Int1D, _Int1D])
assert_type(np.unique(AR_LIKE_f8, return_index=True, return_inverse=True), tuple[_Array1D[Any], _Int1D, _IntND])
assert_type(np.unique(AR_f8, return_index=True, return_inverse=True, axis=0), tuple[npt.NDArray[np.float64], _Int1D, _Int1D])
assert_type(np.unique(AR_f8_1d, return_index=True, return_inverse=True, axis=0), tuple[_Array1D[np.float64], _Int1D, _Int1D])
assert_type(np.unique(AR_LIKE_f8, return_index=True, return_inverse=True, axis=0), tuple[npt.NDArray[Any], _Int1D, _Int1D])

assert_type(np.unique(AR_f8, return_index=True, return_counts=True), tuple[_Array1D[np.float64], _Int1D, _Int1D])
assert_type(np.unique(AR_f8_1d, return_index=True, return_counts=True), tuple[_Array1D[np.float64], _Int1D, _Int1D])
assert_type(np.unique(AR_LIKE_f8, return_index=True, return_counts=True), tuple[_Array1D[Any], _Int1D, _Int1D])
assert_type(np.unique(AR_f8, return_index=True, return_counts=True, axis=0), tuple[npt.NDArray[np.float64], _Int1D, _Int1D])
assert_type(np.unique(AR_f8_1d, return_index=True, return_counts=True, axis=0), tuple[_Array1D[np.float64], _Int1D, _Int1D])
assert_type(np.unique(AR_LIKE_f8, return_index=True, return_counts=True, axis=0), tuple[npt.NDArray[Any], _Int1D, _Int1D])

assert_type(np.unique(AR_f8, True, False, True), tuple[_Array1D[np.float64], _Int1D, _Int1D])
assert_type(np.unique(AR_f8_1d, True, False, True), tuple[_Array1D[np.float64], _Int1D, _Int1D])
assert_type(np.unique(AR_LIKE_f8, True, False, True), tuple[_Array1D[Any], _Int1D, _Int1D])
assert_type(np.unique(AR_f8, True, False, True, axis=0), tuple[npt.NDArray[np.float64], _Int1D, _Int1D])
assert_type(np.unique(AR_f8_1d, True, False, True, axis=0), tuple[_Array1D[np.float64], _Int1D, _Int1D])
assert_type(np.unique(AR_LIKE_f8, True, False, True, axis=0), tuple[npt.NDArray[Any], _Int1D, _Int1D])

assert_type(np.unique(AR_f8, False, True, True), tuple[_Array1D[np.float64], _IntND, _Int1D])
assert_type(np.unique(AR_f8_1d, False, True, True), tuple[_Array1D[np.float64], _Int1D, _Int1D])
assert_type(np.unique(AR_LIKE_f8, False, True, True), tuple[_Array1D[Any], _IntND, _Int1D])
assert_type(np.unique(AR_f8, False, True, True, axis=0), tuple[npt.NDArray[np.float64], _Int1D, _Int1D])
assert_type(np.unique(AR_f8_1d, False, True, True, axis=0), tuple[_Array1D[np.float64], _Int1D, _Int1D])
assert_type(np.unique(AR_LIKE_f8, False, True, True, axis=0), tuple[npt.NDArray[Any], _Int1D, _Int1D])

assert_type(np.unique(AR_f8, return_inverse=True, return_counts=True), tuple[_Array1D[np.float64], _IntND, _Int1D])
assert_type(np.unique(AR_f8_1d, return_inverse=True, return_counts=True), tuple[_Array1D[np.float64], _Int1D, _Int1D])
assert_type(np.unique(AR_LIKE_f8, return_inverse=True, return_counts=True), tuple[_Array1D[Any], _IntND, _Int1D])
assert_type(np.unique(AR_f8, return_inverse=True, return_counts=True, axis=0), tuple[npt.NDArray[np.float64], _Int1D, _Int1D])
assert_type(np.unique(AR_f8_1d, return_inverse=True, return_counts=True, axis=0), tuple[_Array1D[np.float64], _Int1D, _Int1D])
assert_type(np.unique(AR_LIKE_f8, return_inverse=True, return_counts=True, axis=0), tuple[npt.NDArray[Any], _Int1D, _Int1D])

assert_type(np.unique(AR_f8, True, True, True), tuple[_Array1D[np.float64], _Int1D, _IntND, _Int1D])
assert_type(np.unique(AR_f8_1d, True, True, True), tuple[_Array1D[np.float64], _Int1D, _Int1D, _Int1D])
assert_type(np.unique(AR_LIKE_f8, True, True, True), tuple[_Array1D[Any], _Int1D, _IntND, _Int1D])
assert_type(np.unique(AR_f8, True, True, True, axis=0), tuple[npt.NDArray[np.float64], _Int1D, _Int1D, _Int1D])
assert_type(np.unique(AR_f8_1d, True, True, True, axis=0), tuple[_Array1D[np.float64], _Int1D, _Int1D, _Int1D])
assert_type(np.unique(AR_LIKE_f8, True, True, True, axis=0), tuple[npt.NDArray[Any], _Int1D, _Int1D, _Int1D])

###

assert_type(np.unique_all(AR_f8), UniqueAllResult[np.float64])
assert_type(np.unique_all(AR_LIKE_f8), UniqueAllResult[Any])
assert_type(np.unique_counts(AR_f8), UniqueCountsResult[np.float64])
assert_type(np.unique_counts(AR_LIKE_f8), UniqueCountsResult[Any])
assert_type(np.unique_inverse(AR_f8), UniqueInverseResult[np.float64])
assert_type(np.unique_inverse(AR_LIKE_f8), UniqueInverseResult[Any])
assert_type(np.unique_values(AR_f8), _Array1D[np.float64])
assert_type(np.unique_values(AR_LIKE_f8), _Array1D[Any])
