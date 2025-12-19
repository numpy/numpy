from typing import Any, Literal as L, NamedTuple, SupportsIndex, TypeVar, overload

import numpy as np
from numpy._typing import (
    ArrayLike,
    NDArray,
    _ArrayLike,
    _ArrayLikeBool_co,
    _ArrayLikeNumber_co,
)

__all__ = [
    "ediff1d",
    "intersect1d",
    "isin",
    "setdiff1d",
    "setxor1d",
    "union1d",
    "unique",
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
]

# Explicitly set all allowed values to prevent accidental castings to
# abstract dtypes (their common super-type).
# Only relevant if two or more arguments are parametrized, (e.g. `setdiff1d`)
# which could result in, for example, `int64` and `float64`producing a
# `number[_64Bit]` array
_AnyScalarT = TypeVar(
    "_AnyScalarT",
    np.bool,
    np.int8, np.int16, np.int32, np.int64, np.intp,
    np.uint8, np.uint16, np.uint32, np.uint64, np.uintp,
    np.float16, np.float32, np.float64, np.longdouble,
    np.complex64, np.complex128, np.clongdouble,
    np.timedelta64, np.datetime64,
    np.bytes_, np.str_, np.void, np.object_,
    np.integer, np.floating, np.complexfloating, np.character,
)  # fmt: skip

type _NumericScalar = np.number | np.timedelta64 | np.object_
type _IntArray = NDArray[np.intp]

###

class UniqueAllResult[ScalarT: np.generic](NamedTuple):
    values: NDArray[ScalarT]
    indices: _IntArray
    inverse_indices: _IntArray
    counts: _IntArray

class UniqueCountsResult[ScalarT: np.generic](NamedTuple):
    values: NDArray[ScalarT]
    counts: _IntArray

class UniqueInverseResult[ScalarT: np.generic](NamedTuple):
    values: NDArray[ScalarT]
    inverse_indices: _IntArray

#
@overload
def ediff1d(
    ary: _ArrayLikeBool_co,
    to_end: ArrayLike | None = None,
    to_begin: ArrayLike | None = None,
) -> NDArray[np.int8]: ...
@overload
def ediff1d[NumericT: _NumericScalar](
    ary: _ArrayLike[NumericT],
    to_end: ArrayLike | None = None,
    to_begin: ArrayLike | None = None,
) -> NDArray[NumericT]: ...
@overload
def ediff1d(
    ary: _ArrayLike[np.datetime64[Any]],
    to_end: ArrayLike | None = None,
    to_begin: ArrayLike | None = None,
) -> NDArray[np.timedelta64]: ...
@overload
def ediff1d(
    ary: _ArrayLikeNumber_co,
    to_end: ArrayLike | None = None,
    to_begin: ArrayLike | None = None,
) -> np.ndarray: ...

#
@overload  # known scalar-type, FFF
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[False] = False,
    return_inverse: L[False] = False,
    return_counts: L[False] = False,
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> NDArray[ScalarT]: ...
@overload  # unknown scalar-type, FFF
def unique(
    ar: ArrayLike,
    return_index: L[False] = False,
    return_inverse: L[False] = False,
    return_counts: L[False] = False,
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> np.ndarray: ...
@overload  # known scalar-type, TFF
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[True],
    return_inverse: L[False] = False,
    return_counts: L[False] = False,
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[ScalarT], _IntArray]: ...
@overload  # unknown scalar-type, TFF
def unique(
    ar: ArrayLike,
    return_index: L[True],
    return_inverse: L[False] = False,
    return_counts: L[False] = False,
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray, _IntArray]: ...
@overload  # known scalar-type, FTF (positional)
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[False],
    return_inverse: L[True],
    return_counts: L[False] = False,
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[ScalarT], _IntArray]: ...
@overload  # known scalar-type, FTF (keyword)
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[False] = False,
    *,
    return_inverse: L[True],
    return_counts: L[False] = False,
    axis: SupportsIndex | None = None,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[ScalarT], _IntArray]: ...
@overload  # unknown scalar-type, FTF (positional)
def unique(
    ar: ArrayLike,
    return_index: L[False],
    return_inverse: L[True],
    return_counts: L[False] = False,
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray, _IntArray]: ...
@overload  # unknown scalar-type, FTF (keyword)
def unique(
    ar: ArrayLike,
    return_index: L[False] = False,
    *,
    return_inverse: L[True],
    return_counts: L[False] = False,
    axis: SupportsIndex | None = None,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray, _IntArray]: ...
@overload  # known scalar-type, FFT (positional)
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[False],
    return_inverse: L[False],
    return_counts: L[True],
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[ScalarT], _IntArray]: ...
@overload  # known scalar-type, FFT (keyword)
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[False] = False,
    return_inverse: L[False] = False,
    *,
    return_counts: L[True],
    axis: SupportsIndex | None = None,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[ScalarT], _IntArray]: ...
@overload  # unknown scalar-type, FFT (positional)
def unique(
    ar: ArrayLike,
    return_index: L[False],
    return_inverse: L[False],
    return_counts: L[True],
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray, _IntArray]: ...
@overload  # unknown scalar-type, FFT (keyword)
def unique(
    ar: ArrayLike,
    return_index: L[False] = False,
    return_inverse: L[False] = False,
    *,
    return_counts: L[True],
    axis: SupportsIndex | None = None,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray, _IntArray]: ...
@overload  # known scalar-type, TTF
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[True],
    return_inverse: L[True],
    return_counts: L[False] = False,
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[ScalarT], _IntArray, _IntArray]: ...
@overload  # unknown scalar-type, TTF
def unique(
    ar: ArrayLike,
    return_index: L[True],
    return_inverse: L[True],
    return_counts: L[False] = False,
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray, _IntArray, _IntArray]: ...
@overload  # known scalar-type, TFT (positional)
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[True],
    return_inverse: L[False],
    return_counts: L[True],
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[ScalarT], _IntArray, _IntArray]: ...
@overload  # known scalar-type, TFT (keyword)
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[True],
    return_inverse: L[False] = False,
    *,
    return_counts: L[True],
    axis: SupportsIndex | None = None,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[ScalarT], _IntArray, _IntArray]: ...
@overload  # unknown scalar-type, TFT (positional)
def unique(
    ar: ArrayLike,
    return_index: L[True],
    return_inverse: L[False],
    return_counts: L[True],
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray, _IntArray, _IntArray]: ...
@overload  # unknown scalar-type, TFT (keyword)
def unique(
    ar: ArrayLike,
    return_index: L[True],
    return_inverse: L[False] = False,
    *,
    return_counts: L[True],
    axis: SupportsIndex | None = None,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray, _IntArray, _IntArray]: ...
@overload  # known scalar-type, FTT (positional)
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[False],
    return_inverse: L[True],
    return_counts: L[True],
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[ScalarT], _IntArray, _IntArray]: ...
@overload  # known scalar-type, FTT (keyword)
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[False] = False,
    *,
    return_inverse: L[True],
    return_counts: L[True],
    axis: SupportsIndex | None = None,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[ScalarT], _IntArray, _IntArray]: ...
@overload  # unknown scalar-type, FTT (positional)
def unique(
    ar: ArrayLike,
    return_index: L[False],
    return_inverse: L[True],
    return_counts: L[True],
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray, _IntArray, _IntArray]: ...
@overload  # unknown scalar-type, FTT (keyword)
def unique(
    ar: ArrayLike,
    return_index: L[False] = False,
    *,
    return_inverse: L[True],
    return_counts: L[True],
    axis: SupportsIndex | None = None,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray, _IntArray, _IntArray]: ...
@overload  # known scalar-type, TTT
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[True],
    return_inverse: L[True],
    return_counts: L[True],
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[ScalarT], _IntArray, _IntArray, _IntArray]: ...
@overload  # unknown scalar-type, TTT
def unique(
    ar: ArrayLike,
    return_index: L[True],
    return_inverse: L[True],
    return_counts: L[True],
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray, _IntArray, _IntArray, _IntArray]: ...

#
@overload
def unique_all[ScalarT: np.generic](x: _ArrayLike[ScalarT]) -> UniqueAllResult[ScalarT]: ...
@overload
def unique_all(x: ArrayLike) -> UniqueAllResult[Any]: ...

#
@overload
def unique_counts[ScalarT: np.generic](x: _ArrayLike[ScalarT]) -> UniqueCountsResult[ScalarT]: ...
@overload
def unique_counts(x: ArrayLike) -> UniqueCountsResult[Any]: ...

#
@overload
def unique_inverse[ScalarT: np.generic](x: _ArrayLike[ScalarT]) -> UniqueInverseResult[ScalarT]: ...
@overload
def unique_inverse(x: ArrayLike) -> UniqueInverseResult[Any]: ...

#
@overload
def unique_values[ScalarT: np.generic](x: _ArrayLike[ScalarT]) -> NDArray[ScalarT]: ...
@overload
def unique_values(x: ArrayLike) -> np.ndarray: ...

#
@overload  # known scalar-type, return_indices=False (default)
def intersect1d(
    ar1: _ArrayLike[_AnyScalarT],
    ar2: _ArrayLike[_AnyScalarT],
    assume_unique: bool = False,
    return_indices: L[False] = False,
) -> NDArray[_AnyScalarT]: ...
@overload  # known scalar-type, return_indices=True (positional)
def intersect1d(
    ar1: _ArrayLike[_AnyScalarT],
    ar2: _ArrayLike[_AnyScalarT],
    assume_unique: bool,
    return_indices: L[True],
) -> tuple[NDArray[_AnyScalarT], _IntArray, _IntArray]: ...
@overload  # known scalar-type, return_indices=True (keyword)
def intersect1d(
    ar1: _ArrayLike[_AnyScalarT],
    ar2: _ArrayLike[_AnyScalarT],
    assume_unique: bool = False,
    *,
    return_indices: L[True],
) -> tuple[NDArray[_AnyScalarT], _IntArray, _IntArray]: ...
@overload  # unknown scalar-type, return_indices=False (default)
def intersect1d(
    ar1: ArrayLike,
    ar2: ArrayLike,
    assume_unique: bool = False,
    return_indices: L[False] = False,
) -> np.ndarray: ...
@overload  # unknown scalar-type, return_indices=True (positional)
def intersect1d(
    ar1: ArrayLike,
    ar2: ArrayLike,
    assume_unique: bool,
    return_indices: L[True],
) -> tuple[np.ndarray, _IntArray, _IntArray]: ...
@overload  # unknown scalar-type, return_indices=True (keyword)
def intersect1d(
    ar1: ArrayLike,
    ar2: ArrayLike,
    assume_unique: bool = False,
    *,
    return_indices: L[True],
) -> tuple[np.ndarray, _IntArray, _IntArray]: ...

#
@overload
def setxor1d(ar1: _ArrayLike[_AnyScalarT], ar2: _ArrayLike[_AnyScalarT], assume_unique: bool = False) -> NDArray[_AnyScalarT]: ...
@overload
def setxor1d(ar1: ArrayLike, ar2: ArrayLike, assume_unique: bool = False) -> np.ndarray: ...

#
@overload
def union1d(ar1: _ArrayLike[_AnyScalarT], ar2: _ArrayLike[_AnyScalarT]) -> NDArray[_AnyScalarT]: ...
@overload
def union1d(ar1: ArrayLike, ar2: ArrayLike) -> np.ndarray: ...

#
@overload
def setdiff1d(ar1: _ArrayLike[_AnyScalarT], ar2: _ArrayLike[_AnyScalarT], assume_unique: bool = False) -> NDArray[_AnyScalarT]: ...
@overload
def setdiff1d(ar1: ArrayLike, ar2: ArrayLike, assume_unique: bool = False) -> np.ndarray: ...

#
def isin(
    element: ArrayLike,
    test_elements: ArrayLike,
    assume_unique: bool = False,
    invert: bool = False,
    *,
    kind: L["sort", "table"] | None = None,
) -> NDArray[np.bool]: ...
