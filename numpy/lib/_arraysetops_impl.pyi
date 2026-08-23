from _typeshed import Incomplete
from collections.abc import Sequence
from typing import Any, Literal as L, NamedTuple, SupportsIndex, TypeVar, overload

import numpy as np
from numpy._typing import (
    ArrayLike,
    NDArray,
    _ArrayLike,
    _ArrayLikeBool_co,
    _ArrayLikeNumber_co,
    _Shape,
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

type _Array0D[ScalarT: np.generic] = np.ndarray[tuple[()], np.dtype[ScalarT]]
type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]
type _Array3D[ScalarT: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[ScalarT]]

type _IntND = NDArray[np.intp]
type _Int1D = _Array1D[np.intp]

type _IntersectResult[ScalarT: np.generic] = tuple[_Array1D[ScalarT], _Int1D, _Int1D]

###

class UniqueAllResult[ScalarT: np.generic](NamedTuple):
    values: _Array1D[ScalarT]
    indices: _Int1D
    inverse_indices: _IntND
    counts: _Int1D

class UniqueCountsResult[ScalarT: np.generic](NamedTuple):
    values: _Array1D[ScalarT]
    counts: _Int1D

class UniqueInverseResult[ScalarT: np.generic](NamedTuple):
    values: _Array1D[ScalarT]
    inverse_indices: NDArray[np.intp]

# keep in sync with `ma.extras.ediff1d`
@overload
def ediff1d(
    ary: _ArrayLikeBool_co,
    to_end: ArrayLike | None = None,
    to_begin: ArrayLike | None = None,
) -> _Array1D[np.int8]: ...
@overload
def ediff1d[NumericT: _NumericScalar](
    ary: _ArrayLike[NumericT],
    to_end: ArrayLike | None = None,
    to_begin: ArrayLike | None = None,
) -> _Array1D[NumericT]: ...
@overload
def ediff1d(
    ary: _ArrayLike[np.datetime64[Any]],
    to_end: ArrayLike | None = None,
    to_begin: ArrayLike | None = None,
) -> _Array1D[np.timedelta64]: ...
@overload
def ediff1d(
    ary: _ArrayLikeNumber_co,
    to_end: ArrayLike | None = None,
    to_begin: ArrayLike | None = None,
) -> _Array1D[Incomplete]: ...

#
@overload  # known array, FFF, axis=<given>
def unique[ShapeT: _Shape, DTypeT: np.dtype](
    ar: np.ndarray[ShapeT, DTypeT],
    return_index: L[False] = False,
    return_inverse: L[False] = False,
    return_counts: L[False] = False,
    *,
    axis: SupportsIndex,
    equal_nan: bool = True,
    sorted: bool = True,
) -> np.ndarray[ShapeT, DTypeT]: ...
@overload  # known scalar-type, FFF, axis=None  (default)
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[False] = False,
    return_inverse: L[False] = False,
    return_counts: L[False] = False,
    axis: None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> _Array1D[ScalarT]: ...
@overload  # known scalar-type, FFF, axis=<given>
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[False] = False,
    return_inverse: L[False] = False,
    return_counts: L[False] = False,
    *,
    axis: SupportsIndex,
    equal_nan: bool = True,
    sorted: bool = True,
) -> NDArray[ScalarT]: ...
@overload  # unknown scalar-type, FFF, axis=None (default)
def unique(
    ar: ArrayLike,
    return_index: L[False] = False,
    return_inverse: L[False] = False,
    return_counts: L[False] = False,
    axis: None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> _Array1D[Any]: ...
@overload  # unknown scalar-type, FFF, axis=<given>
def unique(
    ar: ArrayLike,
    return_index: L[False] = False,
    return_inverse: L[False] = False,
    return_counts: L[False] = False,
    *,
    axis: SupportsIndex,
    equal_nan: bool = True,
    sorted: bool = True,
) -> NDArray[Any]: ...
@overload  # known array, TFF, axis=<given>
def unique[ShapeT: _Shape, DTypeT: np.dtype](
    ar: np.ndarray[ShapeT, DTypeT],
    return_index: L[True],
    return_inverse: L[False] = False,
    return_counts: L[False] = False,
    *,
    axis: SupportsIndex,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray[ShapeT, DTypeT], _Int1D]: ...
@overload  # known scalar-type, TFF, axis=None (default)
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[True],
    return_inverse: L[False] = False,
    return_counts: L[False] = False,
    axis: None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_Array1D[ScalarT], _Int1D]: ...
@overload  # known scalar-type, TFF, axis=<given>
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[True],
    return_inverse: L[False] = False,
    return_counts: L[False] = False,
    *,
    axis: SupportsIndex,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[ScalarT], _Int1D]: ...
@overload  # unknown scalar-type, TFF, axis=None (default)
def unique(
    ar: ArrayLike,
    return_index: L[True],
    return_inverse: L[False] = False,
    return_counts: L[False] = False,
    axis: None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_Array1D[Any], _Int1D]: ...
@overload  # unknown scalar-type, TFF, axis=<given>
def unique(
    ar: ArrayLike,
    return_index: L[True],
    return_inverse: L[False] = False,
    return_counts: L[False] = False,
    *,
    axis: SupportsIndex,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[Any], _Int1D]: ...
@overload  # known array, FTF (positional), axis=None (default)
def unique[ShapeT: _Shape, DTypeT: np.dtype](
    ar: np.ndarray[ShapeT, DTypeT],
    return_index: L[False],
    return_inverse: L[True],
    return_counts: L[False] = False,
    axis: None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray[tuple[int], DTypeT], np.ndarray[ShapeT, np.dtype[np.intp]]]: ...
@overload  # known array, FTF (positional), axis=<given>
def unique[ShapeT: _Shape, DTypeT: np.dtype](
    ar: np.ndarray[ShapeT, DTypeT],
    return_index: L[False],
    return_inverse: L[True],
    return_counts: L[False] = False,
    *,
    axis: SupportsIndex,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray[ShapeT, DTypeT], _Int1D]: ...
@overload  # known scalar-type, FTF (positional), axis=None (default)
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[False],
    return_inverse: L[True],
    return_counts: L[False] = False,
    axis: None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_Array1D[ScalarT], _IntND]: ...
@overload  # known scalar-type, FTF (positional), axis=<given>
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[False],
    return_inverse: L[True],
    return_counts: L[False] = False,
    *,
    axis: SupportsIndex,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[ScalarT], _Int1D]: ...
@overload  # unknown scalar-type, FTF (positional), axis=None (default)
def unique(
    ar: ArrayLike,
    return_index: L[False],
    return_inverse: L[True],
    return_counts: L[False] = False,
    axis: None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_Array1D[Any], _IntND]: ...
@overload  # unknown scalar-type, FTF (positional), axis=<given>
def unique(
    ar: ArrayLike,
    return_index: L[False],
    return_inverse: L[True],
    return_counts: L[False] = False,
    *,
    axis: SupportsIndex,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[Any], _Int1D]: ...
@overload  # known array, FTF (keyword), axis=None (default)
def unique[ShapeT: _Shape, DTypeT: np.dtype](
    ar: np.ndarray[ShapeT, DTypeT],
    return_index: L[False] = False,
    *,
    return_inverse: L[True],
    return_counts: L[False] = False,
    axis: None = None,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray[tuple[int], DTypeT], np.ndarray[ShapeT, np.dtype[np.intp]]]: ...
@overload  # known array, FTF (keyword), axis=<given>
def unique[ShapeT: _Shape, DTypeT: np.dtype](
    ar: np.ndarray[ShapeT, DTypeT],
    return_index: L[False] = False,
    *,
    return_inverse: L[True],
    return_counts: L[False] = False,
    axis: SupportsIndex,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray[ShapeT, DTypeT], _Int1D]: ...
@overload  # known scalar-type, FTF (keyword), axis=None (default)
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[False] = False,
    *,
    return_inverse: L[True],
    return_counts: L[False] = False,
    axis: None = None,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_Array1D[ScalarT], _IntND]: ...
@overload  # known scalar-type, FTF (keyword), axis=<given>
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[False] = False,
    *,
    return_inverse: L[True],
    return_counts: L[False] = False,
    axis: SupportsIndex,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[ScalarT], _Int1D]: ...
@overload  # unknown scalar-type, FTF (keyword), axis=None (default)
def unique(
    ar: ArrayLike,
    return_index: L[False] = False,
    *,
    return_inverse: L[True],
    return_counts: L[False] = False,
    axis: None = None,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_Array1D[Any], _IntND]: ...
@overload  # unknown scalar-type, FTF (keyword), axis=<given>
def unique(
    ar: ArrayLike,
    return_index: L[False] = False,
    *,
    return_inverse: L[True],
    return_counts: L[False] = False,
    axis: SupportsIndex,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[Any], _Int1D]: ...
@overload  # known array, FFT (positional), axis=<given>
def unique[ShapeT: _Shape, DTypeT: np.dtype](
    ar: np.ndarray[ShapeT, DTypeT],
    return_index: L[False],
    return_inverse: L[False],
    return_counts: L[True],
    axis: SupportsIndex,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray[ShapeT, DTypeT], _Int1D]: ...
@overload  # known scalar-type, FFT (positional), axis=None (default)
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[False],
    return_inverse: L[False],
    return_counts: L[True],
    axis: None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_Array1D[ScalarT], _Int1D]: ...
@overload  # known scalar-type, FFT (positional), axis=<given>
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[False],
    return_inverse: L[False],
    return_counts: L[True],
    axis: SupportsIndex,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[ScalarT], _Int1D]: ...
@overload  # unknown scalar-type, FFT (positional), axis=None (default)
def unique(
    ar: ArrayLike,
    return_index: L[False],
    return_inverse: L[False],
    return_counts: L[True],
    axis: None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_Array1D[Any], _Int1D]: ...
@overload  # unknown scalar-type, FFT (positional), axis=<given>
def unique(
    ar: ArrayLike,
    return_index: L[False],
    return_inverse: L[False],
    return_counts: L[True],
    axis: SupportsIndex,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[Any], _Int1D]: ...
@overload  # known array, FFT (keyword), axis=<given>
def unique[ShapeT: _Shape, DTypeT: np.dtype](
    ar: np.ndarray[ShapeT, DTypeT],
    return_index: L[False] = False,
    return_inverse: L[False] = False,
    *,
    return_counts: L[True],
    axis: SupportsIndex,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray[ShapeT, DTypeT], _Int1D]: ...
@overload  # known scalar-type, FFT (keyword), axis=None (default)
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[False] = False,
    return_inverse: L[False] = False,
    *,
    return_counts: L[True],
    axis: None = None,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_Array1D[ScalarT], _Int1D]: ...
@overload  # known scalar-type, FFT (keyword), axis=<given>
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[False] = False,
    return_inverse: L[False] = False,
    *,
    return_counts: L[True],
    axis: SupportsIndex,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[ScalarT], _Int1D]: ...
@overload  # unknown scalar-type, FFT (keyword), axis=None (default)
def unique(
    ar: ArrayLike,
    return_index: L[False] = False,
    return_inverse: L[False] = False,
    *,
    return_counts: L[True],
    axis: None = None,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_Array1D[Any], _Int1D]: ...
@overload  # unknown scalar-type, FFT (keyword), axis=<given>
def unique(
    ar: ArrayLike,
    return_index: L[False] = False,
    return_inverse: L[False] = False,
    *,
    return_counts: L[True],
    axis: SupportsIndex,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[Any], _Int1D]: ...
@overload  # known array, TTF, axis=None (default)
def unique[ShapeT: _Shape, DTypeT: np.dtype](
    ar: np.ndarray[ShapeT, DTypeT],
    return_index: L[True],
    return_inverse: L[True],
    return_counts: L[False] = False,
    axis: None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray[tuple[int], DTypeT], _Int1D, np.ndarray[ShapeT, np.dtype[np.intp]]]: ...
@overload  # known array, TTF, axis=<given>
def unique[ShapeT: _Shape, DTypeT: np.dtype](
    ar: np.ndarray[ShapeT, DTypeT],
    return_index: L[True],
    return_inverse: L[True],
    return_counts: L[False] = False,
    *,
    axis: SupportsIndex,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray[ShapeT, DTypeT], _Int1D, _Int1D]: ...
@overload  # known scalar-type, TTF, axis=None (default)
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[True],
    return_inverse: L[True],
    return_counts: L[False] = False,
    axis: None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_Array1D[ScalarT], _Int1D, _IntND]: ...
@overload  # known scalar-type, TTF, axis=<given>
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[True],
    return_inverse: L[True],
    return_counts: L[False] = False,
    *,
    axis: SupportsIndex,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[ScalarT], _Int1D, _Int1D]: ...
@overload  # unknown scalar-type, TTF, axis=None (default)
def unique(
    ar: ArrayLike,
    return_index: L[True],
    return_inverse: L[True],
    return_counts: L[False] = False,
    axis: None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_Array1D[Any], _Int1D, _IntND]: ...
@overload  # unknown scalar-type, TTF, axis=<given>
def unique(
    ar: ArrayLike,
    return_index: L[True],
    return_inverse: L[True],
    return_counts: L[False] = False,
    *,
    axis: SupportsIndex,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[Any], _Int1D, _Int1D]: ...
@overload  # known array, TFT (positional), axis=<given>
def unique[ShapeT: _Shape, DTypeT: np.dtype](
    ar: np.ndarray[ShapeT, DTypeT],
    return_index: L[True],
    return_inverse: L[False],
    return_counts: L[True],
    axis: SupportsIndex,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray[ShapeT, DTypeT], _Int1D, _Int1D]: ...
@overload  # known scalar-type, TFT (positional), axis=None (default)
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[True],
    return_inverse: L[False],
    return_counts: L[True],
    axis: None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_Array1D[ScalarT], _Int1D, _Int1D]: ...
@overload  # known scalar-type, TFT (positional), axis=<given>
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[True],
    return_inverse: L[False],
    return_counts: L[True],
    axis: SupportsIndex,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[ScalarT], _Int1D, _Int1D]: ...
@overload  # unknown scalar-type, TFT (positional), axis=None (default)
def unique(
    ar: ArrayLike,
    return_index: L[True],
    return_inverse: L[False],
    return_counts: L[True],
    axis: None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_Array1D[Any], _Int1D, _Int1D]: ...
@overload  # unknown scalar-type, TFT (positional), axis=<given>
def unique(
    ar: ArrayLike,
    return_index: L[True],
    return_inverse: L[False],
    return_counts: L[True],
    axis: SupportsIndex,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[Any], _Int1D, _Int1D]: ...
@overload  # known array, TFT (keyword), axis=<given>
def unique[ShapeT: _Shape, DTypeT: np.dtype](
    ar: np.ndarray[ShapeT, DTypeT],
    return_index: L[True],
    return_inverse: L[False] = False,
    *,
    return_counts: L[True],
    axis: SupportsIndex,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray[ShapeT, DTypeT], _Int1D, _Int1D]: ...
@overload  # known scalar-type, TFT (keyword), axis=None (default)
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[True],
    return_inverse: L[False] = False,
    *,
    return_counts: L[True],
    axis: None = None,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_Array1D[ScalarT], _Int1D, _Int1D]: ...
@overload  # known scalar-type, TFT (keyword), axis=<given>
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[True],
    return_inverse: L[False] = False,
    *,
    return_counts: L[True],
    axis: SupportsIndex,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[ScalarT], _Int1D, _Int1D]: ...
@overload  # unknown scalar-type, TFT (keyword), axis=None (default)
def unique(
    ar: ArrayLike,
    return_index: L[True],
    return_inverse: L[False] = False,
    *,
    return_counts: L[True],
    axis: None = None,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_Array1D[Any], _Int1D, _Int1D]: ...
@overload  # unknown scalar-type, TFT (keyword), axis=<given>
def unique(
    ar: ArrayLike,
    return_index: L[True],
    return_inverse: L[False] = False,
    *,
    return_counts: L[True],
    axis: SupportsIndex,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[Any], _Int1D, _Int1D]: ...
@overload  # known array, FTT (positional), axis=None (default)
def unique[ShapeT: _Shape, DTypeT: np.dtype](
    ar: np.ndarray[ShapeT, DTypeT],
    return_index: L[False],
    return_inverse: L[True],
    return_counts: L[True],
    axis: None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray[tuple[int], DTypeT], np.ndarray[ShapeT, np.dtype[np.intp]], _Int1D]: ...
@overload  # known array, FTT (positional), axis=<given>
def unique[ShapeT: _Shape, DTypeT: np.dtype](
    ar: np.ndarray[ShapeT, DTypeT],
    return_index: L[False],
    return_inverse: L[True],
    return_counts: L[True],
    axis: SupportsIndex,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray[ShapeT, DTypeT], _Int1D, _Int1D]: ...
@overload  # known scalar-type, FTT (positional), axis=None (default)
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[False],
    return_inverse: L[True],
    return_counts: L[True],
    axis: None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_Array1D[ScalarT], _IntND, _Int1D]: ...
@overload  # known scalar-type, FTT (positional), axis=<given>
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[False],
    return_inverse: L[True],
    return_counts: L[True],
    axis: SupportsIndex,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[ScalarT], _Int1D, _Int1D]: ...
@overload  # unknown scalar-type, FTT (positional), axis=None (default)
def unique(
    ar: ArrayLike,
    return_index: L[False],
    return_inverse: L[True],
    return_counts: L[True],
    axis: None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_Array1D[Any], _IntND, _Int1D]: ...
@overload  # unknown scalar-type, FTT (positional), axis=<given>
def unique(
    ar: ArrayLike,
    return_index: L[False],
    return_inverse: L[True],
    return_counts: L[True],
    axis: SupportsIndex,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[Any], _Int1D, _Int1D]: ...
@overload  # known array, FTT (keyword), axis=None (default)
def unique[ShapeT: _Shape, DTypeT: np.dtype](
    ar: np.ndarray[ShapeT, DTypeT],
    return_index: L[False] = False,
    *,
    return_inverse: L[True],
    return_counts: L[True],
    axis: None = None,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray[tuple[int], DTypeT], np.ndarray[ShapeT, np.dtype[np.intp]], _Int1D]: ...
@overload  # known array, FTT (keyword), axis=<given>
def unique[ShapeT: _Shape, DTypeT: np.dtype](
    ar: np.ndarray[ShapeT, DTypeT],
    return_index: L[False] = False,
    *,
    return_inverse: L[True],
    return_counts: L[True],
    axis: SupportsIndex,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray[ShapeT, DTypeT], _Int1D, _Int1D]: ...
@overload  # known scalar-type, FTT (keyword), axis=None (default)
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[False] = False,
    *,
    return_inverse: L[True],
    return_counts: L[True],
    axis: None = None,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_Array1D[ScalarT], _IntND, _Int1D]: ...
@overload  # known scalar-type, FTT (keyword), axis=<given>
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[False] = False,
    *,
    return_inverse: L[True],
    return_counts: L[True],
    axis: SupportsIndex,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[ScalarT], _Int1D, _Int1D]: ...
@overload  # unknown scalar-type, FTT (keyword), axis=None (default)
def unique(
    ar: ArrayLike,
    return_index: L[False] = False,
    *,
    return_inverse: L[True],
    return_counts: L[True],
    axis: None = None,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_Array1D[Any], _IntND, _Int1D]: ...
@overload  # unknown scalar-type, FTT (keyword), axis=<given>
def unique(
    ar: ArrayLike,
    return_index: L[False] = False,
    *,
    return_inverse: L[True],
    return_counts: L[True],
    axis: SupportsIndex,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[Any], _Int1D, _Int1D]: ...
@overload  # known array, TTT, axis=None (default)
def unique[ShapeT: _Shape, DTypeT: np.dtype](
    ar: np.ndarray[ShapeT, DTypeT],
    return_index: L[True],
    return_inverse: L[True],
    return_counts: L[True],
    axis: None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray[tuple[int], DTypeT], _Int1D, np.ndarray[ShapeT, np.dtype[np.intp]], _Int1D]: ...
@overload  # known array, TTT, axis=<given>
def unique[ShapeT: _Shape, DTypeT: np.dtype](
    ar: np.ndarray[ShapeT, DTypeT],
    return_index: L[True],
    return_inverse: L[True],
    return_counts: L[True],
    axis: SupportsIndex,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray[ShapeT, DTypeT], _Int1D, _Int1D, _Int1D]: ...
@overload  # known scalar-type, TTT, axis=None (default)
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[True],
    return_inverse: L[True],
    return_counts: L[True],
    axis: None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_Array1D[ScalarT], _Int1D, _IntND, _Int1D]: ...
@overload  # known scalar-type, TTT, axis=<given>
def unique[ScalarT: np.generic](
    ar: _ArrayLike[ScalarT],
    return_index: L[True],
    return_inverse: L[True],
    return_counts: L[True],
    axis: SupportsIndex,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[ScalarT], _Int1D, _Int1D, _Int1D]: ...
@overload  # unknown scalar-type, TTT, axis=None (default)
def unique(
    ar: ArrayLike,
    return_index: L[True],
    return_inverse: L[True],
    return_counts: L[True],
    axis: None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_Array1D[Any], _Int1D, _IntND, _Int1D]: ...
@overload  # unknown scalar-type, TTT, axis=<given>
def unique(
    ar: ArrayLike,
    return_index: L[True],
    return_inverse: L[True],
    return_counts: L[True],
    axis: SupportsIndex,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray, _Int1D, _Int1D, _Int1D]: ...

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
def unique_values[ScalarT: np.generic](x: _ArrayLike[ScalarT]) -> _Array1D[ScalarT]: ...
@overload
def unique_values(x: ArrayLike) -> _Array1D[Incomplete]: ...

# NOTE: we ignore UP047 because inlining `_AnyScalarT` would result in a lot of code duplication

#
@overload  # known scalar-type, return_indices=False (default)
def intersect1d(  # noqa: UP047
    ar1: _ArrayLike[_AnyScalarT],
    ar2: _ArrayLike[_AnyScalarT],
    assume_unique: bool = False,
    return_indices: L[False] = False,
) -> _Array1D[_AnyScalarT]: ...
@overload  # known scalar-type, return_indices=True (positional)
def intersect1d(  # noqa: UP047
    ar1: _ArrayLike[_AnyScalarT],
    ar2: _ArrayLike[_AnyScalarT],
    assume_unique: bool,
    return_indices: L[True],
) -> _IntersectResult[_AnyScalarT]: ...
@overload  # known scalar-type, return_indices=True (keyword)
def intersect1d(  # noqa: UP047
    ar1: _ArrayLike[_AnyScalarT],
    ar2: _ArrayLike[_AnyScalarT],
    assume_unique: bool = False,
    *,
    return_indices: L[True],
) -> _IntersectResult[_AnyScalarT]: ...
@overload  # unknown scalar-type, return_indices=False (default)
def intersect1d(
    ar1: ArrayLike,
    ar2: ArrayLike,
    assume_unique: bool = False,
    return_indices: L[False] = False,
) -> _Array1D[Incomplete]: ...
@overload  # unknown scalar-type, return_indices=True (positional)
def intersect1d(
    ar1: ArrayLike,
    ar2: ArrayLike,
    assume_unique: bool,
    return_indices: L[True],
) -> _IntersectResult[Incomplete]: ...
@overload  # unknown scalar-type, return_indices=True (keyword)
def intersect1d(
    ar1: ArrayLike,
    ar2: ArrayLike,
    assume_unique: bool = False,
    *,
    return_indices: L[True],
) -> _IntersectResult[Incomplete]: ...

#
@overload
def setxor1d(  # noqa: UP047
    ar1: _ArrayLike[_AnyScalarT], ar2: _ArrayLike[_AnyScalarT], assume_unique: bool = False
) -> _Array1D[_AnyScalarT]: ...
@overload
def setxor1d(ar1: ArrayLike, ar2: ArrayLike, assume_unique: bool = False) -> _Array1D[Incomplete]: ...

#
@overload
def union1d(ar1: _ArrayLike[_AnyScalarT], ar2: _ArrayLike[_AnyScalarT]) -> _Array1D[_AnyScalarT]: ...  # noqa: UP047
@overload
def union1d(ar1: ArrayLike, ar2: ArrayLike) -> _Array1D[Incomplete]: ...

#
@overload
def setdiff1d(  # noqa: UP047
    ar1: _ArrayLike[_AnyScalarT], ar2: _ArrayLike[_AnyScalarT], assume_unique: bool = False
) -> _Array1D[_AnyScalarT]: ...
@overload
def setdiff1d(ar1: ArrayLike, ar2: ArrayLike, assume_unique: bool = False) -> _Array1D[Incomplete]: ...

#
@overload  # known shape
def isin[ShapeT: _Shape](
    element: np.ndarray[ShapeT],
    test_elements: ArrayLike,
    assume_unique: bool = False,
    invert: bool = False,
    *,
    kind: L["sort", "table"] | None = None,
) -> np.ndarray[ShapeT, np.dtype[np.bool]]: ...
@overload  # 0d
def isin[ShapeT: _Shape](
    element: complex | np.generic,
    test_elements: ArrayLike,
    assume_unique: bool = False,
    invert: bool = False,
    *,
    kind: L["sort", "table"] | None = None,
) -> _Array0D[np.bool]: ...
@overload  # 1d
def isin[ShapeT: _Shape](
    element: Sequence[complex | np.generic],
    test_elements: ArrayLike,
    assume_unique: bool = False,
    invert: bool = False,
    *,
    kind: L["sort", "table"] | None = None,
) -> _Array1D[np.bool]: ...
@overload  # 2d
def isin[ShapeT: _Shape](
    element: Sequence[Sequence[complex | np.generic]],
    test_elements: ArrayLike,
    assume_unique: bool = False,
    invert: bool = False,
    *,
    kind: L["sort", "table"] | None = None,
) -> _Array2D[np.bool]: ...
@overload  # 3d
def isin[ShapeT: _Shape](
    element: Sequence[Sequence[Sequence[complex | np.generic]]],
    test_elements: ArrayLike,
    assume_unique: bool = False,
    invert: bool = False,
    *,
    kind: L["sort", "table"] | None = None,
) -> _Array3D[np.bool]: ...
@overload  # fallback
def isin(
    element: ArrayLike,
    test_elements: ArrayLike,
    assume_unique: bool = False,
    invert: bool = False,
    *,
    kind: L["sort", "table"] | None = None,
) -> NDArray[np.bool]: ...
