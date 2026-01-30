from _typeshed import Incomplete, SupportsLenAndGetItem
from collections.abc import Callable, Iterator, Sequence
from typing import (
    Any,
    Concatenate,
    Final,
    Literal as L,
    SupportsIndex,
    TypeVar,
    overload,
    override,
)

import numpy as np
from numpy import _CastingKind
from numpy._globals import _NoValueType
from numpy._typing import (
    ArrayLike,
    DTypeLike,
    NDArray,
    _AnyShape,
    _ArrayLike,
    _ArrayLikeBool_co,
    _ArrayLikeComplex_co,
    _ArrayLikeFloat_co,
    _ArrayLikeInt_co,
    _DTypeLike,
    _NestedSequence,
    _NumberLike_co,
    _Shape,
    _ShapeLike,
)
from numpy.lib._function_base_impl import average
from numpy.lib._index_tricks_impl import AxisConcatenator

from .core import MaskedArray, dot

__all__ = [
    "apply_along_axis",
    "apply_over_axes",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "average",
    "clump_masked",
    "clump_unmasked",
    "column_stack",
    "compress_cols",
    "compress_nd",
    "compress_rowcols",
    "compress_rows",
    "corrcoef",
    "count_masked",
    "cov",
    "diagflat",
    "dot",
    "dstack",
    "ediff1d",
    "flatnotmasked_contiguous",
    "flatnotmasked_edges",
    "hsplit",
    "hstack",
    "in1d",
    "intersect1d",
    "isin",
    "mask_cols",
    "mask_rowcols",
    "mask_rows",
    "masked_all",
    "masked_all_like",
    "median",
    "mr_",
    "ndenumerate",
    "notmasked_contiguous",
    "notmasked_edges",
    "polyfit",
    "row_stack",
    "setdiff1d",
    "setxor1d",
    "stack",
    "union1d",
    "unique",
    "vander",
    "vstack",
]

type _MArray[ScalarT: np.generic] = MaskedArray[_AnyShape, np.dtype[ScalarT]]
type _MArray1D[ScalarT: np.generic] = MaskedArray[tuple[int], np.dtype[ScalarT]]
type _MArray2D[ScalarT: np.generic] = MaskedArray[tuple[int, int], np.dtype[ScalarT]]
type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]

type _IntArray = NDArray[np.intp]
type _ScalarNumeric = np.inexact | np.timedelta64 | np.object_
type _InexactDouble = np.float64 | np.longdouble | np.complex128 | np.clongdouble
type _ListSeqND[T] = list[T] | _NestedSequence[list[T]]

# helper aliases for polyfit
type _2Tup[T] = tuple[T, T]
type _5Tup[T] = tuple[T, NDArray[np.float64], NDArray[np.int32], NDArray[np.float64], NDArray[np.float64]]

# Explicitly set all allowed values to prevent accidental castings to
# abstract dtypes (their common super-type).
# Only relevant if two or more arguments are parametrized, (e.g. `setdiff1d`)
# which could result in, for example, `int64` and `float64` producing a
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

###

# keep in sync with `numpy._core.shape_base.atleast_1d`
@overload
def atleast_1d[ScalarT: np.generic](a0: _ArrayLike[ScalarT], /) -> _MArray[ScalarT]: ...
@overload
def atleast_1d[ScalarT1: np.generic, ScalarT2: np.generic](
    a0: _ArrayLike[ScalarT1], a1: _ArrayLike[ScalarT2], /
) -> tuple[_MArray[ScalarT1], _MArray[ScalarT2]]: ...
@overload
def atleast_1d[ScalarT: np.generic](
    a0: _ArrayLike[ScalarT], a1: _ArrayLike[ScalarT], /, *arys: _ArrayLike[ScalarT]
) -> tuple[_MArray[ScalarT], ...]: ...
@overload
def atleast_1d(a0: ArrayLike, /) -> _MArray[Incomplete]: ...
@overload
def atleast_1d(a0: ArrayLike, a1: ArrayLike, /) -> tuple[_MArray[Incomplete], _MArray[Incomplete]]: ...
@overload
def atleast_1d(a0: ArrayLike, a1: ArrayLike, /, *ai: ArrayLike) -> tuple[_MArray[Incomplete], ...]: ...

# keep in sync with `numpy._core.shape_base.atleast_2d`
@overload
def atleast_2d[ScalarT: np.generic](a0: _ArrayLike[ScalarT], /) -> _MArray[ScalarT]: ...
@overload
def atleast_2d[ScalarT1: np.generic, ScalarT2: np.generic](
    a0: _ArrayLike[ScalarT1], a1: _ArrayLike[ScalarT2], /
) -> tuple[_MArray[ScalarT1], _MArray[ScalarT2]]: ...
@overload
def atleast_2d[ScalarT: np.generic](
    a0: _ArrayLike[ScalarT], a1: _ArrayLike[ScalarT], /, *arys: _ArrayLike[ScalarT]
) -> tuple[_MArray[ScalarT], ...]: ...
@overload
def atleast_2d(a0: ArrayLike, /) -> _MArray[Incomplete]: ...
@overload
def atleast_2d(a0: ArrayLike, a1: ArrayLike, /) -> tuple[_MArray[Incomplete], _MArray[Incomplete]]: ...
@overload
def atleast_2d(a0: ArrayLike, a1: ArrayLike, /, *ai: ArrayLike) -> tuple[_MArray[Incomplete], ...]: ...

# keep in sync with `numpy._core.shape_base.atleast_2d`
@overload
def atleast_3d[ScalarT: np.generic](a0: _ArrayLike[ScalarT], /) -> _MArray[ScalarT]: ...
@overload
def atleast_3d[ScalarT1: np.generic, ScalarT2: np.generic](
    a0: _ArrayLike[ScalarT1], a1: _ArrayLike[ScalarT2], /
) -> tuple[_MArray[ScalarT1], _MArray[ScalarT2]]: ...
@overload
def atleast_3d[ScalarT: np.generic](
    a0: _ArrayLike[ScalarT], a1: _ArrayLike[ScalarT], /, *arys: _ArrayLike[ScalarT]
) -> tuple[_MArray[ScalarT], ...]: ...
@overload
def atleast_3d(a0: ArrayLike, /) -> _MArray[Incomplete]: ...
@overload
def atleast_3d(a0: ArrayLike, a1: ArrayLike, /) -> tuple[_MArray[Incomplete], _MArray[Incomplete]]: ...
@overload
def atleast_3d(a0: ArrayLike, a1: ArrayLike, /, *ai: ArrayLike) -> tuple[_MArray[Incomplete], ...]: ...

# keep in sync with `numpy._core.shape_base.vstack`
@overload
def vstack[ScalarT: np.generic](
    tup: Sequence[_ArrayLike[ScalarT]],
    *,
    dtype: None = None,
    casting: _CastingKind = "same_kind"
) -> _MArray[ScalarT]: ...
@overload
def vstack[ScalarT: np.generic](
    tup: Sequence[ArrayLike],
    *,
    dtype: _DTypeLike[ScalarT],
    casting: _CastingKind = "same_kind"
) -> _MArray[ScalarT]: ...
@overload
def vstack(
    tup: Sequence[ArrayLike],
    *,
    dtype: DTypeLike | None = None,
    casting: _CastingKind = "same_kind"
) -> _MArray[Incomplete]: ...

row_stack = vstack

# keep in sync with `numpy._core.shape_base.hstack`
@overload
def hstack[ScalarT: np.generic](
    tup: Sequence[_ArrayLike[ScalarT]],
    *,
    dtype: None = None,
    casting: _CastingKind = "same_kind"
) -> _MArray[ScalarT]: ...
@overload
def hstack[ScalarT: np.generic](
    tup: Sequence[ArrayLike],
    *,
    dtype: _DTypeLike[ScalarT],
    casting: _CastingKind = "same_kind"
) -> _MArray[ScalarT]: ...
@overload
def hstack(
    tup: Sequence[ArrayLike],
    *,
    dtype: DTypeLike | None = None,
    casting: _CastingKind = "same_kind"
) -> _MArray[Incomplete]: ...

# keep in sync with `numpy._core.shape_base_impl.column_stack`
@overload
def column_stack[ScalarT: np.generic](tup: Sequence[_ArrayLike[ScalarT]]) -> _MArray[ScalarT]: ...
@overload
def column_stack(tup: Sequence[ArrayLike]) -> _MArray[Incomplete]: ...

# keep in sync with `numpy._core.shape_base_impl.dstack`
@overload
def dstack[ScalarT: np.generic](tup: Sequence[_ArrayLike[ScalarT]]) -> _MArray[ScalarT]: ...
@overload
def dstack(tup: Sequence[ArrayLike]) -> _MArray[Incomplete]: ...

# keep in sync with `numpy._core.shape_base.stack`
@overload
def stack[ScalarT: np.generic](
    arrays: Sequence[_ArrayLike[ScalarT]],
    axis: SupportsIndex = 0,
    out: None = None,
    *,
    dtype: None = None,
    casting: _CastingKind = "same_kind"
) -> _MArray[ScalarT]: ...
@overload
def stack[ScalarT: np.generic](
    arrays: Sequence[ArrayLike],
    axis: SupportsIndex = 0,
    out: None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    casting: _CastingKind = "same_kind"
) -> _MArray[ScalarT]: ...
@overload
def stack(
    arrays: Sequence[ArrayLike],
    axis: SupportsIndex = 0,
    out: None = None,
    *,
    dtype: DTypeLike | None = None,
    casting: _CastingKind = "same_kind"
) -> _MArray[Incomplete]: ...
@overload
def stack[MArrayT: MaskedArray](
    arrays: Sequence[ArrayLike],
    axis: SupportsIndex,
    out: MArrayT,
    *,
    dtype: DTypeLike | None = None,
    casting: _CastingKind = "same_kind",
) -> MArrayT: ...
@overload
def stack[MArrayT: MaskedArray](
    arrays: Sequence[ArrayLike],
    axis: SupportsIndex = 0,
    *,
    out: MArrayT,
    dtype: DTypeLike | None = None,
    casting: _CastingKind = "same_kind",
) -> MArrayT: ...

# keep in sync with `numpy._core.shape_base_impl.hsplit`
@overload
def hsplit[ScalarT: np.generic](ary: _ArrayLike[ScalarT], indices_or_sections: _ShapeLike) -> list[_MArray[ScalarT]]: ...
@overload
def hsplit(ary: ArrayLike, indices_or_sections: _ShapeLike) -> list[_MArray[Incomplete]]: ...

# keep in sync with `numpy._core.twodim_base_impl.hsplit`
@overload
def diagflat[ScalarT: np.generic](v: _ArrayLike[ScalarT], k: int = 0) -> _MArray[ScalarT]: ...
@overload
def diagflat(v: ArrayLike, k: int = 0) -> _MArray[Incomplete]: ...

#
def count_masked(arr: ArrayLike, axis: SupportsIndex | None = None) -> NDArray[np.intp]: ...

#
@overload
def masked_all[ScalarT: np.generic](shape: _ShapeLike, dtype: _DTypeLike[ScalarT]) -> _MArray[ScalarT]: ...
@overload
def masked_all(shape: _ShapeLike, dtype: DTypeLike = float) -> _MArray[Incomplete]: ...

#
@overload
def masked_all_like[ScalarT: np.generic](arr: _ArrayLike[ScalarT]) -> _MArray[ScalarT]: ...
@overload
def masked_all_like(arr: ArrayLike) -> _MArray[Incomplete]: ...

#
def apply_along_axis[**Tss](
    func1d: Callable[Concatenate[MaskedArray, Tss], ArrayLike],
    axis: SupportsIndex,
    arr: ArrayLike,
    *args: Tss.args,
    **kwargs: Tss.kwargs,
) -> _MArray[Incomplete]: ...

#
@overload
def apply_over_axes[ScalarT: np.generic](
    func: Callable[[MaskedArray, int], _ArrayLike[ScalarT]],
    a: np.ndarray,
    axes: _ShapeLike,
) -> _MArray[ScalarT]: ...
@overload
def apply_over_axes(
    func: Callable[[MaskedArray, int], ArrayLike],
    a: np.ndarray,
    axes: _ShapeLike,
) -> _MArray[Incomplete]: ...

# keep in sync with `lib._function_base_impl.median`
@overload  # known scalar-type, keepdims=False (default)
def median[ScalarT: np.inexact | np.timedelta64](
    a: _ArrayLike[ScalarT],
    axis: None = None,
    out: None = None,
    overwrite_input: bool = False,
    keepdims: L[False] = False,
) -> ScalarT: ...
@overload  # float array-like, keepdims=False (default)
def median(
    a: _ArrayLikeInt_co | _NestedSequence[float] | float,
    axis: None = None,
    out: None = None,
    overwrite_input: bool = False,
    keepdims: L[False] = False,
) -> np.float64: ...
@overload  # complex array-like, keepdims=False (default)
def median(
    a: _ListSeqND[complex],
    axis: None = None,
    out: None = None,
    overwrite_input: bool = False,
    keepdims: L[False] = False,
) -> np.complex128: ...
@overload  # complex scalar, keepdims=False (default)
def median(
    a: complex,
    axis: None = None,
    out: None = None,
    overwrite_input: bool = False,
    keepdims: L[False] = False,
) -> np.complex128 | Any: ...
@overload  # known array-type, keepdims=True
def median[ArrayT: _MArray[_ScalarNumeric]](
    a: ArrayT,
    axis: _ShapeLike | None = None,
    out: None = None,
    overwrite_input: bool = False,
    *,
    keepdims: L[True],
) -> ArrayT: ...
@overload  # known scalar-type, keepdims=True
def median[ScalarT: _ScalarNumeric](
    a: _ArrayLike[ScalarT],
    axis: _ShapeLike | None = None,
    out: None = None,
    overwrite_input: bool = False,
    *,
    keepdims: L[True],
) -> _MArray[ScalarT]: ...
@overload  # known scalar-type, axis=<given>
def median[ScalarT: _ScalarNumeric](
    a: _ArrayLike[ScalarT],
    axis: _ShapeLike,
    out: None = None,
    overwrite_input: bool = False,
    keepdims: bool = False,
) -> _MArray[ScalarT]: ...
@overload  # float array-like, keepdims=True
def median(
    a: _NestedSequence[float],
    axis: _ShapeLike | None = None,
    out: None = None,
    overwrite_input: bool = False,
    *,
    keepdims: L[True],
) -> _MArray[np.float64]: ...
@overload  # float array-like, axis=<given>
def median(
    a: _NestedSequence[float],
    axis: _ShapeLike,
    out: None = None,
    overwrite_input: bool = False,
    keepdims: bool = False,
) -> _MArray[np.float64]: ...
@overload  # complex array-like, keepdims=True
def median(
    a: _ListSeqND[complex],
    axis: _ShapeLike | None = None,
    out: None = None,
    overwrite_input: bool = False,
    *,
    keepdims: L[True],
) -> _MArray[np.complex128]: ...
@overload  # complex array-like, axis=<given>
def median(
    a: _ListSeqND[complex],
    axis: _ShapeLike,
    out: None = None,
    overwrite_input: bool = False,
    keepdims: bool = False,
) -> _MArray[np.complex128]: ...
@overload  # out=<given> (keyword)
def median[ArrayT: np.ndarray](
    a: _ArrayLikeComplex_co | _ArrayLike[np.timedelta64 | np.object_],
    axis: _ShapeLike | None = None,
    *,
    out: ArrayT,
    overwrite_input: bool = False,
    keepdims: bool = False,
) -> ArrayT: ...
@overload  # out=<given> (positional)
def median[ArrayT: np.ndarray](
    a: _ArrayLikeComplex_co | _ArrayLike[np.timedelta64 | np.object_],
    axis: _ShapeLike | None,
    out: ArrayT,
    overwrite_input: bool = False,
    keepdims: bool = False,
) -> ArrayT: ...
@overload  # fallback
def median(
    a: _ArrayLikeComplex_co | _ArrayLike[np.timedelta64 | np.object_],
    axis: _ShapeLike | None = None,
    out: None = None,
    overwrite_input: bool = False,
    keepdims: bool = False,
) -> Incomplete: ...

#
@overload
def compress_nd[ScalarT: np.generic](x: _ArrayLike[ScalarT], axis: _ShapeLike | None = None) -> NDArray[ScalarT]: ...
@overload
def compress_nd(x: ArrayLike, axis: _ShapeLike | None = None) -> NDArray[Incomplete]: ...

#
@overload
def compress_rowcols[ScalarT: np.generic](x: _ArrayLike[ScalarT], axis: int | None = None) -> _Array2D[ScalarT]: ...
@overload
def compress_rowcols(x: ArrayLike, axis: int | None = None) -> _Array2D[Incomplete]: ...

#
@overload
def compress_rows[ScalarT: np.generic](a: _ArrayLike[ScalarT]) -> _Array2D[ScalarT]: ...
@overload
def compress_rows(a: ArrayLike) -> _Array2D[Incomplete]: ...

#
@overload
def compress_cols[ScalarT: np.generic](a: _ArrayLike[ScalarT]) -> _Array2D[ScalarT]: ...
@overload
def compress_cols(a: ArrayLike) -> _Array2D[Incomplete]: ...

#
def mask_rowcols(a: ArrayLike, axis: SupportsIndex | None = None) -> _MArray[Incomplete]: ...
def mask_rows(a: ArrayLike, axis: _NoValueType = ...) -> _MArray[Incomplete]: ...
def mask_cols(a: ArrayLike, axis: _NoValueType = ...) -> _MArray[Incomplete]: ...

# keep in sync with `lib._arraysetops_impl.ediff1d`
@overload
def ediff1d(
    arr: _ArrayLikeBool_co,
    to_end: ArrayLike | None = None,
    to_begin: ArrayLike | None = None,
) -> _MArray1D[np.int8]: ...
@overload
def ediff1d[NumericT: _ScalarNumeric](
    arr: _ArrayLike[NumericT],
    to_end: ArrayLike | None = None,
    to_begin: ArrayLike | None = None,
) -> _MArray1D[NumericT]: ...
@overload
def ediff1d(
    arr: _ArrayLike[np.datetime64[Any]],
    to_end: ArrayLike | None = None,
    to_begin: ArrayLike | None = None,
) -> _MArray1D[np.timedelta64]: ...
@overload
def ediff1d(
    arr: _ArrayLikeComplex_co,
    to_end: ArrayLike | None = None,
    to_begin: ArrayLike | None = None,
) -> _MArray1D[Incomplete]: ...

# keep in sync with `lib._arraysetops_impl.unique`, minus `return_counts`
@overload  # known scalar-type, FF
def unique[ScalarT: np.generic](
    ar1: _ArrayLike[ScalarT],
    return_index: L[False] = False,
    return_inverse: L[False] = False,
) -> _MArray[ScalarT]: ...
@overload  # unknown scalar-type, FF
def unique(
    ar1: ArrayLike,
    return_index: L[False] = False,
    return_inverse: L[False] = False,
) -> _MArray[Incomplete]: ...
@overload  # known scalar-type, TF
def unique[ScalarT: np.generic](
    ar1: _ArrayLike[ScalarT],
    return_index: L[True],
    return_inverse: L[False] = False,
) -> tuple[_MArray[ScalarT], _IntArray]: ...
@overload  # unknown scalar-type, TFF
def unique(
    ar1: ArrayLike,
    return_index: L[True],
    return_inverse: L[False] = False,
) -> tuple[_MArray[Incomplete], _IntArray]: ...
@overload  # known scalar-type, FT (positional)
def unique[ScalarT: np.generic](
    ar1: _ArrayLike[ScalarT],
    return_index: L[False],
    return_inverse: L[True],
) -> tuple[_MArray[ScalarT], _IntArray]: ...
@overload  # known scalar-type, FT (keyword)
def unique[ScalarT: np.generic](
    ar1: _ArrayLike[ScalarT],
    return_index: L[False] = False,
    *,
    return_inverse: L[True],
) -> tuple[_MArray[ScalarT], _IntArray]: ...
@overload  # unknown scalar-type, FT (positional)
def unique(
    ar1: ArrayLike,
    return_index: L[False],
    return_inverse: L[True],
) -> tuple[_MArray[Incomplete], _IntArray]: ...
@overload  # unknown scalar-type, FT (keyword)
def unique(
    ar1: ArrayLike,
    return_index: L[False] = False,
    *,
    return_inverse: L[True],
) -> tuple[_MArray[Incomplete], _IntArray]: ...
@overload  # known scalar-type, TT
def unique[ScalarT: np.generic](
    ar1: _ArrayLike[ScalarT],
    return_index: L[True],
    return_inverse: L[True],
) -> tuple[_MArray[ScalarT], _IntArray, _IntArray]: ...
@overload  # unknown scalar-type, TT
def unique(
    ar1: ArrayLike,
    return_index: L[True],
    return_inverse: L[True],
) -> tuple[_MArray[Incomplete], _IntArray, _IntArray]: ...

# keep in sync with `lib._arraysetops_impl.intersect1d`
@overload  # known scalar-type, return_indices=False (default)
def intersect1d(
    ar1: _ArrayLike[_AnyScalarT], ar2: _ArrayLike[_AnyScalarT], assume_unique: bool = False
) -> _MArray1D[_AnyScalarT]: ...
@overload  # unknown scalar-type, return_indices=False (default)
def intersect1d(ar1: ArrayLike, ar2: ArrayLike, assume_unique: bool = False) -> _MArray1D[Incomplete]: ...

# keep in sync with `lib._arraysetops_impl.setxor1d`
@overload
def setxor1d(
    ar1: _ArrayLike[_AnyScalarT], ar2: _ArrayLike[_AnyScalarT], assume_unique: bool = False
) -> _MArray1D[_AnyScalarT]: ...
@overload
def setxor1d(ar1: ArrayLike, ar2: ArrayLike, assume_unique: bool = False) -> _MArray1D[Incomplete]: ...

# keep in sync with `lib._arraysetops_impl.union1d`
@overload
def union1d(ar1: _ArrayLike[_AnyScalarT], ar2: _ArrayLike[_AnyScalarT]) -> _MArray1D[_AnyScalarT]: ...
@overload
def union1d(ar1: ArrayLike, ar2: ArrayLike) -> _MArray1D[Incomplete]: ...

# keep in sync with `lib._arraysetops_impl.setdiff1d`
@overload
def setdiff1d(
    ar1: _ArrayLike[_AnyScalarT], ar2: _ArrayLike[_AnyScalarT], assume_unique: bool = False
) -> _MArray1D[_AnyScalarT]: ...
@overload
def setdiff1d(ar1: ArrayLike, ar2: ArrayLike, assume_unique: bool = False) -> _MArray1D[Incomplete]: ...

#
def in1d(ar1: ArrayLike, ar2: ArrayLike, assume_unique: bool = False, invert: bool = False) -> _MArray1D[np.bool]: ...

# keep in sync with `lib._arraysetops_impl.isin`
def isin(
    element: ArrayLike, test_elements: ArrayLike, assume_unique: bool = False, invert: bool = False
) -> _MArray[np.bool]: ...

# keep in sync with `corrcoef`
def cov(
    x: ArrayLike,
    y: ArrayLike | None = None,
    rowvar: bool = True,
    bias: bool = False,
    allow_masked: bool = True,
    ddof: int | None = None
) -> _MArray[Incomplete]: ...

# keep in sync with `cov`
def corrcoef(x: ArrayLike, y: ArrayLike | None = None, rowvar: bool = True, allow_masked: bool = True) -> _MArray[Incomplete]: ...

class MAxisConcatenator(AxisConcatenator):
    __slots__ = ()

    # keep in sync with `ma.core.concatenate`
    @override  # type: ignore[override]
    @overload
    @staticmethod
    def concatenate[ScalarT: np.generic](arrays: _ArrayLike[ScalarT], axis: SupportsIndex | None = 0) -> _MArray[ScalarT]: ...
    @overload
    @staticmethod
    def concatenate(arrays: SupportsLenAndGetItem[ArrayLike], axis: SupportsIndex | None = 0) -> _MArray[Incomplete]: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    #
    @override
    @classmethod
    def makemat(cls, /, arr: ArrayLike) -> _MArray[Incomplete]: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleVariableOverride]

class mr_class(MAxisConcatenator):
    __slots__ = ()
    def __init__(self) -> None: ...

mr_: Final[mr_class] = ...

#
@overload
def ndenumerate[ShapeT: _Shape, ScalarT: np.generic](
    a: np.ndarray[ShapeT, np.dtype[ScalarT]],
    compressed: bool = True,
) -> Iterator[tuple[ShapeT, ScalarT]]: ...
@overload
def ndenumerate[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    compressed: bool = True,
) -> Iterator[tuple[_AnyShape, ScalarT]]: ...
@overload
def ndenumerate(
    a: ArrayLike,
    compressed: bool = True,
) -> Iterator[tuple[_AnyShape, Incomplete]]: ...

#
@overload
def flatnotmasked_edges[ScalarT: np.generic](a: _ArrayLike[ScalarT]) -> _Array1D[ScalarT] | None: ...
@overload
def flatnotmasked_edges(a: ArrayLike) -> _Array1D[Incomplete] | None: ...

#
@overload
def notmasked_edges[ScalarT: np.generic](a: _ArrayLike[ScalarT], axis: None = None) -> _Array1D[ScalarT] | None: ...
@overload
def notmasked_edges(a: ArrayLike, axis: None = None) -> _Array1D[Incomplete] | None: ...
@overload
def notmasked_edges(a: ArrayLike, axis: SupportsIndex) -> Incomplete: ...

#
def flatnotmasked_contiguous(a: ArrayLike) -> list[slice[int, int, None]]: ...

#
@overload
def notmasked_contiguous(a: ArrayLike, axis: None = None) -> list[slice[int, int, None]]: ...
@overload
def notmasked_contiguous(a: ArrayLike, axis: SupportsIndex) -> list[Incomplete]: ...

#
def _ezclump(mask: np.ndarray) -> list[slice[int, int, None]]: ...  # undocumented
def clump_unmasked(a: np.ndarray) -> list[slice[int, int, None]]: ...
def clump_masked(a: np.ndarray) -> list[slice[int, int, None]]: ...

# keep in sync with `lib._twodim_base_impl.vander`
@overload
def vander[ScalarT: np.number | np.object_](x: _ArrayLike[ScalarT], n: int | None = None) -> _Array2D[ScalarT]: ...
@overload
def vander(x: _ArrayLike[np.bool] | list[int], n: int | None = None) -> _Array2D[np.int_]: ...
@overload
def vander(x: list[float], n: int | None = None) -> _Array2D[np.float64]: ...
@overload
def vander(x: list[complex], n: int | None = None) -> _Array2D[np.complex128]: ...
@overload  # fallback
def vander(x: Sequence[_NumberLike_co], n: int | None = None) -> _Array2D[Any]: ...

# keep roughly in sync with `lib._polynomial_impl.polyfit`
@overload  # float dtype, cov: False (default)
def polyfit(
    x: _ArrayLikeFloat_co,
    y: _ArrayLikeFloat_co,
    deg: int,
    rcond: bool | None = None,
    full: L[False] = False,
    w: _ArrayLikeFloat_co | None = None,
    cov: L[False] = False
) -> NDArray[np.float64]: ...
@overload  # float dtype, cov: True | "unscaled" (keyword)
def polyfit(
    x: _ArrayLikeFloat_co,
    y: _ArrayLikeFloat_co,
    deg: int,
    rcond: bool | None = None,
    full: L[False] = False,
    w: _ArrayLikeFloat_co | None = None,
    *,
    cov: L[True, "unscaled"],
) -> _2Tup[NDArray[np.float64]]: ...
@overload  # float dtype, full: True (keyword)
def polyfit(
    x: _ArrayLikeFloat_co,
    y: _ArrayLikeFloat_co,
    deg: int,
    rcond: bool | None = None,
    *,
    full: L[True],
    w: _ArrayLikeFloat_co | None = None,
    cov: bool | L["unscaled"] = False,
) -> _5Tup[NDArray[np.float64]]: ...
@overload  # complex dtype, cov: False (default)
def polyfit(
    x: _ArrayLikeComplex_co,
    y: _ArrayLikeComplex_co,
    deg: int,
    rcond: bool | None = None,
    full: L[False] = False,
    w: _ArrayLikeFloat_co | None = None,
    cov: L[False] = False
) -> NDArray[Incomplete]: ...
@overload  # complex dtype, cov: True | "unscaled" (keyword)
def polyfit(
    x: _ArrayLikeComplex_co,
    y: _ArrayLikeComplex_co,
    deg: int,
    rcond: bool | None = None,
    full: L[False] = False,
    w: _ArrayLikeFloat_co | None = None,
    *,
    cov: L[True, "unscaled"],
) -> _2Tup[NDArray[np.complex128 | Any]]: ...
@overload  # complex dtype, full: True (keyword)
def polyfit(
    x: _ArrayLikeComplex_co,
    y: _ArrayLikeComplex_co,
    deg: int,
    rcond: bool | None = None,
    *,
    full: L[True],
    w: _ArrayLikeFloat_co | None = None,
    cov: bool | L["unscaled"] = False,
) -> _5Tup[NDArray[np.complex128 | Any]]: ...
