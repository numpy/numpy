from _typeshed import Incomplete
from collections.abc import Callable, Sequence
from typing import Any, Concatenate, Literal as L, SupportsIndex, overload

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
    _ArrayLikeInt_co,
    _DTypeLike,
    _NestedSequence,
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
type _MArray1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]

type _ScalarNumeric = np.inexact | np.timedelta64 | np.object_
type _ListSeqND[T] = list[T] | _NestedSequence[list[T]]

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
def median[ArrayT: NDArray[_ScalarNumeric]](
    a: ArrayT,
    axis: _ShapeLike | None = None,
    out: None = None,
    overwrite_input: bool = False,
    *,
    keepdims: L[True],
) -> ArrayT: ...
@overload  # known scalar-type, keepdims=True_ArrayLikeNumber_co
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

# TODO: everything below
# mypy: disable-error-code=no-untyped-def

def unique(ar1, return_index=False, return_inverse=False): ...
def intersect1d(ar1, ar2, assume_unique=False): ...
def setxor1d(ar1, ar2, assume_unique=False): ...
def in1d(ar1, ar2, assume_unique=False, invert=False): ...
def isin(element, test_elements, assume_unique=False, invert=False): ...
def union1d(ar1, ar2): ...
def setdiff1d(ar1, ar2, assume_unique=False): ...
def cov(x, y=None, rowvar=True, bias=False, allow_masked=True, ddof=None): ...
def corrcoef(x, y=None, rowvar=True, allow_masked=True): ...

class MAxisConcatenator(AxisConcatenator):
    __slots__ = ()
    @staticmethod
    def concatenate(arrays: Incomplete, axis: int = 0) -> Incomplete: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @classmethod
    def makemat(cls, arr: Incomplete) -> Incomplete: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleVariableOverride]

class mr_class(MAxisConcatenator):
    __slots__ = ()

    def __init__(self) -> None: ...

mr_: mr_class

def ndenumerate(a, compressed=True): ...
def flatnotmasked_edges(a): ...
def notmasked_edges(a, axis=None): ...
def flatnotmasked_contiguous(a): ...
def notmasked_contiguous(a, axis=None): ...
def clump_unmasked(a): ...
def clump_masked(a): ...
def vander(x, n=None): ...
def polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False): ...
