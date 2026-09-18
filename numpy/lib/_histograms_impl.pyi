from _typeshed import Incomplete
from collections.abc import Sequence
from typing import Any, Literal as L, Never, SupportsIndex, overload

import numpy as np
from numpy._typing import (
    ArrayLike,
    NDArray,
    _ArrayLike,
    _ArrayLikeComplex_co,
    _ArrayLikeFloat64_co,
    _ArrayLikeFloat_co,
    _ArrayLikeInt_co,
    _ArrayLikeObject_co,
    _NestedSequence,
)

__all__ = ["histogram", "histogramdd", "histogram_bin_edges"]

###

type _BinKind = L["auto", "fd", "doane", "scott", "stone", "rice", "sturges", "sqrt"]

type _2Tuple[T] = tuple[T, T]
type _3Tuple[T] = tuple[T, T, T]

type _Range = _2Tuple[float]
type _NestedList[T] = list[T] | _NestedSequence[list[T]]

type _WeightsLike = _ArrayLikeComplex_co | _ArrayLikeObject_co
type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]
type _Array3D[ScalarT: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[ScalarT]]
# workaround for mypy and pyright not following the typing spec for overloads
type _ArrayJustND[ScalarT: np.generic] = np.ndarray[tuple[Never, Never, Never, Never], np.dtype[ScalarT]]

type _ArrayLike1D[ScalarT: np.generic] = _Array1D[ScalarT] | Sequence[ScalarT]
type _HistogramResult[HistT: np.generic, EdgeT: np.generic] = tuple[_Array1D[HistT], _Array1D[EdgeT]]

###

# NOTE: The return type can also be complex or `object_`, not only floating like the docstring suggests.
@overload  # dtype +float64
def histogram_bin_edges(
    a: _ArrayLikeInt_co | _NestedSequence[float],
    bins: _BinKind | SupportsIndex | ArrayLike = 10,
    range: _Range | None = None,
    weights: _WeightsLike | None = None,
) -> _Array1D[np.float64]: ...
@overload  # dtype ~complex
def histogram_bin_edges(
    a: _NestedList[complex],
    bins: _BinKind | SupportsIndex | ArrayLike = 10,
    range: _Range | None = None,
    weights: _WeightsLike | None = None,
) -> _Array1D[np.complex128]: ...
@overload  # dtype known
def histogram_bin_edges[ScalarT: np.inexact | np.object_](
    a: _ArrayLike[ScalarT],
    bins: _BinKind | SupportsIndex | ArrayLike = 10,
    range: _Range | None = None,
    weights: _WeightsLike | None = None,
) -> _Array1D[ScalarT]: ...
@overload  # dtype unknown
def histogram_bin_edges(
    a: _ArrayLikeComplex_co,
    bins: _BinKind | SupportsIndex | ArrayLike = 10,
    range: _Range | None = None,
    weights: _WeightsLike | None = None,
) -> _Array1D[Incomplete]: ...

# There are 4 groups of 2 + 3 overloads (2 for density=True, 3 for density=False) = 20 in total
@overload  # a: +float64, density: True (keyword), weights: +float | None (default)
def histogram(
    a: _ArrayLikeInt_co | _NestedSequence[float],
    bins: _BinKind | SupportsIndex | ArrayLike = 10,
    range: _Range | None = None,
    *,
    density: L[True],
    weights: _ArrayLikeFloat_co | None = None,
) -> _HistogramResult[np.float64, np.float64]: ...
@overload  # a: +float64, density: True (keyword), weights: +complex
def histogram(
    a: _ArrayLikeInt_co | _NestedSequence[float],
    bins: _BinKind | SupportsIndex | ArrayLike = 10,
    range: _Range | None = None,
    *,
    density: L[True],
    weights: _ArrayLike[np.complexfloating] | _NestedList[complex],
) -> _HistogramResult[np.complex128, np.float64]: ...
@overload  # a: +float64, density: False (default), weights: ~int | None (default)
def histogram(
    a: _ArrayLikeInt_co | _NestedSequence[float],
    bins: _BinKind | SupportsIndex | ArrayLike = 10,
    range: _Range | None = None,
    density: L[False] | None = None,
    weights: _NestedSequence[int] | None = None,
) -> _HistogramResult[np.intp, np.float64]: ...
@overload  # a: +float64, density: False (default), weights: known (keyword)
def histogram[WeightsT: np.bool | np.number | np.timedelta64](
    a: _ArrayLikeInt_co | _NestedSequence[float],
    bins: _BinKind | SupportsIndex | ArrayLike = 10,
    range: _Range | None = None,
    density: L[False] | None = None,
    *,
    weights: _ArrayLike[WeightsT],
) -> _HistogramResult[WeightsT, np.float64]: ...
@overload  # a: +float64, density: False (default), weights: unknown (keyword)
def histogram(
    a: _ArrayLikeInt_co | _NestedSequence[float],
    bins: _BinKind | SupportsIndex | ArrayLike = 10,
    range: _Range | None = None,
    density: L[False] | None = None,
    *,
    weights: _WeightsLike,
) -> _HistogramResult[Incomplete, np.float64]: ...
@overload  # a: ~complex, density: True (keyword), weights: +float | None (default)
def histogram(
    a: _NestedList[complex],
    bins: _BinKind | SupportsIndex | ArrayLike = 10,
    range: _Range | None = None,
    *,
    density: L[True],
    weights: _ArrayLikeFloat_co | None = None,
) -> _HistogramResult[np.float64, np.complex128]: ...
@overload  # a: ~complex, density: True (keyword), weights: +complex
def histogram(
    a: _NestedList[complex],
    bins: _BinKind | SupportsIndex | ArrayLike = 10,
    range: _Range | None = None,
    *,
    density: L[True],
    weights: _ArrayLike[np.complexfloating] | _NestedList[complex],
) -> _HistogramResult[np.complex128, np.complex128]: ...
@overload  # a: ~complex, density: False (default), weights: ~int | None (default)
def histogram(
    a: _NestedList[complex],
    bins: _BinKind | SupportsIndex | ArrayLike = 10,
    range: _Range | None = None,
    density: L[False] | None = None,
    weights: _NestedSequence[int] | None = None,
) -> _HistogramResult[np.intp, np.complex128]: ...
@overload  # a: ~complex, density: False (default), weights: known (keyword)
def histogram[WeightsT: np.bool | np.number | np.timedelta64](
    a: _NestedList[complex],
    bins: _BinKind | SupportsIndex | ArrayLike = 10,
    range: _Range | None = None,
    density: L[False] | None = None,
    *,
    weights: _ArrayLike[WeightsT],
) -> _HistogramResult[WeightsT, np.complex128]: ...
@overload  # a: ~complex, density: False (default), weights: unknown (keyword)
def histogram(
    a: _NestedList[complex],
    bins: _BinKind | SupportsIndex | ArrayLike = 10,
    range: _Range | None = None,
    density: L[False] | None = None,
    *,
    weights: _WeightsLike,
) -> _HistogramResult[Incomplete, np.complex128]: ...
@overload  # a: known, density: True (keyword), weights: +float | None (default)
def histogram[ScalarT: np.inexact | np.object_](
    a: _ArrayLike[ScalarT],
    bins: _BinKind | SupportsIndex | ArrayLike = 10,
    range: _Range | None = None,
    *,
    density: L[True],
    weights: _ArrayLikeFloat_co | None = None,
) -> _HistogramResult[np.float64, ScalarT]: ...
@overload  # a: known, density: True (keyword), weights: +complex
def histogram[ScalarT: np.inexact | np.object_](
    a: _ArrayLike[ScalarT],
    bins: _BinKind | SupportsIndex | ArrayLike = 10,
    range: _Range | None = None,
    *,
    density: L[True],
    weights: _ArrayLike[np.complexfloating] | _NestedList[complex],
) -> _HistogramResult[np.complex128, ScalarT]: ...
@overload  # a: known, density: False (default), weights: ~int | None (default)
def histogram[ScalarT: np.inexact | np.object_](
    a: _ArrayLike[ScalarT],
    bins: _BinKind | SupportsIndex | ArrayLike = 10,
    range: _Range | None = None,
    density: L[False] | None = None,
    weights: _NestedSequence[int] | None = None,
) -> _HistogramResult[np.intp, ScalarT]: ...
@overload  # a: known, density: False (default), weights: known (keyword)
def histogram[ScalarT: np.inexact | np.object_, WeightsT: np.bool | np.number | np.timedelta64](
    a: _ArrayLike[ScalarT],
    bins: _BinKind | SupportsIndex | ArrayLike = 10,
    range: _Range | None = None,
    density: L[False] | None = None,
    *,
    weights: _ArrayLike[WeightsT],
) -> _HistogramResult[WeightsT, ScalarT]: ...
@overload  # a: known, density: False (default), weights: unknown (keyword)
def histogram[ScalarT: np.inexact | np.object_](
    a: _ArrayLike[ScalarT],
    bins: _BinKind | SupportsIndex | ArrayLike = 10,
    range: _Range | None = None,
    density: L[False] | None = None,
    *,
    weights: _WeightsLike,
) -> _HistogramResult[Incomplete, ScalarT]: ...
@overload  # a: unknown, density: True (keyword), weights: +float | None (default)
def histogram(
    a: _ArrayLikeComplex_co,
    bins: _BinKind | SupportsIndex | ArrayLike = 10,
    range: _Range | None = None,
    *,
    density: L[True],
    weights: _ArrayLikeFloat_co | None = None,
) -> _HistogramResult[np.float64, Incomplete]: ...
@overload  # a: unknown, density: True (keyword), weights: +complex
def histogram(
    a: _ArrayLikeComplex_co,
    bins: _BinKind | SupportsIndex | ArrayLike = 10,
    range: _Range | None = None,
    *,
    density: L[True],
    weights: _ArrayLike[np.complexfloating] | _NestedList[complex],
) -> _HistogramResult[np.complex128, Incomplete]: ...
@overload  # a: unknown, density: False (default), weights: int | None (default)
def histogram(
    a: _ArrayLikeComplex_co,
    bins: _BinKind | SupportsIndex | ArrayLike = 10,
    range: _Range | None = None,
    density: L[False] | None = None,
    weights: _NestedSequence[int] | None = None,
) -> _HistogramResult[np.intp, Incomplete]: ...
@overload  # a: unknown, density: False (default), weights: known (keyword)
def histogram[WeightsT: np.bool | np.number | np.timedelta64](
    a: _ArrayLikeComplex_co,
    bins: _BinKind | SupportsIndex | ArrayLike = 10,
    range: _Range | None = None,
    density: L[False] | None = None,
    *,
    weights: _ArrayLike[WeightsT],
) -> _HistogramResult[WeightsT, Incomplete]: ...
@overload  # a: unknown, density: False (default), weights: unknown (keyword)
def histogram(
    a: _ArrayLikeComplex_co,
    bins: _BinKind | SupportsIndex | ArrayLike = 10,
    range: _Range | None = None,
    density: L[False] | None = None,
    *,
    weights: _WeightsLike,
) -> _HistogramResult[Incomplete, Incomplete]: ...

# unlike `histogram`, `weights` must be safe-castable to f64
@overload  # ?d T  (workaround)
def histogramdd[ScalarT: np.inexact](
    sample: _ArrayJustND[ScalarT],
    bins: SupportsIndex | ArrayLike = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[NDArray[np.float64], list[_Array1D[ScalarT]]]: ...
@overload  # ?d +f64  (workaround)
def histogramdd(
    sample: _ArrayJustND[np.integer | np.bool | np.object_],
    bins: SupportsIndex | ArrayLike = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[NDArray[np.float64], list[_Array1D[np.float64]]]: ...
@overload  # 1d T
def histogramdd[ScalarT: np.inexact](
    sample: _ArrayLike1D[ScalarT],
    bins: SupportsIndex | ArrayLike = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[_Array1D[np.float64], list[_Array1D[ScalarT]]]: ...
@overload  # 1d +f64
def histogramdd(
    sample: _Array1D[np.integer | np.bool] | Sequence[float | np.integer | np.bool],
    bins: SupportsIndex | ArrayLike = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[_Array1D[np.float64], list[_Array1D[np.float64]]]: ...
@overload  # 1d ~c128
def histogramdd(
    sample: list[complex],
    bins: SupportsIndex | ArrayLike = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[_Array1D[np.float64], list[_Array1D[np.complex128]]]: ...
@overload  # 1d ?
def histogramdd(
    sample: _ArrayLike1D[np.number | np.bool] | Sequence[complex],
    bins: SupportsIndex | ArrayLike = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[_Array1D[np.float64], list[_Array1D[Any]]]: ...
@overload  # (1d, 1d) T
def histogramdd[ScalarT: np.inexact](
    sample: _2Tuple[_ArrayLike1D[ScalarT]],
    bins: SupportsIndex | ArrayLike = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[_Array2D[np.float64], list[_Array1D[ScalarT]]]: ...
@overload  # (1d, 1d) +f64
def histogramdd(
    sample: _2Tuple[_Array1D[np.integer | np.bool] | Sequence[float | np.integer | np.bool]],
    bins: SupportsIndex | ArrayLike = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[_Array2D[np.float64], list[_Array1D[np.float64]]]: ...
@overload  # (1d, 1d) ~c128
def histogramdd(
    sample: _2Tuple[list[complex]],
    bins: SupportsIndex | ArrayLike = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[_Array2D[np.float64], list[_Array1D[np.complex128]]]: ...
@overload  # (1d, 1d) ?
def histogramdd(
    sample: _2Tuple[_ArrayLike1D[np.number | np.bool] | Sequence[complex]],
    bins: SupportsIndex | ArrayLike = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[_Array2D[np.float64], list[_Array1D[Any]]]: ...
@overload  # (1d, 1d, 1d) T
def histogramdd[ScalarT: np.inexact](
    sample: _3Tuple[_ArrayLike1D[ScalarT]],
    bins: SupportsIndex | ArrayLike = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[_Array3D[np.float64], list[_Array1D[ScalarT]]]: ...
@overload  # (1d, 1d, 1d) +f64
def histogramdd(
    sample: _3Tuple[_Array1D[np.integer | np.bool] | Sequence[float | np.integer | np.bool]],
    bins: SupportsIndex | ArrayLike = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[_Array3D[np.float64], list[_Array1D[np.float64]]]: ...
@overload  # (1d, 1d, 1d) ~c128
def histogramdd(
    sample: _3Tuple[list[complex]],
    bins: SupportsIndex | ArrayLike = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[_Array3D[np.float64], list[_Array1D[np.complex128]]]: ...
@overload  # (1d, 1d, 1d) ?
def histogramdd(
    sample: _3Tuple[_ArrayLike1D[np.number | np.bool] | Sequence[complex]],
    bins: SupportsIndex | ArrayLike = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[_Array3D[np.float64], list[_Array1D[Any]]]: ...
@overload  # ?d +f64
def histogramdd(
    sample: _ArrayLikeInt_co | _NestedSequence[float] | _ArrayLikeObject_co,
    bins: SupportsIndex | ArrayLike = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[NDArray[np.float64], list[_Array1D[np.float64]]]: ...
@overload  # ?d ~c128
def histogramdd(
    sample: _NestedList[complex],
    bins: SupportsIndex | ArrayLike = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[NDArray[np.float64], list[_Array1D[np.complex128]]]: ...
@overload  # ?d T
def histogramdd[ScalarT: np.inexact](
    sample: _ArrayLike[ScalarT],
    bins: SupportsIndex | ArrayLike = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[NDArray[np.float64], list[_Array1D[ScalarT]]]: ...
@overload  # ?d ?  (fallback)
def histogramdd(
    sample: _ArrayLikeComplex_co,
    bins: SupportsIndex | ArrayLike = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[NDArray[np.float64], list[_Array1D[Any]]]: ...
