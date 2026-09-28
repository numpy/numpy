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
type _ToBins = SupportsIndex | Sequence[SupportsIndex] | _Array1D[np.integer]

###

# NOTE: The return type can also be complex or `object_`, not only floating like the docstring suggests.
@overload  # Nd, 1d T
def histogram_bin_edges[ScalarT: np.number | np.object_](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    bins: _ArrayLike1D[ScalarT],
    range: _Range | None = None,
    weights: _WeightsLike | None = None,
) -> _Array1D[ScalarT]: ...
@overload  # Nd, 1d ~int
def histogram_bin_edges(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    bins: Sequence[int],
    range: _Range | None = None,
    weights: _WeightsLike | None = None,
) -> _Array1D[np.int_]: ...
@overload  # Nd, 1d ~float
def histogram_bin_edges(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    bins: list[float],
    range: _Range | None = None,
    weights: _WeightsLike | None = None,
) -> _Array1D[np.float64]: ...
@overload  # Nd +f64
def histogram_bin_edges(
    a: _ArrayLikeInt_co | _NestedSequence[float],
    bins: str | SupportsIndex = 10,
    range: _Range | None = None,
    weights: _WeightsLike | None = None,
) -> _Array1D[np.float64]: ...
@overload  # Nd ~complex
def histogram_bin_edges(
    a: _NestedList[complex],
    bins: str | SupportsIndex = 10,
    range: _Range | None = None,
    weights: _WeightsLike | None = None,
) -> _Array1D[np.complex128]: ...
@overload  # Nd T
def histogram_bin_edges[ScalarT: np.inexact | np.object_](
    a: _ArrayLike[ScalarT],
    bins: str | SupportsIndex = 10,
    range: _Range | None = None,
    weights: _WeightsLike | None = None,
) -> _Array1D[ScalarT]: ...
@overload  # Nd  (fallback)
def histogram_bin_edges(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    bins: SupportsIndex | ArrayLike = 10,
    range: _Range | None = None,
    weights: _WeightsLike | None = None,
) -> _Array1D[Any]: ...

# There are 7 groups of 2 + 3 overloads (2 for density=True, 3 for density=False) = 35 in total
@overload  # Nd, 1d T, density=True
def histogram[ScalarT: np.number | np.object_](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    bins: _ArrayLike1D[ScalarT],
    range: _Range | None = None,
    *,
    density: L[True],
    weights: _ArrayLikeFloat_co | None = None,
) -> _HistogramResult[np.float64, ScalarT]: ...
@overload  # Nd, 1d T, density=True, weights=+c128
def histogram[ScalarT: np.number | np.object_](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    bins: _ArrayLike1D[ScalarT],
    range: _Range | None = None,
    *,
    density: L[True],
    weights: _ArrayLike[np.complexfloating] | _NestedList[complex],
) -> _HistogramResult[np.complex128, ScalarT]: ...
@overload  # Nd, 1d T
def histogram[ScalarT: np.number | np.object_](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    bins: _ArrayLike1D[ScalarT],
    range: _Range | None = None,
    density: L[False] | None = None,
    weights: _NestedSequence[int] | None = None,
) -> _HistogramResult[np.intp, ScalarT]: ...
@overload  # Nd, 1d T, weights=<known>
def histogram[ScalarT: np.number | np.object_, WeightsT: np.bool | np.number | np.timedelta64](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    bins: _ArrayLike1D[ScalarT],
    range: _Range | None = None,
    density: L[False] | None = None,
    *,
    weights: _ArrayLike[WeightsT],
) -> _HistogramResult[WeightsT, ScalarT]: ...
@overload  # Nd, 1d T, weights=<unknown>
def histogram[ScalarT: np.number | np.object_](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    bins: _ArrayLike1D[ScalarT],
    range: _Range | None = None,
    density: L[False] | None = None,
    *,
    weights: _WeightsLike,
) -> _HistogramResult[Any, ScalarT]: ...
@overload  # Nd, 1d ~int, density=True
def histogram(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    bins: Sequence[int],
    range: _Range | None = None,
    *,
    density: L[True],
    weights: _ArrayLikeFloat_co | None = None,
) -> _HistogramResult[np.float64, np.int_]: ...
@overload  # Nd, 1d ~int, density=True, weights=+c128
def histogram(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    bins: Sequence[int],
    range: _Range | None = None,
    *,
    density: L[True],
    weights: _ArrayLike[np.complexfloating] | _NestedList[complex],
) -> _HistogramResult[np.complex128, np.int_]: ...
@overload  # Nd, 1d ~int
def histogram(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    bins: Sequence[int],
    range: _Range | None = None,
    density: L[False] | None = None,
    weights: _NestedSequence[int] | None = None,
) -> _HistogramResult[np.intp, np.int_]: ...
@overload  # Nd, 1d ~int, weights=<known>
def histogram[WeightsT: np.bool | np.number | np.timedelta64](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    bins: Sequence[int],
    range: _Range | None = None,
    density: L[False] | None = None,
    *,
    weights: _ArrayLike[WeightsT],
) -> _HistogramResult[WeightsT, np.int_]: ...
@overload  # Nd, 1d ~int, weights=<unknown>
def histogram(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    bins: Sequence[int],
    range: _Range | None = None,
    density: L[False] | None = None,
    *,
    weights: _WeightsLike,
) -> _HistogramResult[Any, np.int_]: ...
@overload  # Nd, 1d ~float, density=True
def histogram(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    bins: list[float],
    range: _Range | None = None,
    *,
    density: L[True],
    weights: _ArrayLikeFloat_co | None = None,
) -> _HistogramResult[np.float64, np.float64]: ...
@overload  # Nd, 1d ~float, density=True, weights=+c128
def histogram(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    bins: list[float],
    range: _Range | None = None,
    *,
    density: L[True],
    weights: _ArrayLike[np.complexfloating] | _NestedList[complex],
) -> _HistogramResult[np.complex128, np.float64]: ...
@overload  # Nd, 1d ~float
def histogram(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    bins: list[float],
    range: _Range | None = None,
    density: L[False] | None = None,
    weights: _NestedSequence[int] | None = None,
) -> _HistogramResult[np.intp, np.float64]: ...
@overload  # Nd, 1d ~float, weights=<known>
def histogram[WeightsT: np.bool | np.number | np.timedelta64](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    bins: list[float],
    range: _Range | None = None,
    density: L[False] | None = None,
    *,
    weights: _ArrayLike[WeightsT],
) -> _HistogramResult[WeightsT, np.float64]: ...
@overload  # Nd, 1d ~float, weights=<unknown>
def histogram(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    bins: list[float],
    range: _Range | None = None,
    density: L[False] | None = None,
    *,
    weights: _WeightsLike,
) -> _HistogramResult[Any, np.float64]: ...
@overload  # Nd +f64, density=True
def histogram(
    a: _ArrayLikeInt_co | _NestedSequence[float],
    bins: str | SupportsIndex = 10,
    range: _Range | None = None,
    *,
    density: L[True],
    weights: _ArrayLikeFloat_co | None = None,
) -> _HistogramResult[np.float64, np.float64]: ...
@overload  # Nd +f64, density=True, weights=+c128
def histogram(
    a: _ArrayLikeInt_co | _NestedSequence[float],
    bins: str | SupportsIndex = 10,
    range: _Range | None = None,
    *,
    density: L[True],
    weights: _ArrayLike[np.complexfloating] | _NestedList[complex],
) -> _HistogramResult[np.complex128, np.float64]: ...
@overload  # Nd +f64
def histogram(
    a: _ArrayLikeInt_co | _NestedSequence[float],
    bins: str | SupportsIndex = 10,
    range: _Range | None = None,
    density: L[False] | None = None,
    weights: _NestedSequence[int] | None = None,
) -> _HistogramResult[np.intp, np.float64]: ...
@overload  # Nd +f64, weights=<known>
def histogram[WeightsT: np.bool | np.number | np.timedelta64](
    a: _ArrayLikeInt_co | _NestedSequence[float],
    bins: str | SupportsIndex = 10,
    range: _Range | None = None,
    density: L[False] | None = None,
    *,
    weights: _ArrayLike[WeightsT],
) -> _HistogramResult[WeightsT, np.float64]: ...
@overload  # Nd +f64, weights=<unknown>
def histogram(
    a: _ArrayLikeInt_co | _NestedSequence[float],
    bins: str | SupportsIndex = 10,
    range: _Range | None = None,
    density: L[False] | None = None,
    *,
    weights: _WeightsLike,
) -> _HistogramResult[Any, np.float64]: ...
@overload  # Nd ~complex, density=True
def histogram(
    a: _NestedList[complex],
    bins: str | SupportsIndex = 10,
    range: _Range | None = None,
    *,
    density: L[True],
    weights: _ArrayLikeFloat_co | None = None,
) -> _HistogramResult[np.float64, np.complex128]: ...
@overload  # Nd ~complex, density=True, weights=+c128
def histogram(
    a: _NestedList[complex],
    bins: str | SupportsIndex = 10,
    range: _Range | None = None,
    *,
    density: L[True],
    weights: _ArrayLike[np.complexfloating] | _NestedList[complex],
) -> _HistogramResult[np.complex128, np.complex128]: ...
@overload  # Nd ~complex
def histogram(
    a: _NestedList[complex],
    bins: str | SupportsIndex = 10,
    range: _Range | None = None,
    density: L[False] | None = None,
    weights: _NestedSequence[int] | None = None,
) -> _HistogramResult[np.intp, np.complex128]: ...
@overload  # Nd ~complex, weights=<known>
def histogram[WeightsT: np.bool | np.number | np.timedelta64](
    a: _NestedList[complex],
    bins: str | SupportsIndex = 10,
    range: _Range | None = None,
    density: L[False] | None = None,
    *,
    weights: _ArrayLike[WeightsT],
) -> _HistogramResult[WeightsT, np.complex128]: ...
@overload  # Nd ~complex, weights=<unknown>
def histogram(
    a: _NestedList[complex],
    bins: str | SupportsIndex = 10,
    range: _Range | None = None,
    density: L[False] | None = None,
    *,
    weights: _WeightsLike,
) -> _HistogramResult[Any, np.complex128]: ...
@overload  # Nd T, density=True
def histogram[ScalarT: np.inexact | np.object_](
    a: _ArrayLike[ScalarT],
    bins: str | SupportsIndex = 10,
    range: _Range | None = None,
    *,
    density: L[True],
    weights: _ArrayLikeFloat_co | None = None,
) -> _HistogramResult[np.float64, ScalarT]: ...
@overload  # Nd T, density=True, weights=+c128
def histogram[ScalarT: np.inexact | np.object_](
    a: _ArrayLike[ScalarT],
    bins: str | SupportsIndex = 10,
    range: _Range | None = None,
    *,
    density: L[True],
    weights: _ArrayLike[np.complexfloating] | _NestedList[complex],
) -> _HistogramResult[np.complex128, ScalarT]: ...
@overload  # Nd T
def histogram[ScalarT: np.inexact | np.object_](
    a: _ArrayLike[ScalarT],
    bins: str | SupportsIndex = 10,
    range: _Range | None = None,
    density: L[False] | None = None,
    weights: _NestedSequence[int] | None = None,
) -> _HistogramResult[np.intp, ScalarT]: ...
@overload  # Nd T, weights=<known>
def histogram[ScalarT: np.inexact | np.object_, WeightsT: np.bool | np.number | np.timedelta64](
    a: _ArrayLike[ScalarT],
    bins: str | SupportsIndex = 10,
    range: _Range | None = None,
    density: L[False] | None = None,
    *,
    weights: _ArrayLike[WeightsT],
) -> _HistogramResult[WeightsT, ScalarT]: ...
@overload  # Nd T, weights=<unknown>
def histogram[ScalarT: np.inexact | np.object_](
    a: _ArrayLike[ScalarT],
    bins: str | SupportsIndex = 10,
    range: _Range | None = None,
    density: L[False] | None = None,
    *,
    weights: _WeightsLike,
) -> _HistogramResult[Any, ScalarT]: ...
@overload  # Nd, density=True
def histogram(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    bins: SupportsIndex | ArrayLike = 10,
    range: _Range | None = None,
    *,
    density: L[True],
    weights: _ArrayLikeFloat_co | None = None,
) -> _HistogramResult[np.float64, Any]: ...
@overload  # Nd, density=True, weights=+c128
def histogram(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    bins: SupportsIndex | ArrayLike = 10,
    range: _Range | None = None,
    *,
    density: L[True],
    weights: _ArrayLike[np.complexfloating] | _NestedList[complex],
) -> _HistogramResult[np.complex128, Any]: ...
@overload  # Nd
def histogram(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    bins: SupportsIndex | ArrayLike = 10,
    range: _Range | None = None,
    density: L[False] | None = None,
    weights: _NestedSequence[int] | None = None,
) -> _HistogramResult[np.intp, Any]: ...
@overload  # Nd, weights=<known>
def histogram[WeightsT: np.bool | np.number | np.timedelta64](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    bins: SupportsIndex | ArrayLike = 10,
    range: _Range | None = None,
    density: L[False] | None = None,
    *,
    weights: _ArrayLike[WeightsT],
) -> _HistogramResult[WeightsT, Any]: ...
@overload  # Nd, weights=<unknown>  (fallback)
def histogram(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    bins: SupportsIndex | ArrayLike = 10,
    range: _Range | None = None,
    density: L[False] | None = None,
    *,
    weights: _WeightsLike,
) -> _HistogramResult[Any, Any]: ...

# unlike `histogram`, `weights` must be safe-castable to f64
@overload  # Nd, 2d T
def histogramdd[ScalarT: np.number | np.object_](
    sample: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    bins: Sequence[_ArrayLike1D[ScalarT]],
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[NDArray[np.float64], list[_Array1D[ScalarT]]]: ...
@overload  # Nd, 2d ~int
def histogramdd(
    sample: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    bins: Sequence[Sequence[int]],
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[NDArray[np.float64], list[_Array1D[np.int_]]]: ...
@overload  # Nd, 2d ~float
def histogramdd(
    sample: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    bins: Sequence[list[float]],
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[NDArray[np.float64], list[_Array1D[np.float64]]]: ...
@overload  # ?d T  (workaround)
def histogramdd[ScalarT: np.inexact](
    sample: _ArrayJustND[ScalarT],
    bins: _ToBins = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[NDArray[np.float64], list[_Array1D[ScalarT]]]: ...
@overload  # ?d +f64  (workaround)
def histogramdd(
    sample: _ArrayJustND[np.integer | np.bool | np.object_],
    bins: _ToBins = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[NDArray[np.float64], list[_Array1D[np.float64]]]: ...
@overload  # 1d T
def histogramdd[ScalarT: np.inexact](
    sample: _ArrayLike1D[ScalarT],
    bins: _ToBins = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[_Array1D[np.float64], list[_Array1D[ScalarT]]]: ...
@overload  # 1d +f64
def histogramdd(
    sample: _Array1D[np.integer | np.bool] | Sequence[float | np.integer | np.bool],
    bins: _ToBins = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[_Array1D[np.float64], list[_Array1D[np.float64]]]: ...
@overload  # 1d ~c128
def histogramdd(
    sample: list[complex],
    bins: _ToBins = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[_Array1D[np.float64], list[_Array1D[np.complex128]]]: ...
@overload  # 1d ?
def histogramdd(
    sample: _ArrayLike1D[np.number | np.bool] | Sequence[complex],
    bins: _ToBins = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[_Array1D[np.float64], list[_Array1D[Any]]]: ...
@overload  # (1d, 1d) T
def histogramdd[ScalarT: np.inexact](
    sample: _2Tuple[_ArrayLike1D[ScalarT]],
    bins: _ToBins = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[_Array2D[np.float64], list[_Array1D[ScalarT]]]: ...
@overload  # (1d, 1d) +f64
def histogramdd(
    sample: _2Tuple[_Array1D[np.integer | np.bool] | Sequence[float | np.integer | np.bool]],
    bins: _ToBins = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[_Array2D[np.float64], list[_Array1D[np.float64]]]: ...
@overload  # (1d, 1d) ~c128
def histogramdd(
    sample: _2Tuple[list[complex]],
    bins: _ToBins = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[_Array2D[np.float64], list[_Array1D[np.complex128]]]: ...
@overload  # (1d, 1d) ?
def histogramdd(
    sample: _2Tuple[_ArrayLike1D[np.number | np.bool] | Sequence[complex]],
    bins: _ToBins = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[_Array2D[np.float64], list[_Array1D[Any]]]: ...
@overload  # (1d, 1d, 1d) T
def histogramdd[ScalarT: np.inexact](
    sample: _3Tuple[_ArrayLike1D[ScalarT]],
    bins: _ToBins = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[_Array3D[np.float64], list[_Array1D[ScalarT]]]: ...
@overload  # (1d, 1d, 1d) +f64
def histogramdd(
    sample: _3Tuple[_Array1D[np.integer | np.bool] | Sequence[float | np.integer | np.bool]],
    bins: _ToBins = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[_Array3D[np.float64], list[_Array1D[np.float64]]]: ...
@overload  # (1d, 1d, 1d) ~c128
def histogramdd(
    sample: _3Tuple[list[complex]],
    bins: _ToBins = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[_Array3D[np.float64], list[_Array1D[np.complex128]]]: ...
@overload  # (1d, 1d, 1d) ?
def histogramdd(
    sample: _3Tuple[_ArrayLike1D[np.number | np.bool] | Sequence[complex]],
    bins: _ToBins = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[_Array3D[np.float64], list[_Array1D[Any]]]: ...
@overload  # ?d +f64
def histogramdd(
    sample: _ArrayLikeInt_co | _NestedSequence[float] | _ArrayLikeObject_co,
    bins: _ToBins = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[NDArray[np.float64], list[_Array1D[np.float64]]]: ...
@overload  # ?d ~c128
def histogramdd(
    sample: _NestedList[complex],
    bins: _ToBins = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[NDArray[np.float64], list[_Array1D[np.complex128]]]: ...
@overload  # ?d T
def histogramdd[ScalarT: np.inexact](
    sample: _ArrayLike[ScalarT],
    bins: _ToBins = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[NDArray[np.float64], list[_Array1D[ScalarT]]]: ...
@overload  # ?d ?  (fallback)
def histogramdd(
    sample: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    bins: SupportsIndex | ArrayLike = 10,
    range: Sequence[_Range] | None = None,
    density: bool | None = None,
    weights: _ArrayLikeFloat64_co | None = None,
) -> tuple[NDArray[np.float64], list[_Array1D[Any]]]: ...
