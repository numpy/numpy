from _typeshed import Incomplete
from collections.abc import Sequence
from typing import Any, Literal as L, SupportsIndex, overload

import numpy as np
from numpy._typing import (
    ArrayLike,
    NDArray,
    _ArrayLike,
    _ArrayLikeInt_co,
    _ArrayLikeNumber_co,
    _ArrayLikeObject_co,
    _NestedSequence,
)

__all__ = ["histogram", "histogramdd", "histogram_bin_edges"]

###

type _BinKind = L["auto", "fd", "doane", "scott", "stone", "rice", "sturges", "sqrt"]

type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _WeightsLike = _ArrayLikeNumber_co | _ArrayLikeObject_co

###

# NOTE: The return type can also be complex or `object_`, not only floating like the docstring suggests.
@overload  # +float64
def histogram_bin_edges(
    a: _ArrayLikeInt_co | _NestedSequence[float],
    bins: _BinKind | SupportsIndex | ArrayLike = 10,
    range: tuple[float, float] | None = None,
    weights: _WeightsLike | None = None,
) -> _Array1D[np.float64]: ...
@overload  # ~complex
def histogram_bin_edges(
    a: list[complex] | _NestedSequence[list[complex]],
    bins: _BinKind | SupportsIndex | ArrayLike = 10,
    range: tuple[float, float] | None = None,
    weights: _WeightsLike | None = None,
) -> _Array1D[np.complex128]: ...
@overload  # ~inexact | object_
def histogram_bin_edges[ScalarT: np.inexact | np.object_](
    a: _ArrayLike[ScalarT],
    bins: _BinKind | SupportsIndex | ArrayLike = 10,
    range: tuple[float, float] | None = None,
    weights: _WeightsLike | None = None,
) -> _Array1D[ScalarT]: ...
@overload  # fallback
def histogram_bin_edges(
    a: _ArrayLikeNumber_co,
    bins: _BinKind | SupportsIndex | ArrayLike = 10,
    range: tuple[float, float] | None = None,
    weights: _WeightsLike | None = None,
) -> _Array1D[Incomplete]: ...

#
def histogram(
    a: ArrayLike,
    bins: _BinKind | SupportsIndex | ArrayLike = 10,
    range: tuple[float, float] | None = None,
    density: bool | None = None,
    weights: ArrayLike | None = None,
) -> tuple[NDArray[Any], NDArray[Any]]: ...

def histogramdd(
    sample: ArrayLike,
    bins: SupportsIndex | ArrayLike = 10,
    range: Sequence[tuple[float, float]] | None = None,
    density: bool | None = None,
    weights: ArrayLike | None = None,
) -> tuple[NDArray[Any], tuple[NDArray[Any], ...]]: ...
