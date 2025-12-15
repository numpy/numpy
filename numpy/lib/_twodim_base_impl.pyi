from _typeshed import Incomplete
from collections.abc import Callable, Sequence
from typing import Any, Literal as L, Never, Protocol, overload, type_check_only

import numpy as np
from numpy import _OrderCF
from numpy._typing import (
    ArrayLike,
    DTypeLike,
    NDArray,
    _ArrayLike,
    _DTypeLike,
    _NumberLike_co,
    _ScalarLike_co,
    _SupportsArray,
    _SupportsArrayFunc,
)

__all__ = [
    "diag",
    "diagflat",
    "eye",
    "fliplr",
    "flipud",
    "tri",
    "triu",
    "tril",
    "vander",
    "histogram2d",
    "mask_indices",
    "tril_indices",
    "tril_indices_from",
    "triu_indices",
    "triu_indices_from",
]

###

type _Int_co = np.integer | np.bool
type _Float_co = np.floating | _Int_co
type _Number_co = np.number | np.bool

type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]
# Workaround for mypy's and pyright's lack of compliance with the typing spec for
# overloads for gradual types. This works because only `Any` and `Never` are assignable
# to `Never`.
type _ArrayNoD[ScalarT: np.generic] = np.ndarray[tuple[Never] | tuple[Never, Never], np.dtype[ScalarT]]

type _ArrayLike1D[ScalarT: np.generic] = _SupportsArray[np.dtype[ScalarT]] | Sequence[ScalarT]
type _ArrayLike1DInt_co = _SupportsArray[np.dtype[_Int_co]] | Sequence[int | _Int_co]
type _ArrayLike1DFloat_co = _SupportsArray[np.dtype[_Float_co]] | Sequence[float | _Float_co]
type _ArrayLike2DFloat_co = _SupportsArray[np.dtype[_Float_co]] | Sequence[_ArrayLike1DFloat_co]
type _ArrayLike1DNumber_co = _SupportsArray[np.dtype[_Number_co]] | Sequence[complex | _Number_co]

# The returned arrays dtype must be compatible with `np.equal`
type _MaskFunc[_T] = Callable[[NDArray[np.int_], _T], NDArray[_Number_co | np.timedelta64 | np.datetime64 | np.object_]]

type _Indices2D = tuple[_Array1D[np.intp], _Array1D[np.intp]]
type _Histogram2D[ScalarT: np.generic] = tuple[_Array1D[np.float64], _Array1D[ScalarT], _Array1D[ScalarT]]

@type_check_only
class _HasShapeAndNDim(Protocol):
    @property  # TODO: require 2d shape once shape-typing has matured
    def shape(self) -> tuple[int, ...]: ...
    @property
    def ndim(self) -> int: ...

###

# keep in sync with `flipud`
@overload
def fliplr[ArrayT: np.ndarray](m: ArrayT) -> ArrayT: ...
@overload
def fliplr[ScalarT: np.generic](m: _ArrayLike[ScalarT]) -> NDArray[ScalarT]: ...
@overload
def fliplr(m: ArrayLike) -> NDArray[Any]: ...

# keep in sync with `fliplr`
@overload
def flipud[ArrayT: np.ndarray](m: ArrayT) -> ArrayT: ...
@overload
def flipud[ScalarT: np.generic](m: _ArrayLike[ScalarT]) -> NDArray[ScalarT]: ...
@overload
def flipud(m: ArrayLike) -> NDArray[Any]: ...

#
@overload
def eye(
    N: int,
    M: int | None = None,
    k: int = 0,
    dtype: None = ...,  # = float  # stubdefaulter: ignore[missing-default]
    order: _OrderCF = "C",
    *,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> _Array2D[np.float64]: ...
@overload
def eye[ScalarT: np.generic](
    N: int,
    M: int | None,
    k: int,
    dtype: _DTypeLike[ScalarT],
    order: _OrderCF = "C",
    *,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> _Array2D[ScalarT]: ...
@overload
def eye[ScalarT: np.generic](
    N: int,
    M: int | None = None,
    k: int = 0,
    *,
    dtype: _DTypeLike[ScalarT],
    order: _OrderCF = "C",
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> _Array2D[ScalarT]: ...
@overload
def eye(
    N: int,
    M: int | None = None,
    k: int = 0,
    dtype: DTypeLike | None = ...,  # = float
    order: _OrderCF = "C",
    *,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> _Array2D[Incomplete]: ...

#
@overload
def diag[ScalarT: np.generic](v: _ArrayNoD[ScalarT] | Sequence[Sequence[ScalarT]], k: int = 0) -> NDArray[ScalarT]: ...
@overload
def diag[ScalarT: np.generic](v: _Array2D[ScalarT] | Sequence[Sequence[ScalarT]], k: int = 0) -> _Array1D[ScalarT]: ...
@overload
def diag[ScalarT: np.generic](v: _Array1D[ScalarT] | Sequence[ScalarT], k: int = 0) -> _Array2D[ScalarT]: ...
@overload
def diag(v: Sequence[Sequence[_ScalarLike_co]], k: int = 0) -> _Array1D[Incomplete]: ...
@overload
def diag(v: Sequence[_ScalarLike_co], k: int = 0) -> _Array2D[Incomplete]: ...
@overload
def diag[ScalarT: np.generic](v: _ArrayLike[ScalarT], k: int = 0) -> NDArray[ScalarT]: ...
@overload
def diag(v: ArrayLike, k: int = 0) -> NDArray[Incomplete]: ...

# keep in sync with `numpy.ma.extras.diagflat`
@overload
def diagflat[ScalarT: np.generic](v: _ArrayLike[ScalarT], k: int = 0) -> _Array2D[ScalarT]: ...
@overload
def diagflat(v: ArrayLike, k: int = 0) -> _Array2D[Incomplete]: ...

#
@overload
def tri(
    N: int,
    M: int | None = None,
    k: int = 0,
    dtype: None = ...,  # = float  # stubdefaulter: ignore[missing-default]
    *,
    like: _SupportsArrayFunc | None = None
) -> _Array2D[np.float64]: ...
@overload
def tri[ScalarT: np.generic](
    N: int,
    M: int | None,
    k: int,
    dtype: _DTypeLike[ScalarT],
    *,
    like: _SupportsArrayFunc | None = None
) -> _Array2D[ScalarT]: ...
@overload
def tri[ScalarT: np.generic](
    N: int,
    M: int | None = None,
    k: int = 0,
    *,
    dtype: _DTypeLike[ScalarT],
    like: _SupportsArrayFunc | None = None
) -> _Array2D[ScalarT]: ...
@overload
def tri(
    N: int,
    M: int | None = None,
    k: int = 0,
    dtype: DTypeLike | None = ...,  # = float
    *,
    like: _SupportsArrayFunc | None = None
) -> _Array2D[Any]: ...

# keep in sync with `triu`
@overload
def tril[ArrayT: np.ndarray](m: ArrayT, k: int = 0) -> ArrayT: ...
@overload
def tril[ScalarT: np.generic](m: _ArrayLike[ScalarT], k: int = 0) -> NDArray[ScalarT]: ...
@overload
def tril(m: ArrayLike, k: int = 0) -> NDArray[Any]: ...

# keep in sync with `tril`
@overload
def triu[ArrayT: np.ndarray](m: ArrayT, k: int = 0) -> ArrayT: ...
@overload
def triu[ScalarT: np.generic](m: _ArrayLike[ScalarT], k: int = 0) -> NDArray[ScalarT]: ...
@overload
def triu(m: ArrayLike, k: int = 0) -> NDArray[Any]: ...

# we use `list` (invariant) instead of `Sequence` (covariant) to avoid overlap
@overload
def vander[ScalarT: np.number | np.object_](x: _ArrayLike1D[ScalarT], N: int | None = None, increasing: bool = False) -> _Array2D[ScalarT]: ...
@overload
def vander(x: _ArrayLike1D[np.bool] | list[int], N: int | None = None, increasing: bool = False) -> _Array2D[np.int_]: ...
@overload
def vander(x: list[float], N: int | None = None, increasing: bool = False) -> _Array2D[np.float64]: ...
@overload
def vander(x: list[complex], N: int | None = None, increasing: bool = False) -> _Array2D[np.complex128]: ...
@overload  # fallback
def vander(x: Sequence[_NumberLike_co], N: int | None = None, increasing: bool = False) -> _Array2D[Any]: ...

#
@overload
def histogram2d[ScalarT: np.complexfloating](
    x: _ArrayLike1D[ScalarT],
    y: _ArrayLike1D[ScalarT | _Float_co],
    bins: int | Sequence[int] = 10,
    range: _ArrayLike2DFloat_co | None = None,
    density: bool | None = None,
    weights: _ArrayLike1DFloat_co | None = None,
) -> _Histogram2D[ScalarT]: ...
@overload
def histogram2d[ScalarT: np.complexfloating](
    x: _ArrayLike1D[ScalarT | _Float_co],
    y: _ArrayLike1D[ScalarT],
    bins: int | Sequence[int] = 10,
    range: _ArrayLike2DFloat_co | None = None,
    density: bool | None = None,
    weights: _ArrayLike1DFloat_co | None = None,
) -> _Histogram2D[ScalarT]: ...
@overload
def histogram2d[ScalarT: np.inexact](
    x: _ArrayLike1D[ScalarT],
    y: _ArrayLike1D[ScalarT | _Int_co],
    bins: int | Sequence[int] = 10,
    range: _ArrayLike2DFloat_co | None = None,
    density: bool | None = None,
    weights: _ArrayLike1DFloat_co | None = None,
) -> _Histogram2D[ScalarT]: ...
@overload
def histogram2d[ScalarT: np.inexact](
    x: _ArrayLike1D[ScalarT | _Int_co],
    y: _ArrayLike1D[ScalarT],
    bins: int | Sequence[int] = 10,
    range: _ArrayLike2DFloat_co | None = None,
    density: bool | None = None,
    weights: _ArrayLike1DFloat_co | None = None,
) -> _Histogram2D[ScalarT]: ...
@overload
def histogram2d(
    x: _ArrayLike1DInt_co | Sequence[float],
    y: _ArrayLike1DInt_co | Sequence[float],
    bins: int | Sequence[int] = 10,
    range: _ArrayLike2DFloat_co | None = None,
    density: bool | None = None,
    weights: _ArrayLike1DFloat_co | None = None,
) -> _Histogram2D[np.float64]: ...
@overload
def histogram2d(
    x: Sequence[complex],
    y: Sequence[complex],
    bins: int | Sequence[int] = 10,
    range: _ArrayLike2DFloat_co | None = None,
    density: bool | None = None,
    weights: _ArrayLike1DFloat_co | None = None,
) -> _Histogram2D[np.complex128 | Any]: ...
@overload
def histogram2d[ScalarT: _Number_co](
    x: _ArrayLike1DNumber_co,
    y: _ArrayLike1DNumber_co,
    bins: _ArrayLike1D[ScalarT] | Sequence[_ArrayLike1D[ScalarT]],
    range: _ArrayLike2DFloat_co | None = None,
    density: bool | None = None,
    weights: _ArrayLike1DFloat_co | None = None,
) -> _Histogram2D[ScalarT]: ...
@overload
def histogram2d[ScalarT: np.inexact, BinsScalarT: _Number_co](
    x: _ArrayLike1D[ScalarT],
    y: _ArrayLike1D[ScalarT],
    bins: Sequence[_ArrayLike1D[BinsScalarT] | int],
    range: _ArrayLike2DFloat_co | None = None,
    density: bool | None = None,
    weights: _ArrayLike1DFloat_co | None = None,
) -> _Histogram2D[ScalarT | BinsScalarT]: ...
@overload
def histogram2d[ScalarT: np.inexact](
    x: _ArrayLike1D[ScalarT],
    y: _ArrayLike1D[ScalarT],
    bins: Sequence[_ArrayLike1DNumber_co | int],
    range: _ArrayLike2DFloat_co | None = None,
    density: bool | None = None,
    weights: _ArrayLike1DFloat_co | None = None,
) -> _Histogram2D[ScalarT | Any]: ...
@overload
def histogram2d[ScalarT: _Number_co](
    x: _ArrayLike1DInt_co | Sequence[float],
    y: _ArrayLike1DInt_co | Sequence[float],
    bins: Sequence[_ArrayLike1D[ScalarT] | int],
    range: _ArrayLike2DFloat_co | None = None,
    density: bool | None = None,
    weights: _ArrayLike1DFloat_co | None = None,
) -> _Histogram2D[np.float64 | ScalarT]: ...
@overload
def histogram2d(
    x: _ArrayLike1DInt_co | Sequence[float],
    y: _ArrayLike1DInt_co | Sequence[float],
    bins: Sequence[_ArrayLike1DNumber_co | int],
    range: _ArrayLike2DFloat_co | None = None,
    density: bool | None = None,
    weights: _ArrayLike1DFloat_co | None = None,
) -> _Histogram2D[np.float64 | Any]: ...
@overload
def histogram2d[ScalarT: _Number_co](
    x: Sequence[complex],
    y: Sequence[complex],
    bins: Sequence[_ArrayLike1D[ScalarT] | int],
    range: _ArrayLike2DFloat_co | None = None,
    density: bool | None = None,
    weights: _ArrayLike1DFloat_co | None = None,
) -> _Histogram2D[np.complex128 | ScalarT]: ...
@overload
def histogram2d(
    x: Sequence[complex],
    y: Sequence[complex],
    bins: Sequence[_ArrayLike1DNumber_co | int],
    range: _ArrayLike2DFloat_co | None = None,
    density: bool | None = None,
    weights: _ArrayLike1DFloat_co | None = None,
) -> _Histogram2D[np.complex128 | Any]: ...
@overload
def histogram2d(
    x: _ArrayLike1DNumber_co,
    y: _ArrayLike1DNumber_co,
    bins: Sequence[Sequence[int]],
    range: _ArrayLike2DFloat_co | None = None,
    density: bool | None = None,
    weights: _ArrayLike1DFloat_co | None = None,
) -> _Histogram2D[np.int_]: ...
@overload
def histogram2d(
    x: _ArrayLike1DNumber_co,
    y: _ArrayLike1DNumber_co,
    bins: Sequence[Sequence[float]],
    range: _ArrayLike2DFloat_co | None = None,
    density: bool | None = None,
    weights: _ArrayLike1DFloat_co | None = None,
) -> _Histogram2D[np.float64 | Any]: ...
@overload
def histogram2d(
    x: _ArrayLike1DNumber_co,
    y: _ArrayLike1DNumber_co,
    bins: Sequence[Sequence[complex]],
    range: _ArrayLike2DFloat_co | None = None,
    density: bool | None = None,
    weights: _ArrayLike1DFloat_co | None = None,
) -> _Histogram2D[np.complex128 | Any]: ...
@overload
def histogram2d(
    x: _ArrayLike1DNumber_co,
    y: _ArrayLike1DNumber_co,
    bins: Sequence[_ArrayLike1DNumber_co | int] | int,
    range: _ArrayLike2DFloat_co | None = None,
    density: bool | None = None,
    weights: _ArrayLike1DFloat_co | None = None,
) -> _Histogram2D[Any]: ...

# NOTE: we're assuming/demanding here the `mask_func` returns
# an ndarray of shape `(n, n)`; otherwise there is the possibility
# of the output tuple having more or less than 2 elements
@overload
def mask_indices(n: int, mask_func: _MaskFunc[int], k: int = 0) -> _Indices2D: ...
@overload
def mask_indices[T](n: int, mask_func: _MaskFunc[T], k: T) -> _Indices2D: ...

#
def tril_indices(n: int, k: int = 0, m: int | None = None) -> _Indices2D: ...
def triu_indices(n: int, k: int = 0, m: int | None = None) -> _Indices2D: ...

# these will accept anything with `shape: tuple[int, int]` and `ndim: int` attributes
def tril_indices_from(arr: _HasShapeAndNDim, k: int = 0) -> _Indices2D: ...
def triu_indices_from(arr: _HasShapeAndNDim, k: int = 0) -> _Indices2D: ...
