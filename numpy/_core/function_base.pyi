from _typeshed import Incomplete
from typing import Any, Literal as L, SupportsIndex, overload

import numpy as np
from numpy._typing import (
    DTypeLike,
    NDArray,
    _ArrayLikeComplex_co,
    _ArrayLikeFloat_co,
    _ComplexLike_co,
    _DTypeLike,
)
from numpy._typing._array_like import _DualArrayLike

__all__ = ["geomspace", "linspace", "logspace"]

type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _ToFloat64 = float | np.integer | np.bool  # `np.float64` is assignable to `float`
type _ToArrayFloat64 = _DualArrayLike[np.dtype[np.float64 | np.integer | np.bool], float]

###

@overload
def linspace(
    start: _ToFloat64,
    stop: _ToFloat64,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    retstep: L[False] = False,
    dtype: None = None,
    axis: SupportsIndex = 0,
    *,
    device: L["cpu"] | None = None,
) -> _Array1D[np.float64]: ...
@overload
def linspace(
    start: complex,
    stop: complex,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    retstep: L[False] = False,
    dtype: None = None,
    axis: SupportsIndex = 0,
    *,
    device: L["cpu"] | None = None,
) -> _Array1D[np.complex128 | Any]: ...
@overload
def linspace[ScalarT: np.generic](
    start: _ComplexLike_co,
    stop: _ComplexLike_co,
    num: SupportsIndex,
    endpoint: bool,
    retstep: L[False],
    dtype: _DTypeLike[ScalarT],
    axis: SupportsIndex = 0,
    *,
    device: L["cpu"] | None = None,
) -> _Array1D[ScalarT]: ...
@overload
def linspace[ScalarT: np.generic](
    start: _ComplexLike_co,
    stop: _ComplexLike_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    retstep: L[False] = False,
    *,
    dtype: _DTypeLike[ScalarT],
    axis: SupportsIndex = 0,
    device: L["cpu"] | None = None,
) -> _Array1D[ScalarT]: ...
@overload
def linspace(
    start: _ToArrayFloat64,
    stop: _ToArrayFloat64,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    retstep: L[False] = False,
    dtype: None = None,
    axis: SupportsIndex = 0,
    *,
    device: L["cpu"] | None = None,
) -> NDArray[np.float64]: ...
@overload
def linspace(
    start: _ArrayLikeFloat_co,
    stop: _ArrayLikeFloat_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    retstep: L[False] = False,
    dtype: None = None,
    axis: SupportsIndex = 0,
    *,
    device: L["cpu"] | None = None,
) -> NDArray[np.float64 | Any]: ...
@overload
def linspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    retstep: L[False] = False,
    dtype: None = None,
    axis: SupportsIndex = 0,
    *,
    device: L["cpu"] | None = None,
) -> NDArray[np.complex128 | Any]: ...
@overload
def linspace[ScalarT: np.generic](
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex,
    endpoint: bool,
    retstep: L[False],
    dtype: _DTypeLike[ScalarT],
    axis: SupportsIndex = 0,
    *,
    device: L["cpu"] | None = None,
) -> NDArray[ScalarT]: ...
@overload
def linspace[ScalarT: np.generic](
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    retstep: L[False] = False,
    *,
    dtype: _DTypeLike[ScalarT],
    axis: SupportsIndex = 0,
    device: L["cpu"] | None = None,
) -> NDArray[ScalarT]: ...
@overload
def linspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    retstep: L[False] = False,
    dtype: DTypeLike | None = None,
    axis: SupportsIndex = 0,
    *,
    device: L["cpu"] | None = None,
) -> NDArray[Incomplete]: ...
@overload
def linspace(
    start: _ToFloat64,
    stop: _ToFloat64,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    *,
    retstep: L[True],
    dtype: None = None,
    axis: SupportsIndex = 0,
    device: L["cpu"] | None = None,
) -> tuple[_Array1D[np.float64], np.float64]: ...
@overload
def linspace(
    start: complex,
    stop: complex,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    *,
    retstep: L[True],
    dtype: None = None,
    axis: SupportsIndex = 0,
    device: L["cpu"] | None = None,
) -> tuple[_Array1D[np.complex128 | Any], np.complex128 | Any]: ...
@overload
def linspace[ScalarT: np.generic](
    start: _ComplexLike_co,
    stop: _ComplexLike_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    *,
    retstep: L[True],
    dtype: _DTypeLike[ScalarT],
    axis: SupportsIndex = 0,
    device: L["cpu"] | None = None,
) -> tuple[_Array1D[ScalarT], ScalarT]: ...
@overload
def linspace(
    start: _ToArrayFloat64,
    stop: _ToArrayFloat64,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    *,
    retstep: L[True],
    dtype: None = None,
    axis: SupportsIndex = 0,
    device: L["cpu"] | None = None,
) -> tuple[NDArray[np.float64], np.float64]: ...
@overload
def linspace(
    start: _ArrayLikeFloat_co,
    stop: _ArrayLikeFloat_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    *,
    retstep: L[True],
    dtype: None = None,
    axis: SupportsIndex = 0,
    device: L["cpu"] | None = None,
) -> tuple[NDArray[np.float64 | Any], np.float64 | Any]: ...
@overload
def linspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    *,
    retstep: L[True],
    dtype: None = None,
    axis: SupportsIndex = 0,
    device: L["cpu"] | None = None,
) -> tuple[NDArray[np.complex128 | Any], np.complex128 | Any]: ...
@overload
def linspace[ScalarT: np.generic](
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    *,
    retstep: L[True],
    dtype: _DTypeLike[ScalarT],
    axis: SupportsIndex = 0,
    device: L["cpu"] | None = None,
) -> tuple[NDArray[ScalarT], ScalarT]: ...
@overload
def linspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    *,
    retstep: L[True],
    dtype: DTypeLike | None = None,
    axis: SupportsIndex = 0,
    device: L["cpu"] | None = None,
) -> tuple[NDArray[Incomplete], Incomplete]: ...

#
@overload
def logspace(
    start: _ToFloat64,
    stop: _ToFloat64,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    base: _ToFloat64 = 10.0,
    dtype: None = None,
    axis: SupportsIndex = 0,
) -> _Array1D[np.float64]: ...
@overload
def logspace(
    start: complex,
    stop: complex,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    base: complex = 10.0,
    dtype: None = None,
    axis: SupportsIndex = 0,
) -> _Array1D[np.complex128 | Any]: ...
@overload
def logspace[ScalarT: np.generic](
    start: _ComplexLike_co,
    stop: _ComplexLike_co,
    num: SupportsIndex,
    endpoint: bool,
    base: _ComplexLike_co,
    dtype: _DTypeLike[ScalarT],
    axis: SupportsIndex = 0,
) -> _Array1D[ScalarT]: ...
@overload
def logspace[ScalarT: np.generic](
    start: _ComplexLike_co,
    stop: _ComplexLike_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    base: _ArrayLikeComplex_co = 10.0,
    *,
    dtype: _DTypeLike[ScalarT],
    axis: SupportsIndex = 0,
) -> _Array1D[ScalarT]: ...
@overload
def logspace(
    start: _ToArrayFloat64,
    stop: _ToArrayFloat64,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    base: _ToArrayFloat64 = 10.0,
    dtype: None = None,
    axis: SupportsIndex = 0,
) -> NDArray[np.float64]: ...
@overload
def logspace(
    start: _ArrayLikeFloat_co,
    stop: _ArrayLikeFloat_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    base: _ArrayLikeFloat_co = 10.0,
    dtype: None = None,
    axis: SupportsIndex = 0,
) -> NDArray[np.float64 | Any]: ...
@overload
def logspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    base: _ArrayLikeComplex_co = 10.0,
    dtype: None = None,
    axis: SupportsIndex = 0,
) -> NDArray[np.complex128 | Any]: ...
@overload
def logspace[ScalarT: np.generic](
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex,
    endpoint: bool,
    base: _ArrayLikeComplex_co,
    dtype: _DTypeLike[ScalarT],
    axis: SupportsIndex = 0,
) -> NDArray[ScalarT]: ...
@overload
def logspace[ScalarT: np.generic](
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    base: _ArrayLikeComplex_co = 10.0,
    *,
    dtype: _DTypeLike[ScalarT],
    axis: SupportsIndex = 0,
) -> NDArray[ScalarT]: ...
@overload
def logspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    base: _ArrayLikeComplex_co = 10.0,
    dtype: DTypeLike | None = None,
    axis: SupportsIndex = 0,
) -> NDArray[Incomplete]: ...

#
@overload
def geomspace(
    start: _ToFloat64,
    stop: _ToFloat64,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    dtype: None = None,
    axis: SupportsIndex = 0,
) -> _Array1D[np.float64]: ...
@overload
def geomspace(
    start: complex,
    stop: complex,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    dtype: None = None,
    axis: SupportsIndex = 0,
) -> _Array1D[np.complex128 | Any]: ...
@overload
def geomspace[ScalarT: np.generic](
    start: _ComplexLike_co,
    stop: _ComplexLike_co,
    num: SupportsIndex,
    endpoint: bool,
    dtype: _DTypeLike[ScalarT],
    axis: SupportsIndex = 0,
) -> _Array1D[ScalarT]: ...
@overload
def geomspace[ScalarT: np.generic](
    start: _ComplexLike_co,
    stop: _ComplexLike_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    *,
    dtype: _DTypeLike[ScalarT],
    axis: SupportsIndex = 0,
) -> _Array1D[ScalarT]: ...
@overload
def geomspace(
    start: _ToArrayFloat64,
    stop: _ToArrayFloat64,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    dtype: None = None,
    axis: SupportsIndex = 0,
) -> NDArray[np.float64]: ...
@overload
def geomspace(
    start: _ArrayLikeFloat_co,
    stop: _ArrayLikeFloat_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    dtype: None = None,
    axis: SupportsIndex = 0,
) -> NDArray[np.float64 | Any]: ...
@overload
def geomspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    dtype: None = None,
    axis: SupportsIndex = 0,
) -> NDArray[np.complex128 | Any]: ...
@overload
def geomspace[ScalarT: np.generic](
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex,
    endpoint: bool,
    dtype: _DTypeLike[ScalarT],
    axis: SupportsIndex = 0,
) -> NDArray[ScalarT]: ...
@overload
def geomspace[ScalarT: np.generic](
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    *,
    dtype: _DTypeLike[ScalarT],
    axis: SupportsIndex = 0,
) -> NDArray[ScalarT]: ...
@overload
def geomspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    dtype: DTypeLike | None = None,
    axis: SupportsIndex = 0,
) -> NDArray[Incomplete]: ...

#
def add_newdoc(
    place: str,
    obj: str,
    doc: str | tuple[str, str] | list[tuple[str, str]],
    warn_on_python: bool = True,
) -> None: ...
