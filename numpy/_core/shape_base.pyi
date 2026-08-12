from collections.abc import Sequence
from typing import Any, Never, SupportsIndex, overload

import numpy as np
from numpy import _CastingKind
from numpy._typing import ArrayLike, DTypeLike, NDArray, _ArrayLike, _DTypeLike

__all__ = [
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "block",
    "hstack",
    "stack",
    "unstack",
    "vstack",
]

type _Array0D[ScalarT: np.generic] = np.ndarray[tuple[()], np.dtype[ScalarT]]
type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]
type _Array3D[ScalarT: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[ScalarT]]
type _Array4D[ScalarT: np.generic] = np.ndarray[tuple[int, int, int, int], np.dtype[ScalarT]]

# input only
type _AtLeast2D = tuple[int, int, *tuple[Any, ...]]
type _ToJustND[ScalarT: np.generic] = np.ndarray[tuple[Never, Never, Never, Never], np.dtype[ScalarT]]
type _To0D[ScalarT: np.generic] = ScalarT | np.ndarray[tuple[()], np.dtype[ScalarT]]
type _To1D[ScalarT: np.generic] = ScalarT | np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]
type _To2D[ScalarT: np.generic] = ScalarT | np.ndarray[tuple[()] | tuple[int] | tuple[int, int], np.dtype[ScalarT]]

###

# keep in sync with `numpy.ma.extras.atleast_1d`
@overload
def atleast_1d[ArrayT: _Array1D[Any] | _Array2D[Any] | _Array3D[Any]](a0: ArrayT, /) -> ArrayT: ...
@overload
def atleast_1d[ScalarT: np.generic](a0: _Array0D[ScalarT], /) -> _Array1D[ScalarT]: ...
@overload
def atleast_1d[ScalarT: np.generic](a0: ScalarT, /) -> _Array1D[ScalarT]: ...
@overload
def atleast_1d[ScalarT: np.generic](a0: _ArrayLike[ScalarT], /) -> NDArray[ScalarT]: ...
@overload
def atleast_1d[ScalarT1: np.generic, ScalarT2: np.generic](
    a0: _ArrayLike[ScalarT1], a1: _ArrayLike[ScalarT2], /
) -> tuple[NDArray[ScalarT1], NDArray[ScalarT2]]: ...
@overload
def atleast_1d[ScalarT: np.generic](
    a0: _ArrayLike[ScalarT], a1: _ArrayLike[ScalarT], /, *arys: _ArrayLike[ScalarT]
) -> tuple[NDArray[ScalarT], ...]: ...
@overload
def atleast_1d(a0: ArrayLike, /) -> NDArray[Any]: ...
@overload
def atleast_1d(a0: ArrayLike, a1: ArrayLike, /) -> tuple[NDArray[Any], NDArray[Any]]: ...
@overload
def atleast_1d(a0: ArrayLike, a1: ArrayLike, /, *ai: ArrayLike) -> tuple[NDArray[Any], ...]: ...

# keep in sync with `numpy.ma.extras.atleast_2d`
@overload
def atleast_2d[ArrayT: _Array2D[Any] | _Array3D[Any]](a0: ArrayT, /) -> ArrayT: ...
@overload
def atleast_2d[ScalarT: np.generic](a0: _Array0D[ScalarT] | _Array1D[ScalarT], /) -> _Array2D[ScalarT]: ...
@overload
def atleast_2d[ScalarT: np.generic](a0: ScalarT, /) -> _Array2D[ScalarT]: ...
@overload
def atleast_2d[ScalarT: np.generic](a0: _ArrayLike[ScalarT], /) -> NDArray[ScalarT]: ...
@overload
def atleast_2d[ScalarT1: np.generic, ScalarT2: np.generic](
    a0: _ArrayLike[ScalarT1], a1: _ArrayLike[ScalarT2], /
) -> tuple[NDArray[ScalarT1], NDArray[ScalarT2]]: ...
@overload
def atleast_2d[ScalarT: np.generic](
    a0: _ArrayLike[ScalarT], a1: _ArrayLike[ScalarT], /, *arys: _ArrayLike[ScalarT]
) -> tuple[NDArray[ScalarT], ...]: ...
@overload
def atleast_2d(a0: ArrayLike, /) -> NDArray[Any]: ...
@overload
def atleast_2d(a0: ArrayLike, a1: ArrayLike, /) -> tuple[NDArray[Any], NDArray[Any]]: ...
@overload
def atleast_2d(a0: ArrayLike, a1: ArrayLike, /, *ai: ArrayLike) -> tuple[NDArray[Any], ...]: ...

# keep in sync with `numpy.ma.extras.atleast_3d`
@overload
def atleast_3d[ArrayT: _Array3D[Any]](a0: ArrayT, /) -> ArrayT: ...
@overload
def atleast_3d[ScalarT: np.generic](a0: _Array0D[ScalarT] | _Array1D[ScalarT] | _Array2D[ScalarT], /) -> _Array3D[ScalarT]: ...
@overload
def atleast_3d[ScalarT: np.generic](a0: ScalarT, /) -> _Array3D[ScalarT]: ...
@overload
def atleast_3d[ScalarT: np.generic](a0: _ArrayLike[ScalarT], /) -> NDArray[ScalarT]: ...
@overload
def atleast_3d[ScalarT1: np.generic, ScalarT2: np.generic](
    a0: _ArrayLike[ScalarT1], a1: _ArrayLike[ScalarT2], /
) -> tuple[NDArray[ScalarT1], NDArray[ScalarT2]]: ...
@overload
def atleast_3d[ScalarT: np.generic](
    a0: _ArrayLike[ScalarT], a1: _ArrayLike[ScalarT], /, *arys: _ArrayLike[ScalarT]
) -> tuple[NDArray[ScalarT], ...]: ...
@overload
def atleast_3d(a0: ArrayLike, /) -> NDArray[Any]: ...
@overload
def atleast_3d(a0: ArrayLike, a1: ArrayLike, /) -> tuple[NDArray[Any], NDArray[Any]]: ...
@overload
def atleast_3d(a0: ArrayLike, a1: ArrayLike, /, *ai: ArrayLike) -> tuple[NDArray[Any], ...]: ...

# used by numpy.lib._shape_base_impl
def _arrays_for_stack_dispatcher[T](arrays: Sequence[T]) -> tuple[T, ...]: ...

# keep in sync with `hstack` and `numpy.ma.extras.vstack`
@overload  # >=2d
def vstack[ShapeT: _AtLeast2D, DTypeT: np.dtype](
    tup: Sequence[np.ndarray[ShapeT, DTypeT]],
    *,
    dtype: None = None,
    casting: _CastingKind = "same_kind",
) -> np.ndarray[ShapeT, DTypeT]: ...
@overload  # >=2d, dtype=<known>
def vstack[ShapeT: _AtLeast2D, ScalarT: np.generic](
    tup: Sequence[np.ndarray[ShapeT]],
    *,
    dtype: _DTypeLike[ScalarT],
    casting: _CastingKind = "same_kind",
) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
@overload  # >=2d, dtype=<unknown>
def vstack[ShapeT: _AtLeast2D](
    tup: Sequence[np.ndarray[ShapeT]],
    *,
    dtype: DTypeLike,
    casting: _CastingKind = "same_kind",
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # <=2d
def vstack[ScalarT: np.generic](
    tup: Sequence[_To2D[ScalarT]],
    *,
    dtype: None = None,
    casting: _CastingKind = "same_kind",
) -> _Array2D[ScalarT]: ...
@overload  # <=2d, dtype=<known>
def vstack[ScalarT: np.generic](
    tup: Sequence[_To2D[np.generic]],
    *,
    dtype: _DTypeLike[ScalarT],
    casting: _CastingKind = "same_kind",
) -> _Array2D[ScalarT]: ...
@overload  # <=2d, dtype=<unknown>
def vstack(
    tup: Sequence[_To2D[np.generic]],
    *,
    dtype: DTypeLike,
    casting: _CastingKind = "same_kind",
) -> _Array2D[Any]: ...
@overload  # ?d
def vstack[ScalarT: np.generic](
    tup: Sequence[_ArrayLike[ScalarT]],
    *,
    dtype: None = None,
    casting: _CastingKind = "same_kind",
) -> NDArray[ScalarT]: ...
@overload  # ?d, dtype=<known>
def vstack[ScalarT: np.generic](
    tup: Sequence[ArrayLike],
    *,
    dtype: _DTypeLike[ScalarT],
    casting: _CastingKind = "same_kind",
) -> NDArray[ScalarT]: ...
@overload  # fallback
def vstack(
    tup: Sequence[ArrayLike],
    *,
    dtype: DTypeLike | None = None,
    casting: _CastingKind = "same_kind",
) -> NDArray[Any]: ...

# keep in sync with `vstack` and `numpy.ma.extras.hstack`
@overload  # >=2d
def hstack[ShapeT: _AtLeast2D, DTypeT: np.dtype](
    tup: Sequence[np.ndarray[ShapeT, DTypeT]],
    *,
    dtype: None = None,
    casting: _CastingKind = "same_kind",
) -> np.ndarray[ShapeT, DTypeT]: ...
@overload  # >=2d, dtype=<known>
def hstack[ShapeT: _AtLeast2D, ScalarT: np.generic](
    tup: Sequence[np.ndarray[ShapeT]],
    *,
    dtype: _DTypeLike[ScalarT],
    casting: _CastingKind = "same_kind",
) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
@overload  # >=2d, dtype=<unknown>
def hstack[ShapeT: _AtLeast2D](
    tup: Sequence[np.ndarray[ShapeT]],
    *,
    dtype: DTypeLike,
    casting: _CastingKind = "same_kind",
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # <=1d
def hstack[ScalarT: np.generic](
    tup: Sequence[_To1D[ScalarT]],
    *,
    dtype: None = None,
    casting: _CastingKind = "same_kind",
) -> _Array1D[ScalarT]: ...
@overload  # <=1d, dtype=<known>
def hstack[ScalarT: np.generic](
    tup: Sequence[_To1D[np.generic]],
    *,
    dtype: _DTypeLike[ScalarT],
    casting: _CastingKind = "same_kind",
) -> _Array1D[ScalarT]: ...
@overload  # <=1d, dtype=<unknown>
def hstack(
    tup: Sequence[_To1D[np.generic]],
    *,
    dtype: DTypeLike,
    casting: _CastingKind = "same_kind",
) -> _Array1D[Any]: ...
@overload  # ?d
def hstack[ScalarT: np.generic](
    tup: Sequence[_ArrayLike[ScalarT]],
    *,
    dtype: None = None,
    casting: _CastingKind = "same_kind",
) -> NDArray[ScalarT]: ...
@overload  # ?d, dtype=<known>
def hstack[ScalarT: np.generic](
    tup: Sequence[ArrayLike],
    *,
    dtype: _DTypeLike[ScalarT],
    casting: _CastingKind = "same_kind",
) -> NDArray[ScalarT]: ...
@overload  # fallback
def hstack(
    tup: Sequence[ArrayLike],
    *,
    dtype: DTypeLike | None = None,
    casting: _CastingKind = "same_kind",
) -> NDArray[Any]: ...

# keep in sync with `numpy.ma.extras.stack`
@overload  # ?d  (workaround overload)
def stack[ScalarT: np.generic](
    arrays: Sequence[_ToJustND[ScalarT]],
    axis: SupportsIndex = 0,
    out: None = None,
    *,
    dtype: None = None,
    casting: _CastingKind = "same_kind",
) -> NDArray[ScalarT]: ...
@overload  # ?d, dtype=<known>  (workaround overload)
def stack[ScalarT: np.generic](
    arrays: Sequence[_ToJustND[np.generic]],
    axis: SupportsIndex = 0,
    out: None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    casting: _CastingKind = "same_kind",
) -> NDArray[ScalarT]: ...
@overload  # ?d, dtype=<unknown>  (workaround overload)
def stack(
    arrays: Sequence[_ToJustND[np.generic]],
    axis: SupportsIndex = 0,
    out: None = None,
    *,
    dtype: DTypeLike,
    casting: _CastingKind = "same_kind",
) -> NDArray[Any]: ...
@overload  # 0d -> 1d
def stack[ScalarT: np.generic](
    arrays: Sequence[_To0D[ScalarT]],
    axis: SupportsIndex = 0,
    out: None = None,
    *,
    dtype: None = None,
    casting: _CastingKind = "same_kind",
) -> _Array1D[ScalarT]: ...
@overload  # 0d -> 1d, dtype=<known>
def stack[ScalarT: np.generic](
    arrays: Sequence[_To0D[np.generic]],
    axis: SupportsIndex = 0,
    out: None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    casting: _CastingKind = "same_kind",
) -> _Array1D[ScalarT]: ...
@overload  # 0d -> 1d, dtype=<unknown>
def stack(
    arrays: Sequence[_To0D[np.generic]],
    axis: SupportsIndex = 0,
    out: None = None,
    *,
    dtype: DTypeLike,
    casting: _CastingKind = "same_kind",
) -> _Array1D[Any]: ...
@overload  # 1d -> 2d
def stack[ScalarT: np.generic](
    arrays: Sequence[_Array1D[ScalarT]],
    axis: SupportsIndex = 0,
    out: None = None,
    *,
    dtype: None = None,
    casting: _CastingKind = "same_kind",
) -> _Array2D[ScalarT]: ...
@overload  # 1d -> 2d, dtype=<known>
def stack[ScalarT: np.generic](
    arrays: Sequence[_Array1D[np.generic]],
    axis: SupportsIndex = 0,
    out: None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    casting: _CastingKind = "same_kind",
) -> _Array2D[ScalarT]: ...
@overload  # 1d -> 2d, dtype=<unknown>
def stack(
    arrays: Sequence[_Array1D[np.generic]],
    axis: SupportsIndex = 0,
    out: None = None,
    *,
    dtype: DTypeLike,
    casting: _CastingKind = "same_kind",
) -> _Array2D[Any]: ...
@overload  # 2d -> 3d
def stack[ScalarT: np.generic](
    arrays: Sequence[_Array2D[ScalarT]],
    axis: SupportsIndex = 0,
    out: None = None,
    *,
    dtype: None = None,
    casting: _CastingKind = "same_kind",
) -> _Array3D[ScalarT]: ...
@overload  # 2d -> 3d, dtype=<known>
def stack[ScalarT: np.generic](
    arrays: Sequence[_Array2D[np.generic]],
    axis: SupportsIndex = 0,
    out: None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    casting: _CastingKind = "same_kind",
) -> _Array3D[ScalarT]: ...
@overload  # 2d -> 3d, dtype=<unknown>
def stack(
    arrays: Sequence[_Array2D[np.generic]],
    axis: SupportsIndex = 0,
    out: None = None,
    *,
    dtype: DTypeLike,
    casting: _CastingKind = "same_kind",
) -> _Array3D[Any]: ...
@overload  # 3d -> 4d
def stack[ScalarT: np.generic](
    arrays: Sequence[_Array3D[ScalarT]],
    axis: SupportsIndex = 0,
    out: None = None,
    *,
    dtype: None = None,
    casting: _CastingKind = "same_kind",
) -> _Array4D[ScalarT]: ...
@overload  # 3d -> 4d, dtype=<known>
def stack[ScalarT: np.generic](
    arrays: Sequence[_Array3D[np.generic]],
    axis: SupportsIndex = 0,
    out: None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    casting: _CastingKind = "same_kind",
) -> _Array4D[ScalarT]: ...
@overload  # 3d -> 4d, dtype=<unknown>
def stack(
    arrays: Sequence[_Array3D[np.generic]],
    axis: SupportsIndex = 0,
    out: None = None,
    *,
    dtype: DTypeLike,
    casting: _CastingKind = "same_kind",
) -> _Array4D[Any]: ...
@overload  # ?d
def stack[ScalarT: np.generic](
    arrays: Sequence[_ArrayLike[ScalarT]],
    axis: SupportsIndex = 0,
    out: None = None,
    *,
    dtype: None = None,
    casting: _CastingKind = "same_kind",
) -> NDArray[ScalarT]: ...
@overload  # ?d, dtype=<known>
def stack[ScalarT: np.generic](
    arrays: Sequence[ArrayLike],
    axis: SupportsIndex = 0,
    out: None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    casting: _CastingKind = "same_kind",
) -> NDArray[ScalarT]: ...
@overload  # fallback
def stack(
    arrays: Sequence[ArrayLike],
    axis: SupportsIndex = 0,
    out: None = None,
    *,
    dtype: DTypeLike | None = None,
    casting: _CastingKind = "same_kind",
) -> NDArray[Any]: ...
@overload  # out=<given>  (positional)
def stack[OutT: np.ndarray](
    arrays: Sequence[ArrayLike],
    axis: SupportsIndex,
    out: OutT,
    *,
    dtype: DTypeLike | None = None,
    casting: _CastingKind = "same_kind",
) -> OutT: ...
@overload  # out=<given> (keyword)
def stack[OutT: np.ndarray](
    arrays: Sequence[ArrayLike],
    axis: SupportsIndex = 0,
    *,
    out: OutT,
    dtype: DTypeLike | None = None,
    casting: _CastingKind = "same_kind",
) -> OutT: ...

@overload
def unstack[ScalarT: np.generic](
    array: _ArrayLike[ScalarT],
    /,
    *,
    axis: int = 0,
) -> tuple[NDArray[ScalarT], ...]: ...
@overload
def unstack(
    array: ArrayLike,
    /,
    *,
    axis: int = 0,
) -> tuple[NDArray[Any], ...]: ...

@overload
def block[ScalarT: np.generic](arrays: _ArrayLike[ScalarT]) -> NDArray[ScalarT]: ...
@overload
def block(arrays: ArrayLike) -> NDArray[Any]: ...
