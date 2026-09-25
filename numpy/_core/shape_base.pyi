from collections.abc import Sequence
from typing import Any, Never, SupportsIndex, overload

import numpy as np
from numpy import _CastingKind, _ScalarNotObject
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
type _AtLeast1D = tuple[int, *tuple[Any, ...]]
type _AtLeast2D = tuple[int, int, *tuple[Any, ...]]
type _AtLeast3D = tuple[int, int, int, *tuple[Any, ...]]
type _Sequence2[T] = Sequence[Sequence[T]]
type _Sequence3[T] = Sequence[Sequence[Sequence[T]]]
type _ToJustND[ScalarT: np.generic] = np.ndarray[tuple[Never, Never, Never, Never], np.dtype[ScalarT]]
type _To0D[ScalarT: np.generic] = ScalarT | np.ndarray[tuple[()], np.dtype[ScalarT]]
type _To1D[ScalarT: np.generic] = ScalarT | np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]
type _To2D[ScalarT: np.generic] = ScalarT | np.ndarray[tuple[()] | tuple[int] | tuple[int, int], np.dtype[ScalarT]]
type _To3D[ScalarT: np.generic] = ScalarT | np.ndarray[
    tuple[()] | tuple[int] | tuple[int, int] | tuple[int, int, int], np.dtype[ScalarT]
]

###

#
@overload  # >=1d T
def atleast_1d[ArrayT: np.ndarray[_AtLeast1D]](a0: ArrayT, /) -> ArrayT: ...
@overload  # <=1d T
def atleast_1d[ScalarT: np.generic](a0: _To0D[ScalarT] | Sequence[ScalarT], /) -> _Array1D[ScalarT]: ...
@overload  # <=1d bool
def atleast_1d(a0: bool | Sequence[bool], /) -> _Array1D[np.bool]: ...
@overload  # 0d ~int
def atleast_1d(a0: int, /) -> _Array1D[np.int_ | Any]: ...
@overload  # 0d ~float
def atleast_1d(a0: float, /) -> _Array1D[np.float64 | Any]: ...
@overload  # 0d ~complex
def atleast_1d(a0: complex, /) -> _Array1D[np.complex128 | Any]: ...
@overload  # 1d ~int
def atleast_1d(a0: list[int], /) -> _Array1D[np.int_]: ...
@overload  # 1d ~float
def atleast_1d(a0: list[float], /) -> _Array1D[np.float64]: ...
@overload  # 1d ~complex
def atleast_1d(a0: list[complex], /) -> _Array1D[np.complex128]: ...
@overload  # ?d T
def atleast_1d[ScalarT: np.generic](a0: _ArrayLike[ScalarT], /) -> NDArray[ScalarT]: ...
@overload  # ?d
def atleast_1d(a0: ArrayLike, /) -> NDArray[Any]: ...
@overload  # >=1d T, >=1d T
def atleast_1d[ArrayT0: np.ndarray[_AtLeast1D], ArrayT1: np.ndarray[_AtLeast1D]](
    a0: ArrayT0,
    a1: ArrayT1,
    /,
) -> tuple[ArrayT0, ArrayT1]: ...
@overload  # ?d T, ?d T
def atleast_1d[ScalarT0: np.generic, ScalarT1: np.generic](
    a0: _ArrayLike[ScalarT0],
    a1: _ArrayLike[ScalarT1],
    /,
) -> tuple[NDArray[ScalarT0], NDArray[ScalarT1]]: ...
@overload  # ?d, ?d
def atleast_1d(
    a0: ArrayLike,
    a1: ArrayLike,
    /,
) -> tuple[NDArray[Any], NDArray[Any]]: ...
@overload  # ?d T, *?d T
def atleast_1d[ScalarT: np.generic](
    a0: _ArrayLike[ScalarT],
    a1: _ArrayLike[ScalarT],
    /,
    *ai: _ArrayLike[ScalarT],
) -> tuple[NDArray[ScalarT], ...]: ...
@overload  # ?d, *?d
def atleast_1d(
    a0: ArrayLike,
    a1: ArrayLike,
    /,
    *ai: ArrayLike,
) -> tuple[NDArray[Any], ...]: ...

#
@overload  # >=2d T
def atleast_2d[ArrayT: np.ndarray[_AtLeast2D]](a0: ArrayT, /) -> ArrayT: ...
@overload  # <=2d T
def atleast_2d[ScalarT: np.generic](
    a0: _To1D[ScalarT] | Sequence[ScalarT] | Sequence[Sequence[ScalarT]], /
) -> _Array2D[ScalarT]: ...
@overload  # <=2d bool
def atleast_2d(a0: bool | Sequence[bool] | Sequence[Sequence[bool]], /) -> _Array2D[np.bool]: ...
@overload  # 0d ~int
def atleast_2d(a0: int, /) -> _Array2D[np.int_ | Any]: ...
@overload  # 0d ~float
def atleast_2d(a0: float, /) -> _Array2D[np.float64 | Any]: ...
@overload  # 0d ~complex
def atleast_2d(a0: complex, /) -> _Array2D[np.complex128 | Any]: ...
@overload  # 1d | 2d ~int
def atleast_2d(a0: list[int] | Sequence[list[int]], /) -> _Array2D[np.int_]: ...
@overload  # 1d | 2d ~float
def atleast_2d(a0: list[float] | Sequence[list[float]], /) -> _Array2D[np.float64]: ...
@overload  # 1d | 2d ~complex
def atleast_2d(a0: list[complex] | Sequence[list[complex]], /) -> _Array2D[np.complex128]: ...
@overload  # ?d T
def atleast_2d[ScalarT: np.generic](a0: _ArrayLike[ScalarT], /) -> NDArray[ScalarT]: ...
@overload  # ?d
def atleast_2d(a0: ArrayLike, /) -> NDArray[Any]: ...
@overload  # >=2d T, >=2d T
def atleast_2d[ArrayT0: np.ndarray[_AtLeast2D], ArrayT1: np.ndarray[_AtLeast2D]](
    a0: ArrayT0,
    a1: ArrayT1,
    /,
) -> tuple[ArrayT0, ArrayT1]: ...
@overload  # ?d T, ?d T
def atleast_2d[ScalarT0: np.generic, ScalarT1: np.generic](
    a0: _ArrayLike[ScalarT0],
    a1: _ArrayLike[ScalarT1],
    /,
) -> tuple[NDArray[ScalarT0], NDArray[ScalarT1]]: ...
@overload  # ?d, ?d
def atleast_2d(
    a0: ArrayLike,
    a1: ArrayLike,
    /,
) -> tuple[NDArray[Any], NDArray[Any]]: ...
@overload  # ?d T, *?d T
def atleast_2d[ScalarT: np.generic](
    a0: _ArrayLike[ScalarT],
    a1: _ArrayLike[ScalarT],
    /,
    *ai: _ArrayLike[ScalarT],
) -> tuple[NDArray[ScalarT], ...]: ...
@overload  # ?d, *?d
def atleast_2d(
    a0: ArrayLike,
    a1: ArrayLike,
    /,
    *ai: ArrayLike,
) -> tuple[NDArray[Any], ...]: ...

#
@overload  # >=3d T
def atleast_3d[ArrayT: np.ndarray[_AtLeast3D]](a0: ArrayT, /) -> ArrayT: ...
@overload  # <=3d T
def atleast_3d[ScalarT: np.generic](
    a0: _To2D[ScalarT] | Sequence[ScalarT] | Sequence[Sequence[ScalarT]] | Sequence[Sequence[Sequence[ScalarT]]], /
) -> _Array3D[ScalarT]: ...
@overload  # <=3d bool
def atleast_3d(
    a0: bool | Sequence[bool] | Sequence[Sequence[bool]] | Sequence[Sequence[Sequence[bool]]], /
) -> _Array3D[np.bool]: ...
@overload  # 0d ~int
def atleast_3d(a0: int, /) -> _Array3D[np.int_ | Any]: ...
@overload  # 0d ~float
def atleast_3d(a0: float, /) -> _Array3D[np.float64 | Any]: ...
@overload  # 0d ~complex
def atleast_3d(a0: complex, /) -> _Array3D[np.complex128 | Any]: ...
@overload  # 1d | 2d | 3d ~int
def atleast_3d(a0: list[int] | Sequence[list[int]] | Sequence[Sequence[list[int]]], /) -> _Array3D[np.int_]: ...
@overload  # 1d | 2d | 3d ~float
def atleast_3d(a0: list[float] | Sequence[list[float]] | Sequence[Sequence[list[float]]], /) -> _Array3D[np.float64]: ...
@overload  # 1d | 2d | 3d ~complex
def atleast_3d(a0: list[complex] | Sequence[list[complex]] | Sequence[Sequence[list[complex]]], /) -> _Array3D[np.complex128]: ...
@overload  # ?d T
def atleast_3d[ScalarT: np.generic](a0: _ArrayLike[ScalarT], /) -> NDArray[ScalarT]: ...
@overload  # ?d
def atleast_3d(a0: ArrayLike, /) -> NDArray[Any]: ...
@overload  # >=3d T, >=3d T
def atleast_3d[ArrayT0: np.ndarray[_AtLeast3D], ArrayT1: np.ndarray[_AtLeast3D]](
    a0: ArrayT0,
    a1: ArrayT1,
    /,
) -> tuple[ArrayT0, ArrayT1]: ...
@overload  # ?d T, ?d T
def atleast_3d[ScalarT0: np.generic, ScalarT1: np.generic](
    a0: _ArrayLike[ScalarT0],
    a1: _ArrayLike[ScalarT1],
    /,
) -> tuple[NDArray[ScalarT0], NDArray[ScalarT1]]: ...
@overload  # ?d, ?d
def atleast_3d(
    a0: ArrayLike,
    a1: ArrayLike,
    /,
) -> tuple[NDArray[Any], NDArray[Any]]: ...
@overload  # ?d T, *?d T
def atleast_3d[ScalarT: np.generic](
    a0: _ArrayLike[ScalarT],
    a1: _ArrayLike[ScalarT],
    /,
    *ai: _ArrayLike[ScalarT],
) -> tuple[NDArray[ScalarT], ...]: ...
@overload  # ?d, *?d
def atleast_3d(
    a0: ArrayLike,
    a1: ArrayLike,
    /,
    *ai: ArrayLike,
) -> tuple[NDArray[Any], ...]: ...

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

#
@overload  # ?d  (workaround)
def unstack[DTypeT: np.dtype](
    array: np.ndarray[tuple[Never, Never, Never, Never], DTypeT],
    /,
    *,
    axis: int = 0,
) -> tuple[np.ndarray[tuple[Any, ...], DTypeT], ...]: ...
@overload  # 1d T \ object_
def unstack[ScalarT: _ScalarNotObject](
    array: _Array1D[ScalarT],
    /,
    *,
    axis: int = 0,
) -> tuple[ScalarT, ...]: ...
@overload  # 1d object_[T]
def unstack[ItemT](
    array: _Array1D[np.object_[ItemT]],
    /,
    *,
    axis: int = 0,
) -> tuple[ItemT, ...]: ...
@overload  # 1d StringDType
def unstack(
    array: np.ndarray[tuple[int], np.dtypes.StringDType],
    /,
    *,
    axis: int = 0,
) -> tuple[str, ...]: ...
@overload  # 2d
def unstack[DTypeT: np.dtype](
    array: np.ndarray[tuple[int, int], DTypeT],
    /,
    *,
    axis: int = 0,
) -> tuple[np.ndarray[tuple[int], DTypeT], ...]: ...
@overload  # 3d
def unstack[DTypeT: np.dtype](
    array: np.ndarray[tuple[int, int, int], DTypeT],
    /,
    *,
    axis: int = 0,
) -> tuple[np.ndarray[tuple[int, int], DTypeT], ...]: ...
@overload  # 4d
def unstack[DTypeT: np.dtype](
    array: np.ndarray[tuple[int, int, int, int], DTypeT],
    /,
    *,
    axis: int = 0,
) -> tuple[np.ndarray[tuple[int, int, int], DTypeT], ...]: ...
@overload  # >=5d
def unstack[DTypeT: np.dtype](
    array: np.ndarray[tuple[int, int, int, int, int, *tuple[int, ...]], DTypeT],
    /,
    *,
    axis: int = 0,
) -> tuple[np.ndarray[tuple[Any, ...], DTypeT], ...]: ...
@overload  # ?d  (fallback)
def unstack(
    array: np.ndarray[_AtLeast1D, Any],
    /,
    *,
    axis: int = 0,
) -> tuple[Any, ...]: ...

#
@overload  # known array
def block[ArrayT: np.ndarray](arrays: ArrayT) -> ArrayT: ...
@overload  # [?d T]  (workaround)
def block[ScalarT: np.generic](
    arrays: Sequence[_ToJustND[ScalarT]] | _Sequence2[_ToJustND[ScalarT]] | _Sequence3[_ToJustND[ScalarT]],
) -> NDArray[ScalarT]: ...
@overload  # [<=1d T]
def block[ScalarT: np.generic](arrays: Sequence[_To1D[ScalarT]]) -> _Array1D[ScalarT]: ...
@overload  # [<=1d bool]
def block(arrays: Sequence[bool | _To1D[np.bool]]) -> _Array1D[np.bool]: ...
@overload  # [~int]
def block(arrays: list[int]) -> _Array1D[np.int_]: ...
@overload  # [~float]
def block(arrays: list[float]) -> _Array1D[np.float64]: ...
@overload  # [~complex]
def block(arrays: list[complex]) -> _Array1D[np.complex128]: ...
@overload  # [<=1d]
def block(arrays: Sequence[complex | _To1D[np.number | np.bool]]) -> _Array1D[Any]: ...
@overload  # [2d T] | [[<=2d T]]
def block[ScalarT: np.generic](arrays: Sequence[_Array2D[ScalarT]] | _Sequence2[_To2D[ScalarT]]) -> _Array2D[ScalarT]: ...
@overload  # [[<=2d bool]]
def block(arrays: _Sequence2[bool | _To2D[np.bool]]) -> _Array2D[np.bool]: ...
@overload  # [[~int]]
def block(arrays: Sequence[list[int]]) -> _Array2D[np.int_]: ...
@overload  # [[~float]]
def block(arrays: Sequence[list[float]]) -> _Array2D[np.float64]: ...
@overload  # [[~complex]]
def block(arrays: Sequence[list[complex]]) -> _Array2D[np.complex128]: ...
@overload  # [[<=2d]]
def block(arrays: _Sequence2[complex | _To2D[np.number | np.bool]]) -> _Array2D[Any]: ...
@overload  # [3d T] | [[3d T]] | [[[<=3d T]]]
def block[ScalarT: np.generic](
    arrays: Sequence[_Array3D[ScalarT]] | _Sequence2[_Array3D[ScalarT]] | _Sequence3[_To3D[ScalarT]],
) -> _Array3D[ScalarT]: ...
@overload  # [[[<=3d bool]]]
def block(arrays: _Sequence3[bool | _To3D[np.bool]]) -> _Array3D[np.bool]: ...
@overload  # [[[~int]]]
def block(arrays: _Sequence2[list[int]]) -> _Array3D[np.int_]: ...
@overload  # [[[~float]]]
def block(arrays: _Sequence2[list[float]]) -> _Array3D[np.float64]: ...
@overload  # [[[~complex]]]
def block(arrays: _Sequence2[list[complex]]) -> _Array3D[np.complex128]: ...
@overload  # [[[<=3d]]]
def block(arrays: _Sequence3[complex | _To3D[np.number | np.bool]]) -> _Array3D[Any]: ...
@overload  # ?d T
def block[ScalarT: np.generic](arrays: _ArrayLike[ScalarT]) -> NDArray[ScalarT]: ...
@overload  # ?d  (fallback)
def block(arrays: ArrayLike) -> NDArray[Any]: ...
