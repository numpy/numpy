from collections.abc import Sequence
from typing import Any, SupportsIndex, overload

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

# keep in sync with `numpy.ma.extras.atleast_1d`
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

# keep in sync with `numpy.ma.extras.vstack`
@overload
def vstack[ScalarT: np.generic](
    tup: Sequence[_ArrayLike[ScalarT]],
    *,
    dtype: None = None,
    casting: _CastingKind = "same_kind"
) -> NDArray[ScalarT]: ...
@overload
def vstack[ScalarT: np.generic](
    tup: Sequence[ArrayLike],
    *,
    dtype: _DTypeLike[ScalarT],
    casting: _CastingKind = "same_kind"
) -> NDArray[ScalarT]: ...
@overload
def vstack(
    tup: Sequence[ArrayLike],
    *,
    dtype: DTypeLike | None = None,
    casting: _CastingKind = "same_kind"
) -> NDArray[Any]: ...

# keep in sync with `numpy.ma.extras.hstack`
@overload
def hstack[ScalarT: np.generic](
    tup: Sequence[_ArrayLike[ScalarT]],
    *,
    dtype: None = None,
    casting: _CastingKind = "same_kind"
) -> NDArray[ScalarT]: ...
@overload
def hstack[ScalarT: np.generic](
    tup: Sequence[ArrayLike],
    *,
    dtype: _DTypeLike[ScalarT],
    casting: _CastingKind = "same_kind"
) -> NDArray[ScalarT]: ...
@overload
def hstack(
    tup: Sequence[ArrayLike],
    *,
    dtype: DTypeLike | None = None,
    casting: _CastingKind = "same_kind"
) -> NDArray[Any]: ...

# keep in sync with `numpy.ma.extras.stack`
@overload
def stack[ScalarT: np.generic](
    arrays: Sequence[_ArrayLike[ScalarT]],
    axis: SupportsIndex = 0,
    out: None = None,
    *,
    dtype: None = None,
    casting: _CastingKind = "same_kind"
) -> NDArray[ScalarT]: ...
@overload
def stack[ScalarT: np.generic](
    arrays: Sequence[ArrayLike],
    axis: SupportsIndex = 0,
    out: None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    casting: _CastingKind = "same_kind"
) -> NDArray[ScalarT]: ...
@overload
def stack(
    arrays: Sequence[ArrayLike],
    axis: SupportsIndex = 0,
    out: None = None,
    *,
    dtype: DTypeLike | None = None,
    casting: _CastingKind = "same_kind"
) -> NDArray[Any]: ...
@overload
def stack[OutT: np.ndarray](
    arrays: Sequence[ArrayLike],
    axis: SupportsIndex,
    out: OutT,
    *,
    dtype: DTypeLike | None = None,
    casting: _CastingKind = "same_kind",
) -> OutT: ...
@overload
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
