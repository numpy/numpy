from collections.abc import Callable, Sequence
from typing import Any, Concatenate, Protocol, SupportsIndex, overload, type_check_only
from typing_extensions import deprecated

import numpy as np
from numpy import (
    _CastingKind,
    complexfloating,
    floating,
    integer,
    object_,
    signedinteger,
    ufunc,
    unsignedinteger,
)
from numpy._typing import (
    ArrayLike,
    DTypeLike,
    NDArray,
    _ArrayLike,
    _ArrayLikeBool_co,
    _ArrayLikeComplex_co,
    _ArrayLikeFloat_co,
    _ArrayLikeInt_co,
    _ArrayLikeObject_co,
    _ArrayLikeUInt_co,
    _ShapeLike,
)

__all__ = [
    "column_stack",
    "row_stack",
    "dstack",
    "array_split",
    "split",
    "hsplit",
    "vsplit",
    "dsplit",
    "apply_over_axes",
    "expand_dims",
    "apply_along_axis",
    "kron",
    "tile",
    "take_along_axis",
    "put_along_axis",
]

# Signature of `__array_wrap__`
@type_check_only
class _ArrayWrap(Protocol):
    def __call__(
        self,
        array: NDArray[Any],
        context: tuple[ufunc, tuple[Any, ...], int] | None = ...,
        return_scalar: bool = ...,
        /,
    ) -> Any: ...

@type_check_only
class _SupportsArrayWrap(Protocol):
    @property
    def __array_wrap__(self) -> _ArrayWrap: ...

###

def take_along_axis[ScalarT: np.generic](
    arr: ScalarT | NDArray[ScalarT],
    indices: NDArray[integer],
    axis: int | None = -1,
) -> NDArray[ScalarT]: ...

def put_along_axis[ScalarT: np.generic](
    arr: NDArray[ScalarT],
    indices: NDArray[integer],
    values: ArrayLike,
    axis: int | None,
) -> None: ...

@overload
def apply_along_axis[**Tss, ScalarT: np.generic](
    func1d: Callable[Concatenate[np.ndarray, Tss], _ArrayLike[ScalarT]],
    axis: SupportsIndex,
    arr: ArrayLike,
    *args: Tss.args,
    **kwargs: Tss.kwargs,
) -> NDArray[ScalarT]: ...
@overload
def apply_along_axis[**Tss](
    func1d: Callable[Concatenate[NDArray[Any], Tss], Any],
    axis: SupportsIndex,
    arr: ArrayLike,
    *args: Tss.args,
    **kwargs: Tss.kwargs,
) -> NDArray[Any]: ...

def apply_over_axes[ScalarT: np.generic](
    func: Callable[[np.ndarray, int], NDArray[ScalarT]],
    a: ArrayLike,
    axes: int | Sequence[int],
) -> NDArray[ScalarT]: ...

@overload
def expand_dims[ScalarT: np.generic](a: _ArrayLike[ScalarT], axis: _ShapeLike) -> NDArray[ScalarT]: ...
@overload
def expand_dims(a: ArrayLike, axis: _ShapeLike) -> NDArray[Any]: ...

# Deprecated in NumPy 2.0, 2023-08-18
@deprecated("`row_stack` alias is deprecated. Use `np.vstack` directly.")
def row_stack(
    tup: Sequence[ArrayLike],
    *,
    dtype: DTypeLike | None = None,
    casting: _CastingKind = "same_kind",
) -> NDArray[Any]: ...

# keep in sync with `numpy.ma.extras.column_stack`
@overload
def column_stack[ScalarT: np.generic](tup: Sequence[_ArrayLike[ScalarT]]) -> NDArray[ScalarT]: ...
@overload
def column_stack(tup: Sequence[ArrayLike]) -> NDArray[Any]: ...

# keep in sync with `numpy.ma.extras.dstack`
@overload
def dstack[ScalarT: np.generic](tup: Sequence[_ArrayLike[ScalarT]]) -> NDArray[ScalarT]: ...
@overload
def dstack(tup: Sequence[ArrayLike]) -> NDArray[Any]: ...

@overload
def array_split[ScalarT: np.generic](
    ary: _ArrayLike[ScalarT],
    indices_or_sections: _ShapeLike,
    axis: SupportsIndex = 0,
) -> list[NDArray[ScalarT]]: ...
@overload
def array_split(
    ary: ArrayLike,
    indices_or_sections: _ShapeLike,
    axis: SupportsIndex = 0,
) -> list[NDArray[Any]]: ...

@overload
def split[ScalarT: np.generic](
    ary: _ArrayLike[ScalarT],
    indices_or_sections: _ShapeLike,
    axis: SupportsIndex = 0,
) -> list[NDArray[ScalarT]]: ...
@overload
def split(
    ary: ArrayLike,
    indices_or_sections: _ShapeLike,
    axis: SupportsIndex = 0,
) -> list[NDArray[Any]]: ...

# keep in sync with `numpy.ma.extras.hsplit`
@overload
def hsplit[ScalarT: np.generic](
    ary: _ArrayLike[ScalarT],
    indices_or_sections: _ShapeLike,
) -> list[NDArray[ScalarT]]: ...
@overload
def hsplit(
    ary: ArrayLike,
    indices_or_sections: _ShapeLike,
) -> list[NDArray[Any]]: ...

@overload
def vsplit[ScalarT: np.generic](
    ary: _ArrayLike[ScalarT],
    indices_or_sections: _ShapeLike,
) -> list[NDArray[ScalarT]]: ...
@overload
def vsplit(
    ary: ArrayLike,
    indices_or_sections: _ShapeLike,
) -> list[NDArray[Any]]: ...

@overload
def dsplit[ScalarT: np.generic](
    ary: _ArrayLike[ScalarT],
    indices_or_sections: _ShapeLike,
) -> list[NDArray[ScalarT]]: ...
@overload
def dsplit(
    ary: ArrayLike,
    indices_or_sections: _ShapeLike,
) -> list[NDArray[Any]]: ...

@overload
def get_array_wrap(*args: _SupportsArrayWrap) -> _ArrayWrap: ...
@overload
def get_array_wrap(*args: object) -> _ArrayWrap | None: ...

@overload
def kron(a: _ArrayLikeBool_co, b: _ArrayLikeBool_co) -> NDArray[np.bool]: ...
@overload
def kron(a: _ArrayLikeUInt_co, b: _ArrayLikeUInt_co) -> NDArray[unsignedinteger]: ...
@overload
def kron(a: _ArrayLikeInt_co, b: _ArrayLikeInt_co) -> NDArray[signedinteger]: ...
@overload
def kron(a: _ArrayLikeFloat_co, b: _ArrayLikeFloat_co) -> NDArray[floating]: ...
@overload
def kron(a: _ArrayLikeComplex_co, b: _ArrayLikeComplex_co) -> NDArray[complexfloating]: ...
@overload
def kron(a: _ArrayLikeObject_co, b: Any) -> NDArray[object_]: ...
@overload
def kron(a: Any, b: _ArrayLikeObject_co) -> NDArray[object_]: ...

@overload
def tile[ScalarT: np.generic](
    A: _ArrayLike[ScalarT],
    reps: int | Sequence[int],
) -> NDArray[ScalarT]: ...
@overload
def tile(
    A: ArrayLike,
    reps: int | Sequence[int],
) -> NDArray[Any]: ...
