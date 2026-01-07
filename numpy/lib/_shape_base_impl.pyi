from _typeshed import Incomplete
from collections.abc import Callable, Sequence
from typing import (
    Any,
    Concatenate,
    Protocol,
    Self,
    SupportsIndex,
    overload,
    type_check_only,
)

import numpy as np
from numpy._typing import (
    ArrayLike,
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
        context: tuple[np.ufunc, tuple[Any, ...], int] | None = ...,
        return_scalar: bool = ...,
        /,
    ) -> Any: ...

@type_check_only
class _SupportsArrayWrap(Protocol):
    @property
    def __array_wrap__(self) -> _ArrayWrap: ...

# Protocol for array-like objects that preserve their type through split operations.
# Requires shape for size, ndim for dimensional checks in hsplit/vsplit/dsplit,
# swapaxes for axis manipulation, and __getitem__ for slicing.
@type_check_only
class _SupportsSplitOps(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def ndim(self) -> int: ...
    def swapaxes(self, axis1: int, axis2: int, /) -> Self: ...
    def __getitem__(self, key: Any, /) -> Self: ...

###

def take_along_axis[ScalarT: np.generic](
    arr: ScalarT | NDArray[ScalarT],
    indices: NDArray[np.integer],
    axis: int | None = -1,
) -> NDArray[ScalarT]: ...

#
def put_along_axis[ScalarT: np.generic](
    arr: NDArray[ScalarT],
    indices: NDArray[np.integer],
    values: ArrayLike,
    axis: int | None,
) -> None: ...

#
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
    func1d: Callable[Concatenate[np.ndarray, Tss], Any],
    axis: SupportsIndex,
    arr: ArrayLike,
    *args: Tss.args,
    **kwargs: Tss.kwargs,
) -> NDArray[Any]: ...

#
def apply_over_axes[ScalarT: np.generic](
    func: Callable[[np.ndarray, int], NDArray[ScalarT]],
    a: ArrayLike,
    axes: _ShapeLike,
) -> NDArray[ScalarT]: ...

#
@overload
def expand_dims[ScalarT: np.generic](a: _ArrayLike[ScalarT], axis: _ShapeLike) -> NDArray[ScalarT]: ...
@overload
def expand_dims(a: ArrayLike, axis: _ShapeLike) -> NDArray[Incomplete]: ...

# keep in sync with `numpy.ma.extras.column_stack`
@overload
def column_stack[ScalarT: np.generic](tup: Sequence[_ArrayLike[ScalarT]]) -> NDArray[ScalarT]: ...
@overload
def column_stack(tup: Sequence[ArrayLike]) -> NDArray[Incomplete]: ...

# keep in sync with `numpy.ma.extras.dstack`
@overload
def dstack[ScalarT: np.generic](tup: Sequence[_ArrayLike[ScalarT]]) -> NDArray[ScalarT]: ...
@overload
def dstack(tup: Sequence[ArrayLike]) -> NDArray[Incomplete]: ...

#
@overload
def array_split[SplitableT: _SupportsSplitOps](
    ary: SplitableT,
    indices_or_sections: _ShapeLike,
    axis: SupportsIndex = 0,
) -> list[SplitableT]: ...
@overload
def array_split[ScalarT: np.generic](
    ary: _ArrayLike[ScalarT],
    indices_or_sections: _ShapeLike,
    axis: SupportsIndex = 0,
) -> list[NDArray[ScalarT]]: ...
@overload
def array_split(ary: ArrayLike, indices_or_sections: _ShapeLike, axis: SupportsIndex = 0) -> list[NDArray[Incomplete]]: ...

#
@overload
def split[SplitableT: _SupportsSplitOps](
    ary: SplitableT,
    indices_or_sections: _ShapeLike,
    axis: SupportsIndex = 0,
) -> list[SplitableT]: ...
@overload
def split[ScalarT: np.generic](
    ary: _ArrayLike[ScalarT],
    indices_or_sections: _ShapeLike,
    axis: SupportsIndex = 0,
) -> list[NDArray[ScalarT]]: ...
@overload
def split(ary: ArrayLike, indices_or_sections: _ShapeLike, axis: SupportsIndex = 0) -> list[NDArray[Incomplete]]: ...

# keep in sync with `numpy.ma.extras.hsplit`
@overload
def hsplit[SplitableT: _SupportsSplitOps](ary: SplitableT, indices_or_sections: _ShapeLike) -> list[SplitableT]: ...
@overload
def hsplit[ScalarT: np.generic](ary: _ArrayLike[ScalarT], indices_or_sections: _ShapeLike) -> list[NDArray[ScalarT]]: ...
@overload
def hsplit(ary: ArrayLike, indices_or_sections: _ShapeLike) -> list[NDArray[Incomplete]]: ...

#
@overload
def vsplit[SplitableT: _SupportsSplitOps](ary: SplitableT, indices_or_sections: _ShapeLike) -> list[SplitableT]: ...
@overload
def vsplit[ScalarT: np.generic](ary: _ArrayLike[ScalarT], indices_or_sections: _ShapeLike) -> list[NDArray[ScalarT]]: ...
@overload
def vsplit(ary: ArrayLike, indices_or_sections: _ShapeLike) -> list[NDArray[Incomplete]]: ...

#
@overload
def dsplit[SplitableT: _SupportsSplitOps](ary: SplitableT, indices_or_sections: _ShapeLike) -> list[SplitableT]: ...
@overload
def dsplit[ScalarT: np.generic](ary: _ArrayLike[ScalarT], indices_or_sections: _ShapeLike) -> list[NDArray[ScalarT]]: ...
@overload
def dsplit(ary: ArrayLike, indices_or_sections: _ShapeLike) -> list[NDArray[Incomplete]]: ...

#
@overload
def kron(a: _ArrayLikeBool_co, b: _ArrayLikeBool_co) -> NDArray[np.bool]: ...
@overload
def kron(a: _ArrayLikeUInt_co, b: _ArrayLikeUInt_co) -> NDArray[np.unsignedinteger]: ...
@overload
def kron(a: _ArrayLikeInt_co, b: _ArrayLikeInt_co) -> NDArray[np.signedinteger]: ...
@overload
def kron(a: _ArrayLikeFloat_co, b: _ArrayLikeFloat_co) -> NDArray[np.floating]: ...
@overload
def kron(a: _ArrayLikeComplex_co, b: _ArrayLikeComplex_co) -> NDArray[np.complexfloating]: ...
@overload
def kron(a: _ArrayLikeObject_co, b: object) -> NDArray[np.object_]: ...
@overload
def kron(a: object, b: _ArrayLikeObject_co) -> NDArray[np.object_]: ...

#
@overload
def tile[ScalarT: np.generic](A: _ArrayLike[ScalarT], reps: int | Sequence[int]) -> NDArray[ScalarT]: ...
@overload
def tile(A: ArrayLike, reps: int | Sequence[int]) -> NDArray[Incomplete]: ...
