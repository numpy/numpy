from _typeshed import Incomplete
from collections.abc import Callable, Sequence
from typing import (
    Any,
    Concatenate,
    Never,
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
    _AnyShape,
    _ArrayLike,
    _ArrayLikeBool_co,
    _ArrayLikeComplex_co,
    _ArrayLikeFloat_co,
    _ArrayLikeInt,
    _ArrayLikeInt_co,
    _ArrayLikeObject_co,
    _ArrayLikeUInt_co,
    _ScalarLike_co,
    _Shape,
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

type _JustAnyShape = tuple[Never, Never, Never, Never]  # workaround for microsoft/pyright#10232

type _0d = tuple[()]
type _1d = tuple[int]
type _2d = tuple[int, int]
type _3d = tuple[int, int, int]
type _4d = tuple[int, int, int, int]
type _5d = tuple[int, int, int, int, int]
type _6d = tuple[int, int, int, int, int, int]

type _Min1D = tuple[int, *tuple[int, ...]]
type _Min2D = tuple[int, int, *tuple[int, ...]]
type _Min3D = tuple[int, int, int, *tuple[int, ...]]

type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]

type _To1D[ScalarT: np.generic] = np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]] | ScalarT
type _To2D[ScalarT: np.generic] = np.ndarray[tuple[()] | tuple[int] | tuple[int, int], np.dtype[ScalarT]] | ScalarT
type _To3D[ScalarT: np.generic] = (
    np.ndarray[tuple[()] | tuple[int] | tuple[int, int] | tuple[int, int, int], np.dtype[ScalarT]] | ScalarT
)

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
@overload  # Nd -> Nd
def expand_dims[ShapeT: _Shape, DTypeT: np.dtype](
    a: np.ndarray[ShapeT, DTypeT],
    axis: tuple[()],
) -> np.ndarray[ShapeT, DTypeT]: ...
@overload  # ?d -> ?d  (workaround)
def expand_dims[DTypeT: np.dtype](
    a: np.ndarray[_JustAnyShape, DTypeT],
    axis: int | tuple[int, ...],
) -> np.ndarray[_AnyShape, DTypeT]: ...
@overload  # 0d -> 1d
def expand_dims[ScalarT: np.generic](
    a: ScalarT | np.ndarray[_0d, np.dtype[ScalarT]],
    axis: int | tuple[int],
) -> np.ndarray[_1d, np.dtype[ScalarT]]: ...
@overload  # 0d -> 2d
def expand_dims[ScalarT: np.generic](
    a: ScalarT | np.ndarray[_0d, np.dtype[ScalarT]],
    axis: tuple[int, int],
) -> np.ndarray[_2d, np.dtype[ScalarT]]: ...
@overload  # 1d -> 2d
def expand_dims[DTypeT: np.dtype](
    a: np.ndarray[_1d, DTypeT],
    axis: int | tuple[int],
) -> np.ndarray[_2d, DTypeT]: ...
@overload  # 1d -> 3d
def expand_dims[DTypeT: np.dtype](
    a: np.ndarray[_1d, DTypeT],
    axis: tuple[int, int],
) -> np.ndarray[_3d, DTypeT]: ...
@overload  # 2d -> 3d
def expand_dims[DTypeT: np.dtype](
    a: np.ndarray[_2d, DTypeT],
    axis: int | tuple[int],
) -> np.ndarray[_3d, DTypeT]: ...
@overload  # 2d -> 4d
def expand_dims[DTypeT: np.dtype](
    a: np.ndarray[_2d, DTypeT],
    axis: tuple[int, int],
) -> np.ndarray[_4d, DTypeT]: ...
@overload  # 3d -> 4d
def expand_dims[DTypeT: np.dtype](
    a: np.ndarray[_3d, DTypeT],
    axis: int | tuple[int],
) -> np.ndarray[_4d, DTypeT]: ...
@overload  # 3d -> 5d
def expand_dims[DTypeT: np.dtype](
    a: np.ndarray[_3d, DTypeT],
    axis: tuple[int, int],
) -> np.ndarray[_5d, DTypeT]: ...
@overload  # 4d -> 5d
def expand_dims[DTypeT: np.dtype](
    a: np.ndarray[_4d, DTypeT],
    axis: int | tuple[int],
) -> np.ndarray[_5d, DTypeT]: ...
@overload  # 4d -> 6d
def expand_dims[DTypeT: np.dtype](
    a: np.ndarray[_4d, DTypeT],
    axis: tuple[int, int],
) -> np.ndarray[_6d, DTypeT]: ...
@overload  # Nd -> ?d
def expand_dims[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    axis: int | tuple[int, ...],
) -> NDArray[ScalarT]: ...
@overload  # fallback
def expand_dims(a: ArrayLike, axis: int | tuple[int, ...]) -> NDArray[Any]: ...

# keep in sync with `numpy.ma.extras.column_stack`
@overload  # >=2d, known dtype
def column_stack[ShapeT: _Min2D, DTypeT: np.dtype](
    tup: Sequence[np.ndarray[ShapeT, DTypeT]],
) -> np.ndarray[ShapeT, DTypeT]: ...
@overload  # <=2d, known dtype
def column_stack[ScalarT: np.generic](tup: Sequence[_To2D[ScalarT]]) -> _Array2D[ScalarT]: ...
@overload  # ?d, known dtype
def column_stack[ScalarT: np.generic](tup: Sequence[_ArrayLike[ScalarT]]) -> NDArray[ScalarT]: ...
@overload  # fallback
def column_stack(tup: Sequence[ArrayLike]) -> NDArray[Any]: ...

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
@overload  # ?d, known dtype, (workaround overload)
def tile[DTypeT: np.dtype](
    A: np.ndarray[_JustAnyShape, DTypeT],
    reps: _ArrayLikeInt,
) -> np.ndarray[_AnyShape, DTypeT]: ...
@overload  # >=1d, known dtype, <=1d reps
def tile[ArrayT: np.ndarray[_Min1D]](A: ArrayT, reps: int | tuple[()] | tuple[int]) -> ArrayT: ...
@overload  # >=2d, known dtype, 2d reps
def tile[ArrayT: np.ndarray[_Min2D]](A: ArrayT, reps: tuple[int, int]) -> ArrayT: ...
@overload  # >=3d, known dtype, 3d reps
def tile[ArrayT: np.ndarray[_Min3D]](A: ArrayT, reps: tuple[int, int, int]) -> ArrayT: ...
@overload  # <=1d, known dtype, >=1d reps
def tile[ScalarT: np.generic, ShapeT: (_1d, _2d, _3d, _4d, _5d, _6d)](  # constraints avoid `Literal` propagation
    A: _To1D[ScalarT],
    reps: ShapeT,
) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
@overload  # <=2d, known dtype, >=2d reps
def tile[ScalarT: np.generic, ShapeT: (_2d, _3d, _4d, _5d, _6d)](
    A: _To2D[ScalarT],
    reps: ShapeT,
) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
@overload  # <=3d, known dtype, >=3d reps
def tile[ScalarT: np.generic, ShapeT: (_3d, _4d, _5d, _6d)](
    A: _To3D[ScalarT],
    reps: ShapeT,
) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
@overload  # <=1d, unknown dtype, >=1d reps
def tile[ShapeT: (_1d, _2d, _3d, _4d, _5d, _6d)](
    A: Sequence[_ScalarLike_co] | _ScalarLike_co,
    reps: ShapeT,
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # <=1d, unknown dtype, 1d reps
def tile(
    A: Sequence[_ScalarLike_co] | _ScalarLike_co,
    reps: int,
) -> np.ndarray[tuple[int], np.dtype[Any]]: ...
@overload  # 2d, unknown dtype, >=2d reps
def tile[ShapeT: (_2d, _3d, _4d, _5d, _6d)](
    A: Sequence[Sequence[_ScalarLike_co]],
    reps: ShapeT,
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # 3d, unknown dtype, >=3d reps
def tile[ShapeT: (_3d, _4d, _5d, _6d)](
    A: Sequence[Sequence[Sequence[_ScalarLike_co]]],
    reps: ShapeT,
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # ?d, known dtype
def tile[ScalarT: np.generic](A: _ArrayLike[ScalarT], reps: _ArrayLikeInt) -> NDArray[ScalarT]: ...
@overload  # ?d, unknown dtype
def tile(A: ArrayLike, reps: _ArrayLikeInt) -> NDArray[Incomplete]: ...
