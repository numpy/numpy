from _typeshed import Incomplete
from collections.abc import Callable, Sequence
from typing import (
    Any,
    Concatenate,
    Never,
    Protocol,
    Self,
    SupportsIndex,
    TypeVar,
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
type _Array3D[ScalarT: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[ScalarT]]
type _ArrayMax2D[ScalarT: np.generic] = np.ndarray[tuple[int] | tuple[int, int], np.dtype[ScalarT]]
type _ArrayJustND[ScalarT: np.generic] = np.ndarray[_JustAnyShape, np.dtype[ScalarT]]

type _To0D[ScalarT: np.generic] = np.ndarray[tuple[()], np.dtype[ScalarT]] | ScalarT
type _To1D[ScalarT: np.generic] = np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]] | ScalarT
type _To2D[ScalarT: np.generic] = np.ndarray[tuple[()] | tuple[int] | tuple[int, int], np.dtype[ScalarT]] | ScalarT
type _To3D[ScalarT: np.generic] = (
    np.ndarray[tuple[()] | tuple[int] | tuple[int, int] | tuple[int, int, int], np.dtype[ScalarT]] | ScalarT
)

type _Func1D[ScalarT: np.generic, **Tss, ReturnT] = Callable[Concatenate[_Array1D[ScalarT], Tss], ReturnT]

_AnyNumberT = TypeVar(
    "_AnyNumberT",
    np.bool,
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float16, np.float32, np.float64, np.longdouble,
    np.complex64, np.complex128, np.clongdouble,
    np.object_,
)

###

@overload  # Nd T, Nd, axis=<given>
def take_along_axis[ScalarT: np.generic, ShapeT: _Shape](
    arr: ScalarT | NDArray[ScalarT],
    indices: np.ndarray[ShapeT, np.dtype[np.integer]],
    axis: int = -1,
) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
@overload  # Nd T, 1d, axis=None
def take_along_axis[ScalarT: np.generic](
    arr: ScalarT | NDArray[ScalarT],
    indices: NDArray[np.integer],
    axis: None,
) -> _Array1D[ScalarT]: ...

#
def put_along_axis[ScalarT: np.generic](
    arr: NDArray[ScalarT],
    indices: NDArray[np.integer],
    values: ArrayLike,
    axis: int | None,
) -> None: ...

#
@overload  # (1d T) -> ?d T, ?d T  (workaround)
def apply_along_axis[ScalarT: np.generic, **Tss, ResultT: np.generic](
    func1d: _Func1D[ScalarT, Tss, _ArrayJustND[ResultT]],
    axis: SupportsIndex,
    arr: NDArray[ScalarT],
    *args: Tss.args,
    **kwargs: Tss.kwargs,
) -> NDArray[ResultT]: ...
@overload  # (1d T) -> ?d T, ?d T  (workaround)
def apply_along_axis[ScalarT: np.generic, **Tss, ResultT: np.generic](
    func1d: _Func1D[ScalarT, Tss, _ArrayLike[ResultT]],
    axis: SupportsIndex,
    arr: _ArrayJustND[ScalarT],
    *args: Tss.args,
    **kwargs: Tss.kwargs,
) -> NDArray[ResultT]: ...
@overload  # (1d T) -> bool, ?d T  (workaround)
def apply_along_axis[ScalarT: np.generic, **Tss](
    func1d: _Func1D[ScalarT, Tss, bool],
    axis: SupportsIndex,
    arr: _ArrayJustND[ScalarT],
    *args: Tss.args,
    **kwargs: Tss.kwargs,
) -> NDArray[np.bool]: ...
@overload  # (1d T) -> ~int, ?d T  (workaround)
def apply_along_axis[ScalarT: np.generic, **Tss](
    func1d: _Func1D[ScalarT, Tss, int],
    axis: SupportsIndex,
    arr: _ArrayJustND[ScalarT],
    *args: Tss.args,
    **kwargs: Tss.kwargs,
) -> NDArray[np.int_]: ...
@overload  # (1d T) -> ~float, ?d T  (workaround)
def apply_along_axis[ScalarT: np.generic, **Tss](
    func1d: _Func1D[ScalarT, Tss, float],
    axis: SupportsIndex,
    arr: _ArrayJustND[ScalarT],
    *args: Tss.args,
    **kwargs: Tss.kwargs,
) -> NDArray[np.float64]: ...
@overload  # (1d T) -> ~complex, ?d T  (workaround)
def apply_along_axis[ScalarT: np.generic, **Tss](
    func1d: _Func1D[ScalarT, Tss, complex],
    axis: SupportsIndex,
    arr: _ArrayJustND[ScalarT],
    *args: Tss.args,
    **kwargs: Tss.kwargs,
) -> NDArray[np.complex128]: ...
@overload  # (1d T) -> 0d T, 1d T
def apply_along_axis[ScalarT: np.generic, **Tss, ResultT: np.generic](
    func1d: _Func1D[ScalarT, Tss, _To0D[ResultT]],
    axis: SupportsIndex,
    arr: _Array1D[ScalarT],
    *args: Tss.args,
    **kwargs: Tss.kwargs,
) -> np.ndarray[tuple[()], np.dtype[ResultT]]: ...
@overload  # (1d T) -> 1d T, 1d T
def apply_along_axis[ScalarT: np.generic, **Tss, ResultT: np.generic](
    func1d: _Func1D[ScalarT, Tss, _Array1D[ResultT]],
    axis: SupportsIndex,
    arr: _Array1D[ScalarT],
    *args: Tss.args,
    **kwargs: Tss.kwargs,
) -> _Array1D[ResultT]: ...
@overload  # (1d T) -> 0d T, 2d T
def apply_along_axis[ScalarT: np.generic, **Tss, ResultT: np.generic](
    func1d: _Func1D[ScalarT, Tss, _To0D[ResultT]],
    axis: SupportsIndex,
    arr: _Array2D[ScalarT],
    *args: Tss.args,
    **kwargs: Tss.kwargs,
) -> _Array1D[ResultT]: ...
@overload  # (1d T) -> 1d T, 2d T
def apply_along_axis[ScalarT: np.generic, **Tss, ResultT: np.generic](
    func1d: _Func1D[ScalarT, Tss, _Array1D[ResultT]],
    axis: SupportsIndex,
    arr: _Array2D[ScalarT],
    *args: Tss.args,
    **kwargs: Tss.kwargs,
) -> _Array2D[ResultT]: ...
@overload  # (1d T) -> 2d T, 2d T
def apply_along_axis[ScalarT: np.generic, **Tss, ResultT: np.generic](
    func1d: _Func1D[ScalarT, Tss, _Array2D[ResultT]],
    axis: SupportsIndex,
    arr: _Array2D[ScalarT],
    *args: Tss.args,
    **kwargs: Tss.kwargs,
) -> _Array3D[ResultT]: ...
@overload  # (1d T) -> bool, 2d T
def apply_along_axis[ScalarT: np.generic, **Tss](
    func1d: _Func1D[ScalarT, Tss, bool],
    axis: SupportsIndex,
    arr: _Array2D[ScalarT],
    *args: Tss.args,
    **kwargs: Tss.kwargs,
) -> _Array1D[np.bool]: ...
@overload  # (1d T) -> ~int, 2d T
def apply_along_axis[ScalarT: np.generic, **Tss](
    func1d: _Func1D[ScalarT, Tss, int],
    axis: SupportsIndex,
    arr: _Array2D[ScalarT],
    *args: Tss.args,
    **kwargs: Tss.kwargs,
) -> _Array1D[np.int_]: ...
@overload  # (1d T) -> ~float, 2d T
def apply_along_axis[ScalarT: np.generic, **Tss](
    func1d: _Func1D[ScalarT, Tss, float],
    axis: SupportsIndex,
    arr: _Array2D[ScalarT],
    *args: Tss.args,
    **kwargs: Tss.kwargs,
) -> _Array1D[np.float64]: ...
@overload  # (1d T) -> ~complex, 2d T
def apply_along_axis[ScalarT: np.generic, **Tss](
    func1d: _Func1D[ScalarT, Tss, complex],
    axis: SupportsIndex,
    arr: _Array2D[ScalarT],
    *args: Tss.args,
    **kwargs: Tss.kwargs,
) -> _Array1D[np.complex128]: ...
@overload  # (1d T) -> 0d T, 3d T
def apply_along_axis[ScalarT: np.generic, **Tss, ResultT: np.generic](
    func1d: _Func1D[ScalarT, Tss, _To0D[ResultT]],
    axis: SupportsIndex,
    arr: _Array3D[ScalarT],
    *args: Tss.args,
    **kwargs: Tss.kwargs,
) -> _Array2D[ResultT]: ...
@overload  # (1d T) -> 1d T, 3d T
def apply_along_axis[ScalarT: np.generic, **Tss, ResultT: np.generic](
    func1d: _Func1D[ScalarT, Tss, _Array1D[ResultT]],
    axis: SupportsIndex,
    arr: _Array3D[ScalarT],
    *args: Tss.args,
    **kwargs: Tss.kwargs,
) -> _Array3D[ResultT]: ...
@overload  # (1d T) -> bool, 3d T
def apply_along_axis[ScalarT: np.generic, **Tss](
    func1d: _Func1D[ScalarT, Tss, bool],
    axis: SupportsIndex,
    arr: _Array3D[ScalarT],
    *args: Tss.args,
    **kwargs: Tss.kwargs,
) -> _Array2D[np.bool]: ...
@overload  # (1d T) -> ~int, 3d T
def apply_along_axis[ScalarT: np.generic, **Tss](
    func1d: _Func1D[ScalarT, Tss, int],
    axis: SupportsIndex,
    arr: _Array3D[ScalarT],
    *args: Tss.args,
    **kwargs: Tss.kwargs,
) -> _Array2D[np.int_]: ...
@overload  # (1d T) -> ~float, 3d T
def apply_along_axis[ScalarT: np.generic, **Tss](
    func1d: _Func1D[ScalarT, Tss, float],
    axis: SupportsIndex,
    arr: _Array3D[ScalarT],
    *args: Tss.args,
    **kwargs: Tss.kwargs,
) -> _Array2D[np.float64]: ...
@overload  # (1d T) -> ~complex, 3d T
def apply_along_axis[ScalarT: np.generic, **Tss](
    func1d: _Func1D[ScalarT, Tss, complex],
    axis: SupportsIndex,
    arr: _Array3D[ScalarT],
    *args: Tss.args,
    **kwargs: Tss.kwargs,
) -> _Array2D[np.complex128]: ...
@overload  # (1d) -> ?d T, ?d
def apply_along_axis[**Tss, ResultT: np.generic](
    func1d: _Func1D[Any, Tss, _ArrayLike[ResultT]],
    axis: SupportsIndex,
    arr: ArrayLike,
    *args: Tss.args,
    **kwargs: Tss.kwargs,
) -> NDArray[ResultT]: ...
@overload  # (1d) -> ?, ?d  (fallback)
def apply_along_axis[**Tss](
    func1d: _Func1D[Any, Tss, Any],
    axis: SupportsIndex,
    arr: ArrayLike,
    *args: Tss.args,
    **kwargs: Tss.kwargs,
) -> NDArray[Any]: ...

#
@overload  # (Nd T) -> ?d T, Nd T
def apply_over_axes[ShapeT: _Shape, ScalarT: np.generic, ResultT: np.generic](
    func: Callable[[np.ndarray[ShapeT, np.dtype[ScalarT]], int], NDArray[ResultT]],
    a: np.ndarray[ShapeT, np.dtype[ScalarT]],
    axes: _ShapeLike,
) -> np.ndarray[ShapeT, np.dtype[ResultT]]: ...
@overload  # (1d T) -> ?d T, 1d T
def apply_over_axes[ScalarT: np.generic, ResultT: np.generic](
    func: Callable[[_Array1D[ScalarT], int], NDArray[ResultT] | ResultT],
    a: _Array1D[ScalarT],
    axes: _ShapeLike,
) -> _Array1D[ResultT]: ...

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

#
@overload  # >=3d T
def dstack[ShapeT: _Min3D, DTypeT: np.dtype](tup: Sequence[np.ndarray[ShapeT, DTypeT]]) -> np.ndarray[ShapeT, DTypeT]: ...
@overload  # <=3d T
def dstack[ScalarT: np.generic](tup: Sequence[_To3D[ScalarT]]) -> _Array3D[ScalarT]: ...
@overload  # ?d T
def dstack[ScalarT: np.generic](tup: Sequence[_ArrayLike[ScalarT]]) -> NDArray[ScalarT]: ...
@overload  # fallback
def dstack(tup: Sequence[ArrayLike]) -> NDArray[Any]: ...

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
@overload  # ?d T, ?d T  (workaround)
def kron(  # noqa: UP047
    a: _ArrayJustND[_AnyNumberT],
    b: _ArrayLike[_AnyNumberT],
) -> NDArray[_AnyNumberT]: ...
@overload  # ?d T, ?d T  (workaround)
def kron(  # noqa: UP047
    a: _ArrayLike[_AnyNumberT],
    b: _ArrayJustND[_AnyNumberT],
) -> NDArray[_AnyNumberT]: ...
@overload  # 1d T, 1d T
def kron(  # noqa: UP047
    a: _Array1D[_AnyNumberT],
    b: _Array1D[_AnyNumberT],
) -> _Array1D[_AnyNumberT]: ...
@overload  # 1d T, 2d T
def kron(  # noqa: UP047
    a: _Array1D[_AnyNumberT],
    b: _Array2D[_AnyNumberT],
) -> _Array2D[_AnyNumberT]: ...
@overload  # 2d T, <=2d T
def kron(  # noqa: UP047
    a: _Array2D[_AnyNumberT],
    b: _ArrayMax2D[_AnyNumberT],
) -> _Array2D[_AnyNumberT]: ...
@overload  # <=2d T, 3d T
def kron(  # noqa: UP047
    a: _ArrayMax2D[_AnyNumberT],
    b: _Array3D[_AnyNumberT],
) -> _Array3D[_AnyNumberT]: ...
@overload  # 3d T, <=3d T
def kron(  # noqa: UP047
    a: _Array3D[_AnyNumberT],
    b: np.ndarray[tuple[int] | tuple[int, int] | tuple[int, int, int], np.dtype[_AnyNumberT]],
) -> _Array3D[_AnyNumberT]: ...
@overload  # ?d T, ?d T
def kron(  # noqa: UP047
    a: _ArrayLike[_AnyNumberT],
    b: _ArrayLike[_AnyNumberT],
) -> NDArray[_AnyNumberT]: ...
@overload  # ?d bool, ?d bool
def kron(a: _ArrayLikeBool_co, b: _ArrayLikeBool_co) -> NDArray[np.bool]: ...
@overload  # ?d +int, ?d +int
def kron(a: _ArrayLikeInt_co, b: _ArrayLikeInt_co) -> NDArray[np.int_ | Any]: ...
@overload  # ?d +f64, ?d +f64
def kron(a: _ArrayLikeFloat_co, b: _ArrayLikeFloat_co) -> NDArray[np.float64 | Any]: ...
@overload  # ?d +c128, ?d +c128
def kron(a: _ArrayLikeComplex_co, b: _ArrayLikeComplex_co) -> NDArray[np.complex128 | Any]: ...
@overload  # ?d ~object_, ?d
def kron(a: _ArrayLikeObject_co, b: object) -> NDArray[np.object_]: ...
@overload  # ?d, ?d ~object_
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
