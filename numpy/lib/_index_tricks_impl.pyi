from _typeshed import Incomplete, SupportsLenAndGetItem
from collections.abc import Sequence
from typing import (
    Any,
    ClassVar,
    Final,
    Generic,
    Literal as L,
    Never,
    Self,
    SupportsIndex,
    final,
    overload,
    override,
)
from typing_extensions import TypeVar

import numpy as np
from numpy import _CastingKind
from numpy._core.multiarray import ravel_multi_index, unravel_index
from numpy._typing import (
    ArrayLike,
    DTypeLike,
    NDArray,
    _AnyShape,
    _ArrayLike,
    _DTypeLike,
    _IntLike_co,
    _NestedSequence,
    _ScalarLike_co,
    _SupportsArray,
)

__all__ = [
    "ravel_multi_index",
    "unravel_index",
    "mgrid",
    "ogrid",
    "r_",
    "c_",
    "s_",
    "index_exp",
    "ix_",
    "ndenumerate",
    "ndindex",
    "fill_diagonal",
    "diag_indices",
    "diag_indices_from",
]

###

_ScalarT_co = TypeVar("_ScalarT_co", bound=np.generic, default=Any, covariant=True)
_BoolT_co = TypeVar("_BoolT_co", bound=bool, default=bool, covariant=True)
_AxisT_co = TypeVar("_AxisT_co", bound=int, default=L[0], covariant=True)
_MatrixT_co = TypeVar("_MatrixT_co", bound=bool, default=L[False], covariant=True)
_NDMinT_co = TypeVar("_NDMinT_co", bound=int, default=L[1], covariant=True)
_Trans1DT_co = TypeVar("_Trans1DT_co", bound=int, default=L[-1], covariant=True)

type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]
type _Array3D[ScalarT: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[ScalarT]]
type _Array4D[ScalarT: np.generic] = np.ndarray[tuple[int, int, int, int], np.dtype[ScalarT]]

type _Int1D = _Array1D[np.intp]

type _ToArray1D[ScalarT: np.generic] = _Array1D[ScalarT] | Sequence[ScalarT]

type _SliceInt = slice[int | None, int | None, int | None]
type _SliceFloat = slice[float | None, float | None, complex | None]

type _ItemOrTuple[T] = T | tuple[T, *tuple[T, ...]]
type _To1D[ScalarT: np.generic] = ScalarT | np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]
type _To2D[ScalarT: np.generic] = ScalarT | np.ndarray[tuple[()] | tuple[int] | tuple[int, int], np.dtype[ScalarT]]
type _ToPy1D[T, StepT] = T | Sequence[T] | slice[T | None, T | None, StepT | None]

type _Tuple2[T] = tuple[T, T]
type _Tuple3[T] = tuple[T, T, T]

type _JustAnyShape = tuple[Never, Never, Never, Never, Never]  # workaround for microsoft/pyright#10232

###

class ndenumerate(Generic[_ScalarT_co]):
    iter: np.flatiter[NDArray[_ScalarT_co]]

    @overload
    def __init__[ScalarT: np.generic](
        self: ndenumerate[ScalarT],
        arr: _NestedSequence[_SupportsArray[np.dtype[ScalarT]]] | _SupportsArray[np.dtype[ScalarT]],
    ) -> None: ...
    @overload
    def __init__(self: ndenumerate[np.str_], arr: str | _NestedSequence[str]) -> None: ...
    @overload
    def __init__(self: ndenumerate[np.bytes_], arr: bytes | _NestedSequence[bytes]) -> None: ...
    @overload
    def __init__(self: ndenumerate[np.bool], arr: bool | _NestedSequence[bool]) -> None: ...
    @overload
    def __init__(self: ndenumerate[np.intp], arr: int | _NestedSequence[int]) -> None: ...
    @overload
    def __init__(self: ndenumerate[np.float64], arr: float | _NestedSequence[float]) -> None: ...
    @overload
    def __init__(self: ndenumerate[np.complex128], arr: complex | _NestedSequence[complex]) -> None: ...
    @overload
    def __init__(self: ndenumerate[Incomplete], arr: object) -> None: ...

    # The first overload is a (semi-)workaround for a mypy bug (tested with v1.10 and v1.11)
    @overload
    def __next__(
        self: ndenumerate[np.bool | np.number | np.flexible | np.datetime64 | np.timedelta64],
        /,
    ) -> tuple[_AnyShape, _ScalarT_co]: ...
    @overload
    def __next__(self: ndenumerate[np.object_], /) -> tuple[_AnyShape, Incomplete]: ...
    @overload
    def __next__(self, /) -> tuple[_AnyShape, _ScalarT_co]: ...

    #
    def __iter__(self) -> Self: ...

class ndindex:
    @overload
    def __init__(self, shape: tuple[SupportsIndex, ...], /) -> None: ...
    @overload
    def __init__(self, /, *shape: SupportsIndex) -> None: ...

    #
    def __iter__(self) -> Self: ...
    def __next__(self) -> _AnyShape: ...

class nd_grid:  # undocumented
    __slots__ = ("sparse",)

    sparse: Final[bool]

    def __init__(self, sparse: bool = False) -> None: ...
    def __getitem__(self, key: slice, /) -> NDArray[Any]: ...

@final
class MGridClass(nd_grid):
    __slots__ = ()

    def __init__(self) -> None: ...

    #
    @override
    @overload  # +int
    def __getitem__(self, key: _SliceInt, /) -> _Array1D[np.int_]: ...
    @overload  # +float
    def __getitem__(self, key: _SliceFloat, /) -> _Array1D[np.float64 | Any]: ...
    @overload  # ?
    def __getitem__(self, key: slice, /) -> _Array1D[Any]: ...
    @overload  # (+int, +int)
    def __getitem__(self, key: _Tuple2[_SliceInt], /) -> _Array3D[np.int_]: ...
    @overload  # (+float, +float)
    def __getitem__(self, key: _Tuple2[_SliceFloat], /) -> _Array3D[np.float64 | Any]: ...
    @overload  # (?, ?)
    def __getitem__(self, key: _Tuple2[slice], /) -> _Array3D[Any]: ...
    @overload  # (+int, +int, +int)
    def __getitem__(self, key: _Tuple3[_SliceInt], /) -> _Array4D[np.int_]: ...
    @overload  # (+float, +float, +float)
    def __getitem__(self, key: _Tuple3[_SliceFloat], /) -> _Array4D[np.float64 | Any]: ...
    @overload  # (?, ?, ?)
    def __getitem__(self, key: _Tuple3[slice], /) -> _Array4D[Any]: ...
    @overload  # fallback
    def __getitem__(self, key: slice | Sequence[slice], /) -> NDArray[Any]: ...

@final
class OGridClass(nd_grid):
    __slots__ = ()

    def __init__(self) -> None: ...

    #
    @override
    @overload  # +int
    def __getitem__(self, key: _SliceInt, /) -> _Array1D[np.int_]: ...
    @overload  # +float
    def __getitem__(self, key: _SliceFloat, /) -> _Array1D[np.float64 | Any]: ...
    @overload  # ?
    def __getitem__(self, key: slice, /) -> _Array1D[Any]: ...
    @overload  # (+int, +int)
    def __getitem__(self, key: _Tuple2[_SliceInt], /) -> _Tuple2[_Array2D[np.int_]]: ...
    @overload  # (+float, +float)
    def __getitem__(self, key: _Tuple2[_SliceFloat], /) -> _Tuple2[_Array2D[np.float64 | Any]]: ...
    @overload  # (?, ?)
    def __getitem__(self, key: _Tuple2[slice], /) -> _Tuple2[_Array2D[Any]]: ...
    @overload  # (+int, +int, +int)
    def __getitem__(self, key: _Tuple3[_SliceInt], /) -> _Tuple3[_Array3D[np.int_]]: ...
    @overload  # (+float, +float, +float)
    def __getitem__(self, key: _Tuple3[_SliceFloat], /) -> _Tuple3[_Array3D[np.float64 | Any]]: ...
    @overload  # (?, ?, ?)
    def __getitem__(self, key: _Tuple3[slice], /) -> _Tuple3[_Array3D[Any]]: ...
    @overload  # fallback
    def __getitem__(self, key: Sequence[slice], /) -> tuple[NDArray[Any], ...]: ...

class AxisConcatenator(Generic[_AxisT_co, _MatrixT_co, _NDMinT_co, _Trans1DT_co]):
    __slots__ = "axis", "matrix", "ndmin", "trans1d"

    makemat: ClassVar[type[np.matrix[tuple[int, int], np.dtype]]]

    axis: _AxisT_co
    matrix: _MatrixT_co
    ndmin: _NDMinT_co
    trans1d: _Trans1DT_co

    # NOTE: mypy does not understand that these default values are the same as the
    # TypeVar defaults. Since the workaround would require us to write 16 overloads,
    # we ignore the assignment type errors here.
    def __init__(
        self,
        /,
        axis: _AxisT_co = 0,  # type: ignore[assignment]
        matrix: _MatrixT_co = False,  # type: ignore[assignment]
        ndmin: _NDMinT_co = 1,  # type: ignore[assignment]
        trans1d: _Trans1DT_co = -1,  # type: ignore[assignment]
    ) -> None: ...

    def __getitem__(self, key: Incomplete, /) -> Incomplete: ...
    def __len__(self, /) -> L[0]: ...

    # Keep in sync with _core.multiarray.concatenate
    @staticmethod
    @overload
    def concatenate[ScalarT: np.generic](
        arrays: _ArrayLike[ScalarT],
        /,
        axis: SupportsIndex | None = 0,
        out: None = None,
        *,
        dtype: None = None,
        casting: _CastingKind | None = "same_kind",
    ) -> NDArray[ScalarT]: ...
    @staticmethod
    @overload
    def concatenate[ScalarT: np.generic](
        arrays: SupportsLenAndGetItem[ArrayLike],
        /,
        axis: SupportsIndex | None = 0,
        out: None = None,
        *,
        dtype: _DTypeLike[ScalarT],
        casting: _CastingKind | None = "same_kind",
    ) -> NDArray[ScalarT]: ...
    @staticmethod
    @overload
    def concatenate(
        arrays: SupportsLenAndGetItem[ArrayLike],
        /,
        axis: SupportsIndex | None = 0,
        out: None = None,
        *,
        dtype: DTypeLike | None = None,
        casting: _CastingKind | None = "same_kind",
    ) -> NDArray[Incomplete]: ...
    @staticmethod
    @overload
    def concatenate[OutT: np.ndarray](
        arrays: SupportsLenAndGetItem[ArrayLike],
        /,
        axis: SupportsIndex | None = 0,
        *,
        out: OutT,
        dtype: DTypeLike | None = None,
        casting: _CastingKind | None = "same_kind",
    ) -> OutT: ...
    @staticmethod
    @overload
    def concatenate[OutT: np.ndarray](
        arrays: SupportsLenAndGetItem[ArrayLike],
        /,
        axis: SupportsIndex | None,
        out: OutT,
        *,
        dtype: DTypeLike | None = None,
        casting: _CastingKind | None = "same_kind",
    ) -> OutT: ...

@final
class RClass(AxisConcatenator[L[0], L[False], L[1], L[-1]]):
    __slots__ = ()

    def __init__(self, /) -> None: ...

    # keep in sync with `CClass.__getitem__` (-1 ndim)
    @overload  # >=2d
    def __getitem__[ShapeT: tuple[int, int, *tuple[Any, ...]], DTypeT: np.dtype](
        self,
        key: np.ndarray[ShapeT, DTypeT],
        /,
    ) -> np.ndarray[ShapeT, DTypeT]: ...
    @overload  # >=2d
    def __getitem__[ShapeT: tuple[int, int, *tuple[Any, ...]], DTypeT: np.dtype](
        self,
        key: tuple[np.ndarray[ShapeT, DTypeT], *tuple[np.ndarray[ShapeT, DTypeT], ...]],
        /,
    ) -> np.ndarray[ShapeT, DTypeT]: ...
    @overload  # 1d T
    def __getitem__[ScalarT: np.generic](
        self,
        key: _ItemOrTuple[_To1D[ScalarT]],
        /,
    ) -> _Array1D[ScalarT]: ...
    @overload  # 1d +int
    def __getitem__(
        self,
        key: _ItemOrTuple[_To1D[np.integer | np.bool] | _ToPy1D[int, int]],
        /,
    ) -> _Array1D[np.int_]: ...
    @overload  # 1d +f64
    def __getitem__(
        self,
        key: _ItemOrTuple[_To1D[np.float64 | np.float32 | np.float16 | np.integer | np.bool] | _ToPy1D[float, complex]],
        /,
    ) -> _Array1D[np.float64 | Any]: ...
    @overload  # 1d +c128
    def __getitem__(
        self,
        key: _ItemOrTuple[_To1D[np.complex128 | np.float64 | np.integer | np.bool] | _ToPy1D[complex, complex]],
        /,
    ) -> _Array1D[np.complex128 | Any]: ...
    @overload  # ?d T
    def __getitem__[ScalarT: np.generic](
        self,
        key: _ItemOrTuple[_ArrayLike[ScalarT]],
        /,
    ) -> NDArray[ScalarT]: ...
    @overload  # ?d  (fallback)
    def __getitem__(self, key: _ItemOrTuple[ArrayLike | slice], /) -> NDArray[Any]: ...

@final
class CClass(AxisConcatenator[L[-1], L[False], L[2], L[0]]):
    __slots__ = ()

    def __init__(self, /) -> None: ...

    # keep in sync with `RClass.__getitem__` (+1 ndim)
    @overload  # >=3d
    def __getitem__[ShapeT: tuple[int, int, int, *tuple[Any, ...]], DTypeT: np.dtype](
        self,
        key: np.ndarray[ShapeT, DTypeT],
        /,
    ) -> np.ndarray[ShapeT, DTypeT]: ...
    @overload  # >=3d
    def __getitem__[ShapeT: tuple[int, int, int, *tuple[Any, ...]], DTypeT: np.dtype](
        self,
        key: tuple[np.ndarray[ShapeT, DTypeT], *tuple[np.ndarray[ShapeT, DTypeT], ...]],
        /,
    ) -> np.ndarray[ShapeT, DTypeT]: ...
    @overload  # 2d T
    def __getitem__[ScalarT: np.generic](
        self,
        key: _ItemOrTuple[_To2D[ScalarT]],
        /,
    ) -> _Array2D[ScalarT]: ...
    @overload  # 2d +int
    def __getitem__(
        self,
        key: _ItemOrTuple[_To2D[np.integer | np.bool] | _ToPy1D[int, int]],
        /,
    ) -> _Array2D[np.int_]: ...
    @overload  # 2d +f64
    def __getitem__(
        self,
        key: _ItemOrTuple[_To2D[np.float64 | np.float32 | np.float16 | np.integer | np.bool] | _ToPy1D[float, complex]],
        /,
    ) -> _Array2D[np.float64 | Any]: ...
    @overload  # 2d +c128
    def __getitem__(
        self,
        key: _ItemOrTuple[_To2D[np.complex128 | np.float64 | np.integer | np.bool] | _ToPy1D[complex, complex]],
        /,
    ) -> _Array2D[np.complex128 | Any]: ...
    @overload  # ?d T
    def __getitem__[ScalarT: np.generic](
        self,
        key: _ItemOrTuple[_ArrayLike[ScalarT]],
        /,
    ) -> NDArray[ScalarT]: ...
    @overload  # ?d  (fallback)
    def __getitem__(self, key: _ItemOrTuple[ArrayLike | slice], /) -> NDArray[Any]: ...

class IndexExpression(Generic[_BoolT_co]):
    __slots__ = ("maketuple",)

    maketuple: _BoolT_co
    def __init__(self, maketuple: _BoolT_co) -> None: ...
    @overload
    def __getitem__[TupleT: tuple[Any, ...]](self, item: TupleT) -> TupleT: ...
    @overload
    def __getitem__[T](self: IndexExpression[L[True]], item: T) -> tuple[T]: ...
    @overload
    def __getitem__[T](self: IndexExpression[L[False]], item: T) -> T: ...

# only the `int` sequences have special-cased shape-type overloads, because this is the
# most common use case and the others would require too many overloads to be worth it.
@overload  # 0
def ix_() -> tuple[()]: ...
@overload  # 1 +int
def ix_(arg0: Sequence[int], /) -> tuple[_Array1D[np.int_]]: ...
@overload  # 1 ScalarT
def ix_[ScalarT: np.generic](
    arg0: _ToArray1D[ScalarT],
    /,
) -> tuple[_Array1D[ScalarT]]: ...
@overload  # 2 +int
def ix_(
    arg0: Sequence[int],
    arg1: Sequence[int],
    /,
) -> tuple[_Array2D[np.int_], _Array2D[np.int_]]: ...
@overload  # 2 ScalarT
def ix_[ScalarT: np.generic](
    arg0: _ToArray1D[ScalarT],
    arg1: _ToArray1D[ScalarT],
    /,
) -> tuple[_Array2D[ScalarT], _Array2D[ScalarT]]: ...
@overload  # 3 +int
def ix_(
    arg0: Sequence[int],
    arg1: Sequence[int],
    arg2: Sequence[int],
    /,
) -> tuple[_Array3D[np.int_], _Array3D[np.int_], _Array3D[np.int_]]: ...
@overload  # 3 ScalarT
def ix_[ScalarT: np.generic](
    arg0: _ToArray1D[ScalarT],
    arg1: _ToArray1D[ScalarT],
    arg2: _ToArray1D[ScalarT],
    /,
) -> tuple[_Array3D[ScalarT], _Array3D[ScalarT], _Array3D[ScalarT]]: ...
@overload  # N +int
def ix_(
    arg0: Sequence[int],
    arg1: Sequence[int],
    arg2: Sequence[int],
    /,
    *args: Sequence[int],
) -> tuple[NDArray[np.int_], ...]: ...
@overload  # N ScalarT
def ix_[ScalarT: np.generic](
    arg0: _ToArray1D[ScalarT],
    arg1: _ToArray1D[ScalarT],
    arg2: _ToArray1D[ScalarT],
    /,
    *args: _ToArray1D[ScalarT],
) -> tuple[NDArray[ScalarT], ...]: ...
@overload  # N float
def ix_(arg0: list[float], /, *args: Sequence[float]) -> tuple[NDArray[np.float64], ...]: ...
@overload  # N complex
def ix_(arg0: list[complex], /, *args: Sequence[complex]) -> tuple[NDArray[np.complex128], ...]: ...
@overload  # N bytes
def ix_(arg0: Sequence[bytes], /, *args: Sequence[bytes]) -> tuple[NDArray[np.bytes_], ...]: ...
@overload  # N str
def ix_(arg0: Sequence[str], /, *args: Sequence[str]) -> tuple[NDArray[np.str_], ...]: ...
@overload  # fallback
def ix_(
    arg0: Sequence[_ScalarLike_co] | _Array1D[Any],
    /,
    *args: Sequence[_ScalarLike_co] | _Array1D[Any],
) -> tuple[NDArray[Any], ...]: ...

#
def fill_diagonal(a: NDArray[Any], val: object, wrap: bool = False) -> None: ...

#
@overload
def diag_indices(n: _IntLike_co, ndim: L[0]) -> tuple[()]: ...
@overload
def diag_indices(n: _IntLike_co, ndim: L[1]) -> tuple[_Int1D]: ...
@overload
def diag_indices(n: _IntLike_co, ndim: L[2] = 2) -> tuple[_Int1D, _Int1D]: ...
@overload
def diag_indices(n: _IntLike_co, ndim: L[3]) -> tuple[_Int1D, _Int1D, _Int1D]: ...
@overload
def diag_indices(n: _IntLike_co, ndim: int) -> tuple[_Int1D, ...]: ...

#
@overload  # ?d  (workaround)
def diag_indices_from(arr: np.ndarray[_JustAnyShape]) -> tuple[_Int1D, _Int1D, *tuple[_Int1D, ...]]: ...
@overload  # 2d
def diag_indices_from(arr: np.ndarray[tuple[int, int]]) -> tuple[_Int1D, _Int1D]: ...
@overload  # 3d
def diag_indices_from(arr: np.ndarray[tuple[int, int, int]]) -> tuple[_Int1D, _Int1D, _Int1D]: ...
@overload  # 4d
def diag_indices_from(arr: np.ndarray[tuple[int, int, int, int]]) -> tuple[_Int1D, _Int1D, _Int1D, _Int1D]: ...
@overload  # >=2d (fallback)
def diag_indices_from(arr: np.ndarray[tuple[int, int, *tuple[int, ...]]]) -> tuple[_Int1D, _Int1D, *tuple[_Int1D, ...]]: ...

#
mgrid: Final[MGridClass] = ...
ogrid: Final[OGridClass] = ...

r_: Final[RClass] = ...
c_: Final[CClass] = ...

index_exp: Final[IndexExpression[L[True]]] = ...
s_: Final[IndexExpression[L[False]]] = ...
