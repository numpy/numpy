from _typeshed import SupportsBool
from collections.abc import Sequence
from typing import (
    Any,
    Literal,
    Never,
    Protocol,
    SupportsIndex,
    TypedDict,
    Unpack,
    overload,
    type_check_only,
)

import numpy as np
from numpy import (
    _CastingKind,
    _ModeKind,
    _OrderACF,
    _OrderKACF,
    _PartitionKind,
    _SortKind,
    _SortSide,
    intp,
)
from numpy._globals import _NoValueType
from numpy._typing import (
    ArrayLike,
    DTypeLike,
    NDArray,
    _AnyShape,
    _ArrayLike,
    _ArrayLikeBool_co,
    _ArrayLikeComplex_co,
    _ArrayLikeFloat_co,
    _ArrayLikeInt,
    _ArrayLikeInt_co,
    _ArrayLikeObject_co,
    _BoolLike_co,
    _DTypeLike,
    _FloatLike_co,
    _IntLike_co,
    _NestedSequence,
    _NumberLike_co,
    _ScalarLike_co,
    _Shape,
    _ShapeLike,
    _SupportsArray,
)
from numpy._typing._array_like import _DualArrayLike

__all__ = [
    "all",
    "amax",
    "amin",
    "any",
    "argmax",
    "argmin",
    "argpartition",
    "argsort",
    "around",
    "choose",
    "clip",
    "compress",
    "cumprod",
    "cumsum",
    "cumulative_prod",
    "cumulative_sum",
    "diagonal",
    "mean",
    "max",
    "min",
    "matrix_transpose",
    "ndim",
    "nonzero",
    "partition",
    "prod",
    "ptp",
    "put",
    "ravel",
    "repeat",
    "reshape",
    "resize",
    "round",
    "searchsorted",
    "shape",
    "size",
    "sort",
    "squeeze",
    "std",
    "sum",
    "swapaxes",
    "take",
    "top_k",
    "trace",
    "transpose",
    "var",
]

@type_check_only
class _SupportsShape[ShapeT_co: _Shape](Protocol):
    # NOTE: it matters that `self` is positional only
    @property
    def shape(self, /) -> ShapeT_co: ...

@type_check_only
class _UFuncKwargs(TypedDict, total=False):
    where: _ArrayLikeBool_co | None
    order: _OrderKACF
    subok: bool
    signature: str | tuple[str | None, ...]
    casting: _CastingKind

# a "sequence" that isn't a string, bytes, bytearray, or memoryview
type _PyArray[_T] = list[_T] | tuple[_T, ...]
# `int` also covers `bool`
type _PyScalar = complex | bytes | str

type _0D = tuple[()]
type _1D = tuple[int]
type _2D = tuple[int, int]
type _3D = tuple[int, int, int]
type _4D = tuple[int, int, int, int]

type _Array1D[ScalarT: np.generic] = np.ndarray[_1D, np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[_2D, np.dtype[ScalarT]]
type _Array3D[ScalarT: np.generic] = np.ndarray[_3D, np.dtype[ScalarT]]
type _Array4D[ScalarT: np.generic] = np.ndarray[_4D, np.dtype[ScalarT]]
# workaround for mypy's and pyright's typing spec non-compliance regarding overloads
type _ArrayJustND[ScalarT: np.generic] = np.ndarray[tuple[Never, Never, Never, Never], np.dtype[ScalarT]]

type _ToArray1D2[ScalarT: np.generic, T] = _Array1D[ScalarT] | Sequence[T]
type _ToArray2D2[ScalarT: np.generic, T] = _Array2D[ScalarT] | Sequence[Sequence[T]]
type _ToArray3D2[ScalarT: np.generic, T] = _Array3D[ScalarT] | Sequence[Sequence[Sequence[T]]]
type _ToArray4D2[ScalarT: np.generic, T] = _Array4D[ScalarT] | Sequence[Sequence[Sequence[Sequence[T]]]]

type _ToArray1D[ScalarT: np.generic] = _ToArray1D2[ScalarT, ScalarT]
type _ToArray2D[ScalarT: np.generic] = _ToArray2D2[ScalarT, ScalarT]
type _ToArray3D[ScalarT: np.generic] = _ToArray3D2[ScalarT, ScalarT]
type _ToArray4D[ScalarT: np.generic] = _ToArray4D2[ScalarT, ScalarT]

type _ToNumeric1D = _ToArray1D2[np.number | np.bool | np.object_ | np.timedelta64, complex]
type _ToNumeric2D = _ToArray2D2[np.number | np.bool | np.object_ | np.timedelta64, complex]
type _ToNumeric3D = _ToArray3D2[np.number | np.bool | np.object_ | np.timedelta64, complex]
type _ToNumeric4D = _ToArray4D2[np.number | np.bool | np.object_ | np.timedelta64, complex]

type _ArrayLikeMultiplicative_co = _DualArrayLike[np.dtype[np.number | np.bool | np.object_], complex]
type _ArrayLikeNumeric_co = _DualArrayLike[np.dtype[np.number | np.bool | np.object_ | np.timedelta64], complex]

@type_check_only
class _CanLE(Protocol):
    def __le__(self, other: Any, /) -> SupportsBool: ...

@type_check_only
class _CanGE(Protocol):
    def __ge__(self, other: Any, /) -> SupportsBool: ...

type _Orderable = _CanLE | _CanGE

@type_check_only
class _CanArray[ArrayT: np.ndarray](Protocol):
    def __array__(self, /) -> ArrayT: ...

type _ArrayLike1D = _CanArray[_Array1D[Any]] | Sequence[_ScalarLike_co]
type _ArrayLikeInt1D = _CanArray[_Array1D[np.integer]] | Sequence[int | np.integer]

###

# TODO: Fix overlapping overloads: https://github.com/numpy/numpy/issues/27032
@overload
def take[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    indices: _IntLike_co,
    axis: None = None,
    out: None = None,
    mode: _ModeKind = "raise",
) -> ScalarT: ...
@overload
def take(
    a: ArrayLike,
    indices: _IntLike_co,
    axis: SupportsIndex | None = None,
    out: None = None,
    mode: _ModeKind = "raise",
) -> Any: ...
@overload
def take[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    indices: _ArrayLikeInt_co,
    axis: SupportsIndex | None = None,
    out: None = None,
    mode: _ModeKind = "raise",
) -> NDArray[ScalarT]: ...
@overload
def take(
    a: ArrayLike,
    indices: _ArrayLikeInt_co,
    axis: SupportsIndex | None = None,
    out: None = None,
    mode: _ModeKind = "raise",
) -> NDArray[Any]: ...
@overload
def take[ArrayT: np.ndarray](
    a: ArrayLike,
    indices: _ArrayLikeInt_co,
    axis: SupportsIndex | None,
    out: ArrayT,
    mode: _ModeKind = "raise",
) -> ArrayT: ...
@overload
def take[ArrayT: np.ndarray](
    a: ArrayLike,
    indices: _ArrayLikeInt_co,
    axis: SupportsIndex | None = None,
    *,
    out: ArrayT,
    mode: _ModeKind = "raise",
) -> ArrayT: ...

def top_k(
    a: ArrayLike,
    k: int,
    /,
    *,
    axis: int = -1,
    mode: Literal["largest", "smallest"] = "largest",
    sorted: bool = True,
) -> tuple[NDArray[Any], NDArray[intp]]: ...

# keep in sync with `ma.core.reshape`
@overload  # shape: index
def reshape[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    /,
    shape: SupportsIndex,
    order: _OrderACF = "C",
    *,
    copy: bool | None = None,
) -> _Array1D[ScalarT]: ...
@overload  # shape: ~ShapeT
def reshape[ScalarT: np.generic, ShapeT: _Shape](
    a: _ArrayLike[ScalarT],
    /,
    shape: ShapeT,
    order: _OrderACF = "C",
    *,
    copy: bool | None = None,
) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
@overload  # shape: Sequence[index]
def reshape[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    /,
    shape: Sequence[SupportsIndex],
    order: _OrderACF = "C",
    *,
    copy: bool | None = None,
) -> NDArray[ScalarT]: ...
@overload  # shape: index
def reshape(
    a: ArrayLike,
    /,
    shape: SupportsIndex,
    order: _OrderACF = "C",
    *,
    copy: bool | None = None,
) -> np.ndarray[_1D]: ...
@overload  # shape: ~ShapeT
def reshape[ShapeT: _Shape](
    a: ArrayLike,
    /,
    shape: ShapeT,
    order: _OrderACF = "C",
    *,
    copy: bool | None = None,
) -> np.ndarray[ShapeT]: ...
@overload  # shape: Sequence[index]
def reshape(
    a: ArrayLike,
    /,
    shape: Sequence[SupportsIndex],
    order: _OrderACF = "C",
    *,
    copy: bool | None = None,
) -> NDArray[Any]: ...

# keep in sync with `ma.core.choose`
@overload
def choose(
    a: _IntLike_co,
    choices: ArrayLike,
    out: None = None,
    mode: _ModeKind = "raise",
) -> Any: ...
@overload
def choose[ScalarT: np.generic](
    a: _ArrayLikeInt_co,
    choices: _ArrayLike[ScalarT],
    out: None = None,
    mode: _ModeKind = "raise",
) -> NDArray[ScalarT]: ...
@overload
def choose(
    a: _ArrayLikeInt_co,
    choices: ArrayLike,
    out: None = None,
    mode: _ModeKind = "raise",
) -> NDArray[Any]: ...
@overload
def choose[ArrayT: np.ndarray](
    a: _ArrayLikeInt_co,
    choices: ArrayLike,
    out: ArrayT,
    mode: _ModeKind = "raise",
) -> ArrayT: ...

# keep in sync with `ma.core.repeat`
@overload
def repeat[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    repeats: _ArrayLikeInt_co,
    axis: None = None,
) -> _Array1D[ScalarT]: ...
@overload
def repeat[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    repeats: _ArrayLikeInt_co,
    axis: SupportsIndex,
) -> NDArray[ScalarT]: ...
@overload
def repeat(
    a: ArrayLike,
    repeats: _ArrayLikeInt_co,
    axis: None = None,
) -> _Array1D[Any]: ...
@overload
def repeat(
    a: ArrayLike,
    repeats: _ArrayLikeInt_co,
    axis: SupportsIndex,
) -> NDArray[Any]: ...

# keep in sync with `ma.core.put`
def put(
    a: NDArray[Any],
    ind: _ArrayLikeInt_co,
    v: ArrayLike,
    mode: _ModeKind = "raise",
) -> None: ...

# keep in sync with `ndarray.swapaxes` and `ma.core.swapaxes`
@overload
def swapaxes[ArrayT: np.ndarray](a: ArrayT, axis1: SupportsIndex, axis2: SupportsIndex) -> ArrayT: ...
@overload
def swapaxes[ScalarT: np.generic](a: _ArrayLike[ScalarT], axis1: SupportsIndex, axis2: SupportsIndex) -> NDArray[ScalarT]: ...
@overload
def swapaxes(a: ArrayLike, axis1: SupportsIndex, axis2: SupportsIndex) -> NDArray[Any]: ...

# keep in sync with `matrix_transpose`  (but start at 1d lists)
@overload  # known array
def transpose[ArrayT: np.ndarray](a: ArrayT, axes: _ShapeLike | None = None) -> ArrayT: ...
@overload  # 1d bool
def transpose(a: list[bool], axes: _ShapeLike | None = None) -> _Array1D[np.bool]: ...
@overload  # 1d int
def transpose(a: list[int], axes: _ShapeLike | None = None) -> _Array1D[np.int_]: ...
@overload  # 1d float
def transpose(a: list[float], axes: _ShapeLike | None = None) -> _Array1D[np.float64]: ...
@overload  # 1d complex
def transpose(a: list[complex], axes: _ShapeLike | None = None) -> _Array1D[np.complex128]: ...
@overload  # 2d bool
def transpose(a: Sequence[list[bool]], axes: _ShapeLike | None = None) -> _Array2D[np.bool]: ...
@overload  # 2d int
def transpose(a: Sequence[list[int]], axes: _ShapeLike | None = None) -> _Array2D[np.int_]: ...
@overload  # 2d float
def transpose(a: Sequence[list[float]], axes: _ShapeLike | None = None) -> _Array2D[np.float64]: ...
@overload  # 2d complex
def transpose(a: Sequence[list[complex]], axes: _ShapeLike | None = None) -> _Array2D[np.complex128]: ...
@overload  # known dtype
def transpose[ScalarT: np.generic](a: _ArrayLike[ScalarT], axes: _ShapeLike | None = None) -> NDArray[ScalarT]: ...
@overload  # fallback
def transpose(a: ArrayLike, axes: _ShapeLike | None = None) -> NDArray[Any]: ...

# keep in sync with `transpose`  (but start at 2d lists)
@overload  # known array
def matrix_transpose[ArrayT: np.ndarray](x: ArrayT, /) -> ArrayT: ...
@overload  # 2d bool
def matrix_transpose(x: Sequence[list[bool]], /) -> _Array2D[np.bool]: ...
@overload  # 2d int
def matrix_transpose(x: Sequence[list[int]], /) -> _Array2D[np.int_]: ...
@overload  # 2d float
def matrix_transpose(x: Sequence[list[float]], /) -> _Array2D[np.float64]: ...
@overload  # 2d complex
def matrix_transpose(x: Sequence[list[complex]], /) -> _Array2D[np.complex128]: ...
@overload  # 3d bool
def matrix_transpose(x: Sequence[Sequence[list[bool]]], /) -> _Array3D[np.bool]: ...
@overload  # 3d int
def matrix_transpose(x: Sequence[Sequence[list[int]]], /) -> _Array3D[np.int_]: ...
@overload  # 3d float
def matrix_transpose(x: Sequence[Sequence[list[float]]], /) -> _Array3D[np.float64]: ...
@overload  # 3d complex
def matrix_transpose(x: Sequence[Sequence[list[complex]]], /) -> _Array3D[np.complex128]: ...
@overload  # known dtype
def matrix_transpose[ScalarT: np.generic](x: _ArrayLike[ScalarT], /) -> NDArray[ScalarT]: ...
@overload  # fallback
def matrix_transpose(x: ArrayLike, /) -> NDArray[Any]: ...

#
@overload  # Nd
def partition[ArrayT: np.ndarray](
    a: ArrayT,
    kth: _ArrayLikeInt,
    axis: SupportsIndex = -1,
    kind: _PartitionKind | _NoValueType = ...,
    order: str | Sequence[str] | None = None,
    descending: bool | _NoValueType = ...,
) -> ArrayT: ...
@overload  # ?d
def partition[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    kth: _ArrayLikeInt,
    axis: SupportsIndex = -1,
    kind: _PartitionKind | _NoValueType = ...,
    order: str | Sequence[str] | None = None,
    descending: bool | _NoValueType = ...,
) -> NDArray[ScalarT]: ...
@overload  # axis: None
def partition[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    kth: _ArrayLikeInt,
    axis: None,
    kind: _PartitionKind | _NoValueType = ...,
    order: str | Sequence[str] | None = None,
    descending: bool | _NoValueType = ...,
) -> _Array1D[ScalarT]: ...
@overload  # fallback
def partition(
    a: ArrayLike,
    kth: _ArrayLikeInt,
    axis: SupportsIndex = -1,
    kind: _PartitionKind | _NoValueType = ...,
    order: str | Sequence[str] | None = None,
    descending: bool | _NoValueType = ...,
) -> NDArray[Any]: ...
@overload  # fallback, axis: None
def partition(
    a: ArrayLike,
    kth: _ArrayLikeInt,
    axis: None,
    kind: _PartitionKind | _NoValueType = ...,
    order: str | Sequence[str] | None = None,
    descending: bool | _NoValueType = ...,
) -> _Array1D[Any]: ...

# keep roughly in sync with `ndarray.argpartition`
@overload  # axis: None
def argpartition(
    a: ArrayLike,
    kth: _ArrayLikeInt,
    axis: None,
    kind: _PartitionKind | _NoValueType = ...,
    order: None = None,
    descending: bool | _NoValueType = ...,
) -> np.ndarray[tuple[int], np.dtype[np.intp]]: ...
@overload  # known shape, axis: index (default)
def argpartition[ShapeT: _Shape](
    a: np.ndarray[ShapeT],
    kth: _ArrayLikeInt,
    axis: SupportsIndex = -1,
    kind: _PartitionKind | _NoValueType = ...,
    order: None = None,
    descending: bool | _NoValueType = ...,
) -> np.ndarray[ShapeT, np.dtype[np.intp]]: ...
@overload  # 1d array-like, axis: index (default)
def argpartition(
    a: Sequence[np.generic | complex],
    kth: _ArrayLikeInt,
    axis: SupportsIndex = -1,
    kind: _PartitionKind | _NoValueType = ...,
    order: None = None,
    descending: bool | _NoValueType = ...,
) -> np.ndarray[tuple[int], np.dtype[np.intp]]: ...
@overload  # 2d array-like, axis: index (default)
def argpartition(
    a: Sequence[Sequence[np.generic | complex]],
    kth: _ArrayLikeInt,
    axis: SupportsIndex = -1,
    kind: _PartitionKind | _NoValueType = ...,
    order: None = None,
    descending: bool | _NoValueType = ...,
) -> np.ndarray[tuple[int, int], np.dtype[np.intp]]: ...
@overload  # ?d array-like, axis: index (default)
def argpartition(
    a: ArrayLike,
    kth: _ArrayLikeInt,
    axis: SupportsIndex = -1,
    kind: _PartitionKind | _NoValueType = ...,
    order: None = None,
    descending: bool | _NoValueType = ...,
) -> NDArray[np.intp]: ...
@overload  # void, axis: None
def argpartition(
    a: _SupportsArray[np.dtype[np.void]],
    kth: _ArrayLikeInt,
    axis: None,
    kind: _PartitionKind | _NoValueType = ...,
    order: str | Sequence[str] | None = None,
    descending: bool | _NoValueType = ...,
) -> np.ndarray[tuple[int], np.dtype[intp]]: ...
@overload  # void, axis: index (default)
def argpartition[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.void]],
    kth: _ArrayLikeInt,
    axis: SupportsIndex = -1,
    kind: _PartitionKind | _NoValueType = ...,
    order: str | Sequence[str] | None = None,
    descending: bool | _NoValueType = ...,
) -> np.ndarray[ShapeT, np.dtype[np.intp]]: ...

#
@overload  # known array, axis=<given> (default)
def sort[ArrayT: np.ndarray](
    a: ArrayT,
    axis: SupportsIndex = -1,
    kind: _SortKind | None = None,
    order: str | Sequence[str] | None = None,
    *,
    stable: bool | None = None,
    descending: bool | _NoValueType = ...,
) -> ArrayT: ...
@overload  # 1d, bool
def sort[ScalarT: np.generic](
    a: list[bool],
    axis: SupportsIndex | None = -1,
    kind: _SortKind | None = None,
    order: None = None,
    *,
    stable: bool | None = None,
    descending: bool | _NoValueType = ...,
) -> _Array1D[np.bool]: ...
@overload  # 1d, int
def sort[ScalarT: np.generic](
    a: list[int],
    axis: SupportsIndex | None = -1,
    kind: _SortKind | None = None,
    order: None = None,
    *,
    stable: bool | None = None,
    descending: bool | _NoValueType = ...,
) -> _Array1D[np.int_]: ...
@overload  # 1d, float
def sort[ScalarT: np.generic](
    a: list[float],
    axis: SupportsIndex | None = -1,
    kind: _SortKind | None = None,
    order: None = None,
    *,
    stable: bool | None = None,
    descending: bool | _NoValueType = ...,
) -> _Array1D[np.float64]: ...
@overload  # 1d, complex
def sort[ScalarT: np.generic](
    a: list[complex],
    axis: SupportsIndex | None = -1,
    kind: _SortKind | None = None,
    order: None = None,
    *,
    stable: bool | None = None,
    descending: bool | _NoValueType = ...,
) -> _Array1D[np.complex128]: ...
@overload  # 2d, bool, axis=<given> (default)
def sort[ScalarT: np.generic](
    a: Sequence[list[bool]],
    axis: SupportsIndex = -1,
    kind: _SortKind | None = None,
    order: None = None,
    *,
    stable: bool | None = None,
    descending: bool | _NoValueType = ...,
) -> _Array2D[np.bool]: ...
@overload  # 2d, int, axis=<given> (default)
def sort[ScalarT: np.generic](
    a: Sequence[list[int]],
    axis: SupportsIndex = -1,
    kind: _SortKind | None = None,
    order: None = None,
    *,
    stable: bool | None = None,
    descending: bool | _NoValueType = ...,
) -> _Array2D[np.int_]: ...
@overload  # 2d, float, axis=<given> (default)
def sort[ScalarT: np.generic](
    a: Sequence[list[float]],
    axis: SupportsIndex = -1,
    kind: _SortKind | None = None,
    order: None = None,
    *,
    stable: bool | None = None,
    descending: bool | _NoValueType = ...,
) -> _Array2D[np.float64]: ...
@overload  # 2d, complex, axis=<given> (default)
def sort[ScalarT: np.generic](
    a: Sequence[list[complex]],
    axis: SupportsIndex = -1,
    kind: _SortKind | None = None,
    order: None = None,
    *,
    stable: bool | None = None,
    descending: bool | _NoValueType = ...,
) -> _Array2D[np.complex128]: ...
@overload  # ?d, known dtype, axis=<given> (default)
def sort[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    axis: SupportsIndex = -1,
    kind: _SortKind | None = None,
    order: str | Sequence[str] | None = None,
    *,
    stable: bool | None = None,
    descending: bool | _NoValueType = ...,
) -> NDArray[ScalarT]: ...
@overload  # ?d, known dtype, axis=None
def sort[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    axis: None,
    kind: _SortKind | None = None,
    order: str | Sequence[str] | None = None,
    *,
    stable: bool | None = None,
    descending: bool | _NoValueType = ...,
) -> _Array1D[ScalarT]: ...
@overload  # ?d, unknown dtype, axis=<given> (default)
def sort(
    a: ArrayLike,
    axis: SupportsIndex = -1,
    kind: _SortKind | None = None,
    order: str | Sequence[str] | None = None,
    *,
    stable: bool | None = None,
    descending: bool | _NoValueType = ...,
) -> NDArray[Any]: ...
@overload  # ?d, unknown dtype, axis=None
def sort(
    a: ArrayLike,
    axis: None,
    kind: _SortKind | None = None,
    order: str | Sequence[str] | None = None,
    *,
    stable: bool | None = None,
    descending: bool | _NoValueType = ...,
) -> _Array1D[Any]: ...

#
@overload  # known shape, axis=<given> (default)
def argsort[ShapeT: _Shape](
    a: np.ndarray[ShapeT],
    axis: SupportsIndex = -1,
    kind: _SortKind | None = None,
    order: str | Sequence[str] | None = None,
    *,
    stable: bool | None = None,
    descending: bool | _NoValueType = ...,
) -> np.ndarray[ShapeT, np.dtype[np.intp]]: ...
@overload  # 1d, axis=<given> (default)
def argsort(
    a: Sequence[_ScalarLike_co],
    axis: SupportsIndex = -1,
    kind: _SortKind | None = None,
    order: str | Sequence[str] | None = None,
    *,
    stable: bool | None = None,
    descending: bool | _NoValueType = ...,
) -> _Array1D[np.intp]: ...
@overload  # 2d, axis=<given> (default)
def argsort(
    a: Sequence[Sequence[_ScalarLike_co]],
    axis: SupportsIndex = -1,
    kind: _SortKind | None = None,
    order: str | Sequence[str] | None = None,
    *,
    stable: bool | None = None,
    descending: bool | _NoValueType = ...,
) -> _Array2D[np.intp]: ...
@overload  # 3d, axis=<given> (default)
def argsort(
    a: Sequence[Sequence[Sequence[_ScalarLike_co]]],
    axis: SupportsIndex = -1,
    kind: _SortKind | None = None,
    order: str | Sequence[str] | None = None,
    *,
    stable: bool | None = None,
    descending: bool | _NoValueType = ...,
) -> _Array3D[np.intp]: ...
@overload  # ?d, axis=<given> (default)
def argsort(
    a: ArrayLike,
    axis: SupportsIndex = -1,
    kind: _SortKind | None = None,
    order: str | Sequence[str] | None = None,
    *,
    stable: bool | None = None,
    descending: bool | _NoValueType = ...,
) -> NDArray[np.intp]: ...
@overload  # axis=None
def argsort(
    a: ArrayLike,
    axis: None,
    kind: _SortKind | None = None,
    order: str | Sequence[str] | None = None,
    *,
    stable: bool | None = None,
    descending: bool | _NoValueType = ...,
) -> _Array1D[np.intp]: ...

# keep in sync with `argmin` below
@overload  # ?d
def argmax(
    a: ArrayLike | _NestedSequence[_Orderable],
    axis: None = None,
    out: None = None,
    *,
    keepdims: Literal[False] | _NoValueType = ...,
) -> np.intp: ...
@overload  # ?d, axis: <given>
def argmax(
    a: ArrayLike | _NestedSequence[_Orderable],
    axis: SupportsIndex,
    out: None = None,
    *,
    keepdims: Literal[False] | _NoValueType = ...,
) -> NDArray[np.intp]: ...
@overload  # Nd, keepdims=True
def argmax[ShapeT: _Shape](
    a: np.ndarray[ShapeT],
    axis: SupportsIndex | None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
) -> np.ndarray[ShapeT, np.dtype[np.intp]]: ...
@overload  # ?d, keepdims=True
def argmax(
    a: ArrayLike | _NestedSequence[_Orderable],
    axis: SupportsIndex | None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
) -> NDArray[np.intp]: ...
@overload  # ?d, out: ArrayT
def argmax[ArrayT: NDArray[np.intp]](
    a: ArrayLike | _NestedSequence[_Orderable],
    axis: SupportsIndex | None = None,
    *,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
) -> ArrayT: ...

# keep in sync with `argmax` above
@overload  # ?d
def argmin(
    a: ArrayLike | _NestedSequence[_Orderable],
    axis: None = None,
    out: None = None,
    *,
    keepdims: Literal[False] | _NoValueType = ...,
) -> np.intp: ...
@overload  # ?d, axis: <given>
def argmin(
    a: ArrayLike | _NestedSequence[_Orderable],
    axis: SupportsIndex,
    out: None = None,
    *,
    keepdims: Literal[False] | _NoValueType = ...,
) -> NDArray[np.intp]: ...
@overload  # Nd, keepdims=True
def argmin[ShapeT: _Shape](
    a: np.ndarray[ShapeT],
    axis: SupportsIndex | None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
) -> np.ndarray[ShapeT, np.dtype[np.intp]]: ...
@overload  # ?d, keepdims=True
def argmin(
    a: ArrayLike | _NestedSequence[_Orderable],
    axis: SupportsIndex | None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
) -> NDArray[np.intp]: ...
@overload  # ?d, out: ArrayT
def argmin[ArrayT: NDArray[np.intp]](
    a: ArrayLike | _NestedSequence[_Orderable],
    axis: SupportsIndex | None = None,
    *,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
) -> ArrayT: ...

#
@overload  # Nd
def searchsorted[ShapeT: _Shape](
    a: _ArrayLike1D,
    v: np.ndarray[ShapeT, np.dtype[Any]],
    side: _SortSide = "left",
    sorter: _ArrayLikeInt1D | None = None,
) -> np.ndarray[ShapeT, np.dtype[np.intp]]: ...
@overload  # 0d
def searchsorted(
    a: _ArrayLike1D,
    v: complex | str | np.generic,
    side: _SortSide = "left",
    sorter: _ArrayLikeInt1D | None = None,
) -> np.intp: ...
@overload  # 1d  (`str <; Sequence[str]`, hence the `list[str]`)
def searchsorted(
    a: _ArrayLike1D,
    v: Sequence[complex | np.generic] | list[str],
    side: _SortSide = "left",
    sorter: _ArrayLikeInt1D | None = None,
) -> _Array1D[np.intp]: ...
@overload  # 2d
def searchsorted(
    a: _ArrayLike1D,
    v: Sequence[Sequence[complex | np.generic]] | Sequence[list[str]],
    side: _SortSide = "left",
    sorter: _ArrayLikeInt1D | None = None,
) -> _Array2D[np.intp]: ...
@overload  # 3d
def searchsorted(
    a: _ArrayLike1D,
    v: Sequence[Sequence[Sequence[complex | np.generic]]] | Sequence[Sequence[list[str]]],
    side: _SortSide = "left",
    sorter: _ArrayLikeInt1D | None = None,
) -> _Array3D[np.intp]: ...
@overload  # fallback  (overlaps with 1st overload)
def searchsorted(
    a: _ArrayLike1D,
    v: ArrayLike,
    side: _SortSide = "left",
    sorter: _ArrayLikeInt1D | None = None,
) -> NDArray[np.intp] | Any: ...

# keep in sync with `ma.core.resize`
@overload
def resize[ScalarT: np.generic](a: _ArrayLike[ScalarT], new_shape: SupportsIndex | tuple[SupportsIndex]) -> _Array1D[ScalarT]: ...
@overload
def resize[ScalarT: np.generic, AnyShapeT: (_0D, _1D, _2D, _3D, _4D)](
    a: _ArrayLike[ScalarT],
    new_shape: AnyShapeT,
) -> np.ndarray[AnyShapeT, np.dtype[ScalarT]]: ...
@overload
def resize[ScalarT: np.generic](a: _ArrayLike[ScalarT], new_shape: _ShapeLike) -> NDArray[ScalarT]: ...
@overload
def resize(a: ArrayLike, new_shape: SupportsIndex | tuple[SupportsIndex]) -> np.ndarray[_1D]: ...
@overload
def resize[AnyShapeT: (_0D, _1D, _2D, _3D, _4D)](a: ArrayLike, new_shape: AnyShapeT) -> np.ndarray[AnyShapeT]: ...
@overload
def resize(a: ArrayLike, new_shape: _ShapeLike) -> NDArray[Any]: ...

# TODO: Fix overlapping overloads: https://github.com/numpy/numpy/issues/27032
@overload
def squeeze[ScalarT: np.generic](a: ScalarT, axis: _ShapeLike | None = None) -> ScalarT: ...
@overload
def squeeze[ScalarT: np.generic](a: _ArrayLike[ScalarT], axis: _ShapeLike | None = None) -> NDArray[ScalarT]: ...
@overload
def squeeze(a: ArrayLike, axis: _ShapeLike | None = None) -> NDArray[Any]: ...

# keep in sync with `ma.core.diagonal`
@overload  # ?d  (workaround)
def diagonal[ScalarT: np.generic](
    a: _ArrayJustND[ScalarT],
    offset: SupportsIndex = 0,
    axis1: SupportsIndex = 0,
    axis2: SupportsIndex = 1,
) -> NDArray[ScalarT]: ...
@overload  # 2d
def diagonal[ScalarT: np.generic](
    a: _ToArray2D[ScalarT],
    offset: SupportsIndex = 0,
    axis1: SupportsIndex = 0,
    axis2: SupportsIndex = 1,
) -> _Array1D[ScalarT]: ...
@overload  # 3d
def diagonal[ScalarT: np.generic](
    a: _ToArray3D[ScalarT],
    offset: SupportsIndex = 0,
    axis1: SupportsIndex = 0,
    axis2: SupportsIndex = 1,
) -> _Array2D[ScalarT]: ...
@overload  # Nd
def diagonal[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    offset: SupportsIndex = 0,
    axis1: SupportsIndex = 0,
    axis2: SupportsIndex = 1,
) -> NDArray[ScalarT]: ...
@overload  # fallback
def diagonal(
    a: ArrayLike,
    offset: SupportsIndex = 0,
    axis1: SupportsIndex = 0,
    axis2: SupportsIndex = 1,
) -> NDArray[Any]: ...

# keep in sync with `ma.core.trace`
@overload
def trace(
    a: ArrayLike,  # >= 2D array
    offset: SupportsIndex = 0,
    axis1: SupportsIndex = 0,
    axis2: SupportsIndex = 1,
    dtype: DTypeLike | None = None,
    out: None = None,
) -> Any: ...
@overload
def trace[ArrayT: np.ndarray](
    a: ArrayLike,  # >= 2D array
    offset: SupportsIndex,
    axis1: SupportsIndex,
    axis2: SupportsIndex,
    dtype: DTypeLike | None,
    out: ArrayT,
) -> ArrayT: ...
@overload
def trace[ArrayT: np.ndarray](
    a: ArrayLike,  # >= 2D array
    offset: SupportsIndex = 0,
    axis1: SupportsIndex = 0,
    axis2: SupportsIndex = 1,
    dtype: DTypeLike | None = None,
    *,
    out: ArrayT,
) -> ArrayT: ...

@overload
def ravel[ScalarT: np.generic](a: _ArrayLike[ScalarT], order: _OrderKACF = "C") -> _Array1D[ScalarT]: ...
@overload
def ravel(a: bytes | _NestedSequence[bytes], order: _OrderKACF = "C") -> _Array1D[np.bytes_]: ...
@overload
def ravel(a: str | _NestedSequence[str], order: _OrderKACF = "C") -> _Array1D[np.str_]: ...
@overload
def ravel(a: bool | _NestedSequence[bool], order: _OrderKACF = "C") -> _Array1D[np.bool]: ...
@overload
def ravel(a: int | _NestedSequence[int], order: _OrderKACF = "C") -> _Array1D[np.int_ | Any]: ...
@overload
def ravel(a: float | _NestedSequence[float], order: _OrderKACF = "C") -> _Array1D[np.float64 | Any]: ...
@overload
def ravel(a: complex | _NestedSequence[complex], order: _OrderKACF = "C") -> _Array1D[np.complex128 | Any]: ...
@overload
def ravel(a: ArrayLike, order: _OrderKACF = "C") -> np.ndarray[_1D]: ...

# keep in sync with the 1-arg overloads of `_core.multiarray.where`
@overload  # ?d  (workaround)
def nonzero(a: _ArrayJustND[Any]) -> tuple[_Array1D[np.intp], ...]: ...
@overload  # 1d
def nonzero(a: _ToArray1D[Any]) -> tuple[_Array1D[np.intp]]: ...
@overload  # 2d
def nonzero(a: _ToArray2D[Any]) -> tuple[_Array1D[np.intp], _Array1D[np.intp]]: ...
@overload  # 3d
def nonzero(a: _ToArray3D[Any]) -> tuple[_Array1D[np.intp], _Array1D[np.intp], _Array1D[np.intp]]: ...
@overload  # Nd  (fallback)
def nonzero(a: _ArrayLike[Any]) ->  tuple[_Array1D[np.intp], ...]: ...

# `collections.abc.Sequence` can't be used here because `bytes` and `str` are
# subtypes of it, which would make the return types incompatible.
@overload  # this prevents `Any` from being returned with Pyright
def shape(a: _SupportsShape[Never]) -> _AnyShape: ...
@overload
def shape[ShapeT: _Shape](a: _SupportsShape[ShapeT]) -> ShapeT: ...
@overload
def shape(a: _PyScalar) -> tuple[()]: ...
@overload  # an unbound type variable is used because `list` is invariant
def shape[ScalarT: _PyScalar](a: _PyArray[ScalarT]) -> _1D: ...
@overload
def shape[ScalarT: _PyScalar](a: Sequence[_PyArray[ScalarT]]) -> _2D: ...
@overload
def shape[ScalarT: _PyScalar](a: Sequence[Sequence[_PyArray[ScalarT]]]) -> _3D: ...
@overload  # this will be skipped by typecheckers that don't support PEP 688
def shape(a: memoryview | bytearray) -> _1D: ...
@overload
def shape(a: ArrayLike) -> _AnyShape: ...

#
@overload
def compress[ScalarT: np.generic](
    condition: _ArrayLikeBool_co,  # 1D bool array
    a: _ArrayLike[ScalarT],
    axis: SupportsIndex | None = None,
    out: None = None,
) -> NDArray[ScalarT]: ...
@overload
def compress(
    condition: _ArrayLikeBool_co,  # 1D bool array
    a: ArrayLike,
    axis: SupportsIndex | None = None,
    out: None = None,
) -> NDArray[Any]: ...
@overload
def compress[ArrayT: np.ndarray](
    condition: _ArrayLikeBool_co,  # 1D bool array
    a: ArrayLike,
    axis: SupportsIndex | None,
    out: ArrayT,
) -> ArrayT: ...
@overload
def compress[ArrayT: np.ndarray](
    condition: _ArrayLikeBool_co,  # 1D bool array
    a: ArrayLike,
    axis: SupportsIndex | None = None,
    *,
    out: ArrayT,
) -> ArrayT: ...

#
@overload  # known array/scalar
def clip[ScalarOrArrayT: np.generic | np.ndarray](
    a: ScalarOrArrayT,
    a_min: _ScalarLike_co | _NoValueType | None = ...,
    a_max: _ScalarLike_co | _NoValueType | None = ...,
    out: None = None,
    *,
    dtype: None = None,
    min: ArrayLike | _NoValueType | None = ...,
    max: ArrayLike | _NoValueType | None = ...,
    **kwargs: Unpack[_UFuncKwargs],
) -> ScalarOrArrayT: ...
@overload  # known array, dtype=<known>
def clip[ShapeT: _Shape, ScalarT: np.generic](
    a: np.ndarray[ShapeT],
    a_min: _ScalarLike_co | _NoValueType | None = ...,
    a_max: _ScalarLike_co | _NoValueType | None = ...,
    out: None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    min: ArrayLike | _NoValueType | None = ...,
    max: ArrayLike | _NoValueType | None = ...,
    **kwargs: Unpack[_UFuncKwargs],
) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
@overload  # known array, dtype=<unknown>
def clip[ShapeT: _Shape](
    a: np.ndarray[ShapeT],
    a_min: _ScalarLike_co | _NoValueType | None = ...,
    a_max: _ScalarLike_co | _NoValueType | None = ...,
    out: None = None,
    *,
    dtype: DTypeLike,
    min: ArrayLike | _NoValueType | None = ...,
    max: ArrayLike | _NoValueType | None = ...,
    **kwargs: Unpack[_UFuncKwargs],
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # known array-like
def clip[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    a_min: ArrayLike | _NoValueType | None = ...,
    a_max: ArrayLike | _NoValueType | None = ...,
    out: None = None,
    *,
    dtype: None = None,
    min: ArrayLike | _NoValueType | None = ...,
    max: ArrayLike | _NoValueType | None = ...,
    **kwargs: Unpack[_UFuncKwargs],
) -> NDArray[ScalarT]: ...
@overload  # unknown scalar-like, dtype=<known>
def clip[ScalarT: np.generic](
    a: _ScalarLike_co,
    a_min: _ScalarLike_co | _NoValueType | None = ...,
    a_max: _ScalarLike_co | _NoValueType | None = ...,
    out: None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    min: ArrayLike | _NoValueType | None = ...,
    max: ArrayLike | _NoValueType | None = ...,
    **kwargs: Unpack[_UFuncKwargs],
) -> ScalarT: ...
@overload  # unknown >0d array-like, dtype=<known>
def clip[ScalarT: np.generic](
    a: _NestedSequence[_ScalarLike_co],
    a_min: ArrayLike | _NoValueType | None = ...,
    a_max: ArrayLike | _NoValueType | None = ...,
    out: None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    min: ArrayLike | _NoValueType | None = ...,
    max: ArrayLike | _NoValueType | None = ...,
    **kwargs: Unpack[_UFuncKwargs],
) -> NDArray[ScalarT]: ...
@overload  # unknown >0d array-like
def clip(
    a: _NestedSequence[_ScalarLike_co],
    a_min: ArrayLike | _NoValueType | None = ...,
    a_max: ArrayLike | _NoValueType | None = ...,
    out: None = None,
    *,
    dtype: DTypeLike | None = None,
    min: ArrayLike | _NoValueType | None = ...,
    max: ArrayLike | _NoValueType | None = ...,
    **kwargs: Unpack[_UFuncKwargs],
) -> NDArray[Any]: ...
@overload  # out=<given>  (positional)
def clip[ArrayT: np.ndarray](
    a: ArrayLike,
    a_min: ArrayLike | None,
    a_max: ArrayLike | None,
    out: ArrayT,
    *,
    dtype: DTypeLike | None = None,
    min: ArrayLike | _NoValueType | None = ...,
    max: ArrayLike | _NoValueType | None = ...,
    **kwargs: Unpack[_UFuncKwargs],
) -> ArrayT: ...
@overload  # out=<given>  (keyword)
def clip[ArrayT: np.ndarray](
    a: ArrayLike,
    a_min: ArrayLike | _NoValueType | None = ...,
    a_max: ArrayLike | _NoValueType | None = ...,
    *,
    out: ArrayT,
    dtype: DTypeLike | None = None,
    min: ArrayLike | _NoValueType | None = ...,
    max: ArrayLike | _NoValueType | None = ...,
    **kwargs: Unpack[_UFuncKwargs],
) -> ArrayT: ...
@overload  # fallback
def clip(
    a: ArrayLike,
    a_min: ArrayLike | _NoValueType | None = ...,
    a_max: ArrayLike | _NoValueType | None = ...,
    out: None = None,
    *,
    dtype: DTypeLike | None = None,
    min: ArrayLike | _NoValueType | None = ...,
    max: ArrayLike | _NoValueType | None = ...,
    **kwargs: Unpack[_UFuncKwargs],
) -> Any: ...

# keep in sync with `any`
@overload
def all(
    a: ArrayLike | None,
    axis: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.bool: ...
@overload  # axis: int
def all[ShapeT: _Shape](
    a: ArrayLike,
    axis: int,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.bool]: ...
@overload  # axis: (int, ...)
def all[ShapeT: _Shape](
    a: ArrayLike,
    axis: tuple[int, ...],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.bool] | Any: ...
@overload  # Nd, keepdims: True
def all[ShapeT: _Shape](
    a: np.ndarray[ShapeT],
    axis: int | tuple[int, ...] | None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.ndarray[ShapeT, np.dtype[np.bool]]: ...
@overload  # ?d, keepdims: True
def all[ShapeT: _Shape](
    a: ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.bool]: ...
@overload  # out: <given> (keyword)
def all[ArrayT: np.ndarray](
    a: ArrayLike | None,
    axis: int | tuple[int, ...] | None = None,
    *,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ArrayT: ...
@overload  # out: <given> (positional)
def all[ArrayT: np.ndarray](
    a: ArrayLike | None,
    axis: int | tuple[int, ...] | None,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ArrayT: ...

# keep in sync with `all`
@overload
def any(
    a: ArrayLike | None,
    axis: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.bool: ...
@overload  # axis: int
def any[ShapeT: _Shape](
    a: ArrayLike,
    axis: int,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.bool]: ...
@overload  # axis: (int, ...)
def any[ShapeT: _Shape](
    a: ArrayLike,
    axis: tuple[int, ...],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.bool] | Any: ...
@overload  # Nd, keepdims: True
def any[ShapeT: _Shape](
    a: np.ndarray[ShapeT],
    axis: int | tuple[int, ...] | None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.ndarray[ShapeT, np.dtype[np.bool]]: ...
@overload  # ?d, keepdims: True
def any[ShapeT: _Shape](
    a: ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.bool]: ...
@overload  # out: <given> (keyword)
def any[ArrayT: np.ndarray](
    a: ArrayLike | None,
    axis: int | tuple[int, ...] | None = None,
    *,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ArrayT: ...
@overload  # out: <given> (positional)
def any[ArrayT: np.ndarray](
    a: ArrayLike | None,
    axis: int | tuple[int, ...] | None,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ArrayT: ...

# keep in sync with `cumprod` below
@overload
def cumsum[ScalarT: np.inexact | np.timedelta64 | np.object_](
    a: _ArrayLike[ScalarT],
    axis: None = None,
    dtype: None = None,
    out: None = None,
) -> _Array1D[ScalarT]: ...
@overload
def cumsum[ArrayT: np.ndarray[Any, np.dtype[np.inexact | np.timedelta64 | np.object_] | np.dtypes.StringDType]](
    a: ArrayT,
    axis: SupportsIndex,
    dtype: None = None,
    out: None = None,
) -> ArrayT: ...
@overload
def cumsum[ScalarT: np.inexact | np.timedelta64 | np.object_](
    a: _ArrayLike[ScalarT],
    axis: SupportsIndex,
    dtype: None = None,
    out: None = None,
) -> NDArray[ScalarT]: ...
@overload
def cumsum(
    a: _ArrayLikeInt_co,
    axis: None = None,
    dtype: None = None,
    out: None = None,
) -> _Array1D[np.int_]: ...
@overload
def cumsum[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.bool | np.integer]],
    axis: SupportsIndex,
    dtype: None = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.int_]]: ...
@overload
def cumsum(
    a: _ArrayLikeInt_co,
    axis: SupportsIndex,
    dtype: None = None,
    out: None = None,
) -> NDArray[np.int_]: ...
@overload
def cumsum(
    a: ArrayLike,
    axis: None = None,
    dtype: None = None,
    out: None = None,
) -> _Array1D[Any]: ...
@overload
def cumsum(
    a: ArrayLike,
    axis: SupportsIndex,
    dtype: None = None,
    out: None = None,
) -> NDArray[Any]: ...
@overload
def cumsum[ScalarT: np.generic](
    a: ArrayLike,
    axis: None,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
) -> _Array1D[ScalarT]: ...
@overload
def cumsum[ScalarT: np.generic](
    a: ArrayLike,
    axis: None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
) -> _Array1D[ScalarT]: ...
@overload
def cumsum[ScalarT: np.generic](
    a: ArrayLike,
    axis: SupportsIndex,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
) -> NDArray[ScalarT]: ...
@overload
def cumsum(
    a: ArrayLike,
    axis: None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
) -> _Array1D[Any]: ...
@overload
def cumsum(
    a: ArrayLike,
    axis: SupportsIndex,
    dtype: DTypeLike | None = None,
    out: None = None,
) -> NDArray[Any]: ...
@overload
def cumsum[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: SupportsIndex | None,
    dtype: DTypeLike | None,
    out: ArrayT,
) -> ArrayT: ...
@overload
def cumsum[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: SupportsIndex | None = None,
    dtype: DTypeLike | None = None,
    *,
    out: ArrayT,
) -> ArrayT: ...

# keep in sync with `cumulative_prod` below
@overload
def cumulative_sum[ScalarT: np.inexact | np.timedelta64 | np.object_](
    x: _ArrayLike[ScalarT],
    /,
    *,
    axis: None = None,
    dtype: None = None,
    out: None = None,
    include_initial: bool = False,
) -> _Array1D[ScalarT]: ...
@overload
def cumulative_sum[ArrayT: np.ndarray[Any, np.dtype[np.inexact | np.timedelta64 | np.object_] | np.dtypes.StringDType]](
    x: ArrayT,
    /,
    *,
    axis: SupportsIndex,
    dtype: None = None,
    out: None = None,
    include_initial: bool = False,
) -> ArrayT: ...
@overload
def cumulative_sum[ScalarT: np.inexact | np.timedelta64 | np.object_](
    x: _ArrayLike[ScalarT],
    /,
    *,
    axis: SupportsIndex,
    dtype: None = None,
    out: None = None,
    include_initial: bool = False,
) -> NDArray[ScalarT]: ...
@overload
def cumulative_sum(
    x: _ArrayLikeInt_co,
    /,
    *,
    axis: None = None,
    dtype: None = None,
    out: None = None,
    include_initial: bool = False,
) -> _Array1D[np.int_]: ...
@overload
def cumulative_sum[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.bool | np.integer]],
    /,
    *,
    axis: SupportsIndex,
    dtype: None = None,
    out: None = None,
    include_initial: bool = False,
) -> np.ndarray[ShapeT, np.dtype[np.int_]]: ...
@overload
def cumulative_sum(
    x: _ArrayLikeInt_co,
    /,
    *,
    axis: SupportsIndex,
    dtype: None = None,
    out: None = None,
    include_initial: bool = False,
) -> NDArray[np.int_]: ...
@overload
def cumulative_sum(
    x: ArrayLike,
    /,
    *,
    axis: None = None,
    dtype: None = None,
    out: None = None,
    include_initial: bool = False,
) -> _Array1D[Any]: ...
@overload
def cumulative_sum(
    x: ArrayLike,
    /,
    *,
    axis: SupportsIndex,
    dtype: None = None,
    out: None = None,
    include_initial: bool = False,
) -> NDArray[Any]: ...
@overload
def cumulative_sum[ScalarT: np.generic](
    x: ArrayLike,
    /,
    *,
    axis: None = None,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    include_initial: bool = False,
) -> _Array1D[ScalarT]: ...
@overload
def cumulative_sum[ScalarT: np.generic](
    x: ArrayLike,
    /,
    *,
    axis: SupportsIndex,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    include_initial: bool = False,
) -> NDArray[ScalarT]: ...
@overload
def cumulative_sum(
    x: ArrayLike,
    /,
    *,
    axis: None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    include_initial: bool = False,
) -> _Array1D[Any]: ...
@overload
def cumulative_sum(
    x: ArrayLike,
    /,
    *,
    axis: SupportsIndex,
    dtype: DTypeLike | None = None,
    out: None = None,
    include_initial: bool = False,
) -> NDArray[Any]: ...
@overload
def cumulative_sum[ArrayT: np.ndarray](
    x: ArrayLike,
    /,
    *,
    axis: SupportsIndex | None = None,
    dtype: DTypeLike | None = None,
    out: ArrayT,
    include_initial: bool = False,
) -> ArrayT: ...

#
@overload  # ~builtins.int
def ptp(
    a: _NestedSequence[list[int]] | list[int],
    axis: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
) -> np.int_: ...
@overload  # ~builtins.int, axis: <given>
def ptp(
    a: _NestedSequence[list[int]] | list[int],
    axis: int | tuple[int, ...],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
) -> NDArray[np.int_]: ...
@overload  # ~builtins.int, keepdims=True
def ptp(
    a: _NestedSequence[list[int]] | list[int],
    axis: int | tuple[int, ...] | None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
) -> NDArray[np.int_]: ...
@overload  # ~builtins.float
def ptp(
    a: _NestedSequence[list[float]] | list[float],
    axis: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
) -> np.float64: ...
@overload  # ~builtins.float, axis: <given>
def ptp(
    a: _NestedSequence[list[float]] | list[float],
    axis: int | tuple[int, ...],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
) -> NDArray[np.float64]: ...
@overload  # ~builtins.float, keepdims=True
def ptp(
    a: _NestedSequence[list[float]] | list[float],
    axis: int | tuple[int, ...] | None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
) -> NDArray[np.float64]: ...
@overload  # ~builtins.complex
def ptp(
    a: _NestedSequence[list[complex]] | list[complex],
    axis: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
) -> np.complex128: ...
@overload  # ~builtins.complex, axis: <given>
def ptp(
    a: _NestedSequence[list[complex]] | list[complex],
    axis: int | tuple[int, ...],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
) -> NDArray[np.complex128]: ...
@overload  # ~builtins.complex, keepdims=True
def ptp(
    a: _NestedSequence[list[complex]] | list[complex],
    axis: int | tuple[int, ...] | None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
) -> NDArray[np.complex128]: ...
@overload  # ~number | timedelta64
def ptp[ScalarT: np.number | np.timedelta64](
    a: _ArrayLike[ScalarT],
    axis: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
) -> ScalarT: ...
@overload  # ~number | timedelta64 | object_, axis: <given>
def ptp[ScalarT: np.number | np.timedelta64 | np.object_](
    a: _ArrayLike[ScalarT],
    axis: int | tuple[int, ...],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
) -> NDArray[ScalarT]: ...
@overload  # ~number | timedelta64 | datetime64 | object_, keepdims=True
def ptp[ArrayT: NDArray[np.number | np.timedelta64 | np.object_]](
    a: ArrayT,
    axis: int | tuple[int, ...] | None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
) -> ArrayT: ...
@overload  # datetime64
def ptp(
    a: _ArrayLike[np.datetime64],
    axis: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
) -> np.timedelta64[Any]: ...
@overload  # datetime64, axis: <given>
def ptp(
    a: _ArrayLike[np.datetime64],
    axis: int | tuple[int, ...],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
) -> NDArray[np.timedelta64]: ...
@overload  # datetime64, keepdims=True
def ptp[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.datetime64]],
    axis: int | tuple[int, ...] | None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
) -> np.ndarray[ShapeT, np.dtype[np.timedelta64]]: ...
@overload  # object_
def ptp(
    a: _ArrayLike[np.object_],
    axis: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
) -> Any: ...
@overload  # out: ArrayT
def ptp[ArrayT: np.ndarray](
    a: _ArrayLikeNumeric_co,
    axis: int | tuple[int, ...] | None = None,
    *,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
) -> ArrayT: ...
@overload  # fallback
def ptp(
    a: _ArrayLikeNumeric_co,
    axis: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
) -> Any: ...
@overload  # fallback, axis: <given>
def ptp(
    a: _ArrayLikeNumeric_co,
    axis: int | tuple[int, ...],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
) -> NDArray[Any]: ...
@overload  # fallback, keepdims=True
def ptp(
    a: _ArrayLikeNumeric_co,
    axis: int | tuple[int, ...] | None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
) -> NDArray[Any]: ...

# keep in sync with `amin` below
@overload  # sequence of just `Any` (workaround)
def amax(
    a: _NestedSequence[Never],
    axis: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> Any: ...
@overload  # builtins.bool
def amax(
    a: _NestedSequence[bool],
    axis: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.bool: ...
@overload  # builtins.bool, axis: <given>
def amax(
    a: _NestedSequence[bool],
    axis: int | tuple[int, ...],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.bool]: ...
@overload  # builtins.bool, keepdims=True
def amax(
    a: _NestedSequence[bool],
    axis: int | tuple[int, ...] | None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.bool]: ...
@overload  # ~builtins.int
def amax(
    a: _NestedSequence[list[int]] | list[int],
    axis: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.int_: ...
@overload  # ~builtins.int, axis: <given>
def amax(
    a: _NestedSequence[list[int]] | list[int],
    axis: int | tuple[int, ...],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.int_]: ...
@overload  # ~builtins.int, keepdims=True
def amax(
    a: _NestedSequence[list[int]] | list[int],
    axis: int | tuple[int, ...] | None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.int_]: ...
@overload  # ~builtins.float
def amax(
    a: _NestedSequence[list[float]] | list[float],
    axis: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.float64: ...
@overload  # ~builtins.float, axis: <given>
def amax(
    a: _NestedSequence[list[float]] | list[float],
    axis: int | tuple[int, ...],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.float64]: ...
@overload  # ~builtins.float, keepdims=True
def amax(
    a: _NestedSequence[list[float]] | list[float],
    axis: int | tuple[int, ...] | None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.float64]: ...
@overload  # ~builtins.complex
def amax(
    a: _NestedSequence[list[complex]] | list[complex],
    axis: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.complex128: ...
@overload  # ~builtins.complex, axis: <given>
def amax(
    a: _NestedSequence[list[complex]] | list[complex],
    axis: int | tuple[int, ...],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.complex128]: ...
@overload  # ~builtins.complex, keepdims=True
def amax(
    a: _NestedSequence[list[complex]] | list[complex],
    axis: int | tuple[int, ...] | None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.complex128]: ...
@overload  # +number | timedelta64 | datetime64
def amax[ScalarT: np.number | np.bool | np.timedelta64 | np.datetime64](
    a: _ArrayLike[ScalarT],
    axis: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ScalarT: ...
@overload  # +number | timedelta64 | datetime64 | object_, axis: <given>
def amax[ScalarT: np.number | np.bool | np.timedelta64 | np.datetime64 | np.object_](
    a: _ArrayLike[ScalarT],
    axis: int | tuple[int, ...],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[ScalarT]: ...
@overload  # +number | timedelta64 | datetime64 | object_, keepdims=True
def amax[ArrayT: NDArray[np.number | np.bool | np.timedelta64 | np.datetime64 | np.object_]](
    a: ArrayT,
    axis: int | tuple[int, ...] | None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ArrayT: ...
@overload  # object_
def amax(
    a: _ArrayLike[np.object_],
    axis: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> Any: ...
@overload  # out: ArrayT
def amax[ArrayT: np.ndarray](
    a: _ArrayLikeNumeric_co | _NestedSequence[_Orderable],
    axis: int | tuple[int, ...] | None = None,
    *,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ArrayT: ...
@overload  # fallback
def amax(
    a: _ArrayLikeNumeric_co | _NestedSequence[_Orderable],
    axis: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> Any: ...
@overload  # fallback, axis: <given>
def amax(
    a: _ArrayLikeNumeric_co | _NestedSequence[_Orderable],
    axis: int | tuple[int, ...],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[Any]: ...
@overload  # fallback, keepdims=True
def amax(
    a: _ArrayLikeNumeric_co | _NestedSequence[_Orderable],
    axis: int | tuple[int, ...] | None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[Any]: ...

max = amax

# keep in sync with `amax` above
@overload  # sequence of just `Any` (workaround)
def amin(
    a: _NestedSequence[Never],
    axis: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> Any: ...
@overload  # builtins.bool
def amin(
    a: _NestedSequence[bool],
    axis: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.bool: ...
@overload  # builtins.bool, axis: <given>
def amin(
    a: _NestedSequence[bool],
    axis: int | tuple[int, ...],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.bool]: ...
@overload  # builtins.bool, keepdims=True
def amin(
    a: _NestedSequence[bool],
    axis: int | tuple[int, ...] | None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.bool]: ...
@overload  # ~builtins.int
def amin(
    a: _NestedSequence[list[int]] | list[int],
    axis: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.int_: ...
@overload  # ~builtins.int, axis: <given>
def amin(
    a: _NestedSequence[list[int]] | list[int],
    axis: int | tuple[int, ...],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.int_]: ...
@overload  # ~builtins.int, keepdims=True
def amin(
    a: _NestedSequence[list[int]] | list[int],
    axis: int | tuple[int, ...] | None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.int_]: ...
@overload  # ~builtins.float
def amin(
    a: _NestedSequence[list[float]] | list[float],
    axis: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.float64: ...
@overload  # ~builtins.float, axis: <given>
def amin(
    a: _NestedSequence[list[float]] | list[float],
    axis: int | tuple[int, ...],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.float64]: ...
@overload  # ~builtins.float, keepdims=True
def amin(
    a: _NestedSequence[list[float]] | list[float],
    axis: int | tuple[int, ...] | None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.float64]: ...
@overload  # ~builtins.complex
def amin(
    a: _NestedSequence[list[complex]] | list[complex],
    axis: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.complex128: ...
@overload  # ~builtins.complex, axis: <given>
def amin(
    a: _NestedSequence[list[complex]] | list[complex],
    axis: int | tuple[int, ...],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.complex128]: ...
@overload  # ~builtins.complex, keepdims=True
def amin(
    a: _NestedSequence[list[complex]] | list[complex],
    axis: int | tuple[int, ...] | None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.complex128]: ...
@overload  # +number | timedelta64 | datetime64
def amin[ScalarT: np.number | np.bool | np.timedelta64 | np.datetime64](
    a: _ArrayLike[ScalarT],
    axis: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ScalarT: ...
@overload  # +number | timedelta64 | datetime64 | object_, axis: <given>
def amin[ScalarT: np.number | np.bool | np.timedelta64 | np.datetime64 | np.object_](
    a: _ArrayLike[ScalarT],
    axis: int | tuple[int, ...],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[ScalarT]: ...
@overload  # +number | timedelta64 | datetime64 | object_, keepdims=True
def amin[ArrayT: NDArray[np.number | np.bool | np.timedelta64 | np.datetime64 | np.object_]](
    a: ArrayT,
    axis: int | tuple[int, ...] | None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ArrayT: ...
@overload  # object_
def amin(
    a: _ArrayLike[np.object_],
    axis: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> Any: ...
@overload  # out: ArrayT
def amin[ArrayT: np.ndarray](
    a: _ArrayLikeNumeric_co | _NestedSequence[_Orderable],
    axis: int | tuple[int, ...] | None = None,
    *,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ArrayT: ...
@overload  # fallback
def amin(
    a: _ArrayLikeNumeric_co | _NestedSequence[_Orderable],
    axis: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> Any: ...
@overload  # fallback, axis: <given>
def amin(
    a: _ArrayLikeNumeric_co | _NestedSequence[_Orderable],
    axis: int | tuple[int, ...],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[Any]: ...
@overload  # fallback, keepdims=True
def amin(
    a: _ArrayLikeNumeric_co | _NestedSequence[_Orderable],
    axis: int | tuple[int, ...] | None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[Any]: ...

min = amin

# keep in sync with `cumsum` above (minus `timedelta64` and `StringDType`)
@overload
def cumprod[ScalarT: np.inexact | np.object_](
    a: _ArrayLike[ScalarT],
    axis: None = None,
    dtype: None = None,
    out: None = None,
) -> _Array1D[ScalarT]: ...
@overload
def cumprod[ArrayT: NDArray[np.inexact | np.object_]](
    a: ArrayT,
    axis: SupportsIndex,
    dtype: None = None,
    out: None = None,
) -> ArrayT: ...
@overload
def cumprod[ScalarT: np.inexact | np.object_](
    a: _ArrayLike[ScalarT],
    axis: SupportsIndex,
    dtype: None = None,
    out: None = None,
) -> NDArray[ScalarT]: ...
@overload
def cumprod(
    a: _ArrayLikeInt_co,
    axis: None = None,
    dtype: None = None,
    out: None = None,
) -> _Array1D[np.int_]: ...
@overload
def cumprod[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.bool | np.integer]],
    axis: SupportsIndex,
    dtype: None = None,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.int_]]: ...
@overload
def cumprod(
    a: _ArrayLikeInt_co,
    axis: SupportsIndex,
    dtype: None = None,
    out: None = None,
) -> NDArray[np.int_]: ...
@overload
def cumprod(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: None = None,
    dtype: None = None,
    out: None = None,
) -> _Array1D[Any]: ...
@overload
def cumprod(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: SupportsIndex,
    dtype: None = None,
    out: None = None,
) -> NDArray[Any]: ...
@overload
def cumprod[ScalarT: np.number | np.bool | np.object_](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: None,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
) -> _Array1D[ScalarT]: ...
@overload
def cumprod[ScalarT: np.number | np.bool | np.object_](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
) -> _Array1D[ScalarT]: ...
@overload
def cumprod[ScalarT: np.number | np.bool | np.object_](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: SupportsIndex,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
) -> NDArray[ScalarT]: ...
@overload
def cumprod(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
) -> _Array1D[Any]: ...
@overload
def cumprod(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: SupportsIndex,
    dtype: DTypeLike | None = None,
    out: None = None,
) -> NDArray[Any]: ...
@overload
def cumprod[ArrayT: np.ndarray](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: SupportsIndex | None,
    dtype: DTypeLike | None,
    out: ArrayT,
) -> ArrayT: ...
@overload
def cumprod[ArrayT: np.ndarray](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: SupportsIndex | None = None,
    dtype: DTypeLike | None = None,
    *,
    out: ArrayT,
) -> ArrayT: ...

# keep in sync with `cumulative_sum` above (minus `timedelta64` and `StringDType`)
@overload
def cumulative_prod[ScalarT: np.inexact | np.object_](
    x: _ArrayLike[ScalarT],
    /,
    *,
    axis: None = None,
    dtype: None = None,
    out: None = None,
    include_initial: bool = False,
) -> _Array1D[ScalarT]: ...
@overload
def cumulative_prod[ArrayT: NDArray[np.inexact | np.object_]](
    x: ArrayT,
    /,
    *,
    axis: SupportsIndex,
    dtype: None = None,
    out: None = None,
    include_initial: bool = False,
) -> ArrayT: ...
@overload
def cumulative_prod[ScalarT: np.inexact | np.object_](
    x: _ArrayLike[ScalarT],
    /,
    *,
    axis: SupportsIndex,
    dtype: None = None,
    out: None = None,
    include_initial: bool = False,
) -> NDArray[ScalarT]: ...
@overload
def cumulative_prod(
    x: _ArrayLikeInt_co,
    /,
    *,
    axis: None = None,
    dtype: None = None,
    out: None = None,
    include_initial: bool = False,
) -> _Array1D[np.int_]: ...
@overload
def cumulative_prod[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.bool | np.integer]],
    /,
    *,
    axis: SupportsIndex,
    dtype: None = None,
    out: None = None,
    include_initial: bool = False,
) -> np.ndarray[ShapeT, np.dtype[np.int_]]: ...
@overload
def cumulative_prod(
    x: _ArrayLikeInt_co,
    /,
    *,
    axis: SupportsIndex,
    dtype: None = None,
    out: None = None,
    include_initial: bool = False,
) -> NDArray[np.int_]: ...
@overload
def cumulative_prod(
    x: ArrayLike,
    /,
    *,
    axis: None = None,
    dtype: None = None,
    out: None = None,
    include_initial: bool = False,
) -> _Array1D[Any]: ...
@overload
def cumulative_prod(
    x: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    /,
    *,
    axis: SupportsIndex,
    dtype: None = None,
    out: None = None,
    include_initial: bool = False,
) -> NDArray[Any]: ...
@overload
def cumulative_prod[ScalarT: np.number | np.bool | np.object_](
    x: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    /,
    *,
    axis: None = None,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    include_initial: bool = False,
) -> _Array1D[ScalarT]: ...
@overload
def cumulative_prod[ScalarT: np.number | np.bool | np.object_](
    x: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    /,
    *,
    axis: SupportsIndex,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    include_initial: bool = False,
) -> NDArray[ScalarT]: ...
@overload
def cumulative_prod(
    x: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    /,
    *,
    axis: None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    include_initial: bool = False,
) -> _Array1D[Any]: ...
@overload
def cumulative_prod(
    x: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    /,
    *,
    axis: SupportsIndex,
    dtype: DTypeLike | None = None,
    out: None = None,
    include_initial: bool = False,
) -> NDArray[Any]: ...
@overload
def cumulative_prod[ArrayT: np.ndarray](
    x: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    /,
    *,
    axis: SupportsIndex | None = None,
    dtype: DTypeLike | None = None,
    out: ArrayT,
    include_initial: bool = False,
) -> ArrayT: ...

def ndim(a: ArrayLike) -> int: ...

def size(a: ArrayLike, axis: int | tuple[int, ...] | None = None) -> int: ...

#
@overload  # known scalar or array
def around[ScalarOrArrayT: np.number | NDArray[np.number | np.object_]](
    a: ScalarOrArrayT,
    decimals: SupportsIndex = 0,
    out: None = None,
) -> ScalarOrArrayT: ...
@overload  # 0d bool
def around(a: _BoolLike_co, decimals: SupportsIndex = 0, out: None = None) -> np.float16: ...
@overload  # 0d int  (overlaps with bool)
def around(a: int, decimals: SupportsIndex = 0, out: None = None) -> np.int_ | Any: ...
@overload  # 0d float  (overlaps with int)
def around(a: float, decimals: SupportsIndex = 0, out: None = None) -> np.float64 | Any: ...
@overload  # 0d complex  (overlaps with float)
def around(a: complex, decimals: SupportsIndex = 0, out: None = None) -> np.complex128 | Any: ...
@overload  # Nd bool
def around[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.bool]],
    decimals: SupportsIndex = 0,
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.float16]]: ...
@overload  # 1d bool
def around(a: list[bool], decimals: SupportsIndex = 0, out: None = None) -> _Array1D[np.float16]: ...
@overload  # 1d int
def around(a: list[int], decimals: SupportsIndex = 0, out: None = None) -> _Array1D[np.int_]: ...
@overload  # 1d float
def around(a: list[float], decimals: SupportsIndex = 0, out: None = None) -> _Array1D[np.float64]: ...
@overload  # 1d complex
def around(a: list[complex], decimals: SupportsIndex = 0, out: None = None) -> _Array1D[np.complex128]: ...
@overload  # 2d bool
def around(a: Sequence[list[bool]], decimals: SupportsIndex = 0, out: None = None) -> _Array2D[np.float16]: ...
@overload  # 2d int
def around(a: Sequence[list[int]], decimals: SupportsIndex = 0, out: None = None) -> _Array2D[np.int_]: ...
@overload  # 2d float
def around(a: Sequence[list[float]], decimals: SupportsIndex = 0, out: None = None) -> _Array2D[np.float64]: ...
@overload  # 2d complex
def around(a: Sequence[list[complex]], decimals: SupportsIndex = 0, out: None = None) -> _Array2D[np.complex128]: ...
@overload  # ?d known dtype
def around[ScalarT: np.number | np.object_](
    a: _ArrayLike[ScalarT],
    decimals: SupportsIndex = 0,
    out: None = None,
) -> NDArray[ScalarT] | Any: ...
@overload  # out=<given>  (postional)
def around[ArrayT: np.ndarray](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    decimals: SupportsIndex,
    out: ArrayT,
) -> ArrayT: ...
@overload  # out=<given>  (keyword)
def around[ArrayT: np.ndarray](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    decimals: SupportsIndex = 0,
    *,
    out: ArrayT,
) -> ArrayT: ...
@overload  # fallback
def around(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    decimals: SupportsIndex = 0,
    out: None = None,
) -> NDArray[Any] | Any: ...

# keep in sync with `sum` below (but without `timedelta64`)
@overload  # ~builtins.float
def prod(
    a: _NestedSequence[list[float]] | list[float],
    axis: None = None,
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _FloatLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.float64: ...
@overload  # ~builtins.float, axis: <given>
def prod(
    a: _NestedSequence[list[float]] | list[float],
    axis: int | tuple[int, ...],
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _FloatLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.float64]: ...
@overload  # ~builtins.float, keepdims=True
def prod(
    a: _NestedSequence[list[float]] | list[float],
    axis: int | tuple[int, ...] | None = None,
    dtype: None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
    initial: _FloatLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.float64]: ...
@overload  # ~builtins.complex
def prod(
    a: _NestedSequence[list[complex]] | list[complex],
    axis: None = None,
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.complex128: ...
@overload  # ~builtins.complex, axis: <given>
def prod(
    a: _NestedSequence[list[complex]] | list[complex],
    axis: int | tuple[int, ...],
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.complex128]: ...
@overload  # ~builtins.complex, keepdims=True
def prod(
    a: _NestedSequence[list[complex]] | list[complex],
    axis: int | tuple[int, ...] | None = None,
    dtype: None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.complex128]: ...
@overload  # ~inexact
def prod[ScalarT: np.inexact](
    a: _ArrayLike[ScalarT],
    axis: None = None,
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ScalarT: ...
@overload  # ~inexact | timedelta64 | object_, axis: <given>
def prod[ScalarT: np.inexact | np.object_](
    a: _ArrayLike[ScalarT],
    axis: int | tuple[int, ...],
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[ScalarT]: ...
@overload  # ~inexact | object_, keepdims=True
def prod[ArrayT: NDArray[np.inexact | np.object_]](
    a: ArrayT,
    axis: int | tuple[int, ...] | None = None,
    dtype: None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ArrayT: ...
@overload  # bool | integer | +builtins.int
def prod(
    a: _DualArrayLike[np.dtype[np.bool | np.integer], int],
    axis: None = None,
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _IntLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.int_: ...
@overload  # bool | np.integer | +builtins.int, axis: <given>
def prod(
    a: _DualArrayLike[np.dtype[np.bool | np.integer], int],
    axis: int | tuple[int, ...],
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _IntLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.int_]: ...
@overload  # bool | np.integer, keepdims=True
def prod[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.bool | np.integer]],
    axis: int | tuple[int, ...] | None = None,
    dtype: None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
    initial: _IntLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.ndarray[ShapeT, np.dtype[np.int_]]: ...
@overload  # object_
def prod(
    a: _SupportsArray[np.dtype[np.object_]],
    axis: None = None,
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> Any: ...
@overload  # dtype: ScalarT
def prod[ScalarT: np.generic](
    a: _ArrayLikeNumeric_co,
    axis: None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ScalarT: ...
@overload  # dtype: ScalarT (keyword), keepdims=True
def prod[ShapeT: _Shape, ScalarT: np.generic](
    a: np.ndarray[ShapeT, np.dtype[np.number | np.bool | np.object_]],
    axis: int | tuple[int, ...] | None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    keepdims: Literal[True],
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
@overload  # dtype: ScalarT (positional), keepdims=True
def prod[ShapeT: _Shape, ScalarT: np.generic](
    a: np.ndarray[ShapeT, np.dtype[np.number | np.bool | np.object_]],
    axis: int | tuple[int, ...] | None,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    *,
    keepdims: Literal[True],
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
@overload  # dtype: ScalarT (keyword), keepdims=True
def prod[ScalarT: np.generic](
    a: _ArrayLikeMultiplicative_co,
    axis: int | tuple[int, ...] | None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    keepdims: Literal[True],
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[ScalarT]: ...
@overload  # axis: <given>, dtype: ScalarT
def prod[ScalarT: np.generic](
    a: _ArrayLikeMultiplicative_co,
    axis: int | tuple[int, ...],
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[ScalarT]: ...
@overload  # out: ArrayT (keyword)
def prod[ArrayT: np.ndarray](
    a: _ArrayLikeMultiplicative_co,
    axis: int | tuple[int, ...] | None = None,
    dtype: DTypeLike | None = None,
    *,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ArrayT: ...
@overload  # out: ArrayT (positional)
def prod[ArrayT: np.ndarray](
    a: _ArrayLikeMultiplicative_co,
    axis: int | tuple[int, ...] | None,
    dtype: DTypeLike | None,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ArrayT: ...
@overload  # fallback
def prod(
    a: _ArrayLikeMultiplicative_co,
    axis: None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> Any: ...
@overload  # fallback, axis: <given>
def prod(
    a: _ArrayLikeMultiplicative_co,
    axis: int | tuple[int, ...],
    dtype: DTypeLike | None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[Any]: ...
@overload  # fallback, keepdims=True
def prod(
    a: _ArrayLikeMultiplicative_co,
    axis: int | tuple[int, ...] | None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[Any]: ...

# keep in sync with `prod` above (but also accept `timedelta64`)
@overload  # ~builtins.float
def sum(
    a: _NestedSequence[list[float]] | list[float],
    axis: None = None,
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _FloatLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.float64: ...
@overload  # ~builtins.float, axis: <given>
def sum(
    a: _NestedSequence[list[float]] | list[float],
    axis: int | tuple[int, ...],
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _FloatLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.float64]: ...
@overload  # ~builtins.float, keepdims=True
def sum(
    a: _NestedSequence[list[float]] | list[float],
    axis: int | tuple[int, ...] | None = None,
    dtype: None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
    initial: _FloatLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.float64]: ...
@overload  # ~builtins.complex
def sum(
    a: _NestedSequence[list[complex]] | list[complex],
    axis: None = None,
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.complex128: ...
@overload  # ~builtins.complex, axis: <given>
def sum(
    a: _NestedSequence[list[complex]] | list[complex],
    axis: int | tuple[int, ...],
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.complex128]: ...
@overload  # ~builtins.complex, keepdims=True
def sum(
    a: _NestedSequence[list[complex]] | list[complex],
    axis: int | tuple[int, ...] | None = None,
    dtype: None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.complex128]: ...
@overload  # ~inexact | timedelta64
def sum[ScalarT: np.inexact | np.timedelta64](
    a: _ArrayLike[ScalarT],
    axis: None = None,
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ScalarT: ...
@overload  # ~inexact | timedelta64 | object_, axis: <given>
def sum[ScalarT: np.inexact | np.timedelta64 | np.object_](
    a: _ArrayLike[ScalarT],
    axis: int | tuple[int, ...],
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[ScalarT]: ...
@overload  # ~number | timedelta64 | object_, keepdims=True
def sum[ArrayT: NDArray[np.inexact | np.timedelta64 | np.object_]](
    a: ArrayT,
    axis: int | tuple[int, ...] | None = None,
    dtype: None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ArrayT: ...
@overload  # bool | integer | +builtins.int
def sum(
    a: _DualArrayLike[np.dtype[np.bool | np.integer], int],
    axis: None = None,
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _IntLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.int_: ...
@overload  # bool | integer | +builtins.int, axis: <given>
def sum(
    a: _DualArrayLike[np.dtype[np.bool | np.integer], int],
    axis: int | tuple[int, ...],
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _IntLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.int_]: ...
@overload  # bool | integer, keepdims=True
def sum[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.bool | np.integer]],
    axis: int | tuple[int, ...] | None = None,
    dtype: None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
    initial: _IntLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.ndarray[ShapeT, np.dtype[np.int_]]: ...
@overload  # object_
def sum(
    a: _SupportsArray[np.dtype[np.object_]],
    axis: None = None,
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> Any: ...
@overload  # dtype: ScalarT
def sum[ScalarT: np.generic](
    a: _ArrayLikeNumeric_co,
    axis: None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ScalarT: ...
@overload  # dtype: ScalarT (keyword), keepdims=True
def sum[ShapeT: _Shape, ScalarT: np.generic](
    a: np.ndarray[ShapeT, np.dtype[np.number | np.bool | np.timedelta64 | np.object_]],
    axis: int | tuple[int, ...] | None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    keepdims: Literal[True],
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
@overload  # dtype: ScalarT (positional), keepdims=True
def sum[ShapeT: _Shape, ScalarT: np.generic](
    a: np.ndarray[ShapeT, np.dtype[np.number | np.bool | np.timedelta64 | np.object_]],
    axis: int | tuple[int, ...] | None,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    *,
    keepdims: Literal[True],
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
@overload  # dtype: ScalarT (keyword), keepdims=True
def sum[ScalarT: np.generic](
    a: _ArrayLikeNumeric_co,
    axis: int | tuple[int, ...] | None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    keepdims: Literal[True],
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[ScalarT]: ...
@overload  # axis: <given>, dtype: ScalarT
def sum[ScalarT: np.generic](
    a: _ArrayLikeNumeric_co,
    axis: int | tuple[int, ...],
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[ScalarT]: ...
@overload  # out: ArrayT (keyword)
def sum[ArrayT: np.ndarray](
    a: _ArrayLikeNumeric_co,
    axis: int | tuple[int, ...] | None = None,
    dtype: DTypeLike | None = None,
    *,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ArrayT: ...
@overload  # out: ArrayT (positional)
def sum[ArrayT: np.ndarray](
    a: _ArrayLikeNumeric_co,
    axis: int | tuple[int, ...] | None,
    dtype: DTypeLike | None,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ArrayT: ...
@overload  # fallback
def sum(
    a: _ArrayLikeNumeric_co,
    axis: None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> Any: ...
@overload  # fallback, axis: <given>
def sum(
    a: _ArrayLikeNumeric_co,
    axis: int | tuple[int, ...],
    dtype: DTypeLike | None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[Any]: ...
@overload  # fallback, keepdims=True
def sum(
    a: _ArrayLikeNumeric_co,
    axis: int | tuple[int, ...] | None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[Any]: ...

#
@overload  # +integer | ~object_ | +builtins.float
def mean(
    a: _DualArrayLike[np.dtype[np.integer | np.bool | np.object_], float],
    axis: None = None,
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.float64: ...
@overload  # +integer | +builtins.float, ?d, axis: <given>
def mean(
    a: _ArrayJustND[np.integer | np.bool],
    axis: int | tuple[int, ...],
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.float64] | Any: ...
@overload  # +integer | +builtins.float, 1d, axis: <single>
def mean(
    a: _ToArray1D2[np.integer | np.bool, float],
    axis: int | tuple[int],
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.float64: ...
@overload  # +integer | +builtins.float, 2d, axis: <single>
def mean(
    a: _ToArray2D2[np.integer | np.bool, float],
    axis: int | tuple[int],
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> _Array1D[np.float64]: ...
@overload  # +integer | +builtins.float, 3d, axis: <single>
def mean(
    a: _ToArray3D2[np.integer | np.bool, float],
    axis: int | tuple[int],
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> _Array2D[np.float64]: ...
@overload  # +integer | +builtins.float, 4d, axis: <single>
def mean(
    a: _ToArray4D2[np.integer | np.bool, float],
    axis: int | tuple[int],
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> _Array3D[np.float64]: ...
@overload  # +integer | +builtins.float, ?d, axis: <given>
def mean(
    a: _DualArrayLike[np.dtype[np.integer | np.bool], float],
    axis: int | tuple[int, ...],
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.float64] | Any: ...
@overload  # +integer, keepdims=True
def mean[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.integer | np.bool]],
    axis: int | tuple[int, ...] | None = None,
    dtype: None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # ~complex
def mean(
    a: _NestedSequence[list[complex]] | list[complex],
    axis: None = None,
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.complex128: ...
@overload  # ~inexact | timedelta64
def mean[ScalarT: np.inexact | np.timedelta64](
    a: _ArrayLike[ScalarT],
    axis: None = None,
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ScalarT: ...
@overload  # ~inexact | timedelta64, ?d, axis: <given>
def mean[ScalarT: np.inexact | np.timedelta64](
    a: _ArrayJustND[ScalarT],
    axis: int | tuple[int, ...],
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[ScalarT] | Any: ...
@overload  # ~inexact | timedelta64, 1d, axis: <single>
def mean[ScalarT: np.inexact | np.timedelta64](
    a: _ToArray1D[ScalarT],
    axis: int | tuple[int],
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ScalarT: ...
@overload  # ~inexact | timedelta64, 2d, axis: <single>
def mean[ScalarT: np.inexact | np.timedelta64](
    a: _ToArray2D[ScalarT],
    axis: int | tuple[int],
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> _Array1D[ScalarT]: ...
@overload  # ~inexact | timedelta64, 3d, axis: <single>
def mean[ScalarT: np.inexact | np.timedelta64](
    a: _ToArray3D[ScalarT],
    axis: int | tuple[int],
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> _Array2D[ScalarT]: ...
@overload  # ~inexact | timedelta64, 4d, axis: <single>
def mean[ScalarT: np.inexact | np.timedelta64](
    a: _ToArray4D[ScalarT],
    axis: int | tuple[int],
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> _Array3D[ScalarT]: ...
@overload  # ~inexact | timedelta64, ?d, axis: <given>
def mean[ScalarT: np.inexact | np.timedelta64](
    a: _ArrayLike[ScalarT],
    axis: int | tuple[int, ...],
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[ScalarT] | Any: ...
@overload  # ~inexact | timedelta64, keepdims=True
def mean[ArrayT: NDArray[np.inexact | np.timedelta64]](
    a: ArrayT,
    axis: int | tuple[int, ...] | None = None,
    dtype: None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ArrayT: ...
@overload  # ~object_, ?d, axis: <given>
def mean(
    a: _ArrayJustND[np.object_],
    axis: int | tuple[int, ...],
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.object_] | Any: ...
@overload  # ~object_, 1d, axis: <single>
def mean(
    a: _Array1D[np.object_],
    axis: int | tuple[int],
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> Any: ...
@overload  # ~object_, 2d, axis: <single>
def mean(
    a: _Array2D[np.object_],
    axis: int | tuple[int],
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> _Array1D[np.object_]: ...
@overload  # ~object_, 3d, axis: <single>
def mean(
    a: _Array3D[np.object_],
    axis: int | tuple[int],
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> _Array2D[np.object_]: ...
@overload  # ~object_, 4d, axis: <single>
def mean(
    a: _Array4D[np.object_],
    axis: int | tuple[int],
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> _Array3D[np.object_]: ...
@overload  # ~object_, ?d, axis: <given>
def mean(
    a: _ArrayLike[np.object_],
    axis: int | tuple[int, ...],
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[np.object_] | Any: ...
@overload  # ~object_, keepdims=True
def mean[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.object_]],
    axis: int | tuple[int, ...] | None = None,
    dtype: None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.ndarray[ShapeT, np.dtype[np.object_]]: ...
@overload  # dtype: ScalarT
def mean[ScalarT: np.generic](
    a: _ArrayLikeNumeric_co,
    axis: None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ScalarT: ...
@overload  # ?d, dtype: ScalarT, axis: <given>
def mean[ScalarT: np.generic](
    a: _ArrayJustND[np.number | np.bool | np.object_ | np.timedelta64],
    axis: int | tuple[int, ...],
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[ScalarT] | Any: ...
@overload  # 1d, dtype: ScalarT, axis: <single>
def mean[ScalarT: np.generic](
    a: _ToNumeric1D,
    axis: int | tuple[int],
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ScalarT: ...
@overload  # 2d, dtype: ScalarT, axis: <single>
def mean[ScalarT: np.generic](
    a: _ToNumeric2D,
    axis: int | tuple[int],
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> _Array1D[ScalarT]: ...
@overload  # 3d, dtype: ScalarT, axis: <single>
def mean[ScalarT: np.generic](
    a: _ToNumeric3D,
    axis: int | tuple[int],
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> _Array2D[ScalarT]: ...
@overload  # 4d, dtype: ScalarT, axis: <single>
def mean[ScalarT: np.generic](
    a: _ToNumeric4D,
    axis: int | tuple[int],
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> _Array3D[ScalarT]: ...
@overload  # ?d, dtype: ScalarT, axis: <given>
def mean[ScalarT: np.generic](
    a: _ArrayLikeNumeric_co,
    axis: int | tuple[int, ...],
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[ScalarT] | Any: ...
@overload  # dtype: ScalarT (keyword), keepdims=True
def mean[ShapeT: _Shape, ScalarT: np.generic](
    a: np.ndarray[ShapeT, np.dtype[np.number | np.bool | np.timedelta64 | np.object_]],
    axis: int | tuple[int, ...] | None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    keepdims: Literal[True],
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
@overload  # dtype: ScalarT (positional), keepdims=True
def mean[ShapeT: _Shape, ScalarT: np.generic](
    a: np.ndarray[ShapeT, np.dtype[np.number | np.bool | np.timedelta64 | np.object_]],
    axis: int | tuple[int, ...] | None,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    *,
    keepdims: Literal[True],
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
@overload  # dtype: ScalarT (keyword), keepdims=True
def mean[ScalarT: np.generic](
    a: _ArrayLikeNumeric_co,
    axis: int | tuple[int, ...] | None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    keepdims: Literal[True],
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[ScalarT]: ...
@overload  # out: ArrayT
def mean[ArrayT: np.ndarray](
    a: _ArrayLikeNumeric_co,
    axis: int | tuple[int, ...] | None = None,
    dtype: DTypeLike | None = None,
    *,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ArrayT: ...
@overload  # fallback
def mean(
    a: _ArrayLikeNumeric_co,
    axis: None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> Any: ...
@overload  # fallback, ?d, axis: <given>
def mean(
    a: _ArrayJustND[np.number | np.bool | np.object_ | np.timedelta64],
    axis: int | tuple[int, ...],
    dtype: DTypeLike | None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[Any] | Any: ...
@overload  # fallback, 1d, axis: <single>
def mean(
    a: _ToNumeric1D,
    axis: int | tuple[int],
    dtype: DTypeLike | None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> Any: ...
@overload  # fallback, 2d, axis: <single>
def mean(
    a: _ToNumeric2D,
    axis: int | tuple[int],
    dtype: DTypeLike | None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> _Array1D[Any]: ...
@overload  # fallback, 3d, axis: <single>
def mean(
    a: _ToNumeric3D,
    axis: int | tuple[int],
    dtype: DTypeLike | None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> _Array2D[Any]: ...
@overload  # fallback, 4d, axis: <single>
def mean(
    a: _ToNumeric4D,
    axis: int | tuple[int],
    dtype: DTypeLike | None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> _Array3D[Any]: ...
@overload  # fallback, ?d, axis: <given>
def mean(
    a: _ArrayLikeNumeric_co,
    axis: int | tuple[int, ...],
    dtype: DTypeLike | None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[Any] | Any: ...
@overload  # fallback, keepdims=True
def mean(
    a: _ArrayLikeNumeric_co,
    axis: int | tuple[int, ...] | None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    *,
    keepdims: Literal[True],
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[Any]: ...

#
@overload  # ~complex128 | +integer | ~object_ | +builtins.float
def std(
    a: _DualArrayLike[np.dtype[np.complex128 | np.integer | np.bool | np.object_], complex],
    axis: None = None,
    dtype: None = None,
    out: None = None,
    ddof: float = 0,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeFloat_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> np.float64: ...
@overload  # ~complex128 | +integer | +builtins.float, axis: <given>
def std(
    a: _DualArrayLike[np.dtype[np.complex128 | np.integer | np.bool], complex],
    axis: int | tuple[int, ...],
    dtype: None = None,
    out: None = None,
    ddof: float = 0,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeFloat_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> NDArray[np.float64]: ...
@overload  # +integer, keepdims=True
def std[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.complex128 | np.integer | np.bool]],
    axis: int | tuple[int, ...] | None = None,
    dtype: None = None,
    out: None = None,
    ddof: float = 0,
    *,
    keepdims: Literal[True],
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # ~floating | timedelta64
def std[ScalarT: np.floating | np.timedelta64](
    a: _ArrayLike[ScalarT],
    axis: None = None,
    dtype: None = None,
    out: None = None,
    ddof: float = 0,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> ScalarT: ...
@overload  # ~floating | timedelta64, axis: <given>
def std[ScalarT: np.floating | np.timedelta64](
    a: _ArrayLike[ScalarT],
    axis: int | tuple[int, ...],
    dtype: None = None,
    out: None = None,
    ddof: float = 0,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> NDArray[ScalarT]: ...
@overload  # ~object_, axis: <given>
def std(
    a: _ArrayLike[np.object_],
    axis: int | tuple[int, ...],
    dtype: None = None,
    out: None = None,
    ddof: float = 0,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> NDArray[np.object_]: ...
@overload  # ~floating | timedelta64, keepdims=True
def std[ArrayT: NDArray[np.floating | np.timedelta64]](
    a: ArrayT,
    axis: int | tuple[int, ...] | None = None,
    dtype: None = None,
    out: None = None,
    *,
    ddof: float = 0,
    keepdims: Literal[True],
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> ArrayT: ...
@overload  # ~object_, keepdims=True
def std[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.object_]],
    axis: int | tuple[int, ...] | None = None,
    dtype: None = None,
    out: None = None,
    *,
    ddof: float = 0,
    keepdims: Literal[True],
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> np.ndarray[ShapeT, np.dtype[np.object_]]: ...
@overload  # dtype: ScalarT
def std[ScalarT: np.generic](
    a: _ArrayLikeNumeric_co,
    axis: None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    ddof: float = 0,
    keepdims: Literal[False] | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> ScalarT: ...
@overload  # dtype: ScalarT (keyword), keepdims=True
def std[ShapeT: _Shape, ScalarT: np.generic](
    a: np.ndarray[ShapeT, np.dtype[np.number | np.bool | np.timedelta64 | np.object_]],
    axis: int | tuple[int, ...] | None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    ddof: float = 0,
    keepdims: Literal[True],
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
@overload  # dtype: ScalarT (positional), keepdims=True
def std[ShapeT: _Shape, ScalarT: np.generic](
    a: np.ndarray[ShapeT, np.dtype[np.number | np.bool | np.timedelta64 | np.object_]],
    axis: int | tuple[int, ...] | None,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    ddof: float = 0,
    *,
    keepdims: Literal[True],
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
@overload  # dtype: ScalarT (keyword), keepdims=True
def std[ScalarT: np.generic](
    a: _ArrayLikeNumeric_co,
    axis: int | tuple[int, ...] | None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    ddof: float = 0,
    keepdims: Literal[True],
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> NDArray[ScalarT]: ...
@overload  # axis: <given>, dtype: ScalarT
def std[ScalarT: np.generic](
    a: _ArrayLikeNumeric_co,
    axis: int | tuple[int, ...],
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    ddof: float = 0,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> NDArray[ScalarT]: ...
@overload  # out: ArrayT
def std[ArrayT: np.ndarray](
    a: _ArrayLikeNumeric_co,
    axis: int | tuple[int, ...] | None = None,
    dtype: DTypeLike | None = None,
    *,
    out: ArrayT,
    ddof: float = 0,
    keepdims: bool | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> ArrayT: ...
@overload  # fallback
def std(
    a: _ArrayLikeNumeric_co,
    axis: None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    ddof: float = 0,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> Any: ...
@overload  # fallback, axis: <given>
def std(
    a: _ArrayLikeNumeric_co,
    axis: int | tuple[int, ...],
    dtype: DTypeLike | None = None,
    out: None = None,
    ddof: float = 0,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> NDArray[Any]: ...
@overload  # fallback, keepdims=True
def std(
    a: _ArrayLikeNumeric_co,
    axis: int | tuple[int, ...] | None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    ddof: float = 0,
    *,
    keepdims: Literal[True],
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> NDArray[Any]: ...

# keep in sync with `std` above
@overload  # ~complex128 | +integer | ~object_ | +builtins.float
def var(
    a: _DualArrayLike[np.dtype[np.complex128 | np.integer | np.bool | np.object_], complex],
    axis: None = None,
    dtype: None = None,
    out: None = None,
    ddof: float = 0,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeFloat_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> np.float64: ...
@overload  # ~complex128 | +integer | +builtins.float, axis: <given>
def var(
    a: _DualArrayLike[np.dtype[np.complex128 | np.integer | np.bool], complex],
    axis: int | tuple[int, ...],
    dtype: None = None,
    out: None = None,
    ddof: float = 0,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeFloat_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> NDArray[np.float64]: ...
@overload  # +integer, keepdims=True
def var[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.complex128 | np.integer | np.bool]],
    axis: int | tuple[int, ...] | None = None,
    dtype: None = None,
    out: None = None,
    ddof: float = 0,
    *,
    keepdims: Literal[True],
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # ~floating | timedelta64
def var[ScalarT: np.floating | np.timedelta64](
    a: _ArrayLike[ScalarT],
    axis: None = None,
    dtype: None = None,
    out: None = None,
    ddof: float = 0,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> ScalarT: ...
@overload  # ~floating | timedelta64, axis: <given>
def var[ScalarT: np.floating | np.timedelta64](
    a: _ArrayLike[ScalarT],
    axis: int | tuple[int, ...],
    dtype: None = None,
    out: None = None,
    ddof: float = 0,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> NDArray[ScalarT]: ...
@overload  # ~object_, axis: <given>
def var(
    a: _ArrayLike[np.object_],
    axis: int | tuple[int, ...],
    dtype: None = None,
    out: None = None,
    ddof: float = 0,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> NDArray[np.object_]: ...
@overload  # ~floating | timedelta64, keepdims=True
def var[ArrayT: NDArray[np.floating | np.timedelta64]](
    a: ArrayT,
    axis: int | tuple[int, ...] | None = None,
    dtype: None = None,
    out: None = None,
    *,
    ddof: float = 0,
    keepdims: Literal[True],
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> ArrayT: ...
@overload  # ~object_, keepdims=True
def var[ShapeT: _Shape](
    a: np.ndarray[ShapeT, np.dtype[np.object_]],
    axis: int | tuple[int, ...] | None = None,
    dtype: None = None,
    out: None = None,
    *,
    ddof: float = 0,
    keepdims: Literal[True],
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> np.ndarray[ShapeT, np.dtype[np.object_]]: ...
@overload  # dtype: ScalarT
def var[ScalarT: np.generic](
    a: _ArrayLikeNumeric_co,
    axis: None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    ddof: float = 0,
    keepdims: Literal[False] | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> ScalarT: ...
@overload  # dtype: ScalarT (keyword), keepdims=True
def var[ShapeT: _Shape, ScalarT: np.generic](
    a: np.ndarray[ShapeT, np.dtype[np.number | np.bool | np.timedelta64 | np.object_]],
    axis: int | tuple[int, ...] | None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    ddof: float = 0,
    keepdims: Literal[True],
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
@overload  # dtype: ScalarT (positional), keepdims=True
def var[ShapeT: _Shape, ScalarT: np.generic](
    a: np.ndarray[ShapeT, np.dtype[np.number | np.bool | np.timedelta64 | np.object_]],
    axis: int | tuple[int, ...] | None,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    ddof: float = 0,
    *,
    keepdims: Literal[True],
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
@overload  # dtype: ScalarT (keyword), keepdims=True
def var[ScalarT: np.generic](
    a: _ArrayLikeNumeric_co,
    axis: int | tuple[int, ...] | None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    ddof: float = 0,
    keepdims: Literal[True],
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> NDArray[ScalarT]: ...
@overload  # axis: <given>, dtype: ScalarT
def var[ScalarT: np.generic](
    a: _ArrayLikeNumeric_co,
    axis: int | tuple[int, ...],
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    ddof: float = 0,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> NDArray[ScalarT]: ...
@overload  # out: ArrayT
def var[ArrayT: np.ndarray](
    a: _ArrayLikeNumeric_co,
    axis: int | tuple[int, ...] | None = None,
    dtype: DTypeLike | None = None,
    *,
    out: ArrayT,
    ddof: float = 0,
    keepdims: bool | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> ArrayT: ...
@overload  # fallback
def var(
    a: _ArrayLikeNumeric_co,
    axis: None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    ddof: float = 0,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> Any: ...
@overload  # fallback, axis: <given>
def var(
    a: _ArrayLikeNumeric_co,
    axis: int | tuple[int, ...],
    dtype: DTypeLike | None = None,
    out: None = None,
    ddof: float = 0,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> NDArray[Any]: ...
@overload  # fallback, keepdims=True
def var(
    a: _ArrayLikeNumeric_co,
    axis: int | tuple[int, ...] | None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    ddof: float = 0,
    *,
    keepdims: Literal[True],
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> NDArray[Any]: ...

round = around
