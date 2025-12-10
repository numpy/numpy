# ruff: noqa: ANN401
from _typeshed import Incomplete
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
    complexfloating,
    float16,
    floating,
    int64,
    int_,
    intp,
    object_,
    uint64,
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
    _ArrayLikeUInt_co,
    _BoolLike_co,
    _ComplexLike_co,
    _DTypeLike,
    _IntLike_co,
    _NestedSequence,
    _NumberLike_co,
    _ScalarLike_co,
    _Shape,
    _ShapeLike,
)

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

#
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

@overload
def transpose[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    axes: _ShapeLike | None = None,
) -> NDArray[ScalarT]: ...
@overload
def transpose(
    a: ArrayLike,
    axes: _ShapeLike | None = None,
) -> NDArray[Any]: ...

@overload
def matrix_transpose[ScalarT: np.generic](x: _ArrayLike[ScalarT], /) -> NDArray[ScalarT]: ...
@overload
def matrix_transpose(x: ArrayLike, /) -> NDArray[Any]: ...

#
@overload
def partition[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    kth: _ArrayLikeInt,
    axis: SupportsIndex | None = -1,
    kind: _PartitionKind = "introselect",
    order: None = None,
) -> NDArray[ScalarT]: ...
@overload
def partition(
    a: _ArrayLike[np.void],
    kth: _ArrayLikeInt,
    axis: SupportsIndex | None = -1,
    kind: _PartitionKind = "introselect",
    order: str | Sequence[str] | None = None,
) -> NDArray[np.void]: ...
@overload
def partition(
    a: ArrayLike,
    kth: _ArrayLikeInt,
    axis: SupportsIndex | None = -1,
    kind: _PartitionKind = "introselect",
    order: str | Sequence[str] | None = None,
) -> NDArray[Any]: ...

#
def argpartition(
    a: ArrayLike,
    kth: _ArrayLikeInt,
    axis: SupportsIndex | None = -1,
    kind: _PartitionKind = "introselect",
    order: str | Sequence[str] | None = None,
) -> NDArray[intp]: ...

#
@overload
def sort[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    axis: SupportsIndex | None = -1,
    kind: _SortKind | None = None,
    order: str | Sequence[str] | None = None,
    *,
    stable: bool | None = None,
) -> NDArray[ScalarT]: ...
@overload
def sort(
    a: ArrayLike,
    axis: SupportsIndex | None = -1,
    kind: _SortKind | None = None,
    order: str | Sequence[str] | None = None,
    *,
    stable: bool | None = None,
) -> NDArray[Any]: ...

def argsort(
    a: ArrayLike,
    axis: SupportsIndex | None = -1,
    kind: _SortKind | None = None,
    order: str | Sequence[str] | None = None,
    *,
    stable: bool | None = None,
) -> NDArray[intp]: ...

@overload
def argmax(
    a: ArrayLike,
    axis: None = None,
    out: None = None,
    *,
    keepdims: Literal[False] | _NoValueType = ...,
) -> intp: ...
@overload
def argmax(
    a: ArrayLike,
    axis: SupportsIndex | None = None,
    out: None = None,
    *,
    keepdims: bool | _NoValueType = ...,
) -> Any: ...
@overload
def argmax[BoolOrIntArrayT: NDArray[np.integer | np.bool]](
    a: ArrayLike,
    axis: SupportsIndex | None,
    out: BoolOrIntArrayT,
    *,
    keepdims: bool | _NoValueType = ...,
) -> BoolOrIntArrayT: ...
@overload
def argmax[BoolOrIntArrayT: NDArray[np.integer | np.bool]](
    a: ArrayLike,
    axis: SupportsIndex | None = None,
    *,
    out: BoolOrIntArrayT,
    keepdims: bool | _NoValueType = ...,
) -> BoolOrIntArrayT: ...

@overload
def argmin(
    a: ArrayLike,
    axis: None = None,
    out: None = None,
    *,
    keepdims: Literal[False] | _NoValueType = ...,
) -> intp: ...
@overload
def argmin(
    a: ArrayLike,
    axis: SupportsIndex | None = None,
    out: None = None,
    *,
    keepdims: bool | _NoValueType = ...,
) -> Any: ...
@overload
def argmin[BoolOrIntArrayT: NDArray[np.integer | np.bool]](
    a: ArrayLike,
    axis: SupportsIndex | None,
    out: BoolOrIntArrayT,
    *,
    keepdims: bool | _NoValueType = ...,
) -> BoolOrIntArrayT: ...
@overload
def argmin[BoolOrIntArrayT: NDArray[np.integer | np.bool]](
    a: ArrayLike,
    axis: SupportsIndex | None = None,
    *,
    out: BoolOrIntArrayT,
    keepdims: bool | _NoValueType = ...,
) -> BoolOrIntArrayT: ...

# TODO: Fix overlapping overloads: https://github.com/numpy/numpy/issues/27032
@overload
def searchsorted(
    a: ArrayLike,
    v: _ScalarLike_co,
    side: _SortSide = "left",
    sorter: _ArrayLikeInt_co | None = None,  # 1D int array
) -> intp: ...
@overload
def searchsorted(
    a: ArrayLike,
    v: ArrayLike,
    side: _SortSide = "left",
    sorter: _ArrayLikeInt_co | None = None,  # 1D int array
) -> NDArray[intp]: ...

#
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
@overload
def diagonal[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    offset: SupportsIndex = 0,
    axis1: SupportsIndex = 0,
    axis2: SupportsIndex = 1,  # >= 2D array
) -> NDArray[ScalarT]: ...
@overload
def diagonal(
    a: ArrayLike,
    offset: SupportsIndex = 0,
    axis1: SupportsIndex = 0,
    axis2: SupportsIndex = 1,  # >= 2D array
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

def nonzero(a: _ArrayLike[Any]) -> tuple[_Array1D[np.intp], ...]: ...

# this prevents `Any` from being returned with Pyright
@overload
def shape(a: _SupportsShape[Never]) -> _AnyShape: ...
@overload
def shape[ShapeT: _Shape](a: _SupportsShape[ShapeT]) -> ShapeT: ...
@overload
def shape(a: _PyScalar) -> tuple[()]: ...
# `collections.abc.Sequence` can't be used hesre, since `bytes` and `str` are
# subtypes of it, which would make the return types incompatible.
@overload
def shape(a: _PyArray[_PyScalar]) -> _1D: ...
@overload
def shape(a: _PyArray[_PyArray[_PyScalar]]) -> _2D: ...
# this overload will be skipped by typecheckers that don't support PEP 688
@overload
def shape(a: memoryview | bytearray) -> _1D: ...
@overload
def shape(a: ArrayLike) -> _AnyShape: ...

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

# TODO: Fix overlapping overloads: https://github.com/numpy/numpy/issues/27032
@overload
def clip[ScalarT: np.generic](
    a: ScalarT,
    a_min: ArrayLike | _NoValueType | None = ...,
    a_max: ArrayLike | _NoValueType | None = ...,
    out: None = None,
    *,
    min: ArrayLike | _NoValueType | None = ...,
    max: ArrayLike | _NoValueType | None = ...,
    dtype: None = None,
    **kwargs: Unpack[_UFuncKwargs],
) -> ScalarT: ...
@overload
def clip(
    a: _ScalarLike_co,
    a_min: ArrayLike | _NoValueType | None = ...,
    a_max: ArrayLike | _NoValueType | None = ...,
    out: None = None,
    *,
    min: ArrayLike | _NoValueType | None = ...,
    max: ArrayLike | _NoValueType | None = ...,
    dtype: None = None,
    **kwargs: Unpack[_UFuncKwargs],
) -> Any: ...
@overload
def clip[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    a_min: ArrayLike | _NoValueType | None = ...,
    a_max: ArrayLike | _NoValueType | None = ...,
    out: None = None,
    *,
    min: ArrayLike | _NoValueType | None = ...,
    max: ArrayLike | _NoValueType | None = ...,
    dtype: None = None,
    **kwargs: Unpack[_UFuncKwargs],
) -> NDArray[ScalarT]: ...
@overload
def clip(
    a: ArrayLike,
    a_min: ArrayLike | _NoValueType | None = ...,
    a_max: ArrayLike | _NoValueType | None = ...,
    out: None = None,
    *,
    min: ArrayLike | _NoValueType | None = ...,
    max: ArrayLike | _NoValueType | None = ...,
    dtype: None = None,
    **kwargs: Unpack[_UFuncKwargs],
) -> NDArray[Any]: ...
@overload
def clip[ArrayT: np.ndarray](
    a: ArrayLike,
    a_min: ArrayLike | None,
    a_max: ArrayLike | None,
    out: ArrayT,
    *,
    min: ArrayLike | _NoValueType | None = ...,
    max: ArrayLike | _NoValueType | None = ...,
    dtype: DTypeLike | None = None,
    **kwargs: Unpack[_UFuncKwargs],
) -> ArrayT: ...
@overload
def clip[ArrayT: np.ndarray](
    a: ArrayLike,
    a_min: ArrayLike | _NoValueType | None = ...,
    a_max: ArrayLike | _NoValueType | None = ...,
    *,
    out: ArrayT,
    min: ArrayLike | _NoValueType | None = ...,
    max: ArrayLike | _NoValueType | None = ...,
    dtype: DTypeLike | None = None,
    **kwargs: Unpack[_UFuncKwargs],
) -> ArrayT: ...
@overload
def clip(
    a: ArrayLike,
    a_min: ArrayLike | _NoValueType | None = ...,
    a_max: ArrayLike | _NoValueType | None = ...,
    out: None = None,
    *,
    min: ArrayLike | _NoValueType | None = ...,
    max: ArrayLike | _NoValueType | None = ...,
    dtype: DTypeLike | None = None,
    **kwargs: Unpack[_UFuncKwargs],
) -> Any: ...

@overload
def sum[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    axis: None = None,
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ScalarT: ...
@overload
def sum[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    axis: None = None,
    dtype: None = None,
    out: None = None,
    keepdims: bool | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ScalarT | NDArray[ScalarT]: ...
@overload
def sum[ScalarT: np.generic](
    a: ArrayLike,
    axis: None,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ScalarT: ...
@overload
def sum[ScalarT: np.generic](
    a: ArrayLike,
    axis: None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ScalarT: ...
@overload
def sum[ScalarT: np.generic](
    a: ArrayLike,
    axis: _ShapeLike | None,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    keepdims: bool | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ScalarT | NDArray[ScalarT]: ...
@overload
def sum[ScalarT: np.generic](
    a: ArrayLike,
    axis: _ShapeLike | None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    keepdims: bool | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ScalarT | NDArray[ScalarT]: ...
@overload
def sum(
    a: ArrayLike,
    axis: _ShapeLike | None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    keepdims: bool | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> Any: ...
@overload
def sum[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: _ShapeLike | None,
    dtype: DTypeLike | None,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ArrayT: ...
@overload
def sum[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: _ShapeLike | None = None,
    dtype: DTypeLike | None = None,
    *,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ArrayT: ...

# keep in sync with `any`
@overload
def all(
    a: ArrayLike | None,
    axis: None = None,
    out: None = None,
    keepdims: Literal[False, 0] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.bool: ...
@overload
def all(
    a: ArrayLike | None,
    axis: int | tuple[int, ...] | None = None,
    out: None = None,
    keepdims: _BoolLike_co | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> Incomplete: ...
@overload
def all[ArrayT: np.ndarray](
    a: ArrayLike | None,
    axis: int | tuple[int, ...] | None,
    out: ArrayT,
    keepdims: _BoolLike_co | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ArrayT: ...
@overload
def all[ArrayT: np.ndarray](
    a: ArrayLike | None,
    axis: int | tuple[int, ...] | None = None,
    *,
    out: ArrayT,
    keepdims: _BoolLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ArrayT: ...

# keep in sync with `all`
@overload
def any(
    a: ArrayLike | None,
    axis: None = None,
    out: None = None,
    keepdims: Literal[False, 0] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.bool: ...
@overload
def any(
    a: ArrayLike | None,
    axis: int | tuple[int, ...] | None = None,
    out: None = None,
    keepdims: _BoolLike_co | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> Incomplete: ...
@overload
def any[ArrayT: np.ndarray](
    a: ArrayLike | None,
    axis: int | tuple[int, ...] | None,
    out: ArrayT,
    keepdims: _BoolLike_co | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ArrayT: ...
@overload
def any[ArrayT: np.ndarray](
    a: ArrayLike | None,
    axis: int | tuple[int, ...] | None = None,
    *,
    out: ArrayT,
    keepdims: _BoolLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ArrayT: ...

#
@overload
def cumsum[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    axis: SupportsIndex | None = None,
    dtype: None = None,
    out: None = None,
) -> NDArray[ScalarT]: ...
@overload
def cumsum(
    a: ArrayLike,
    axis: SupportsIndex | None = None,
    dtype: None = None,
    out: None = None,
) -> NDArray[Any]: ...
@overload
def cumsum[ScalarT: np.generic](
    a: ArrayLike,
    axis: SupportsIndex | None,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
) -> NDArray[ScalarT]: ...
@overload
def cumsum[ScalarT: np.generic](
    a: ArrayLike,
    axis: SupportsIndex | None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
) -> NDArray[ScalarT]: ...
@overload
def cumsum(
    a: ArrayLike,
    axis: SupportsIndex | None = None,
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

@overload
def cumulative_sum[ScalarT: np.generic](
    x: _ArrayLike[ScalarT],
    /,
    *,
    axis: SupportsIndex | None = None,
    dtype: None = None,
    out: None = None,
    include_initial: bool = False,
) -> NDArray[ScalarT]: ...
@overload
def cumulative_sum(
    x: ArrayLike,
    /,
    *,
    axis: SupportsIndex | None = None,
    dtype: None = None,
    out: None = None,
    include_initial: bool = False,
) -> NDArray[Any]: ...
@overload
def cumulative_sum[ScalarT: np.generic](
    x: ArrayLike,
    /,
    *,
    axis: SupportsIndex | None = None,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    include_initial: bool = False,
) -> NDArray[ScalarT]: ...
@overload
def cumulative_sum(
    x: ArrayLike,
    /,
    *,
    axis: SupportsIndex | None = None,
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

@overload
def ptp[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    axis: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
) -> ScalarT: ...
@overload
def ptp(
    a: ArrayLike,
    axis: _ShapeLike | None = None,
    out: None = None,
    keepdims: bool | _NoValueType = ...,
) -> Any: ...
@overload
def ptp[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: _ShapeLike | None,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
) -> ArrayT: ...
@overload
def ptp[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: _ShapeLike | None = None,
    *,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
) -> ArrayT: ...

@overload
def amax[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    axis: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ScalarT: ...
@overload
def amax(
    a: ArrayLike,
    axis: _ShapeLike | None = None,
    out: None = None,
    keepdims: bool | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> Any: ...
@overload
def amax[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: _ShapeLike | None,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ArrayT: ...
@overload
def amax[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: _ShapeLike | None = None,
    *,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ArrayT: ...

@overload
def amin[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    axis: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ScalarT: ...
@overload
def amin(
    a: ArrayLike,
    axis: _ShapeLike | None = None,
    out: None = None,
    keepdims: bool | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> Any: ...
@overload
def amin[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: _ShapeLike | None,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ArrayT: ...
@overload
def amin[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: _ShapeLike | None = None,
    *,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ArrayT: ...

# TODO: `np.prod()``: For object arrays `initial` does not necessarily
# have to be a numerical scalar.
# The only requirement is that it is compatible
# with the `.__mul__()` method(s) of the passed array's elements.
# Note that the same situation holds for all wrappers around
# `np.ufunc.reduce`, e.g. `np.sum()` (`.__add__()`).
# TODO: Fix overlapping overloads: https://github.com/numpy/numpy/issues/27032
@overload
def prod(
    a: _ArrayLikeBool_co,
    axis: None = None,
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> int_: ...
@overload
def prod(
    a: _ArrayLikeUInt_co,
    axis: None = None,
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> uint64: ...
@overload
def prod(
    a: _ArrayLikeInt_co,
    axis: None = None,
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> int64: ...
@overload
def prod(
    a: _ArrayLikeFloat_co,
    axis: None = None,
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> floating: ...
@overload
def prod(
    a: _ArrayLikeComplex_co,
    axis: None = None,
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> complexfloating: ...
@overload
def prod(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: _ShapeLike | None = None,
    dtype: None = None,
    out: None = None,
    keepdims: bool | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> Any: ...
@overload
def prod[ScalarT: np.generic](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: None,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ScalarT: ...
@overload
def prod[ScalarT: np.generic](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ScalarT: ...
@overload
def prod(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: _ShapeLike | None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    keepdims: bool | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> Any: ...
@overload
def prod[ArrayT: np.ndarray](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: _ShapeLike | None,
    dtype: DTypeLike | None,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ArrayT: ...
@overload
def prod[ArrayT: np.ndarray](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: _ShapeLike | None = None,
    dtype: DTypeLike | None = None,
    *,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
    initial: _NumberLike_co | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ArrayT: ...

# TODO: Fix overlapping overloads: https://github.com/numpy/numpy/issues/27032
@overload
def cumprod(
    a: _ArrayLikeBool_co,
    axis: SupportsIndex | None = None,
    dtype: None = None,
    out: None = None,
) -> NDArray[int_]: ...
@overload
def cumprod(
    a: _ArrayLikeUInt_co,
    axis: SupportsIndex | None = None,
    dtype: None = None,
    out: None = None,
) -> NDArray[uint64]: ...
@overload
def cumprod(
    a: _ArrayLikeInt_co,
    axis: SupportsIndex | None = None,
    dtype: None = None,
    out: None = None,
) -> NDArray[int64]: ...
@overload
def cumprod(
    a: _ArrayLikeFloat_co,
    axis: SupportsIndex | None = None,
    dtype: None = None,
    out: None = None,
) -> NDArray[floating]: ...
@overload
def cumprod(
    a: _ArrayLikeComplex_co,
    axis: SupportsIndex | None = None,
    dtype: None = None,
    out: None = None,
) -> NDArray[complexfloating]: ...
@overload
def cumprod(
    a: _ArrayLikeObject_co,
    axis: SupportsIndex | None = None,
    dtype: None = None,
    out: None = None,
) -> NDArray[object_]: ...
@overload
def cumprod[ScalarT: np.generic](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: SupportsIndex | None,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
) -> NDArray[ScalarT]: ...
@overload
def cumprod[ScalarT: np.generic](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: SupportsIndex | None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
) -> NDArray[ScalarT]: ...
@overload
def cumprod(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: SupportsIndex | None = None,
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

# TODO: Fix overlapping overloads: https://github.com/numpy/numpy/issues/27032
@overload
def cumulative_prod(
    x: _ArrayLikeBool_co,
    /,
    *,
    axis: SupportsIndex | None = None,
    dtype: None = None,
    out: None = None,
    include_initial: bool = False,
) -> NDArray[int_]: ...
@overload
def cumulative_prod(
    x: _ArrayLikeUInt_co,
    /,
    *,
    axis: SupportsIndex | None = None,
    dtype: None = None,
    out: None = None,
    include_initial: bool = False,
) -> NDArray[uint64]: ...
@overload
def cumulative_prod(
    x: _ArrayLikeInt_co,
    /,
    *,
    axis: SupportsIndex | None = None,
    dtype: None = None,
    out: None = None,
    include_initial: bool = False,
) -> NDArray[int64]: ...
@overload
def cumulative_prod(
    x: _ArrayLikeFloat_co,
    /,
    *,
    axis: SupportsIndex | None = None,
    dtype: None = None,
    out: None = None,
    include_initial: bool = False,
) -> NDArray[floating]: ...
@overload
def cumulative_prod(
    x: _ArrayLikeComplex_co,
    /,
    *,
    axis: SupportsIndex | None = None,
    dtype: None = None,
    out: None = None,
    include_initial: bool = False,
) -> NDArray[complexfloating]: ...
@overload
def cumulative_prod(
    x: _ArrayLikeObject_co,
    /,
    *,
    axis: SupportsIndex | None = None,
    dtype: None = None,
    out: None = None,
    include_initial: bool = False,
) -> NDArray[object_]: ...
@overload
def cumulative_prod[ScalarT: np.generic](
    x: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    /,
    *,
    axis: SupportsIndex | None = None,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    include_initial: bool = False,
) -> NDArray[ScalarT]: ...
@overload
def cumulative_prod(
    x: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    /,
    *,
    axis: SupportsIndex | None = None,
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

# TODO: Fix overlapping overloads: https://github.com/numpy/numpy/issues/27032
@overload
def around(
    a: _BoolLike_co,
    decimals: SupportsIndex = 0,
    out: None = None,
) -> float16: ...
@overload
def around[NumberOrObjectT: np.number | np.object_](
    a: NumberOrObjectT,
    decimals: SupportsIndex = 0,
    out: None = None,
) -> NumberOrObjectT: ...
@overload
def around(
    a: _ComplexLike_co | object_,
    decimals: SupportsIndex = 0,
    out: None = None,
) -> Any: ...
@overload
def around(
    a: _ArrayLikeBool_co,
    decimals: SupportsIndex = 0,
    out: None = None,
) -> NDArray[float16]: ...
@overload
def around[NumberOrObjectT: np.number | np.object_](
    a: _ArrayLike[NumberOrObjectT],
    decimals: SupportsIndex = 0,
    out: None = None,
) -> NDArray[NumberOrObjectT]: ...
@overload
def around(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    decimals: SupportsIndex = 0,
    out: None = None,
) -> NDArray[Any]: ...
@overload
def around[ArrayT: np.ndarray](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    decimals: SupportsIndex,
    out: ArrayT,
) -> ArrayT: ...
@overload
def around[ArrayT: np.ndarray](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    decimals: SupportsIndex = 0,
    *,
    out: ArrayT,
) -> ArrayT: ...

# TODO: Fix overlapping overloads: https://github.com/numpy/numpy/issues/27032
@overload
def mean(
    a: _ArrayLikeFloat_co,
    axis: None = None,
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> floating: ...
@overload
def mean(
    a: _ArrayLikeComplex_co,
    axis: None = None,
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> complexfloating: ...
@overload
def mean(
    a: _ArrayLike[np.timedelta64],
    axis: None = None,
    dtype: None = None,
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> np.timedelta64: ...
@overload
def mean[ArrayT: np.ndarray](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: _ShapeLike | None,
    dtype: DTypeLike | None,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ArrayT: ...
@overload
def mean[ArrayT: np.ndarray](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: _ShapeLike | None = None,
    dtype: DTypeLike | None = None,
    *,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ArrayT: ...
@overload
def mean[ScalarT: np.generic](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: None,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ScalarT: ...
@overload
def mean[ScalarT: np.generic](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    keepdims: Literal[False] | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ScalarT: ...
@overload
def mean[ScalarT: np.generic](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: _ShapeLike | None,
    dtype: _DTypeLike[ScalarT],
    out: None,
    keepdims: Literal[True, 1],
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> NDArray[ScalarT]: ...
@overload
def mean[ScalarT: np.generic](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: _ShapeLike | None,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    *,
    keepdims: bool | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ScalarT | NDArray[ScalarT]: ...
@overload
def mean[ScalarT: np.generic](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: _ShapeLike | None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    keepdims: bool | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> ScalarT | NDArray[ScalarT]: ...
@overload
def mean(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: _ShapeLike | None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    keepdims: bool | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
) -> Incomplete: ...

@overload
def std(
    a: _ArrayLikeComplex_co,
    axis: None = None,
    dtype: None = None,
    out: None = None,
    ddof: float = 0,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> floating: ...
@overload
def std(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: _ShapeLike | None = None,
    dtype: None = None,
    out: None = None,
    ddof: float = 0,
    keepdims: bool | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _ArrayLikeObject_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> Any: ...
@overload
def std[ScalarT: np.generic](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: None,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    ddof: float = 0,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _ArrayLikeObject_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> ScalarT: ...
@overload
def std[ScalarT: np.generic](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    ddof: float = 0,
    keepdims: Literal[False] | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _ArrayLikeObject_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> ScalarT: ...
@overload
def std(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: _ShapeLike | None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    ddof: float = 0,
    keepdims: bool | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _ArrayLikeObject_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> Any: ...
@overload
def std[ArrayT: np.ndarray](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: _ShapeLike | None,
    dtype: DTypeLike | None,
    out: ArrayT,
    ddof: float = 0,
    keepdims: bool | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _ArrayLikeObject_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> ArrayT: ...
@overload
def std[ArrayT: np.ndarray](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: _ShapeLike | None = None,
    dtype: DTypeLike | None = None,
    *,
    out: ArrayT,
    ddof: float = 0,
    keepdims: bool | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _ArrayLikeObject_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> ArrayT: ...

@overload
def var(
    a: _ArrayLikeComplex_co,
    axis: None = None,
    dtype: None = None,
    out: None = None,
    ddof: float = 0,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> floating: ...
@overload
def var(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: _ShapeLike | None = None,
    dtype: None = None,
    out: None = None,
    ddof: float = 0,
    keepdims: bool | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _ArrayLikeObject_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> Any: ...
@overload
def var[ScalarT: np.generic](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: None,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    ddof: float = 0,
    keepdims: Literal[False] | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _ArrayLikeObject_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> ScalarT: ...
@overload
def var[ScalarT: np.generic](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    out: None = None,
    ddof: float = 0,
    keepdims: Literal[False] | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _ArrayLikeObject_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> ScalarT: ...
@overload
def var(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: _ShapeLike | None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    ddof: float = 0,
    keepdims: bool | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _ArrayLikeObject_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> Any: ...
@overload
def var[ArrayT: np.ndarray](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: _ShapeLike | None,
    dtype: DTypeLike | None,
    out: ArrayT,
    ddof: float = 0,
    keepdims: bool | _NoValueType = ...,
    *,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _ArrayLikeObject_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> ArrayT: ...
@overload
def var[ArrayT: np.ndarray](
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: _ShapeLike | None = None,
    dtype: DTypeLike | None = None,
    *,
    out: ArrayT,
    ddof: float = 0,
    keepdims: bool | _NoValueType = ...,
    where: _ArrayLikeBool_co | _NoValueType = ...,
    mean: _ArrayLikeComplex_co | _ArrayLikeObject_co | _NoValueType = ...,
    correction: float | _NoValueType = ...,
) -> ArrayT: ...

max = amax
min = amin
round = around
