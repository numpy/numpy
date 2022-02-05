import datetime as dt
from collections.abc import Sequence
from typing import Union, Any, overload, TypeVar, Literal

from numpy import (
    ndarray,
    number,
    intp,
    bool_,
    generic,
    _OrderKACF,
    _OrderACF,
    _ModeKind,
    _PartitionKind,
    _SortKind,
    _SortSide,
)
from numpy.typing import (
    DTypeLike,
    ArrayLike,
    _ArrayLike,
    NDArray,
    _ShapeLike,
    _Shape,
    _ArrayLikeBool_co,
    _ArrayLikeInt_co,
    _IntLike_co,
    _NumberLike_co,
    _ScalarLike_co,
)

_SCT = TypeVar("_SCT", bound=generic)
_ArrayType = TypeVar("_ArrayType", bound=NDArray[Any])

__all__: list[str]

def take(
    a: ArrayLike,
    indices: _ArrayLikeInt_co,
    axis: None | int = ...,
    out: None | ndarray = ...,
    mode: _ModeKind = ...,
) -> Any: ...

@overload
def reshape(
    a: _ArrayLike[_SCT],
    newshape: _ShapeLike,
    order: _OrderACF = ...,
) -> NDArray[_SCT]: ...
@overload
def reshape(
    a: ArrayLike,
    newshape: _ShapeLike,
    order: _OrderACF = ...,
) -> NDArray[Any]: ...

@overload
def choose(
    a: _IntLike_co,
    choices: ArrayLike,
    out: None = ...,
    mode: _ModeKind = ...,
) -> Any: ...
@overload
def choose(
    a: _ArrayLikeInt_co,
    choices: _ArrayLike[_SCT],
    out: None = ...,
    mode: _ModeKind = ...,
) -> NDArray[_SCT]: ...
@overload
def choose(
    a: _ArrayLikeInt_co,
    choices: ArrayLike,
    out: None = ...,
    mode: _ModeKind = ...,
) -> NDArray[Any]: ...
@overload
def choose(
    a: _ArrayLikeInt_co,
    choices: ArrayLike,
    out: _ArrayType = ...,
    mode: _ModeKind = ...,
) -> _ArrayType: ...

@overload
def repeat(
    a: _ArrayLike[_SCT],
    repeats: _ArrayLikeInt_co,
    axis: None | int = ...,
) -> NDArray[_SCT]: ...
@overload
def repeat(
    a: ArrayLike,
    repeats: _ArrayLikeInt_co,
    axis: None | int = ...,
) -> NDArray[Any]: ...

def put(
    a: NDArray[Any],
    ind: _ArrayLikeInt_co,
    v: ArrayLike,
    mode: _ModeKind = ...,
) -> None: ...

@overload
def swapaxes(
    a: _ArrayLike[_SCT],
    axis1: int,
    axis2: int,
) -> NDArray[_SCT]: ...
@overload
def swapaxes(
    a: ArrayLike,
    axis1: int,
    axis2: int,
) -> NDArray[Any]: ...

@overload
def transpose(
    a: _ArrayLike[_SCT],
    axes: None | _ShapeLike = ...
) -> NDArray[_SCT]: ...
@overload
def transpose(
    a: ArrayLike,
    axes: None | _ShapeLike = ...
) -> NDArray[Any]: ...

@overload
def partition(
    a: _ArrayLike[_SCT],
    kth: _ArrayLikeInt_co,
    axis: None | int = ...,
    kind: _PartitionKind = ...,
    order: None | str | Sequence[str] = ...,
) -> NDArray[_SCT]: ...
@overload
def partition(
    a: ArrayLike,
    kth: _ArrayLikeInt_co,
    axis: None | int = ...,
    kind: _PartitionKind = ...,
    order: None | str | Sequence[str] = ...,
) -> NDArray[Any]: ...

def argpartition(
    a: ArrayLike,
    kth: _ArrayLikeInt_co,
    axis: None | int = ...,
    kind: _PartitionKind = ...,
    order: None | str | Sequence[str] = ...,
) -> NDArray[intp]: ...

@overload
def sort(
    a: _ArrayLike[_SCT],
    axis: None | int = ...,
    kind: None | _SortKind = ...,
    order: None | str | Sequence[str] = ...,
) -> NDArray[_SCT]: ...
@overload
def sort(
    a: ArrayLike,
    axis: None | int = ...,
    kind: None | _SortKind = ...,
    order: None | str | Sequence[str] = ...,
) -> NDArray[Any]: ...

def argsort(
    a: ArrayLike,
    axis: None | int = ...,
    kind: None | _SortKind = ...,
    order: None | str | Sequence[str] = ...,
) -> NDArray[intp]: ...

@overload
def argmax(
    a: ArrayLike,
    axis: None = ...,
    out: None | ndarray = ...,
    *,
    keepdims: Literal[False] = ...,
) -> intp: ...
@overload
def argmax(
    a: ArrayLike,
    axis: None | int = ...,
    out: None | ndarray = ...,
    *,
    keepdims: bool = ...,
) -> Any: ...

@overload
def argmin(
    a: ArrayLike,
    axis: None = ...,
    out: None | ndarray = ...,
    *,
    keepdims: Literal[False] = ...,
) -> intp: ...
@overload
def argmin(
    a: ArrayLike,
    axis: None | int = ...,
    out: None | ndarray = ...,
    *,
    keepdims: bool = ...,
) -> Any: ...

@overload
def searchsorted(
    a: ArrayLike,
    v: _ScalarLike_co,
    side: _SortSide = ...,
    sorter: None | _ArrayLikeInt_co = ...,  # 1D int array
) -> intp: ...
@overload
def searchsorted(
    a: ArrayLike,
    v: ArrayLike,
    side: _SortSide = ...,
    sorter: None | _ArrayLikeInt_co = ...,  # 1D int array
) -> NDArray[intp]: ...

@overload
def resize(
    a: _ArrayLike[_SCT],
    new_shape: _ShapeLike,
) -> NDArray[_SCT]: ...
@overload
def resize(
    a: ArrayLike,
    new_shape: _ShapeLike,
) -> NDArray[Any]: ...

@overload
def squeeze(
    a: _SCT,
    axis: None | _ShapeLike = ...,
) -> _SCT: ...
@overload
def squeeze(
    a: _ArrayLike[_SCT],
    axis: None | _ShapeLike = ...,
) -> NDArray[_SCT]: ...
@overload
def squeeze(
    a: ArrayLike,
    axis: None | _ShapeLike = ...,
) -> NDArray[Any]: ...

@overload
def diagonal(
    a: _ArrayLike[_SCT],
    offset: int = ...,
    axis1: int = ...,
    axis2: int = ...,  # >= 2D array
) -> NDArray[_SCT]: ...
@overload
def diagonal(
    a: ArrayLike,
    offset: int = ...,
    axis1: int = ...,
    axis2: int = ...,  # >= 2D array
) -> NDArray[Any]: ...

def trace(
    a: ArrayLike,  # >= 2D array
    offset: int = ...,
    axis1: int = ...,
    axis2: int = ...,
    dtype: DTypeLike = ...,
    out: None | ndarray = ...,
) -> Any: ...

@overload
def ravel(a: _ArrayLike[_SCT], order: _OrderKACF = ...) -> NDArray[_SCT]: ...
@overload
def ravel(a: ArrayLike, order: _OrderKACF = ...) -> NDArray[Any]: ...

def nonzero(a: ArrayLike) -> tuple[NDArray[intp], ...]: ...

def shape(a: ArrayLike) -> _Shape: ...

@overload
def compress(
    condition: _ArrayLikeBool_co,  # 1D bool array
    a: _ArrayLike[_SCT],
    axis: None | int = ...,
    out: None = ...,
) -> NDArray[_SCT]: ...
@overload
def compress(
    condition: _ArrayLikeBool_co,  # 1D bool array
    a: ArrayLike,
    axis: None | int = ...,
    out: None = ...,
) -> NDArray[Any]: ...
@overload
def compress(
    condition: _ArrayLikeBool_co,  # 1D bool array
    a: ArrayLike,
    axis: None | int = ...,
    out: _ArrayType = ...,
) -> _ArrayType: ...

@overload
def clip(
    a: ArrayLike,
    a_min: ArrayLike,
    a_max: None | ArrayLike,
    out: None | ndarray = ...,
    **kwargs: Any,
) -> Any: ...
@overload
def clip(
    a: ArrayLike,
    a_min: None,
    a_max: ArrayLike,
    out: None | ndarray = ...,
    **kwargs: Any,
) -> Any: ...

def sum(
    a: ArrayLike,
    axis: _ShapeLike = ...,
    dtype: DTypeLike = ...,
    out: None | ndarray = ...,
    keepdims: bool = ...,
    initial: _NumberLike_co = ...,
    where: _ArrayLikeBool_co = ...,
) -> Any: ...

@overload
def all(
    a: ArrayLike,
    axis: None = ...,
    out: None = ...,
    keepdims: Literal[False] = ...,
    *,
    where: _ArrayLikeBool_co = ...,
) -> bool_: ...
@overload
def all(
    a: ArrayLike,
    axis: None | _ShapeLike = ...,
    out: None | ndarray = ...,
    keepdims: bool = ...,
    *,
    where: _ArrayLikeBool_co = ...,
) -> Any: ...

@overload
def any(
    a: ArrayLike,
    axis: None = ...,
    out: None = ...,
    keepdims: Literal[False] = ...,
    *,
    where: _ArrayLikeBool_co = ...,
) -> bool_: ...
@overload
def any(
    a: ArrayLike,
    axis: None | _ShapeLike = ...,
    out: None | ndarray = ...,
    keepdims: bool = ...,
    *,
    where: _ArrayLikeBool_co = ...,
) -> Any: ...

def cumsum(
    a: ArrayLike,
    axis: None | int = ...,
    dtype: DTypeLike = ...,
    out: None | ndarray = ...,
) -> ndarray: ...

def ptp(
    a: ArrayLike,
    axis: None | _ShapeLike = ...,
    out: None | ndarray = ...,
    keepdims: bool = ...,
) -> Any: ...

def amax(
    a: ArrayLike,
    axis: None | _ShapeLike = ...,
    out: None | ndarray = ...,
    keepdims: bool = ...,
    initial: _NumberLike_co = ...,
    where: _ArrayLikeBool_co = ...,
) -> Any: ...

def amin(
    a: ArrayLike,
    axis: None | _ShapeLike = ...,
    out: None | ndarray = ...,
    keepdims: bool = ...,
    initial: _NumberLike_co = ...,
    where: _ArrayLikeBool_co = ...,
) -> Any: ...

# TODO: `np.prod()``: For object arrays `initial` does not necessarily
# have to be a numerical scalar.
# The only requirement is that it is compatible
# with the `.__mul__()` method(s) of the passed array's elements.

# Note that the same situation holds for all wrappers around
# `np.ufunc.reduce`, e.g. `np.sum()` (`.__add__()`).
def prod(
    a: ArrayLike,
    axis: None | _ShapeLike = ...,
    dtype: DTypeLike = ...,
    out: None | ndarray = ...,
    keepdims: bool = ...,
    initial: _NumberLike_co = ...,
    where: _ArrayLikeBool_co = ...,
) -> Any: ...

def cumprod(
    a: ArrayLike,
    axis: None | int = ...,
    dtype: DTypeLike = ...,
    out: None | ndarray = ...,
) -> ndarray: ...

def ndim(a: ArrayLike) -> int: ...

def size(a: ArrayLike, axis: None | int = ...) -> int: ...

def around(
    a: ArrayLike,
    decimals: int = ...,
    out: None | ndarray = ...,
) -> Any: ...

def mean(
    a: ArrayLike,
    axis: None | _ShapeLike = ...,
    dtype: DTypeLike = ...,
    out: None | ndarray = ...,
    keepdims: bool = ...,
    *,
    where: _ArrayLikeBool_co = ...,
) -> Any: ...

def std(
    a: ArrayLike,
    axis: None | _ShapeLike = ...,
    dtype: DTypeLike = ...,
    out: None | ndarray = ...,
    ddof: int = ...,
    keepdims: bool = ...,
    *,
    where: _ArrayLikeBool_co = ...,
) -> Any: ...

def var(
    a: ArrayLike,
    axis: None | _ShapeLike = ...,
    dtype: DTypeLike = ...,
    out: None | ndarray = ...,
    ddof: int = ...,
    keepdims: bool = ...,
    *,
    where: _ArrayLikeBool_co = ...,
) -> Any: ...
