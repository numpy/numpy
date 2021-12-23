import datetime as dt
from collections.abc import Sequence
from typing import Union, Any, overload, TypeVar, Literal

from numpy import (
    ndarray,
    number,
    integer,
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
    NDArray,
    _ShapeLike,
    _Shape,
    _ArrayLikeBool_co,
    _ArrayLikeInt_co,
    _NumberLike_co,
)

# Various annotations for scalars

# While dt.datetime and dt.timedelta are not technically part of NumPy,
# they are one of the rare few builtin scalars which serve as valid return types.
# See https://github.com/numpy/numpy-stubs/pull/67#discussion_r412604113.
_ScalarNumpy = Union[generic, dt.datetime, dt.timedelta]
_ScalarBuiltin = Union[str, bytes, dt.date, dt.timedelta, bool, int, float, complex]
_Scalar = Union[_ScalarBuiltin, _ScalarNumpy]

# Integers and booleans can generally be used interchangeably
_ScalarGeneric = TypeVar("_ScalarGeneric", bound=generic)

_Number = TypeVar("_Number", bound=number)

# The signature of take() follows a common theme with its overloads:
# 1. A generic comes in; the same generic comes out
# 2. A scalar comes in; a generic comes out
# 3. An array-like object comes in; some keyword ensures that a generic comes out
# 4. An array-like object comes in; an ndarray or generic comes out
def take(
    a: ArrayLike,
    indices: _ArrayLikeInt_co,
    axis: None | int = ...,
    out: None | ndarray = ...,
    mode: _ModeKind = ...,
) -> Any: ...

def reshape(
    a: ArrayLike,
    newshape: _ShapeLike,
    order: _OrderACF = ...,
) -> ndarray: ...

def choose(
    a: _ArrayLikeInt_co,
    choices: ArrayLike,
    out: None | ndarray = ...,
    mode: _ModeKind = ...,
) -> Any: ...

def repeat(
    a: ArrayLike,
    repeats: _ArrayLikeInt_co,
    axis: None | int = ...,
) -> ndarray: ...

def put(
    a: ndarray,
    ind: _ArrayLikeInt_co,
    v: ArrayLike,
    mode: _ModeKind = ...,
) -> None: ...

def swapaxes(
    a: ArrayLike,
    axis1: int,
    axis2: int,
) -> ndarray: ...

def transpose(
    a: ArrayLike,
    axes: None | Sequence[int] | NDArray[Any] = ...
) -> ndarray: ...

def partition(
    a: ArrayLike,
    kth: _ArrayLikeInt_co,
    axis: None | int = ...,
    kind: _PartitionKind = ...,
    order: None | str | Sequence[str] = ...,
) -> ndarray: ...

def argpartition(
    a: ArrayLike,
    kth: _ArrayLikeInt_co,
    axis: None | int = ...,
    kind: _PartitionKind = ...,
    order: None | str | Sequence[str] = ...,
) -> Any: ...

def sort(
    a: ArrayLike,
    axis: None | int = ...,
    kind: None | _SortKind = ...,
    order: None | str | Sequence[str] = ...,
) -> ndarray: ...

def argsort(
    a: ArrayLike,
    axis: None | int = ...,
    kind: None | _SortKind = ...,
    order: None | str | Sequence[str] = ...,
) -> ndarray: ...

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
    v: _Scalar,
    side: _SortSide = ...,
    sorter: None | _ArrayLikeInt_co = ...,  # 1D int array
) -> intp: ...
@overload
def searchsorted(
    a: ArrayLike,
    v: ArrayLike,
    side: _SortSide = ...,
    sorter: None | _ArrayLikeInt_co = ...,  # 1D int array
) -> ndarray: ...

def resize(
    a: ArrayLike,
    new_shape: _ShapeLike,
) -> ndarray: ...

@overload
def squeeze(
    a: _ScalarGeneric,
    axis: None | _ShapeLike = ...,
) -> _ScalarGeneric: ...
@overload
def squeeze(
    a: ArrayLike,
    axis: None | _ShapeLike = ...,
) -> ndarray: ...

def diagonal(
    a: ArrayLike,
    offset: int = ...,
    axis1: int = ...,
    axis2: int = ...,  # >= 2D array
) -> ndarray: ...

def trace(
    a: ArrayLike,  # >= 2D array
    offset: int = ...,
    axis1: int = ...,
    axis2: int = ...,
    dtype: DTypeLike = ...,
    out: None | ndarray = ...,
) -> Any: ...

def ravel(a: ArrayLike, order: _OrderKACF = ...) -> ndarray: ...

def nonzero(a: ArrayLike) -> tuple[ndarray, ...]: ...

def shape(a: ArrayLike) -> _Shape: ...

def compress(
    condition: ArrayLike,  # 1D bool array
    a: ArrayLike,
    axis: None | int = ...,
    out: None | ndarray = ...,
) -> ndarray: ...

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
) -> bool_: ...
@overload
def all(
    a: ArrayLike,
    axis: None | _ShapeLike = ...,
    out: None | ndarray = ...,
    keepdims: bool = ...,
) -> Any: ...

@overload
def any(
    a: ArrayLike,
    axis: None = ...,
    out: None = ...,
    keepdims: Literal[False] = ...,
) -> bool_: ...
@overload
def any(
    a: ArrayLike,
    axis: None | _ShapeLike = ...,
    out: None | ndarray = ...,
    keepdims: bool = ...,
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
) -> Any: ...

def std(
    a: ArrayLike,
    axis: None | _ShapeLike = ...,
    dtype: DTypeLike = ...,
    out: None | ndarray = ...,
    ddof: int = ...,
    keepdims: bool = ...,
) -> Any: ...

def var(
    a: ArrayLike,
    axis: None | _ShapeLike = ...,
    dtype: DTypeLike = ...,
    out: None | ndarray = ...,
    ddof: int = ...,
    keepdims: bool = ...,
) -> Any: ...
