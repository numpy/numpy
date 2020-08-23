import sys
from typing import overload, Tuple, Union, Sequence, Any

from numpy import ndarray, inexact, _NumberLike
from numpy.typing import ArrayLike, DtypeLike, _SupportsArray

if sys.version_info >= (3, 8):
    from typing import SupportsIndex, Literal
else:
    from typing_extensions import SupportsIndex, Literal

# TODO: wait for support for recursive types
_ArrayLikeNested = Sequence[Sequence[Any]]
_ArrayLikeNumber = Union[
    _NumberLike, Sequence[_NumberLike], ndarray, _SupportsArray, _ArrayLikeNested
]
@overload
def linspace(
    start: _ArrayLikeNumber,
    stop: _ArrayLikeNumber,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    retstep: Literal[False] = ...,
    dtype: DtypeLike = ...,
    axis: int = ...,
) -> ndarray: ...
@overload
def linspace(
    start: _ArrayLikeNumber,
    stop: _ArrayLikeNumber,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    retstep: Literal[True] = ...,
    dtype: DtypeLike = ...,
    axis: int = ...,
) -> Tuple[ndarray, inexact]: ...
def logspace(
    start: _ArrayLikeNumber,
    stop: _ArrayLikeNumber,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    base: _ArrayLikeNumber = ...,
    dtype: DtypeLike = ...,
    axis: int = ...,
) -> ndarray: ...
def geomspace(
    start: _ArrayLikeNumber,
    stop: _ArrayLikeNumber,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    dtype: DtypeLike = ...,
    axis: int = ...,
) -> ndarray: ...
