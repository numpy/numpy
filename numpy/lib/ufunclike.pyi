from typing import Any, overload, TypeVar, List, Union

from numpy import floating, bool_, ndarray
from numpy.typing import (
    _ArrayLikeFloat_co,
    _ArrayLikeObject_co,
    _ArrayOrScalar,
)

_ArrayType = TypeVar("_ArrayType", bound=ndarray[Any, Any])

__all__: List[str]

@overload
def fix(
    x: _ArrayLikeFloat_co,
    out: None = ...,
) -> _ArrayOrScalar[floating[Any]]: ...
@overload
def fix(
    x: _ArrayLikeObject_co,
    out: None = ...,
) -> Any: ...
@overload
def fix(
    x: Union[_ArrayLikeFloat_co, _ArrayLikeObject_co],
    out: _ArrayType,
) -> _ArrayType: ...

@overload
def isposinf(
    x: _ArrayLikeFloat_co,
    out: None = ...,
) -> _ArrayOrScalar[bool_]: ...
@overload
def isposinf(
    x: _ArrayLikeFloat_co,
    out: _ArrayType,
) -> _ArrayType: ...

@overload
def isneginf(
    x: _ArrayLikeFloat_co,
    out: None = ...,
) -> _ArrayOrScalar[bool_]: ...
@overload
def isneginf(
    x: _ArrayLikeFloat_co,
    out: _ArrayType,
) -> _ArrayType: ...
