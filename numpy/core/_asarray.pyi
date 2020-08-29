import sys
from typing import TypeVar, Optional, Union, Iterable, Tuple, overload

from numpy import ndarray
from numpy.typing import ArrayLike, DtypeLike

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

_ArrayType = TypeVar("_ArrayType", bound=ndarray)

def asarray(
    a: object,
    dtype: DtypeLike = ...,
    order: Optional[str] = ...,
    *,
    like: ArrayLike = ...
) -> ndarray: ...
@overload
def asanyarray(
    a: _ArrayType,
    dtype: None = ...,
    order: Optional[str] = ...,
    *,
    like: ArrayLike = ...
) -> _ArrayType: ...
@overload
def asanyarray(
    a: object,
    dtype: DtypeLike = ...,
    order: Optional[str] = ...,
    *,
    like: ArrayLike = ...
) -> ndarray: ...
def ascontiguousarray(
    a: object, dtype: DtypeLike = ..., *, like: ArrayLike = ...
) -> ndarray: ...
def asfortranarray(
    a: object, dtype: DtypeLike = ..., *, like: ArrayLike = ...
) -> ndarray: ...

_Requirements = Literal["F", "C", "A", "W", "O"]
_E = Literal["E"]

@overload
def require(
    a: object,
    dtype: DtypeLike = ...,
    requirements: Union[_E, Iterable[Union[_E, _Requirements]]] = ...,
    *,
    like: ArrayLike = ...
) -> ndarray: ...
@overload
def require(
    a: _ArrayType,
    dtype: None = ...,
    requirements: Union[None, _Requirements, Iterable[_Requirements]] = ...,
    *,
    like: ArrayLike = ...
) -> _ArrayType: ...
@overload
def require(
    a: object,
    dtype: DtypeLike = ...,
    requirements: Union[None, _Requirements, Iterable[_Requirements]] = ...,
    *,
    like: ArrayLike = ...
) -> ndarray: ...
