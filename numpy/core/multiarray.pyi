from typing import Any, Optional, overload, TypeVar, List, Type, Union

from numpy import dtype, generic, _OrderKACF
from numpy.typing import ArrayLike, NDArray, DTypeLike, _ShapeLike, _SupportsDType

_SCT = TypeVar("_SCT", bound=generic)

_DTypeLike = Union[
    dtype[_SCT],
    Type[_SCT],
    _SupportsDType[dtype[_SCT]],
]

__all__: List[str]

@overload
def empty_like(
    a: ArrayLike,
    dtype: _DTypeLike[_SCT],
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: Optional[_ShapeLike] = ...,
) -> NDArray[_SCT]: ...
@overload
def empty_like(
    a: ArrayLike,
    dtype: DTypeLike = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: Optional[_ShapeLike] = ...,
) -> NDArray[Any]: ...
