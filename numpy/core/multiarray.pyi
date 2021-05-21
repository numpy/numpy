import sys
from typing import (
    Any,
    Optional,
    overload,
    TypeVar,
    List,
    Type,
    Union,
    Sequence,
    Tuple,
)

from numpy import dtype, generic, intp, _OrderKACF, _OrderCF, _ModeKind
from numpy.typing import (
    ArrayLike,
    NDArray,
    DTypeLike,
    _SupportsDType,
    _ShapeLike,
    _IntLike_co,
    _ArrayLikeInt_co,
)

if sys.version_info >= (3, 8):
    from typing import SupportsIndex
else:
    from typing_extensions import SupportsIndex

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

@overload
def array(
    object: object,
    dtype: _DTypeLike[_SCT],
    *,
    copy: bool = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    ndmin: int = ...,
    like: ArrayLike = ...,
) -> NDArray[_SCT]: ...
@overload
def array(
    object: object,
    dtype: DTypeLike = ...,
    *,
    copy: bool = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    ndmin: int = ...,
    like: ArrayLike = ...,
) -> NDArray[Any]: ...

@overload
def zeros(
    shape: _ShapeLike,
    dtype: _DTypeLike[_SCT],
    order: _OrderCF = ...,
    *,
    like: ArrayLike = ...,
) -> NDArray[_SCT]: ...
@overload
def zeros(
    shape: _ShapeLike,
    dtype: DTypeLike = ...,
    order: _OrderCF = ...,
    *,
    like: ArrayLike = ...,
) -> NDArray[Any]: ...

@overload
def empty(
    shape: _ShapeLike,
    dtype: _DTypeLike[_SCT],
    order: _OrderCF = ...,
    *,
    like: ArrayLike = ...,
) -> NDArray[_SCT]: ...
@overload
def empty(
    shape: _ShapeLike,
    dtype: DTypeLike = ...,
    order: _OrderCF = ...,
    *,
    like: ArrayLike = ...,
) -> NDArray[Any]: ...

@overload
def unravel_index(  # type: ignore[misc]
    indices: _IntLike_co,
    shape: _ShapeLike,
    order: _OrderCF = ...,
) -> Tuple[intp, ...]: ...
@overload
def unravel_index(
    indices: _ArrayLikeInt_co,
    shape: _ShapeLike,
    order: _OrderCF = ...,
) -> Tuple[NDArray[intp], ...]: ...

@overload
def ravel_multi_index(  # type: ignore[misc]
    multi_index: Sequence[_IntLike_co],
    dims: Sequence[SupportsIndex],
    mode: Union[_ModeKind, Tuple[_ModeKind, ...]] = ...,
    order: _OrderCF = ...,
) -> intp: ...
@overload
def ravel_multi_index(
    multi_index: Sequence[_ArrayLikeInt_co],
    dims: Sequence[SupportsIndex],
    mode: Union[_ModeKind, Tuple[_ModeKind, ...]] = ...,
    order: _OrderCF = ...,
) -> NDArray[intp]: ...
