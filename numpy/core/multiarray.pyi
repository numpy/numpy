# TODO: Sort out any and all missing functions in this namespace

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

from numpy import (
    busdaycalendar as busdaycalendar,
    ndarray,
    dtype,
    str_,
    bool_,
    uint8,
    intp,
    float64,
    timedelta64,
    generic,
    unsignedinteger,
    signedinteger,
    floating,
    complexfloating,
    _OrderKACF,
    _OrderCF,
    _CastingKind,
    _ModeKind,
)

from numpy.typing import (
    # Shapes
    _ShapeLike,

    # DTypes
    DTypeLike,
    _SupportsDType,

    # Arrays
    NDArray,
    ArrayLike,
    _SupportsArray,
    _NestedSequence,
    _ArrayLikeBool_co,
    _ArrayLikeUInt_co,
    _ArrayLikeInt_co,
    _ArrayLikeFloat_co,
    _ArrayLikeComplex_co,
    _ArrayLikeTD64_co,
    _ArrayLikeObject_co,
    _IntLike_co,
)

if sys.version_info >= (3, 8):
    from typing import SupportsIndex, Final, Literal as L
else:
    from typing_extensions import SupportsIndex, Final, Literal as L

_SCT = TypeVar("_SCT", bound=generic)
_ArrayType = TypeVar("_ArrayType", bound=NDArray[Any])

_DTypeLike = Union[
    dtype[_SCT],
    Type[_SCT],
    _SupportsDType[dtype[_SCT]],
]
_ArrayLike = _NestedSequence[_SupportsArray[dtype[_SCT]]]

__all__: List[str]

ALLOW_THREADS: Final[int]
BUFSIZE: Final[int]
CLIP: Final[int]
MAXDIMS: Final[int]
MAY_SHARE_BOUNDS: Final[int]
MAY_SHARE_EXACT: Final[int]
RAISE: Final[int]
WRAP: Final[int]
tracemalloc_domain: Final[int]

@overload
def empty_like(
    prototype: _ArrayLike[_SCT],
    dtype: None = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: Optional[_ShapeLike] = ...,
) -> NDArray[_SCT]: ...
@overload
def empty_like(
    prototype: object,
    dtype: None = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: Optional[_ShapeLike] = ...,
) -> NDArray[Any]: ...
@overload
def empty_like(
    prototype: Any,
    dtype: _DTypeLike[_SCT],
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: Optional[_ShapeLike] = ...,
) -> NDArray[_SCT]: ...
@overload
def empty_like(
    prototype: Any,
    dtype: DTypeLike,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: Optional[_ShapeLike] = ...,
) -> NDArray[Any]: ...

@overload
def array(
    object: _ArrayLike[_SCT],
    dtype: None = ...,
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
    dtype: None = ...,
    *,
    copy: bool = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    ndmin: int = ...,
    like: ArrayLike = ...,
) -> NDArray[Any]: ...
@overload
def array(
    object: Any,
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
    object: Any,
    dtype: DTypeLike,
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
    dtype: None = ...,
    order: _OrderCF = ...,
    *,
    like: ArrayLike = ...,
) -> NDArray[float64]: ...
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
    dtype: DTypeLike,
    order: _OrderCF = ...,
    *,
    like: ArrayLike = ...,
) -> NDArray[Any]: ...

@overload
def empty(
    shape: _ShapeLike,
    dtype: None = ...,
    order: _OrderCF = ...,
    *,
    like: ArrayLike = ...,
) -> NDArray[float64]: ...
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
    dtype: DTypeLike,
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

@overload
def concatenate(  # type: ignore[misc]
    __arrays: _ArrayLike[_SCT],
    axis: Optional[SupportsIndex] = ...,
    out: None = ...,
    *,
    dtype: None = ...,
    casting: Optional[_CastingKind] = ...
) -> NDArray[_SCT]: ...
@overload
def concatenate(  # type: ignore[misc]
    __arrays: ArrayLike,
    axis: Optional[SupportsIndex] = ...,
    out: None = ...,
    *,
    dtype: None = ...,
    casting: Optional[_CastingKind] = ...
) -> NDArray[Any]: ...
@overload
def concatenate(  # type: ignore[misc]
    __arrays: ArrayLike,
    axis: Optional[SupportsIndex] = ...,
    out: None = ...,
    *,
    dtype: _DTypeLike[_SCT],
    casting: Optional[_CastingKind] = ...
) -> NDArray[_SCT]: ...
@overload
def concatenate(  # type: ignore[misc]
    __arrays: ArrayLike,
    axis: Optional[SupportsIndex] = ...,
    out: None = ...,
    *,
    dtype: DTypeLike,
    casting: Optional[_CastingKind] = ...
) -> NDArray[Any]: ...
@overload
def concatenate(
    __arrays: ArrayLike,
    axis: Optional[SupportsIndex] = ...,
    out: _ArrayType = ...,
    *,
    dtype: DTypeLike = ...,
    casting: Optional[_CastingKind] = ...
) -> _ArrayType: ...

def inner(
    __a: ArrayLike,
    __b: ArrayLike,
) -> Any: ...

@overload
def where(
    __condition: ArrayLike,
) -> Tuple[NDArray[intp], ...]: ...
@overload
def where(
    __condition: ArrayLike,
    __x: ArrayLike,
    __y: ArrayLike,
) -> NDArray[Any]: ...

def lexsort(
    keys: ArrayLike,
    axis: Optional[SupportsIndex] = ...,
) -> Any: ...

def can_cast(
    from_: Union[ArrayLike, DTypeLike],
    to: DTypeLike,
    casting: Optional[_CastingKind] = ...,
) -> bool: ...

def min_scalar_type(
    __a: ArrayLike,
) -> dtype[Any]: ...

def result_type(
    *arrays_and_dtypes: Union[ArrayLike, DTypeLike],
) -> dtype[Any]: ...

@overload
def dot(a: ArrayLike, b: ArrayLike, out: None = ...) -> Any: ...
@overload
def dot(a: ArrayLike, b: ArrayLike, out: _ArrayType) -> _ArrayType: ...

@overload
def vdot(__a: _ArrayLikeBool_co, __b: _ArrayLikeBool_co) -> bool_: ...  # type: ignore[misc]
@overload
def vdot(__a: _ArrayLikeUInt_co, __b: _ArrayLikeUInt_co) -> unsignedinteger[Any]: ...  # type: ignore[misc]
@overload
def vdot(__a: _ArrayLikeInt_co, __b: _ArrayLikeInt_co) -> signedinteger[Any]: ... # type: ignore[misc]
@overload
def vdot(__a: _ArrayLikeFloat_co, __b: _ArrayLikeFloat_co) -> floating[Any]: ...  # type: ignore[misc]
@overload
def vdot(__a: _ArrayLikeComplex_co, __b: _ArrayLikeComplex_co) -> complexfloating[Any, Any]: ...  # type: ignore[misc]
@overload
def vdot(__a: _ArrayLikeTD64_co, __b: _ArrayLikeTD64_co) -> timedelta64: ...
@overload
def vdot(__a: _ArrayLikeObject_co, __b: Any) -> Any: ...
@overload
def vdot(__a: Any, __b: _ArrayLikeObject_co) -> Any: ...

def bincount(
    __x: ArrayLike,
    weights: Optional[ArrayLike] = ...,
    minlength: SupportsIndex = ...,
) -> NDArray[intp]: ...

def copyto(
    dst: NDArray[Any],
    src: ArrayLike,
    casting: Optional[_CastingKind] = ...,
    where: Optional[_ArrayLikeBool_co] = ...,
) -> None: ...

def putmask(
    a: NDArray[Any],
    mask: _ArrayLikeBool_co,
    values: ArrayLike,
) -> None: ...

def packbits(
    __a: _ArrayLikeInt_co,
    axis: Optional[SupportsIndex] = ...,
    bitorder: L["big", "little"] = ...,
) -> NDArray[uint8]: ...

def unpackbits(
    __a: _ArrayLike[uint8],
    axis: Optional[SupportsIndex] = ...,
    count: Optional[SupportsIndex] = ...,
    bitorder: L["big", "little"] = ...,
) -> NDArray[uint8]: ...

def shares_memory(
    __a: object,
    __b: object,
    max_work: Optional[int] = ...,
) -> bool: ...

def may_share_memory(
    __a: object,
    __b: object,
    max_work: Optional[int] = ...,
) -> bool: ...
