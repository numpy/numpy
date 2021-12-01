from typing import (
    Literal as L,
    overload,
    Tuple,
    Union,
    Any,
    SupportsIndex,
    List,
    Type,
    TypeVar,
)

from numpy import floating, complexfloating, generic, dtype
from numpy.typing import (
    NDArray,
    ArrayLike,
    DTypeLike,
    _SupportsDType,
    _SupportsArray,
    _NumberLike_co,
    _ArrayLikeFloat_co,
    _ArrayLikeComplex_co,
)

_SCT = TypeVar("_SCT", bound=generic)

_DTypeLike = Union[
    dtype[_SCT],
    Type[_SCT],
    _SupportsDType[dtype[_SCT]],
]

__all__: List[str]

@overload
def linspace(
    start: _ArrayLikeFloat_co,
    stop: _ArrayLikeFloat_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    retstep: L[False] = ...,
    dtype: None = ...,
    axis: SupportsIndex = ...,
) -> NDArray[floating[Any]]: ...
@overload
def linspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    retstep: L[False] = ...,
    dtype: None = ...,
    axis: SupportsIndex = ...,
) -> NDArray[complexfloating[Any, Any]]: ...
@overload
def linspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    retstep: L[False] = ...,
    dtype: _DTypeLike[_SCT] = ...,
    axis: SupportsIndex = ...,
) -> NDArray[_SCT]: ...
@overload
def linspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    retstep: L[False] = ...,
    dtype: DTypeLike = ...,
    axis: SupportsIndex = ...,
) -> NDArray[Any]: ...
@overload
def linspace(
    start: _ArrayLikeFloat_co,
    stop: _ArrayLikeFloat_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    retstep: L[True] = ...,
    dtype: None = ...,
    axis: SupportsIndex = ...,
) -> Tuple[NDArray[floating[Any]], floating[Any]]: ...
@overload
def linspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    retstep: L[True] = ...,
    dtype: None = ...,
    axis: SupportsIndex = ...,
) -> Tuple[NDArray[complexfloating[Any, Any]], complexfloating[Any, Any]]: ...
@overload
def linspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    retstep: L[True] = ...,
    dtype: _DTypeLike[_SCT] = ...,
    axis: SupportsIndex = ...,
) -> Tuple[NDArray[_SCT], _SCT]: ...
@overload
def linspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    retstep: L[True] = ...,
    dtype: DTypeLike = ...,
    axis: SupportsIndex = ...,
) -> Tuple[NDArray[Any], Any]: ...

@overload
def logspace(
    start: _ArrayLikeFloat_co,
    stop: _ArrayLikeFloat_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    base: _ArrayLikeFloat_co = ...,
    dtype: None = ...,
    axis: SupportsIndex = ...,
) -> NDArray[floating[Any]]: ...
@overload
def logspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    base: _ArrayLikeComplex_co = ...,
    dtype: None = ...,
    axis: SupportsIndex = ...,
) -> NDArray[complexfloating[Any, Any]]: ...
@overload
def logspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    base: _ArrayLikeComplex_co = ...,
    dtype: _DTypeLike[_SCT] = ...,
    axis: SupportsIndex = ...,
) -> NDArray[_SCT]: ...
@overload
def logspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    base: _ArrayLikeComplex_co = ...,
    dtype: DTypeLike = ...,
    axis: SupportsIndex = ...,
) -> NDArray[Any]: ...

@overload
def geomspace(
    start: _ArrayLikeFloat_co,
    stop: _ArrayLikeFloat_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    dtype: None = ...,
    axis: SupportsIndex = ...,
) -> NDArray[floating[Any]]: ...
@overload
def geomspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    dtype: None = ...,
    axis: SupportsIndex = ...,
) -> NDArray[complexfloating[Any, Any]]: ...
@overload
def geomspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    dtype: _DTypeLike[_SCT] = ...,
    axis: SupportsIndex = ...,
) -> NDArray[_SCT]: ...
@overload
def geomspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    dtype: DTypeLike = ...,
    axis: SupportsIndex = ...,
) -> NDArray[Any]: ...

# Re-exported to `np.lib.function_base`
def add_newdoc(
    place: str,
    obj: str,
    doc: str | Tuple[str, str] | List[Tuple[str, str]],
    warn_on_python: bool = ...,
) -> None: ...
