import sys
from typing import (
    Any,
    Dict,
    List,
    overload,
    Tuple,
    TypeVar,
)

from numpy import ndarray, dtype, generic

from numpy.typing import (
    ArrayLike,
    NDArray,
    _ArrayLikeInt,
    _NestedSequence,
    _SupportsArray,
)

if sys.version_info >= (3, 8):
    from typing import Literal as L, Protocol
else:
    from typing_extensions import Literal as L, Protocol

_SCT = TypeVar("_SCT", bound=generic)

class _ModeFunc(Protocol):
    def __call__(
        self,
        __vector: NDArray[Any],
        __iaxis_pad_width: Tuple[int, int],
        __iaxis: int,
        __kwargs: Dict[str, Any],
    ) -> None: ...

_ModeKind = L[
    "constant",
    "edge",
    "linear_ramp",
    "maximum",
    "mean",
    "median",
    "minimum",
    "reflect",
    "symmetric",
    "wrap",
    "empty",
]

_ArrayLike = _NestedSequence[_SupportsArray[dtype[_SCT]]]

__all__: List[str]

# TODO: In practice each keyword argument is exclusive to one or more
# specific modes. Consider adding more overloads to express this in the future.

# Expand `**kwargs` into explicit keyword-only arguments
@overload
def pad(
    array: _ArrayLike[_SCT],
    pad_width: _ArrayLikeInt,
    mode: _ModeKind = ...,
    *,
    stat_length: None | _ArrayLikeInt = ...,
    constant_values: ArrayLike = ...,
    end_values: ArrayLike = ...,
    reflect_type: L["odd", "even"] = ...,
) -> NDArray[_SCT]: ...
@overload
def pad(
    array: ArrayLike,
    pad_width: _ArrayLikeInt,
    mode: _ModeKind = ...,
    *,
    stat_length: None | _ArrayLikeInt = ...,
    constant_values: ArrayLike = ...,
    end_values: ArrayLike = ...,
    reflect_type: L["odd", "even"] = ...,
) -> NDArray[Any]: ...
@overload
def pad(
    array: _ArrayLike[_SCT],
    pad_width: _ArrayLikeInt,
    mode: _ModeFunc,
    **kwargs: Any,
) -> NDArray[_SCT]: ...
@overload
def pad(
    array: ArrayLike,
    pad_width: _ArrayLikeInt,
    mode: _ModeFunc,
    **kwargs: Any,
) -> NDArray[Any]: ...
