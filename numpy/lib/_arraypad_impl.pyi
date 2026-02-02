from typing import Any, Literal as L, Protocol, overload, type_check_only

import numpy as np
from numpy._typing import ArrayLike, NDArray, _ArrayLike, _ArrayLikeInt

__all__ = ["pad"]

@type_check_only
class _ModeFunc(Protocol):
    def __call__(
        self,
        vector: NDArray[Any],
        iaxis_pad_width: tuple[int, int],
        iaxis: int,
        kwargs: dict[str, Any],
        /,
    ) -> None: ...

type _ModeKind = L[
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

type _PadWidth = (
    _ArrayLikeInt
    | dict[int, int]
    | dict[int, tuple[int, int]]
    | dict[int, int | tuple[int, int]]
)

###

# TODO: In practice each keyword argument is exclusive to one or more
# specific modes. Consider adding more overloads to express this in the future.

# Expand `**kwargs` into explicit keyword-only arguments
@overload
def pad[ScalarT: np.generic](
    array: _ArrayLike[ScalarT],
    pad_width: _PadWidth,
    mode: _ModeKind = "constant",
    *,
    stat_length: _ArrayLikeInt | None = None,
    constant_values: ArrayLike = 0,
    end_values: ArrayLike = 0,
    reflect_type: L["odd", "even"] = "even",
) -> NDArray[ScalarT]: ...
@overload
def pad(
    array: ArrayLike,
    pad_width: _PadWidth,
    mode: _ModeKind = "constant",
    *,
    stat_length: _ArrayLikeInt | None = None,
    constant_values: ArrayLike = 0,
    end_values: ArrayLike = 0,
    reflect_type: L["odd", "even"] = "even",
) -> NDArray[Any]: ...
@overload
def pad[ScalarT: np.generic](
    array: _ArrayLike[ScalarT],
    pad_width: _PadWidth,
    mode: _ModeFunc,
    **kwargs: Any,
) -> NDArray[ScalarT]: ...
@overload
def pad(
    array: ArrayLike,
    pad_width: _PadWidth,
    mode: _ModeFunc,
    **kwargs: Any,
) -> NDArray[Any]: ...
