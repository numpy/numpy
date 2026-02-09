from collections.abc import Iterable
from typing import Any, Literal, overload

from numpy._typing import DTypeLike, NDArray, _SupportsArrayFunc

__all__ = ["require"]

type _Requirements = Literal[
    "C", "C_CONTIGUOUS", "CONTIGUOUS",
    "F", "F_CONTIGUOUS", "FORTRAN",
    "A", "ALIGNED",
    "W", "WRITEABLE",
    "O", "OWNDATA"
]
type _E = Literal["E", "ENSUREARRAY"]
type _RequirementsWithE = _Requirements | _E

@overload
def require[ArrayT: NDArray[Any]](
    a: ArrayT,
    dtype: None = None,
    requirements: _Requirements | Iterable[_Requirements] | None = None,
    *,
    like: _SupportsArrayFunc | None = None
) -> ArrayT: ...
@overload
def require(
    a: object,
    dtype: DTypeLike | None = None,
    requirements: _E | Iterable[_RequirementsWithE] | None = None,
    *,
    like: _SupportsArrayFunc | None = None
) -> NDArray[Any]: ...
@overload
def require(
    a: object,
    dtype: DTypeLike | None = None,
    requirements: _Requirements | Iterable[_Requirements] | None = None,
    *,
    like: _SupportsArrayFunc | None = None
) -> NDArray[Any]: ...
