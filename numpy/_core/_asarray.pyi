from collections.abc import Iterable
from typing import Any, Literal, overload

import numpy as np
from numpy._typing import DTypeLike, NDArray, _DTypeLike, _Shape, _SupportsArrayFunc

__all__ = ["require"]

type _Requirements = Literal[
    "C", "C_CONTIGUOUS", "CONTIGUOUS",
    "F", "F_CONTIGUOUS", "FORTRAN",
    "A", "ALIGNED",
    "W", "WRITEABLE",
    "O", "OWNDATA",
]  # fmt: skip
type _E = Literal["E", "ENSUREARRAY"]
type _RequirementsWithE = _Requirements | _E

###

@overload  # Nd T
def require[ArrayT: NDArray[Any]](
    a: ArrayT,
    dtype: None = None,
    requirements: _Requirements | Iterable[_Requirements] | None = None,
    *,
    like: _SupportsArrayFunc | None = None,
) -> ArrayT: ...
@overload  # Nd T, dtype=<known>
def require[ShapeT: _Shape, ScalarT: np.generic](
    a: np.ndarray[ShapeT, Any],
    dtype: _DTypeLike[ScalarT],
    requirements: _Requirements | Iterable[_Requirements] | None = None,
    *,
    like: _SupportsArrayFunc | None = None,
) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
@overload  # Nd T, dtype=<unknown>
def require[ShapeT: _Shape](
    a: np.ndarray[ShapeT, Any],
    dtype: DTypeLike,
    requirements: _Requirements | Iterable[_Requirements] | None = None,
    *,
    like: _SupportsArrayFunc | None = None,
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # Nd T, requirements=E
def require[ShapeT: _Shape, DTypeT: np.dtype](
    a: np.ndarray[ShapeT, DTypeT],
    dtype: None = None,
    requirements: _E | Iterable[_RequirementsWithE] | None = None,
    *,
    like: _SupportsArrayFunc | None = None,
) -> np.ndarray[ShapeT, DTypeT]: ...
@overload  # ?d, dtype=<known>
def require[ScalarT: np.generic](
    a: object,
    dtype: _DTypeLike[ScalarT],
    requirements: _Requirements | Iterable[_Requirements] | None = None,
    *,
    like: _SupportsArrayFunc | None = None,
) -> NDArray[ScalarT]: ...
@overload  # ?d  (fallback)
def require(
    a: object,
    dtype: DTypeLike | None = None,
    requirements: _RequirementsWithE | Iterable[_RequirementsWithE] | None = None,
    *,
    like: _SupportsArrayFunc | None = None,
) -> NDArray[Any]: ...
