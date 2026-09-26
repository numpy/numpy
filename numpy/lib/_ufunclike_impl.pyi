from collections.abc import Sequence
from typing import Any, overload
from typing_extensions import deprecated

import numpy as np
from numpy._typing import (
    NDArray,
    _ArrayLikeFloat_co,
    _ArrayLikeObject_co,
    _FloatLike_co,
    _NestedSequence,
    _Shape,
)

type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]

###

__all__ = ["fix", "isneginf", "isposinf"]

@overload
@deprecated("numpy.fix is deprecated. Use numpy.trunc instead.")
def fix(x: _FloatLike_co, out: None = None) -> np.floating: ...
@overload
@deprecated("numpy.fix is deprecated. Use numpy.trunc instead.")
def fix(x: _ArrayLikeFloat_co, out: None = None) -> NDArray[np.floating]: ...
@overload
@deprecated("numpy.fix is deprecated. Use numpy.trunc instead.")
def fix(x: _ArrayLikeObject_co, out: None = None) -> NDArray[np.object_]: ...
@overload
@deprecated("numpy.fix is deprecated. Use numpy.trunc instead.")
def fix[ArrayT: np.ndarray](x: _ArrayLikeFloat_co | _ArrayLikeObject_co, out: ArrayT) -> ArrayT: ...

# keep in sync with `isneginf`
@overload  # 0d
def isposinf(x: _FloatLike_co, out: None = None) -> np.bool: ...
@overload  # Nd
def isposinf[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.floating | np.integer | np.bool]],
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.bool]]: ...
@overload  # 1d
def isposinf(x: Sequence[_FloatLike_co], out: None = None) -> _Array1D[np.bool]: ...
@overload  # 2d
def isposinf(x: Sequence[Sequence[_FloatLike_co]], out: None = None) -> _Array2D[np.bool]: ...
@overload  # Nd
def isposinf(x: _NestedSequence[_ArrayLikeFloat_co], out: None = None) -> NDArray[np.bool]: ...
@overload  # ?d  (fallback)
def isposinf(x: _ArrayLikeFloat_co, out: None = None) -> NDArray[np.bool] | Any: ...
@overload  # out=<given>
def isposinf[ArrayT: np.ndarray](x: _ArrayLikeFloat_co, out: ArrayT) -> ArrayT: ...

# keep in sync with `isposinf`
@overload  # 0d
def isneginf(x: _FloatLike_co, out: None = None) -> np.bool: ...
@overload  # Nd
def isneginf[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.floating | np.integer | np.bool]],
    out: None = None,
) -> np.ndarray[ShapeT, np.dtype[np.bool]]: ...
@overload  # 1d
def isneginf(x: Sequence[_FloatLike_co], out: None = None) -> _Array1D[np.bool]: ...
@overload  # 2d
def isneginf(x: Sequence[Sequence[_FloatLike_co]], out: None = None) -> _Array2D[np.bool]: ...
@overload  # Nd
def isneginf(x: _NestedSequence[_ArrayLikeFloat_co], out: None = None) -> NDArray[np.bool]: ...
@overload  # ?d  (fallback)
def isneginf(x: _ArrayLikeFloat_co, out: None = None) -> NDArray[np.bool] | Any: ...
@overload  # out=<given>
def isneginf[ArrayT: np.ndarray](x: _ArrayLikeFloat_co, out: ArrayT) -> ArrayT: ...
