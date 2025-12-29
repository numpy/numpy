from typing import overload
from typing_extensions import deprecated

import numpy as np
from numpy._typing import (
    NDArray,
    _ArrayLikeFloat_co,
    _ArrayLikeObject_co,
    _FloatLike_co,
)

__all__ = ["fix", "isneginf", "isposinf"]

@overload
@deprecated("np.fix will be deprecated in NumPy 2.5 in favor of np.trunc", category=PendingDeprecationWarning)
def fix(x: _FloatLike_co, out: None = None) -> np.floating: ...
@overload
@deprecated("np.fix will be deprecated in NumPy 2.5 in favor of np.trunc", category=PendingDeprecationWarning)
def fix(x: _ArrayLikeFloat_co, out: None = None) -> NDArray[np.floating]: ...
@overload
@deprecated("np.fix will be deprecated in NumPy 2.5 in favor of np.trunc", category=PendingDeprecationWarning)
def fix(x: _ArrayLikeObject_co, out: None = None) -> NDArray[np.object_]: ...
@overload
@deprecated("np.fix will be deprecated in NumPy 2.5 in favor of np.trunc", category=PendingDeprecationWarning)
def fix[ArrayT: np.ndarray](x: _ArrayLikeFloat_co | _ArrayLikeObject_co, out: ArrayT) -> ArrayT: ...

#
@overload
def isposinf(x: _FloatLike_co, out: None = None) -> np.bool: ...
@overload
def isposinf(x: _ArrayLikeFloat_co, out: None = None) -> NDArray[np.bool]: ...
@overload
def isposinf[ArrayT: np.ndarray](x: _ArrayLikeFloat_co, out: ArrayT) -> ArrayT: ...

#
@overload
def isneginf(x: _FloatLike_co, out: None = None) -> np.bool: ...
@overload
def isneginf(x: _ArrayLikeFloat_co, out: None = None) -> NDArray[np.bool]: ...
@overload
def isneginf[ArrayT: np.ndarray](x: _ArrayLikeFloat_co, out: ArrayT) -> ArrayT: ...
