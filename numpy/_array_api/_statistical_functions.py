from __future__ import annotations

from ._array_object import ndarray

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._types import Optional, Tuple, Union, Array

import numpy as np

def max(x: Array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Array:
    return ndarray._new(np.max(x._array, axis=axis, keepdims=keepdims))

def mean(x: Array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Array:
    return ndarray._new(np.asarray(np.mean(x._array, axis=axis, keepdims=keepdims)))

def min(x: Array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Array:
    return ndarray._new(np.min(x._array, axis=axis, keepdims=keepdims))

def prod(x: Array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Array:
    return ndarray._new(np.asarray(np.prod(x._array, axis=axis, keepdims=keepdims)))

def std(x: Array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, correction: Union[int, float] = 0.0, keepdims: bool = False) -> Array:
    # Note: the keyword argument correction is different here
    return ndarray._new(np.asarray(np.std(x._array, axis=axis, ddof=correction, keepdims=keepdims)))

def sum(x: Array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Array:
    return ndarray._new(np.asarray(np.sum(x._array, axis=axis, keepdims=keepdims)))

def var(x: Array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, correction: Union[int, float] = 0.0, keepdims: bool = False) -> Array:
    # Note: the keyword argument correction is different here
    return ndarray._new(np.asarray(np.var(x._array, axis=axis, ddof=correction, keepdims=keepdims)))
