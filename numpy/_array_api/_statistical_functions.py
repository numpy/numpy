from __future__ import annotations

from ._array_object import Array

from typing import Optional, Tuple, Union

import numpy as np

def max(x: Array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Array:
    return Array._new(np.max(x._array, axis=axis, keepdims=keepdims))

def mean(x: Array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Array:
    return Array._new(np.asarray(np.mean(x._array, axis=axis, keepdims=keepdims)))

def min(x: Array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Array:
    return Array._new(np.min(x._array, axis=axis, keepdims=keepdims))

def prod(x: Array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Array:
    return Array._new(np.asarray(np.prod(x._array, axis=axis, keepdims=keepdims)))

def std(x: Array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, correction: Union[int, float] = 0.0, keepdims: bool = False) -> Array:
    # Note: the keyword argument correction is different here
    return Array._new(np.asarray(np.std(x._array, axis=axis, ddof=correction, keepdims=keepdims)))

def sum(x: Array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Array:
    return Array._new(np.asarray(np.sum(x._array, axis=axis, keepdims=keepdims)))

def var(x: Array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, correction: Union[int, float] = 0.0, keepdims: bool = False) -> Array:
    # Note: the keyword argument correction is different here
    return Array._new(np.asarray(np.var(x._array, axis=axis, ddof=correction, keepdims=keepdims)))
