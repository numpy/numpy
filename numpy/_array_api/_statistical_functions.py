from __future__ import annotations

from ._types import Optional, Tuple, Union, array
from ._array_object import ndarray

import numpy as np

def max(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    return ndarray._new(np.max(x._array, axis=axis, keepdims=keepdims))

def mean(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    return ndarray._new(np.asarray(np.mean(x._array, axis=axis, keepdims=keepdims)))

def min(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    return ndarray._new(np.min(x._array, axis=axis, keepdims=keepdims))

def prod(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    return ndarray._new(np.asarray(np.prod(x._array, axis=axis, keepdims=keepdims)))

def std(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, correction: Union[int, float] = 0.0, keepdims: bool = False) -> array:
    # Note: the keyword argument correction is different here
    return ndarray._new(np.asarray(np.std(x._array, axis=axis, ddof=correction, keepdims=keepdims)))

def sum(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    return ndarray._new(np.asarray(np.sum(x._array, axis=axis, keepdims=keepdims)))

def var(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, correction: Union[int, float] = 0.0, keepdims: bool = False) -> array:
    # Note: the keyword argument correction is different here
    return ndarray._new(np.asarray(np.var(x._array, axis=axis, ddof=correction, keepdims=keepdims)))
