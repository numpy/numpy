from __future__ import annotations

from ._types import Optional, Tuple, Union, array

import numpy as np

def max(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    return np.max(x, axis=axis, keepdims=keepdims)

def mean(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    return np.mean(x, axis=axis, keepdims=keepdims)

def min(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    return np.min(x, axis=axis, keepdims=keepdims)

def prod(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    return np.prod(x, axis=axis, keepdims=keepdims)

def std(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, correction: Union[int, float] = 0.0, keepdims: bool = False) -> array:
    # Note: the keyword argument correction is different here
    return np.std(x, axis=axis, ddof=correction, keepdims=keepdims)

def sum(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    return np.sum(x, axis=axis, keepdims=keepdims)

def var(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, correction: Union[int, float] = 0.0, keepdims: bool = False) -> array:
    # Note: the keyword argument correction is different here
    return np.var(x, axis=axis, ddof=correction, keepdims=keepdims)
