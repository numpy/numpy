from __future__ import annotations

from ._types import Optional, Tuple, Union, array

import numpy as np

def max(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    return np.max._implementation(x, axis=axis, keepdims=keepdims)

def mean(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    return np.asarray(np.mean._implementation(x, axis=axis, keepdims=keepdims))

def min(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    return np.min._implementation(x, axis=axis, keepdims=keepdims)

def prod(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    return np.asarray(np.prod._implementation(x, axis=axis, keepdims=keepdims))

def std(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, correction: Union[int, float] = 0.0, keepdims: bool = False) -> array:
    # Note: the keyword argument correction is different here
    return np.asarray(np.std._implementation(x, axis=axis, ddof=correction, keepdims=keepdims))

def sum(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    return np.asarray(np.sum._implementation(x, axis=axis, keepdims=keepdims))

def var(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, correction: Union[int, float] = 0.0, keepdims: bool = False) -> array:
    # Note: the keyword argument correction is different here
    return np.asarray(np.var._implementation(x, axis=axis, ddof=correction, keepdims=keepdims))
