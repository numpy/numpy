from __future__ import annotations

from ._dtypes import (
    _floating_dtypes,
    _numeric_dtypes,
)
from ._array_object import Array

from typing import Optional, Tuple, Union

import numpy as np


def max(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in max")
    return Array._new(np.max(x._array, axis=axis, keepdims=keepdims))


def mean(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in mean")
    return Array._new(np.mean(x._array, axis=axis, keepdims=keepdims))


def min(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in min")
    return Array._new(np.min(x._array, axis=axis, keepdims=keepdims))


def prod(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in prod")
    return Array._new(np.prod(x._array, axis=axis, keepdims=keepdims))


def std(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
) -> Array:
    # Note: the keyword argument correction is different here
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in std")
    return Array._new(np.std(x._array, axis=axis, ddof=correction, keepdims=keepdims))


def sum(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in sum")
    return Array._new(np.sum(x._array, axis=axis, keepdims=keepdims))


def var(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
) -> Array:
    # Note: the keyword argument correction is different here
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in var")
    return Array._new(np.var(x._array, axis=axis, ddof=correction, keepdims=keepdims))
