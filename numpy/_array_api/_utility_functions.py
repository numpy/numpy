from __future__ import annotations

from ._types import Optional, Tuple, Union, array

import numpy as np

def all(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    """
    Array API compatible wrapper for :py:func:`np.all <numpy.all>`.

    See its docstring for more information.
    """
    return np.asarray(np.all(x, axis=axis, keepdims=keepdims))

def any(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    """
    Array API compatible wrapper for :py:func:`np.any <numpy.any>`.

    See its docstring for more information.
    """
    return np.asarray(np.any(x, axis=axis, keepdims=keepdims))
