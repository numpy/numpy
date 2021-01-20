from __future__ import annotations

from ._types import Tuple, array

import numpy as np

def argmax(x: array, /, *, axis: int = None, keepdims: bool = False) -> array:
    """
    Array API compatible wrapper for :py:func:`np.argmax <numpy.argmax>`.

    See its docstring for more information.
    """
    return np.argmax(x, axis=axis, keepdims=keepdims)

def argmin(x: array, /, *, axis: int = None, keepdims: bool = False) -> array:
    """
    Array API compatible wrapper for :py:func:`np.argmin <numpy.argmin>`.

    See its docstring for more information.
    """
    return np.argmin(x, axis=axis, keepdims=keepdims)

def nonzero(x: array, /) -> Tuple[array, ...]:
    """
    Array API compatible wrapper for :py:func:`np.nonzero <numpy.nonzero>`.

    See its docstring for more information.
    """
    return np.nonzero(x)

def where(condition: array, x1: array, x2: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.where <numpy.where>`.

    See its docstring for more information.
    """
    return np.where(condition, x1, x2)
