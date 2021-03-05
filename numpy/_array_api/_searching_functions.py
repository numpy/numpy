from __future__ import annotations

from ._types import Tuple, array
from ._array_object import ndarray

import numpy as np

def argmax(x: array, /, *, axis: int = None, keepdims: bool = False) -> array:
    """
    Array API compatible wrapper for :py:func:`np.argmax <numpy.argmax>`.

    See its docstring for more information.
    """
    # Note: this currently fails as np.argmax does not implement keepdims
    return ndarray._new(np.asarray(np.argmax(x._array, axis=axis, keepdims=keepdims)))

def argmin(x: array, /, *, axis: int = None, keepdims: bool = False) -> array:
    """
    Array API compatible wrapper for :py:func:`np.argmin <numpy.argmin>`.

    See its docstring for more information.
    """
    # Note: this currently fails as np.argmin does not implement keepdims
    return ndarray._new(np.asarray(np.argmin(x._array, axis=axis, keepdims=keepdims)))

def nonzero(x: array, /) -> Tuple[array, ...]:
    """
    Array API compatible wrapper for :py:func:`np.nonzero <numpy.nonzero>`.

    See its docstring for more information.
    """
    return ndarray._new(np.nonzero(x._array))

def where(condition: array, x1: array, x2: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.where <numpy.where>`.

    See its docstring for more information.
    """
    return ndarray._new(np.where(condition._array, x1._array, x2._array))
