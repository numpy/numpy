from __future__ import annotations

from ._array_object import ndarray

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._types import Tuple, Array

import numpy as np

def argmax(x: Array, /, *, axis: int = None, keepdims: bool = False) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.argmax <numpy.argmax>`.

    See its docstring for more information.
    """
    # Note: this currently fails as np.argmax does not implement keepdims
    return ndarray._new(np.asarray(np.argmax(x._array, axis=axis, keepdims=keepdims)))

def argmin(x: Array, /, *, axis: int = None, keepdims: bool = False) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.argmin <numpy.argmin>`.

    See its docstring for more information.
    """
    # Note: this currently fails as np.argmin does not implement keepdims
    return ndarray._new(np.asarray(np.argmin(x._array, axis=axis, keepdims=keepdims)))

def nonzero(x: Array, /) -> Tuple[Array, ...]:
    """
    Array API compatible wrapper for :py:func:`np.nonzero <numpy.nonzero>`.

    See its docstring for more information.
    """
    return ndarray._new(np.nonzero(x._array))

def where(condition: Array, x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.where <numpy.where>`.

    See its docstring for more information.
    """
    return ndarray._new(np.where(condition._array, x1._array, x2._array))
