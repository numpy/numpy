from __future__ import annotations

from ._array_object import ndarray
from ._dtypes import _numeric_dtypes

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._types import Optional, Sequence, Tuple, Union, Array

import numpy as np

# einsum is not yet implemented in the array API spec.

# def einsum():
#     """
#     Array API compatible wrapper for :py:func:`np.einsum <numpy.einsum>`.
#
#     See its docstring for more information.
#     """
#     return np.einsum()

def matmul(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.matmul <numpy.matmul>`.

    See its docstring for more information.
    """
    # Note: the restriction to numeric dtypes only is different from
    # np.matmul.
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in matmul')

    return ndarray._new(np.matmul(x1._array, x2._array))

# Note: axes must be a tuple, unlike np.tensordot where it can be an array or array-like.
def tensordot(x1: Array, x2: Array, /, *, axes: Union[int, Tuple[Sequence[int], Sequence[int]]] = 2) -> Array:
    # Note: the restriction to numeric dtypes only is different from
    # np.tensordot.
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in tensordot')

    return ndarray._new(np.tensordot(x1._array, x2._array, axes=axes))

def transpose(x: Array, /, *, axes: Optional[Tuple[int, ...]] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.transpose <numpy.transpose>`.

    See its docstring for more information.
    """
    return ndarray._new(np.transpose(x._array, axes=axes))

# Note: vecdot is not in NumPy
def vecdot(x1: Array, x2: Array, /, *, axis: Optional[int] = None) -> Array:
    if axis is None:
        axis = -1
    return tensordot(x1, x2, axes=((axis,), (axis,)))
