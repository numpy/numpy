from __future__ import annotations

from ._array_object import ndarray

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._types import List, Tuple, Union, array, dtype
    from collections.abc import Sequence

import numpy as np

def broadcast_arrays(*arrays: Sequence[array]) -> List[array]:
    """
    Array API compatible wrapper for :py:func:`np.broadcast_arrays <numpy.broadcast_arrays>`.

    See its docstring for more information.
    """
    from ._array_object import ndarray
    return [ndarray._new(array) for array in np.broadcast_arrays(*[a._array for a in arrays])]

def broadcast_to(x: array, shape: Tuple[int, ...], /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.broadcast_to <numpy.broadcast_to>`.

    See its docstring for more information.
    """
    from ._array_object import ndarray
    return ndarray._new(np.broadcast_to(x._array, shape))

def can_cast(from_: Union[dtype, array], to: dtype, /) -> bool:
    """
    Array API compatible wrapper for :py:func:`np.can_cast <numpy.can_cast>`.

    See its docstring for more information.
    """
    from ._array_object import ndarray
    if isinstance(from_, ndarray):
        from_ = from_._array
    return np.can_cast(from_, to)

def finfo(type: Union[dtype, array], /) -> finfo_object:
    """
    Array API compatible wrapper for :py:func:`np.finfo <numpy.finfo>`.

    See its docstring for more information.
    """
    return np.finfo(type)

def iinfo(type: Union[dtype, array], /) -> iinfo_object:
    """
    Array API compatible wrapper for :py:func:`np.iinfo <numpy.iinfo>`.

    See its docstring for more information.
    """
    return np.iinfo(type)

def result_type(*arrays_and_dtypes: Sequence[Union[array, dtype]]) -> dtype:
    """
    Array API compatible wrapper for :py:func:`np.result_type <numpy.result_type>`.

    See its docstring for more information.
    """
    return np.result_type(*(a._array if isinstance(a, ndarray) else a for a in arrays_and_dtypes))
