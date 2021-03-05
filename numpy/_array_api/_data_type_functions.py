from __future__ import annotations

from ._types import Union, array, dtype
from ._array_object import ndarray

from collections.abc import Sequence

import numpy as np

def finfo(type: Union[dtype, array], /) -> finfo:
    """
    Array API compatible wrapper for :py:func:`np.finfo <numpy.finfo>`.

    See its docstring for more information.
    """
    return np.finfo(type)

def iinfo(type: Union[dtype, array], /) -> iinfo:
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
