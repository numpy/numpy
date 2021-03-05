from __future__ import annotations

from ._types import Tuple, Union, array
from ._array_object import ndarray

import numpy as np

def unique(x: array, /, *, return_counts: bool = False, return_index: bool = False, return_inverse: bool = False) -> Union[array, Tuple[array, ...]]:
    """
    Array API compatible wrapper for :py:func:`np.unique <numpy.unique>`.

    See its docstring for more information.
    """
    return ndarray._new(np.unique(x._array, return_counts=return_counts, return_index=return_index, return_inverse=return_inverse))
