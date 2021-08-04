from __future__ import annotations

from ._array_object import Array

from typing import Tuple, Union

import numpy as np

def unique(x: Array, /, *, return_counts: bool = False, return_index: bool = False, return_inverse: bool = False) -> Union[Array, Tuple[Array, ...]]:
    """
    Array API compatible wrapper for :py:func:`np.unique <numpy.unique>`.

    See its docstring for more information.
    """
    return Array._new(np.unique(x._array, return_counts=return_counts, return_index=return_index, return_inverse=return_inverse))
