from __future__ import annotations

from ._array_object import ndarray

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._types import Tuple, Union, Array

import numpy as np

def unique(x: Array, /, *, return_counts: bool = False, return_index: bool = False, return_inverse: bool = False) -> Union[Array, Tuple[Array, ...]]:
    """
    Array API compatible wrapper for :py:func:`np.unique <numpy.unique>`.

    See its docstring for more information.
    """
    return ndarray._new(np.unique(x._array, return_counts=return_counts, return_index=return_index, return_inverse=return_inverse))
