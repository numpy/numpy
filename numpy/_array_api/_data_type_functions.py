from __future__ import annotations

from ._array_object import Array

from dataclasses import dataclass
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._types import List, Tuple, Union, Dtype
    from collections.abc import Sequence

import numpy as np

def broadcast_arrays(*arrays: Sequence[Array]) -> List[Array]:
    """
    Array API compatible wrapper for :py:func:`np.broadcast_arrays <numpy.broadcast_arrays>`.

    See its docstring for more information.
    """
    from ._array_object import Array
    return [Array._new(array) for array in np.broadcast_arrays(*[a._array for a in arrays])]

def broadcast_to(x: Array, /, shape: Tuple[int, ...]) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.broadcast_to <numpy.broadcast_to>`.

    See its docstring for more information.
    """
    from ._array_object import Array
    return Array._new(np.broadcast_to(x._array, shape))

def can_cast(from_: Union[Dtype, Array], to: Dtype, /) -> bool:
    """
    Array API compatible wrapper for :py:func:`np.can_cast <numpy.can_cast>`.

    See its docstring for more information.
    """
    from ._array_object import Array
    if isinstance(from_, Array):
        from_ = from_._array
    return np.can_cast(from_, to)

# These are internal objects for the return types of finfo and iinfo, since
# the NumPy versions contain extra data that isn't part of the spec.
@dataclass
class finfo_object:
    bits: int
    # Note: The types of the float data here are float, whereas in NumPy they
    # are scalars of the corresponding float dtype.
    eps: float
    max: float
    min: float
    # Note: smallest_normal is part of the array API spec, but cannot be used
    # until https://github.com/numpy/numpy/pull/18536 is merged.

    # smallest_normal: float

@dataclass
class iinfo_object:
    bits: int
    max: int
    min: int

def finfo(type: Union[Dtype, Array], /) -> finfo_object:
    """
    Array API compatible wrapper for :py:func:`np.finfo <numpy.finfo>`.

    See its docstring for more information.
    """
    fi = np.finfo(type)
    # Note: The types of the float data here are float, whereas in NumPy they
    # are scalars of the corresponding float dtype.
    return finfo_object(
        fi.bits,
        float(fi.eps),
        float(fi.max),
        float(fi.min),
        # TODO: Uncomment this when #18536 is merged.
        # float(fi.smallest_normal),
    )

def iinfo(type: Union[Dtype, Array], /) -> iinfo_object:
    """
    Array API compatible wrapper for :py:func:`np.iinfo <numpy.iinfo>`.

    See its docstring for more information.
    """
    ii = np.iinfo(type)
    return iinfo_object(ii.bits, ii.max, ii.min)

def result_type(*arrays_and_dtypes: Sequence[Union[Array, Dtype]]) -> Dtype:
    """
    Array API compatible wrapper for :py:func:`np.result_type <numpy.result_type>`.

    See its docstring for more information.
    """
    return np.result_type(*(a._array if isinstance(a, Array) else a for a in arrays_and_dtypes))
