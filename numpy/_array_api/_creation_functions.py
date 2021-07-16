from __future__ import annotations


from typing import TYPE_CHECKING, List, Optional, Tuple, Union
if TYPE_CHECKING:
    from ._typing import (Array, Device, Dtype, NestedSequence,
                          SupportsDLPack, SupportsBufferProtocol)
    from collections.abc import Sequence
from ._dtypes import _all_dtypes

import numpy as np

def asarray(obj: Union[Array, float, NestedSequence[bool|int|float], SupportsDLPack, SupportsBufferProtocol], /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None, copy: Optional[bool] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.asarray <numpy.asarray>`.

    See its docstring for more information.
    """
    # _array_object imports in this file are inside the functions to avoid
    # circular imports
    from ._array_object import Array
    if device is not None:
        # Note: Device support is not yet implemented on Array
        raise NotImplementedError("Device support is not yet implemented")
    if copy is not None:
        # Note: copy is not yet implemented in np.asarray
        raise NotImplementedError("The copy keyword argument to asarray is not yet implemented")
    if isinstance(obj, Array) and (dtype is None or obj.dtype == dtype):
        return obj
    if dtype is None and isinstance(obj, int) and (obj > 2**64 or obj < -2**63):
        # Give a better error message in this case. NumPy would convert this
        # to an object array.
        raise OverflowError("Integer out of bounds for array dtypes")
    res = np.asarray(obj, dtype=dtype)
    return Array._new(res)

def arange(start: Union[int, float], /, stop: Optional[Union[int, float]] = None, step: Union[int, float] = 1, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.arange <numpy.arange>`.

    See its docstring for more information.
    """
    from ._array_object import Array
    if device is not None:
        # Note: Device support is not yet implemented on Array
        raise NotImplementedError("Device support is not yet implemented")
    return Array._new(np.arange(start, stop=stop, step=step, dtype=dtype))

def empty(shape: Union[int, Tuple[int, ...]], *, dtype: Optional[Dtype] = None, device: Optional[Device] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.empty <numpy.empty>`.

    See its docstring for more information.
    """
    from ._array_object import Array
    if device is not None:
        # Note: Device support is not yet implemented on Array
        raise NotImplementedError("Device support is not yet implemented")
    return Array._new(np.empty(shape, dtype=dtype))

def empty_like(x: Array, /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.empty_like <numpy.empty_like>`.

    See its docstring for more information.
    """
    from ._array_object import Array
    if device is not None:
        # Note: Device support is not yet implemented on Array
        raise NotImplementedError("Device support is not yet implemented")
    return Array._new(np.empty_like(x._array, dtype=dtype))

def eye(n_rows: int, n_cols: Optional[int] = None, /, *, k: Optional[int] = 0, dtype: Optional[Dtype] = None, device: Optional[Device] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.eye <numpy.eye>`.

    See its docstring for more information.
    """
    from ._array_object import Array
    if device is not None:
        # Note: Device support is not yet implemented on Array
        raise NotImplementedError("Device support is not yet implemented")
    return Array._new(np.eye(n_rows, M=n_cols, k=k, dtype=dtype))

def from_dlpack(x: object, /) -> Array:
    # Note: dlpack support is not yet implemented on Array
    raise NotImplementedError("DLPack support is not yet implemented")

def full(shape: Union[int, Tuple[int, ...]], fill_value: Union[int, float], *, dtype: Optional[Dtype] = None, device: Optional[Device] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.full <numpy.full>`.

    See its docstring for more information.
    """
    from ._array_object import Array
    if device is not None:
        # Note: Device support is not yet implemented on Array
        raise NotImplementedError("Device support is not yet implemented")
    if isinstance(fill_value, Array) and fill_value.ndim == 0:
        fill_value = fill_value._array[...]
    res = np.full(shape, fill_value, dtype=dtype)
    if res.dtype not in _all_dtypes:
        # This will happen if the fill value is not something that NumPy
        # coerces to one of the acceptable dtypes.
        raise TypeError("Invalid input to full")
    return Array._new(res)

def full_like(x: Array, /, fill_value: Union[int, float], *, dtype: Optional[Dtype] = None, device: Optional[Device] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.full_like <numpy.full_like>`.

    See its docstring for more information.
    """
    from ._array_object import Array
    if device is not None:
        # Note: Device support is not yet implemented on Array
        raise NotImplementedError("Device support is not yet implemented")
    res = np.full_like(x._array, fill_value, dtype=dtype)
    if res.dtype not in _all_dtypes:
        # This will happen if the fill value is not something that NumPy
        # coerces to one of the acceptable dtypes.
        raise TypeError("Invalid input to full_like")
    return Array._new(res)

def linspace(start: Union[int, float], stop: Union[int, float], /, num: int, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None, endpoint: bool = True) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.linspace <numpy.linspace>`.

    See its docstring for more information.
    """
    from ._array_object import Array
    if device is not None:
        # Note: Device support is not yet implemented on Array
        raise NotImplementedError("Device support is not yet implemented")
    return Array._new(np.linspace(start, stop, num, dtype=dtype, endpoint=endpoint))

def meshgrid(*arrays: Sequence[Array], indexing: str = 'xy') -> List[Array, ...]:
    """
    Array API compatible wrapper for :py:func:`np.meshgrid <numpy.meshgrid>`.

    See its docstring for more information.
    """
    from ._array_object import Array
    return [Array._new(array) for array in np.meshgrid(*[a._array for a in arrays], indexing=indexing)]

def ones(shape: Union[int, Tuple[int, ...]], *, dtype: Optional[Dtype] = None, device: Optional[Device] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.ones <numpy.ones>`.

    See its docstring for more information.
    """
    from ._array_object import Array
    if device is not None:
        # Note: Device support is not yet implemented on Array
        raise NotImplementedError("Device support is not yet implemented")
    return Array._new(np.ones(shape, dtype=dtype))

def ones_like(x: Array, /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.ones_like <numpy.ones_like>`.

    See its docstring for more information.
    """
    from ._array_object import Array
    if device is not None:
        # Note: Device support is not yet implemented on Array
        raise NotImplementedError("Device support is not yet implemented")
    return Array._new(np.ones_like(x._array, dtype=dtype))

def zeros(shape: Union[int, Tuple[int, ...]], *, dtype: Optional[Dtype] = None, device: Optional[Device] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.zeros <numpy.zeros>`.

    See its docstring for more information.
    """
    from ._array_object import Array
    if device is not None:
        # Note: Device support is not yet implemented on Array
        raise NotImplementedError("Device support is not yet implemented")
    return Array._new(np.zeros(shape, dtype=dtype))

def zeros_like(x: Array, /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.zeros_like <numpy.zeros_like>`.

    See its docstring for more information.
    """
    from ._array_object import Array
    if device is not None:
        # Note: Device support is not yet implemented on Array
        raise NotImplementedError("Device support is not yet implemented")
    return Array._new(np.zeros_like(x._array, dtype=dtype))
