from __future__ import annotations

from ._types import (Optional, SupportsDLPack, SupportsBufferProtocol, Tuple,
                     Union, array, device, dtype)
from ._dtypes import _all_dtypes

import numpy as np

def asarray(obj: Union[float, NestedSequence[bool|int|float], SupportsDLPack, SupportsBufferProtocol], /, *, dtype: Optional[dtype] = None, device: Optional[device] = None, copy: Optional[bool] = None) -> array:
    """
    Array API compatible wrapper for :py:func:`np.asarray <numpy.asarray>`.

    See its docstring for more information.
    """
    from ._array_object import ndarray
    from . import _dtypes
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    if copy is not None:
        # Note: copy is not yet implemented in np.asarray
        raise NotImplementedError("The copy keyword argument to asarray is not yet implemented")
    if isinstance(obj, ndarray):
        return obj
    res = np.asarray(obj, dtype=dtype)
    if res.dtype not in _dtypes._all_dtypes:
        raise TypeError(f"The array_api namespace does not support the dtype '{res.dtype}'")
    return ndarray._new(res)

def arange(start: Union[int, float], /, *, stop: Optional[Union[int, float]] = None, step: Union[int, float] = 1, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    """
    Array API compatible wrapper for :py:func:`np.arange <numpy.arange>`.

    See its docstring for more information.
    """
    from ._array_object import ndarray
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return ndarray._new(np.arange(start, stop=stop, step=step, dtype=dtype))

def empty(shape: Union[int, Tuple[int, ...]], /, *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    """
    Array API compatible wrapper for :py:func:`np.empty <numpy.empty>`.

    See its docstring for more information.
    """
    from ._array_object import ndarray
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return ndarray._new(np.empty(shape, dtype=dtype))

def empty_like(x: array, /, *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    """
    Array API compatible wrapper for :py:func:`np.empty_like <numpy.empty_like>`.

    See its docstring for more information.
    """
    from ._array_object import ndarray
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return ndarray._new(np.empty_like._implementation(x._array, dtype=dtype))

def eye(N: int, /, *, M: Optional[int] = None, k: Optional[int] = 0, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    """
    Array API compatible wrapper for :py:func:`np.eye <numpy.eye>`.

    See its docstring for more information.
    """
    from ._array_object import ndarray
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return ndarray._new(np.eye(N, M=M, k=k, dtype=dtype))

def from_dlpack(x: object, /) -> array:
    # Note: dlpack support is not yet implemented on ndarray
    raise NotImplementedError("DLPack support is not yet implemented")

def full(shape: Union[int, Tuple[int, ...]], fill_value: Union[int, float], /, *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    """
    Array API compatible wrapper for :py:func:`np.full <numpy.full>`.

    See its docstring for more information.
    """
    from ._array_object import ndarray
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    if isinstance(fill_value, ndarray) and fill_value.ndim == 0:
            fill_value = fill_value._array[...]
    res = np.full(shape, fill_value, dtype=dtype)
    if res.dtype not in _all_dtypes:
        # This will happen if the fill value is not something that NumPy
        # coerces to one of the acceptable dtypes.
        raise TypeError("Invalid input to full")
    return ndarray._new(res)

def full_like(x: array, fill_value: Union[int, float], /, *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    """
    Array API compatible wrapper for :py:func:`np.full_like <numpy.full_like>`.

    See its docstring for more information.
    """
    from ._array_object import ndarray
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    res = np.full_like._implementation(x._array, fill_value, dtype=dtype)
    if res.dtype not in _all_dtypes:
        # This will happen if the fill value is not something that NumPy
        # coerces to one of the acceptable dtypes.
        raise TypeError("Invalid input to full_like")
    return ndarray._new(res)

def linspace(start: Union[int, float], stop: Union[int, float], num: int, /, *, dtype: Optional[dtype] = None, device: Optional[device] = None, endpoint: bool = True) -> array:
    """
    Array API compatible wrapper for :py:func:`np.linspace <numpy.linspace>`.

    See its docstring for more information.
    """
    from ._array_object import ndarray
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return ndarray._new(np.linspace(start, stop, num, dtype=dtype, endpoint=endpoint))

def ones(shape: Union[int, Tuple[int, ...]], /, *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    """
    Array API compatible wrapper for :py:func:`np.ones <numpy.ones>`.

    See its docstring for more information.
    """
    from ._array_object import ndarray
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return ndarray._new(np.ones(shape, dtype=dtype))

def ones_like(x: array, /, *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    """
    Array API compatible wrapper for :py:func:`np.ones_like <numpy.ones_like>`.

    See its docstring for more information.
    """
    from ._array_object import ndarray
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return ndarray._new(np.ones_like._implementation(x._array, dtype=dtype))

def zeros(shape: Union[int, Tuple[int, ...]], /, *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    """
    Array API compatible wrapper for :py:func:`np.zeros <numpy.zeros>`.

    See its docstring for more information.
    """
    from ._array_object import ndarray
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return ndarray._new(np.zeros(shape, dtype=dtype))

def zeros_like(x: array, /, *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    """
    Array API compatible wrapper for :py:func:`np.zeros_like <numpy.zeros_like>`.

    See its docstring for more information.
    """
    from ._array_object import ndarray
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return ndarray._new(np.zeros_like._implementation(x._array, dtype=dtype))
