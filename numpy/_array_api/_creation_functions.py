from __future__ import annotations

from ._types import Optional, Tuple, Union, array, device, dtype

import numpy as np

def arange(start: Union[int, float], /, *, stop: Optional[Union[int, float]] = None, step: Union[int, float] = 1, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    """
    Array API compatible wrapper for :py:func:`np.arange <numpy.arange>`.

    See its docstring for more information.
    """
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.arange(start, stop=stop, step=step, dtype=dtype)

def empty(shape: Union[int, Tuple[int, ...]], /, *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    """
    Array API compatible wrapper for :py:func:`np.empty <numpy.empty>`.

    See its docstring for more information.
    """
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.empty(shape, dtype=dtype)

def empty_like(x: array, /, *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    """
    Array API compatible wrapper for :py:func:`np.empty_like <numpy.empty_like>`.

    See its docstring for more information.
    """
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.empty_like(x, dtype=dtype)

def eye(N: int, /, *, M: Optional[int] = None, k: Optional[int] = 0, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    """
    Array API compatible wrapper for :py:func:`np.eye <numpy.eye>`.

    See its docstring for more information.
    """
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.eye(N, M=M, k=k, dtype=dtype)

def full(shape: Union[int, Tuple[int, ...]], fill_value: Union[int, float], /, *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    """
    Array API compatible wrapper for :py:func:`np.full <numpy.full>`.

    See its docstring for more information.
    """
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.full(shape, fill_value, dtype=dtype)

def full_like(x: array, fill_value: Union[int, float], /, *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    """
    Array API compatible wrapper for :py:func:`np.full_like <numpy.full_like>`.

    See its docstring for more information.
    """
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.full_like(x, fill_value, dtype=dtype)

def linspace(start: Union[int, float], stop: Union[int, float], num: int, /, *, dtype: Optional[dtype] = None, device: Optional[device] = None, endpoint: Optional[bool] = True) -> array:
    """
    Array API compatible wrapper for :py:func:`np.linspace <numpy.linspace>`.

    See its docstring for more information.
    """
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.linspace(start, stop, num, dtype=dtype, endpoint=endpoint)

def ones(shape: Union[int, Tuple[int, ...]], /, *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    """
    Array API compatible wrapper for :py:func:`np.ones <numpy.ones>`.

    See its docstring for more information.
    """
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.ones(shape, dtype=dtype)

def ones_like(x: array, /, *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    """
    Array API compatible wrapper for :py:func:`np.ones_like <numpy.ones_like>`.

    See its docstring for more information.
    """
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.ones_like(x, dtype=dtype)

def zeros(shape: Union[int, Tuple[int, ...]], /, *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    """
    Array API compatible wrapper for :py:func:`np.zeros <numpy.zeros>`.

    See its docstring for more information.
    """
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.zeros(shape, dtype=dtype)

def zeros_like(x: array, /, *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    """
    Array API compatible wrapper for :py:func:`np.zeros_like <numpy.zeros_like>`.

    See its docstring for more information.
    """
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.zeros_like(x, dtype=dtype)
