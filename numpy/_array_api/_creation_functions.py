import numpy as np

def arange(start, /, *, stop=None, step=1, dtype=None, device=None):
    """
    Array API compatible wrapper for :py:func:`np.arange <numpy.arange>`.

    See its docstring for more information.
    """
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.arange(start, stop=stop, step=step, dtype=dtype)

def empty(shape, /, *, dtype=None, device=None):
    """
    Array API compatible wrapper for :py:func:`np.empty <numpy.empty>`.

    See its docstring for more information.
    """
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.empty(shape, dtype=dtype)

def empty_like(x, /, *, dtype=None, device=None):
    """
    Array API compatible wrapper for :py:func:`np.empty_like <numpy.empty_like>`.

    See its docstring for more information.
    """
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.empty_like(x, dtype=dtype)

def eye(N, /, *, M=None, k=0, dtype=None, device=None):
    """
    Array API compatible wrapper for :py:func:`np.eye <numpy.eye>`.

    See its docstring for more information.
    """
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.eye(N, M=M, k=k, dtype=dtype)

def full(shape, fill_value, /, *, dtype=None, device=None):
    """
    Array API compatible wrapper for :py:func:`np.full <numpy.full>`.

    See its docstring for more information.
    """
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.full(shape, fill_value, dtype=dtype)

def full_like(x, fill_value, /, *, dtype=None, device=None):
    """
    Array API compatible wrapper for :py:func:`np.full_like <numpy.full_like>`.

    See its docstring for more information.
    """
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.full_like(x, fill_value, dtype=dtype)

def linspace(start, stop, num, /, *, dtype=None, device=None, endpoint=True):
    """
    Array API compatible wrapper for :py:func:`np.linspace <numpy.linspace>`.

    See its docstring for more information.
    """
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.linspace(start, stop, num, dtype=dtype, endpoint=endpoint)

def ones(shape, /, *, dtype=None, device=None):
    """
    Array API compatible wrapper for :py:func:`np.ones <numpy.ones>`.

    See its docstring for more information.
    """
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.ones(shape, dtype=dtype)

def ones_like(x, /, *, dtype=None, device=None):
    """
    Array API compatible wrapper for :py:func:`np.ones_like <numpy.ones_like>`.

    See its docstring for more information.
    """
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.ones_like(x, dtype=dtype)

def zeros(shape, /, *, dtype=None, device=None):
    """
    Array API compatible wrapper for :py:func:`np.zeros <numpy.zeros>`.

    See its docstring for more information.
    """
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.zeros(shape, dtype=dtype)

def zeros_like(x, /, *, dtype=None, device=None):
    """
    Array API compatible wrapper for :py:func:`np.zeros_like <numpy.zeros_like>`.

    See its docstring for more information.
    """
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.zeros_like(x, dtype=dtype)
