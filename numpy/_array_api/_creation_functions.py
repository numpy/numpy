import numpy as np

def arange(start, /, *, stop=None, step=1, dtype=None, device=None):
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.arange(start, stop=stop, step=step, dtype=dtype)

def empty(shape, /, *, dtype=None, device=None):
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.empty(shape, dtype=dtype)

def empty_like(x, /, *, dtype=None, device=None):
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.empty_like(x, dtype=dtype)

def eye(N, /, *, M=None, k=0, dtype=None, device=None):
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.eye(N, M=M, k=k, dtype=dtype)

def full(shape, fill_value, /, *, dtype=None, device=None):
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.full(shape, fill_value, dtype=dtype)

def full_like(x, fill_value, /, *, dtype=None, device=None):
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.full_like(x, fill_value, dtype=dtype)

def linspace(start, stop, num, /, *, dtype=None, device=None, endpoint=True):
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.linspace(start, stop, num, dtype=dtype, endpoint=endpoint)

def ones(shape, /, *, dtype=None, device=None):
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.ones(shape, dtype=dtype)

def ones_like(x, /, *, dtype=None, device=None):
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.ones_like(x, dtype=dtype)

def zeros(shape, /, *, dtype=None, device=None):
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.zeros(shape, dtype=dtype)

def zeros_like(x, /, *, dtype=None, device=None):
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return np.zeros_like(x, dtype=dtype)
