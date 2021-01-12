def arange(start, /, *, stop=None, step=1, dtype=None, device=None):
    from .. import arange
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return arange(start, stop=stop, step=step, dtype=dtype)

def empty(shape, /, *, dtype=None, device=None):
    from .. import empty
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return empty(shape, dtype=dtype)

def empty_like(x, /, *, dtype=None, device=None):
    from .. import empty_like
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return empty_like(x, dtype=dtype)

def eye(N, /, *, M=None, k=0, dtype=None, device=None):
    from .. import eye
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return eye(N, M=M, k=k, dtype=dtype)

def full(shape, fill_value, /, *, dtype=None, device=None):
    from .. import full
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return full(shape, fill_value, dtype=dtype)

def full_like(x, fill_value, /, *, dtype=None, device=None):
    from .. import full_like
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return full_like(x, fill_value, dtype=dtype)

def linspace(start, stop, num, /, *, dtype=None, device=None, endpoint=True):
    from .. import linspace
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return linspace(start, stop, num, dtype=dtype, endpoint=endpoint)

def ones(shape, /, *, dtype=None, device=None):
    from .. import ones
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return ones(shape, dtype=dtype)

def ones_like(x, /, *, dtype=None, device=None):
    from .. import ones_like
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return ones_like(x, dtype=dtype)

def zeros(shape, /, *, dtype=None, device=None):
    from .. import zeros
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return zeros(shape, dtype=dtype)

def zeros_like(x, /, *, dtype=None, device=None):
    from .. import zeros_like
    if device is not None:
        # Note: Device support is not yet implemented on ndarray
        raise NotImplementedError("Device support is not yet implemented")
    return zeros_like(x, dtype=dtype)

__all__ = ['arange', 'empty', 'empty_like', 'eye', 'full', 'full_like', 'linspace', 'ones', 'ones_like', 'zeros', 'zeros_like']
