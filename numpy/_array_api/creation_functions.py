def arange(start, /, *, stop=None, step=1, dtype=None):
    from .. import arange
    return arange(start, stop=stop, step=step, dtype=dtype)

def empty(shape, /, *, dtype=None):
    from .. import empty
    return empty(shape, dtype=dtype)

def empty_like(x, /, *, dtype=None):
    from .. import empty_like
    return empty_like(x, dtype=dtype)

def eye(N, /, *, M=None, k=0, dtype=None):
    from .. import eye
    return eye(N, M=M, k=k, dtype=dtype)

def full(shape, fill_value, /, *, dtype=None):
    from .. import full
    return full(shape, fill_value, dtype=dtype)

def full_like(x, fill_value, /, *, dtype=None):
    from .. import full_like
    return full_like(x, fill_value, dtype=dtype)

def linspace(start, stop, num, /, *, dtype=None, endpoint=True):
    from .. import linspace
    return linspace(start, stop, num, dtype=dtype, endpoint=endpoint)

def ones(shape, /, *, dtype=None):
    from .. import ones
    return ones(shape, dtype=dtype)

def ones_like(x, /, *, dtype=None):
    from .. import ones_like
    return ones_like(x, dtype=dtype)

def zeros(shape, /, *, dtype=None):
    from .. import zeros
    return zeros(shape, dtype=dtype)

def zeros_like(x, /, *, dtype=None):
    from .. import zeros_like
    return zeros_like(x, dtype=dtype)

__all__ = ['arange', 'empty', 'empty_like', 'eye', 'full', 'full_like', 'linspace', 'ones', 'ones_like', 'zeros', 'zeros_like']
