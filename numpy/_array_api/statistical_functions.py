def max(x, /, *, axis=None, keepdims=False):
    from .. import max
    return max(x, axis=axis, keepdims=keepdims)

def mean(x, /, *, axis=None, keepdims=False):
    from .. import mean
    return mean(x, axis=axis, keepdims=keepdims)

def min(x, /, *, axis=None, keepdims=False):
    from .. import min
    return min(x, axis=axis, keepdims=keepdims)

def prod(x, /, *, axis=None, keepdims=False):
    from .. import prod
    return prod(x, axis=axis, keepdims=keepdims)

def std(x, /, *, axis=None, correction=0.0, keepdims=False):
    from .. import std
    return std(x, axis=axis, correction=correction, keepdims=keepdims)

def sum(x, /, *, axis=None, keepdims=False):
    from .. import sum
    return sum(x, axis=axis, keepdims=keepdims)

def var(x, /, *, axis=None, correction=0.0, keepdims=False):
    from .. import var
    return var(x, axis=axis, correction=correction, keepdims=keepdims)

__all__ = ['max', 'mean', 'min', 'prod', 'std', 'sum', 'var']
