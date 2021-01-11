def argmax(x, /, *, axis=None, keepdims=False):
    from .. import argmax
    return argmax(x, axis=axis, keepdims=keepdims)

def argmin(x, /, *, axis=None, keepdims=False):
    from .. import argmin
    return argmin(x, axis=axis, keepdims=keepdims)

def nonzero(x, /):
    from .. import nonzero
    return nonzero(x)

def where(condition, x1, x2, /):
    from .. import where
    return where(condition, x1, x2)

__all__ = ['argmax', 'argmin', 'nonzero', 'where']
