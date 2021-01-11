def all(x, /, *, axis=None, keepdims=False):
    from .. import all
    return all(x, axis=axis, keepdims=keepdims)

def any(x, /, *, axis=None, keepdims=False):
    from .. import any
    return any(x, axis=axis, keepdims=keepdims)

__all__ = ['all', 'any']
