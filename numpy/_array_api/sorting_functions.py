def argsort(x, /, *, axis=-1, descending=False, stable=True):
    from .. import argsort
    return argsort(x, axis=axis, descending=descending, stable=stable)

def sort(x, /, *, axis=-1, descending=False, stable=True):
    from .. import sort
    return sort(x, axis=axis, descending=descending, stable=stable)

__all__ = ['argsort', 'sort']
