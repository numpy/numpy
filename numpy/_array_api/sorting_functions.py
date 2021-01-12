def argsort(x, /, *, axis=-1, descending=False, stable=True):
    from .. import argsort
    from .. import flip
    # Note: this keyword argument is different, and the default is different.
    kind = 'stable' if stable else 'quicksort'
    res = argsort(x, axis=axis, kind=kind)
    if descending:
        res = flip(res, axis=axis)

def sort(x, /, *, axis=-1, descending=False, stable=True):
    from .. import sort
    from .. import flip
    # Note: this keyword argument is different, and the default is different.
    kind = 'stable' if stable else 'quicksort'
    res = sort(x, axis=axis, kind=kind)
    if descending:
        res = flip(res, axis=axis)

__all__ = ['argsort', 'sort']
