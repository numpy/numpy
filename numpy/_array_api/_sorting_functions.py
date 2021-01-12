import numpy as np

def argsort(x, /, *, axis=-1, descending=False, stable=True):
    """
    Array API compatible wrapper for :py:func:`np.argsort <numpy.argsort>`.

    See its docstring for more information.
    """
    # Note: this keyword argument is different, and the default is different.
    kind = 'stable' if stable else 'quicksort'
    res = np.argsort(x, axis=axis, kind=kind)
    if descending:
        res = np.flip(res, axis=axis)
    return res

def sort(x, /, *, axis=-1, descending=False, stable=True):
    """
    Array API compatible wrapper for :py:func:`np.sort <numpy.sort>`.

    See its docstring for more information.
    """
    # Note: this keyword argument is different, and the default is different.
    kind = 'stable' if stable else 'quicksort'
    res = np.sort(x, axis=axis, kind=kind)
    if descending:
        res = np.flip(res, axis=axis)
    return res
