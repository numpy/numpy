import numpy as np

def unique(x, /, *, return_counts=False, return_index=False, return_inverse=False, sorted=True):
    """
    Array API compatible wrapper for :py:func:`np.unique <numpy.unique>`.

    See its docstring for more information.
    """
    return np.unique(x, return_counts=return_counts, return_index=return_index, return_inverse=return_inverse, sorted=sorted)
