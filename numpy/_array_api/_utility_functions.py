import numpy as np

def all(x, /, *, axis=None, keepdims=False):
    """
    Array API compatible wrapper for :py:func:`np.all <numpy.all>`.

    See its docstring for more information.
    """
    return np.all(x, axis=axis, keepdims=keepdims)

def any(x, /, *, axis=None, keepdims=False):
    """
    Array API compatible wrapper for :py:func:`np.any <numpy.any>`.

    See its docstring for more information.
    """
    return np.any(x, axis=axis, keepdims=keepdims)
