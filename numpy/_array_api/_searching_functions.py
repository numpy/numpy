import numpy as np

def argmax(x, /, *, axis=None, keepdims=False):
    """
    Array API compatible wrapper for :py:func:`np.argmax <numpy.argmax>`.

    See its docstring for more information.
    """
    return np.argmax(x, axis=axis, keepdims=keepdims)

def argmin(x, /, *, axis=None, keepdims=False):
    """
    Array API compatible wrapper for :py:func:`np.argmin <numpy.argmin>`.

    See its docstring for more information.
    """
    return np.argmin(x, axis=axis, keepdims=keepdims)

def nonzero(x, /):
    """
    Array API compatible wrapper for :py:func:`np.nonzero <numpy.nonzero>`.

    See its docstring for more information.
    """
    return np.nonzero(x)

def where(condition, x1, x2, /):
    """
    Array API compatible wrapper for :py:func:`np.where <numpy.where>`.

    See its docstring for more information.
    """
    return np.where(condition, x1, x2)
