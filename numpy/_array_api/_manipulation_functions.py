import numpy as np

def concat(arrays, /, *, axis=0):
    """
    Array API compatible wrapper for :py:func:`np.concatenate <numpy.concatenate>`.

    See its docstring for more information.
    """
    # Note: the function name is different here
    return np.concatenate(arrays, axis=axis)

def expand_dims(x, axis, /):
    """
    Array API compatible wrapper for :py:func:`np.expand_dims <numpy.expand_dims>`.

    See its docstring for more information.
    """
    return np.expand_dims(x, axis)

def flip(x, /, *, axis=None):
    """
    Array API compatible wrapper for :py:func:`np.flip <numpy.flip>`.

    See its docstring for more information.
    """
    return np.flip(x, axis=axis)

def reshape(x, shape, /):
    """
    Array API compatible wrapper for :py:func:`np.reshape <numpy.reshape>`.

    See its docstring for more information.
    """
    return np.reshape(x, shape)

def roll(x, shift, /, *, axis=None):
    """
    Array API compatible wrapper for :py:func:`np.roll <numpy.roll>`.

    See its docstring for more information.
    """
    return np.roll(x, shift, axis=axis)

def squeeze(x, /, *, axis=None):
    """
    Array API compatible wrapper for :py:func:`np.squeeze <numpy.squeeze>`.

    See its docstring for more information.
    """
    return np.squeeze(x, axis=axis)

def stack(arrays, /, *, axis=0):
    """
    Array API compatible wrapper for :py:func:`np.stack <numpy.stack>`.

    See its docstring for more information.
    """
    return np.stack(arrays, axis=axis)
