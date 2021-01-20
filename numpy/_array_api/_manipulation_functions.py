from __future__ import annotations

from ._types import Optional, Tuple, Union, array

import numpy as np

def concat(arrays: Tuple[array], /, *, axis: Optional[int] = 0) -> array:
    """
    Array API compatible wrapper for :py:func:`np.concatenate <numpy.concatenate>`.

    See its docstring for more information.
    """
    # Note: the function name is different here
    return np.concatenate(arrays, axis=axis)

def expand_dims(x: array, axis: int, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.expand_dims <numpy.expand_dims>`.

    See its docstring for more information.
    """
    return np.expand_dims(x, axis)

def flip(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> array:
    """
    Array API compatible wrapper for :py:func:`np.flip <numpy.flip>`.

    See its docstring for more information.
    """
    return np.flip(x, axis=axis)

def reshape(x: array, shape: Tuple[int, ...], /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.reshape <numpy.reshape>`.

    See its docstring for more information.
    """
    return np.reshape(x, shape)

def roll(x: array, shift: Union[int, Tuple[int, ...]], /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> array:
    """
    Array API compatible wrapper for :py:func:`np.roll <numpy.roll>`.

    See its docstring for more information.
    """
    return np.roll(x, shift, axis=axis)

def squeeze(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> array:
    """
    Array API compatible wrapper for :py:func:`np.squeeze <numpy.squeeze>`.

    See its docstring for more information.
    """
    return np.squeeze(x, axis=axis)

def stack(arrays: Tuple[array], /, *, axis: Optional[int] = 0) -> array:
    """
    Array API compatible wrapper for :py:func:`np.stack <numpy.stack>`.

    See its docstring for more information.
    """
    return np.stack(arrays, axis=axis)
