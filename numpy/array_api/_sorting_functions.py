from __future__ import annotations
import numpy as np

from ._array_object import Array
from ._dtypes import _real_numeric_dtypes

def argsort(x: Array, /, *, axis: int = -1, descending: bool = False, stable: bool = True) -> Array:
    """
    Argsorts the input Array along the specified axis.

    Args:
        x (Array): The input Array to be sorted.
        axis (int, optional): The axis along which to sort. Default is -1.
        descending (bool, optional): If True, sort in descending order. Default is False.
        stable (bool, optional): If True, use stable sorting algorithm. Default is True.

    Returns:
        Array: A new sorted Array.

    Raises:
        TypeError: If the input Array's dtype is not a real numeric dtype.
    """
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in argsort")
    
    kind = "stable" if stable else "quicksort"
    res = np.argsort(x._array, axis=axis, kind=kind)
    
    if descending:
        res = invert_sort(res, axis)
    
    return Array._new(res)

def sort(x: Array, /, *, axis: int = -1, descending: bool = False, stable: bool = True) -> Array:
    """
    Sorts the input Array along the specified axis.

    Args:
        x (Array): The input Array to be sorted.
        axis (int, optional): The axis along which to sort. Default is -1.
        descending (bool, optional): If True, sort in descending order. Default is False.
        stable (bool, optional): If True, use stable sorting algorithm. Default is True.

    Returns:
        Array: A new sorted Array.

    Raises:
        TypeError: If the input Array's dtype is not a real numeric dtype.
    """
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in sort")
    
    kind = "stable" if stable else "quicksort"
    res = np.sort(x._array, axis=axis, kind=kind)
    
    if descending:
        res = invert_sort(res, axis)
    
    return Array._new(res)

def invert_sort(arr, axis):
    """
    Inverts the sorting order of an array along the specified axis.

    Args:
        arr (np.ndarray): The input array to be inverted.
        axis (int): The axis along which to invert the sorting order.

    Returns:
        np.ndarray: The array with inverted sorting order along the specified axis.
    """
    return np.flip(arr, axis=axis)
