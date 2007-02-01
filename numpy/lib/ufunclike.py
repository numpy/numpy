"""
Module of functions that are like ufuncs in acting on arrays and optionally
storing results in an output array.
"""
__all__ = ['fix', 'isneginf', 'isposinf', 'log2']

import numpy.core.numeric as nx
from numpy.core.numeric import asarray, empty, isinf, signbit, asanyarray
import numpy.core.umath as umath

def fix(x, y=None):
    """ Round x to nearest integer towards zero.
    """
    x = asanyarray(x)
    if y is None:
        y = nx.floor(x)
    else:
        nx.floor(x, y)
    if x.ndim == 0:
        if (x<0):
            y += 1
    else:
        y[x<0] = y[x<0]+1
    return y

def isposinf(x, y=None):
    """Return a boolean array y with y[i] True for x[i] = +Inf.

    If y is an array, the result replaces the contents of y.
    """
    if y is None:
        y = empty(x.shape, dtype=nx.bool_)
    umath.logical_and(isinf(x), ~signbit(x), y)
    return y

def isneginf(x, y=None):
    """Return a boolean array y with y[i] True for x[i] = -Inf.

    If y is an array, the result replaces the contents of y.
    """
    if y is None:
        y = empty(x.shape, dtype=nx.bool_)
    umath.logical_and(isinf(x), signbit(x), y)
    return y

_log2 = umath.log(2)
def log2(x, y=None):
    """Returns the base 2 logarithm of x

    If y is an array, the result replaces the contents of y.
    """
    x = asanyarray(x)
    if y is None:
        y = umath.log(x)
    else:
        umath.log(x, y)
    y /= _log2
    return y
