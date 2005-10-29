"""
Module of functions that are like ufuncs in acting on arrays and optionally
storing results in an output array.
"""
__all__ = ['fix', 'isneginf', 'isposinf', 'sign', 'log2']

import numeric as nx
from numeric import asarray, empty, empty_like, isinf, signbit, zeros
import umath

def fix(x, y=None):
    """ Round x to nearest integer towards zero.
    """
    x = asarray(x)
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

def sign(x, y=None):
    """sign(x) gives an array with shape of x with elexents defined by sign
    function:  where x is less than 0 return -1, where x greater than 0, a=1,
    elsewhere a=0.
    """
    x = asarray(x)
    if y is None:
        y = zeros(x.shape, dtype=nx.int_)
    if x.ndim == 0:
        if x < 0:
            y -= 1
        elif x > 0:
            y += 1
    else:
        y[x<0] = -1
        y[x>0] = 1
    return y

_log2 = umath.log(2)
def log2(x, y=None):
    """Returns the base 2 logarithm of x

    If y is an array, the result replaces the contents of y.
    """
    x = asarray(x)
    if y is None:
        y = umath.log(x)
    else:
        umath.log(x, y)
    y /= _log2
    return y

