
import numeric as _nx
from numeric import asarray, empty, empty_like, isinf, signbit, zeros
import umath

__all__ = ['fix','isneginf','isposinf','sign','log2']

def fix(x, y=None):
    """ Round x to nearest integer towards zero.
    """
    x = asarray(x)
    if y is None:
        y = _nx.floor(x)
    else:
        _nx.floor(x,y)
    if x.ndim == 0:
        if (x<0):
            y += 1
    else:
        y[x<0] = y[x<0]+1
    return y

def isposinf(x, y=None):
    if y is None:
        y = empty(x.shape, dtype='?')
    umath.logical_and(isinf(x), ~signbit(x), y)
    return y
    
def isneginf(x, y=None):
    if y is None:
        y = empty(x.shape, dtype='?')
    umath.logical_and(isinf(x), signbit(x), y)
    return y

def sign(x, y=None):
    """sign(x) gives an array with shape of x with elexents defined by sign
    function:  where x is less than 0 return -1, where x greater than 0, a=1,
    elsewhere a=0.
    """
    x = asarray(x)
    if y is None:
        y = zeros(x.shape, dtype=_nx.int_)
    if x.ndim == 0:
        if x<0:
            y -= 1
        elif x>0:
            y += 1
    else:
        y[x<0] = -1
        y[x>0] = 1
    return y

_log2 = umath.log(2)
def log2(x, y=None):
    x = asarray(x)
    if y is None:
        y = umath.log(x)
    else:
        res = umath.log(x,y)
    y /= _log2
    return y
    
