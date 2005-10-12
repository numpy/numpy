
import numeric as _nx
from numeric import asarray, empty, empty_like, isinf, signbit
import umath

__all__ = ['fix','isneginf','isposinf','sign']

def fix(x, y=None):
    """ Round x to nearest integer towards zero.
    """
    x = asarray(x)    
    if y is None:
        y = _nx.floor(x)
    else:
        _nx.floor(x,y)
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
        y = empty(x.shape, dtype=_nx.int_)
    y[x<0] = -1
    y[x>0] = 1
    y[x==0] = 0
    return y

