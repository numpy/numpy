
import numeric as _nx
from numeric import asarray, empty, empty_like, isinf, signbit
import umath

__all__ = ['fix','mod','isneginf','isposinf','sign']

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

def mod(x,y,z=None):
    """ x - y*floor(x/y)
    
        For numeric arrays, x % y has the same sign as x while
        mod(x,y) has the same sign as y.
    """
    x = asarray(x)
    y = asarray(y)    
    if z is None:
        z = empty_like(x)
    tmp = _nx.floor(x*1.0/y)
    return _nx.subtract(x, y*tmp, z)

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

