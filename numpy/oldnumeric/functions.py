# Functions that should behave the same as Numeric and need changing

import numpy as N
import numpy.core.multiarray as mu
import numpy.core.numeric as nn
from typeconv import convtypecode

__all__ = ['take', 'repeat', 'sum', 'product', 'sometrue', 'alltrue',
           'cumsum', 'cumproduct']
__all__ += ['ones', 'empty', 'identity', 'zeros', 'array', 'asarray', 'nonzero',
            'reshape', 'arange', 'fromstring', 'ravel', 'trace', 'indices',
            'where']

def take(a, indicies, axis=0):
    return N.take(a, indicies, axis)

def repeat(a, repeats, axis=0):
    return N.repeats(a, repeats, axis)

def sum(x, axis=0):
    return N.sum(x, axis)

def product(x, axis=0):
    return N.product(x, axis)

def sometrue(x, axis=0):
    return N.sometrue(x, axis)

def alltrue(x, axis=0):
    return N.alltrue(x, axis)

def cumsum(x, axis=0):
    return N.cumsum(x, axis)

def cumproduct(x, axis=0):
    return N.cumproduct(x, axis)

    
def ones(shape, typecode='l', savespace=0, dtype=None):
    """ones(shape, dtype=int) returns an array of the given
    dimensions which is initialized to all ones.
    """
    dtype = convtypecode(typecode,dtype)
    a = mu.empty(shape, dtype)
    a.fill(1)
    return a

def zeros(shape, typecode='l', savespace=0, dtype=None):
    """zeros(shape, dtype=int) returns an array of the given
    dimensions which is initialized to all zeros
    """
    dtype = convtypecode(typecode,dtype)            
    return mu.zeros(shape, dtype)

def identity(n,typecode='l', dtype=None):
    """identity(n) returns the identity 2-d array of shape n x n.
    """
    dtype = convtypecode(typecode, dtype)
    return nn.identity(n, dtype)

def empty(shape, typecode='l', dtype=None):
    dtype = convtypecode(typecode, dtype)    
    return mu.empty(shape, dtype, order)


def array(sequence, typecode=None, copy=1, savespace=0):
    pass

def asarray(a, typecode=None):
    pass

def nonzero(a):
    pass

def reshape(a, shape):
    pass

def arange(start, stop=None, step=1, typecode=None):
    pass

def fromstring(string, typecode='l', count=-1):
    pass

def ravel(m):
    pass

def trace(a, offset=0, axis1=0, axis2=1):
    pass

def indices(dimensions, typecode=None):
    pass

def where(condition, x, y):
    pass
