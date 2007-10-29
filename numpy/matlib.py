from numpy.core.defmatrix import matrix, asmatrix
from numpy import ndarray, array
import numpy as N
from numpy import *

__version__ = N.__version__

__all__ = N.__all__[:] # copy numpy namespace
__all__ += ['rand', 'randn', 'repmat']

def empty(shape, dtype=None, order='C'):
    """return an empty matrix of the given shape
    """
    return ndarray.__new__(matrix, shape, dtype, order=order)

def ones(shape, dtype=None, order='C'):
    """return a matrix initialized to all ones
    """
    a = ndarray.__new__(matrix, shape, dtype, order=order)
    a.fill(1)
    return a

def zeros(shape, dtype=None, order='C'):
    """return a matrix initialized to all zeros
    """
    a = ndarray.__new__(matrix, shape, dtype, order=order)
    a.fill(0)
    return a

def identity(n,dtype=None):
    """identity(n) returns the identity matrix of shape n x n.
    """
    a = array([1]+n*[0],dtype=dtype)
    b = empty((n,n),dtype=dtype)
    b.flat = a
    return b

def eye(n,M=None, k=0, dtype=float):
    return asmatrix(N.eye(n,M,k,dtype))

def rand(*args):
    if isinstance(args[0], tuple):
        args = args[0]
    return asmatrix(N.random.rand(*args))

def randn(*args):
    if isinstance(args[0], tuple):
        args = args[0]
    return asmatrix(N.random.randn(*args))

def repmat(a, m, n):
    """Repeat a 0-d to 2-d array mxn times
    """
    a = asanyarray(a)
    ndim = a.ndim
    if ndim == 0:
        origrows, origcols = (1,1)
    elif ndim == 1:
        origrows, origcols = (1, a.shape[0])
    else:
        origrows, origcols = a.shape
    rows = origrows * m
    cols = origcols * n
    c = a.reshape(1,a.size).repeat(m, 0).reshape(rows, origcols).repeat(n,0)
    return c.reshape(rows, cols)
