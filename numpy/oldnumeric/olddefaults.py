__all__ = ['ones', 'empty', 'identity', 'zeros', 'eye', 'tri']

import numpy.core.multiarray as mu
import numpy.core.numeric as nn

def ones(shape, dtype=int, order='C'):
    """ones(shape, dtype=int) returns an array of the given
    dimensions which is initialized to all ones.
    """
    a = mu.empty(shape, dtype, order)
    a.fill(1)
    return a

def zeros(shape, dtype=int, order='C'):
    """zeros(shape, dtype=int) returns an array of the given
    dimensions which is initialized to all zeros
    """
    return mu.zeros(shape, dtype, order)

def identity(n,dtype=int):
    """identity(n) returns the identity 2-d array of shape n x n.
    """
    return nn.identity(n, dtype)

def eye(N, M=None, k=0, dtype=int):
    """ eye returns a N-by-M 2-d array where the  k-th diagonal is all ones,
        and everything else is zeros.
    """
    if M is None: M = N
    m = nn.equal(nn.subtract.outer(nn.arange(N), nn.arange(M)),-k)
    if m.dtype != dtype:
        return m.astype(dtype)
    
def tri(N, M=None, k=0, dtype=int):
    """ returns a N-by-M array where all the diagonals starting from
        lower left corner up to the k-th are all ones.
    """
    if M is None: M = N
    m = nn.greater_equal(nn.subtract.outer(nn.arange(N), nn.arange(M)),-k)
    if m.dtype != dtype:
        return m.astype(dtype)

def empty(shape, dtype=int, order='C'):
    return mu.empty(shape, dtype, order)

