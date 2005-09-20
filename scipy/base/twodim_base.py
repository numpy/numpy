""" Basic functions for manipulating 2d arrays

"""

__all__ = ['diag','eye','fliplr','flipud','rot90','tri','triu','tril']

from numeric import *
import sys

def fliplr(m):
    """ returns a 2-d array m with the rows preserved and columns flipped 
        in the left/right direction.  Only works with 2-d arrays.
    """
    m = asarray(m)
    if len(m.shape) != 2:
        raise ValueError, "Input must be 2-d."
    return m[:, ::-1]

def flipud(m):
    """ returns a 2-d array with the columns preserved and rows flipped in
        the up/down direction.  Only works with 2-d arrays.
    """
    m = asarray(m)
    if len(m.shape) != 2:
        raise ValueError, "Input must be 2-d."
    return m[::-1]
    
# reshape(x, m, n) is not used, instead use reshape(x, (m, n))

def rot90(m, k=1):
    """ returns the 2-d array found by rotating m by k*90 degrees in the 
        counterclockwise direction.
    """
    m = asarray(m)
    if len(m.shape) != 2:
        raise ValueError, "Input must be 2-d."
    k = k % 4
    if k == 0: return m
    elif k == 1: return transpose(fliplr(m))
    elif k == 2: return fliplr(flipud(m))
    else: return fliplr(transpose(m))  # k==3
    
def eye(N, M=None, k=0, typecode='d'):
    """ eye returns a N-by-M 2-d array where the  k-th diagonal is all ones, 
        and everything else is zeros.
    """
    if M is None: M = N
    if type(M) == type('d'): 
        typecode = M
        M = N
    m = equal(subtract.outer(arange(N), arange(M)),-k)
    if typecode is None:
        return m
    else:
        return m.astype(typecode)

def diag(v, k=0):
    """ returns the k-th diagonal if v is a array or returns a array 
        with v as the k-th diagonal if v is a vector.
    """
    v = asarray(v)
    s = v.shape
    if len(s)==1:
        n = s[0]+abs(k)
        if k > 0:
            v = concatenate((zeros(k, v.typecode()),v))
        elif k < 0:
            v = concatenate((v,zeros(-k, v.typecode())))
        return eye(n, k=k)*v
    elif len(s)==2:
        v = add.reduce(eye(s[0], s[1], k=k)*v)
        if k > 0: return v[k:]
        elif k < 0: return v[:k]
        else: return v
    else:
            raise ValueError, "Input must be 1- or 2-d."


def tri(N, M=None, k=0, dtype=None):
    """ returns a N-by-M array where all the diagonals starting from
        lower left corner up to the k-th are all ones.
    """
    if M is None: M = N
    if type(M) == type('d'):
        #pearu: any objections to remove this feature?
        #       As tri(N,'d') is equivalent to tri(N,dtype='d')
        dtype = M
        M = N
    m = greater_equal(subtract.outer(arange(N), arange(M)),-k)
    if dtype is None:
        return m
    else:
        return m.astype(dtype)

def tril(m, k=0):
    """ returns the elements on and below the k-th diagonal of m.  k=0 is the
        main diagonal, k > 0 is above and k < 0 is below the main diagonal.
    """
    m = asarray(m)
    out = tri(m.shape[0], m.shape[1], k=k, dtype=m.dtype)*m
    return out

def triu(m, k=0):
    """ returns the elements on and above the k-th diagonal of m.  k=0 is the
        main diagonal, k > 0 is above and k < 0 is below the main diagonal.
    """
    m = asarray(m)
    out = (1-tri(m.shape[0], m.shape[1], k-1, m.dtype))*m
    return out




