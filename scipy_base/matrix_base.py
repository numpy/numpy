""" Basic functions for manipulating 2d arrays

"""

__all__ = ['diag','eye','fliplr','flipud','hankel','rot90','tri',
           'tril','triu','toeplitz','all_mat']
           
# These are from Numeric
import Matrix
from Numeric import *
from fastumath import *
from type_check import isscalar
from index_tricks import mgrid,r_,c_
# Elementary Matrices

# zeros is from matrixmodule in C
# ones is from Numeric.py

    
def fliplr(m):
    """ returns a 2-D matrix m with the rows preserved and columns flipped 
        in the left/right direction.  Only works with 2-D arrays.
    """
    m = asarray(m)
    if len(m.shape) != 2:
        raise ValueError, "Input must be 2-D."
    return m[:, ::-1]

def flipud(m):
    """ returns a 2-D matrix with the columns preserved and rows flipped in
        the up/down direction.  Only works with 2-D arrays.
    """
    m = asarray(m)
    if len(m.shape) != 2:
        raise ValueError, "Input must be 2-D."
    return m[::-1]
    
# reshape(x, m, n) is not used, instead use reshape(x, (m, n))

def rot90(m, k=1):
    """ returns the matrix found by rotating m by k*90 degrees in the 
        counterclockwise direction.
    """
    m = asarray(m)
    if len(m.shape) != 2:
        raise ValueError, "Input must be 2-D."
    k = k % 4
    if k == 0: return m
    elif k == 1: return transpose(fliplr(m))
    elif k == 2: return fliplr(flipud(m))
    else: return fliplr(transpose(m))  # k==3

def tri(N, M=None, k=0, typecode=None):
    """ returns a N-by-M matrix where all the diagonals starting from 
        lower left corner up to the k-th are all ones.
    """
    if M is None: M = N
    if type(M) == type('d'): 
        typecode = M
        M = N
    m = greater_equal(subtract.outer(arange(N), arange(M)),-k)
    if typecode is None:
        return m
    else:
        return m.astype(typecode)
    
def eye(N, M=None, k=0, typecode=None):
    """ eye returns a N-by-M matrix where the  k-th diagonal is all ones, 
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
    """ returns the k-th diagonal if v is a matrix or returns a matrix 
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
            raise ValueError, "Input must be 1- or 2-D."

#-----------------------------------------------------------------------------
# move all these
#-----------------------------------------------------------------------------

def tril(m, k=0):
    """ returns the elements on and below the k-th diagonal of m.  k=0 is the
        main diagonal, k > 0 is above and k < 0 is below the main diagonal.
    """
    svsp = m.spacesaver()
    m = asarray(m,savespace=1)
    out = tri(m.shape[0], m.shape[1], k=k, typecode=m.typecode())*m
    out.savespace(svsp)
    return out

def triu(m, k=0):
    """ returns the elements on and above the k-th diagonal of m.  k=0 is the
        main diagonal, k > 0 is above and k < 0 is below the main diagonal.
    """
    svsp = m.spacesaver()
    m = asarray(m,savespace=1)
    out = (1-tri(m.shape[0], m.shape[1], k-1, m.typecode()))*m
    out.savespace(svsp)
    return out

def toeplitz(c,r=None):
    """Construct a toeplitz matrix (i.e. a matrix with constant diagonals).

    Description:

       toeplitz(c,r) is a non-symmetric Toeplitz matrix with c as its first
       column and r as its first row.

       toeplitz(c) is a symmetric (Hermitian) Toeplitz matrix (r=c). 

    See also: hankel
    """
    if isscalar(c) or isscalar(r):
        return c   
    if r is None:
        r = c
        r[0] = conjugate(r[0])
        c = conjugate(c)
    r,c = map(asarray,(r,c))
    r,c = map(ravel,(r,c))
    rN,cN = map(len,(r,c))
    if r[0] != c[0]:
        print "Warning: column and row values don't agree; column value used."
    vals = r_[r[rN-1:0:-1], c]
    cols = mgrid[0:cN]
    rows = mgrid[rN:0:-1]
    indx = cols[:,NewAxis]*ones((1,rN)) + \
           rows[NewAxis,:]*ones((cN,1)) - 1
    return take(vals, indx)


def hankel(c,r=None):
    """Construct a hankel matrix (i.e. matrix with constant anti-diagonals).

    Description:

      hankel(c,r) is a Hankel matrix whose first column is c and whose
      last row is r.

      hankel(c) is a square Hankel matrix whose first column is C.
      Elements below the first anti-diagonal are zero.

    See also:  toeplitz
    """
    if isscalar(c) or isscalar(r):
        return c   
    if r is None:
        r = zeros(len(c))
    elif r[0] != c[-1]:
        print "Warning: column and row values don't agree; column value used."
    r,c = map(asarray,(r,c))
    r,c = map(ravel,(r,c))
    rN,cN = map(len,(r,c))
    vals = r_[c, r[1:rN]]
    cols = mgrid[1:cN+1]
    rows = mgrid[0:rN]
    indx = cols[:,NewAxis]*ones((1,rN)) + \
           rows[NewAxis,:]*ones((cN,1)) - 1
    return take(vals, indx)

def all_mat(args):
    return map(Matrix.Matrix,args)

#-----------------------------------------------------------------------------
# Test Routines
#-----------------------------------------------------------------------------

def test(level=10):
    from scipy_base.testing import module_test
    module_test(__name__,__file__,level=level)

def test_suite(level=1):
    from scipy_base.testing import module_test_suite
    return module_test_suite(__name__,__file__,level=level)

if __name__ == '__main__':
    print 'float epsilon:',float_epsilon
    print 'float tiny:',float_tiny
    print 'double epsilon:',double_epsilon
    print 'double tiny:',double_tiny
