""" Basic functions for manipulating 2d arrays

"""

__all__ = ['diag','eye','fliplr','flipud','rot90','bmat']

from Numeric import *
from type_check import asarray
import Matrix

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
    
def eye(N, M=None, k=0, typecode='d'):
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


def _from_string(str,gdict,ldict):
    rows = str.split(';')
    rowtup = []
    for row in rows:
        trow = row.split(',')
        coltup = []
        for col in trow:
            col = col.strip()
            try:
                thismat = gdict[col]
            except KeyError:
                try:
                    thismat = ldict[col]
                except KeyError:
                    raise KeyError, "%s not found" % (col,)
                                    
            coltup.append(thismat)
        rowtup.append(concatenate(coltup,axis=-1))
    return concatenate(rowtup,axis=0)

import sys
def bmat(obj,gdict=None,ldict=None):
    """Build a matrix object from string, nested sequence, or array.

    Ex:  F = bmat('A, B; C, D')  
         F = bmat([[A,B],[C,D]])
         F = bmat(r_[c_[A,B],c_[C,D]])

        all produce the same Matrix Object    [ A  B ]
                                              [ C  D ]
                                      
        if A, B, C, and D are appropriately shaped 2-d arrays.
    """
    if isinstance(obj, types.StringType):
        if gdict is None:
            # get previous frame
            frame = sys._getframe().f_back
            glob_dict = frame.f_globals
            loc_dict = frame.f_locals
        else:
            glob_dict = gdict
            loc_dict = ldict
        
        return Matrix.Matrix(_from_string(obj, glob_dict, loc_dict))
    
    if isinstance(obj, (types.TupleType, types.ListType)):
        # [[A,B],[C,D]]
        arr_rows = []
        for row in obj:
            if isinstance(row, ArrayType):  # not 2-d
                return Matrix.Matrix(concatenate(obj,axis=-1))
            else:
                arr_rows.append(concatenate(row,axis=-1))
        return Matrix.Matrix(concatenate(arr_rows,axis=0))
    if isinstance(obj, ArrayType):
        return Matrix.Matrix(obj)

