# This module is for compatibility only.  All functions are defined elsewhere.

from numpy.oldnumeric import *

__all__ = numpy.oldnumeric.__all__

__all__ += ['rand', 'tril', 'trapz', 'hanning', 'rot90', 'triu', 'diff', 'angle', 'roots', 'ptp', 'kaiser', 'randn', 'cumprod', 'diag', 'msort', 'LinearAlgebra', 'RandomArray', 'prod', 'std', 'hamming', 'flipud', 'max', 'blackman', 'corrcoef', 'bartlett', 'eye', 'squeeze', 'sinc', 'tri', 'cov', 'svd', 'min', 'median', 'fliplr', 'eig', 'mean']

import linear_algebra as LinearAlgebra
import random_array as RandomArray
from numpy import tril, trapz as _Ntrapz, hanning, rot90, triu, diff, \
     angle, roots, ptp as _Nptp, kaiser, cumprod as _Ncumprod, \
     diag, msort, prod as _Nprod, std as _Nstd, hamming, flipud, \
     amax as _Nmax, amin as _Nmin, blackman, bartlett, corrcoef as _Ncorrcoef,\
     cov as _Ncov, squeeze, sinc, median, fliplr, mean as _Nmean

from numpy.linalg import eig, svd
from numpy.random import rand, randn
     
from typeconv import oldtype2dtype as o2d

def eye(N, M=None, k=0, typecode=None):
    """ eye returns a N-by-M 2-d array where the  k-th diagonal is all ones,
        and everything else is zeros.
    """
    dtype = o2d[typecode]
    if M is None: M = N
    m = nn.equal(nn.subtract.outer(nn.arange(N), nn.arange(M)),-k)
    if m.dtype != dtype:
        return m.astype(dtype)
    
def tri(N, M=None, k=0, typecode=None):
    """ returns a N-by-M array where all the diagonals starting from
        lower left corner up to the k-th are all ones.
    """
    dtype = o2d[typecode]
    if M is None: M = N
    m = nn.greater_equal(nn.subtract.outer(nn.arange(N), nn.arange(M)),-k)
    if m.dtype != dtype:
        return m.astype(dtype)
    
def trapz(y, x=None, axis=-1):
    return _Ntrapz(y, x, axis=axis)

def ptp(x, axis=0):
    return _Nptp(x, axis)

def cumprod(x, axis=0):
    return _Ncumprod(x, axis)

def max(x, axis=0):
    return _Nmax(x, axis)

def min(x, axis=0):
    return _Nmin(x, axis)

def prod(x, axis=0):
    return _Nprod(x, axis)

def std(x, axis=0):
    return _Nstd(x, axis)

def mean(x, axis=0):
    return _Nmean(x, axis)

def cov(m, y=None, rowvar=0, bias=0):
    return _Ncov(m, y, rowvar, bias)

def corrcoef(x, y=None):
    return _Ncorrcoef(x,y,0,0)


