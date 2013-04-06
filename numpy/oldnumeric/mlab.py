"""This module is for compatibility only.  All functions are defined elsewhere.

"""
from __future__ import division, absolute_import, print_function

__all__ = ['rand', 'tril', 'trapz', 'hanning', 'rot90', 'triu', 'diff', 'angle',
           'roots', 'ptp', 'kaiser', 'randn', 'cumprod', 'diag', 'msort',
           'LinearAlgebra', 'RandomArray', 'prod', 'std', 'hamming', 'flipud',
           'max', 'blackman', 'corrcoef', 'bartlett', 'eye', 'squeeze', 'sinc',
           'tri', 'cov', 'svd', 'min', 'median', 'fliplr', 'eig', 'mean']

import numpy.oldnumeric.linear_algebra as LinearAlgebra
import numpy.oldnumeric.random_array as RandomArray
from numpy import tril, trapz as _Ntrapz, hanning, rot90, triu, diff, \
     angle, roots, ptp as _Nptp, kaiser, cumprod as _Ncumprod, \
     diag, msort, prod as _Nprod, std as _Nstd, hamming, flipud, \
     amax as _Nmax, amin as _Nmin, blackman, bartlett, \
     squeeze, sinc, median, fliplr, mean as _Nmean, transpose

from numpy.linalg import eig, svd
from numpy.random import rand, randn
import numpy as np

from .typeconv import convtypecode

def eye(N, M=None, k=0, typecode=None, dtype=None):
    """ eye returns a N-by-M 2-d array where the  k-th diagonal is all ones,
        and everything else is zeros.
    """
    dtype = convtypecode(typecode, dtype)
    if M is None: M = N
    m = np.equal(np.subtract.outer(np.arange(N), np.arange(M)),-k)
    if m.dtype != dtype:
        return m.astype(dtype)

def tri(N, M=None, k=0, typecode=None, dtype=None):
    """ returns a N-by-M array where all the diagonals starting from
        lower left corner up to the k-th are all ones.
    """
    dtype = convtypecode(typecode, dtype)
    if M is None: M = N
    m = np.greater_equal(np.subtract.outer(np.arange(N), np.arange(M)),-k)
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
    N = asarray(x).shape[axis]
    return _Nstd(x, axis)*sqrt(N/(N-1.))

def mean(x, axis=0):
    return _Nmean(x, axis)

# This is exactly the same cov function as in MLab
def cov(m, y=None, rowvar=0, bias=0):
    if y is None:
        y = m
    else:
        y = y
    if rowvar:
        m = transpose(m)
        y = transpose(y)
    if (m.shape[0] == 1):
        m = transpose(m)
    if (y.shape[0] == 1):
        y = transpose(y)
    N = m.shape[0]
    if (y.shape[0] != N):
        raise ValueError("x and y must have the same number of observations")
    m = m - _Nmean(m,axis=0)
    y = y - _Nmean(y,axis=0)
    if bias:
        fact = N*1.0
    else:
        fact = N-1.0
    return squeeze(dot(transpose(m), conjugate(y)) / fact)

from numpy import sqrt, multiply
def corrcoef(x, y=None):
    c = cov(x, y)
    d = diag(c)
    return c/sqrt(multiply.outer(d,d))

from .compat import *
from .functions import *
from .precision import *
from .ufuncs import *
from .misc import *

from . import compat
from . import precision
from . import functions
from . import misc
from . import ufuncs

import numpy
__version__ = numpy.__version__
del numpy

__all__ += ['__version__']
__all__ += compat.__all__
__all__ += precision.__all__
__all__ += functions.__all__
__all__ += ufuncs.__all__
__all__ += misc.__all__

del compat
del functions
del precision
del ufuncs
del misc
