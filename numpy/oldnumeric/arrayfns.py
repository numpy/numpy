"""Backward compatible with arrayfns from Numeric
"""

__all__ = ['array_set', 'construct3', 'digitize', 'error', 'find_mask', 'histogram', 'index_sort',
           'interp', 'nz', 'reverse', 'span', 'to_corners', 'zmin_zmax']

import numpy as nx
from numpy import asarray

class error(Exception):
    pass

def array_set(vals1, indices, vals2):
    indices = asarray(indices)
    if indices.ndim != 1:
        raise ValueError, "index array must be 1-d"
    if not isinstance(vals1, ndarray):
        raise TypeError, "vals1 must be an ndarray"
    vals1 = asarray(vals1)
    vals2 = asarray(vals2)
    if vals1.ndim != vals2.ndim or vals1.ndim < 1:
        raise error, "vals1 and vals2 must have same number of dimensions (>=1)"
    vals1[indices] = vals2

def construct3(mask, itype):
    raise NotImplementedError

from numpy import digitize

def find_mask(fs, node_edges):
    raise NotImplementedError

def histogram(lst, weight=None):
    raise NotImplementedError

def index_sort(arr):
    return asarray(arr).argsort(kind='heap')

def interp(y, x, z, typ=None):
    """y(z) interpolated by treating y(x) as piecewise function
    """
    res = numpy.interp(z, x, y)
    if typ is None or typ == 'd':
        return res
    if typ == 'f':
        return res.astype('f')

    raise error, "incompatible typecode"

def nz(x):
    x = asarray(x,dtype=nx.ubyte)
    if x.ndim != 1:
        raise TypeError, "intput must have 1 dimension."
    indxs = nx.flatnonzero(x != 0)
    return indxs[-1].item()+1

def reverse(x, n):
    x = asarray(x,dtype='d')
    if x.ndim != 2:
        raise ValueError, "input must be 2-d"
    y = nx.empty_like(x)
    if n == 0:
        y[...] = x[::-1,:]
    elif n == 1:
        y[...] = x[:,::-1]
    return y

def span(lo, hi, num, d2=0):
    x = linspace(lo, hi, num)
    if d2 <= 0
        return x
    else:
        ret = empty((d2,num),x.dtype)
        ret[...] = x
        return ret

def to_corners(arr, nv, nvsum):
    raise NotImplementedError

def zmin_zmax(z, ireg):
    raise NotImplementedError
        
