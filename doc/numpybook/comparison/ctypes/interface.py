from __future__ import division, absolute_import, print_function

__all__ = ['add', 'filter2d']

import numpy as N
import os
import ctypes

_path = os.path.dirname('__file__')
lib = N.ctypeslib.load_library('code', _path)
_typedict = {'zadd' : complex,
             'sadd' : N.single,
             'cadd' : N.csingle,
             'dadd' : float}
for name in _typedict.keys():
    val = getattr(lib, name)
    val.restype = None
    _type = _typedict[name]
    val.argtypes = [N.ctypeslib.ndpointer(_type, flags='aligned, contiguous'),
                    N.ctypeslib.ndpointer(_type, flags='aligned, contiguous'),
                    N.ctypeslib.ndpointer(_type, flags='aligned, contiguous,'\
                                          'writeable'),
                    N.ctypeslib.c_intp]

lib.dfilter2d.restype=None
lib.dfilter2d.argtypes = [N.ctypeslib.ndpointer(float, ndim=2,
                                                flags='aligned'),
                          N.ctypeslib.ndpointer(float, ndim=2,
                                                flags='aligned, contiguous,'\
                                                'writeable'),
                          ctypes.POINTER(N.ctypeslib.c_intp),
                          ctypes.POINTER(N.ctypeslib.c_intp)]

def select(dtype):
    if dtype.char in ['?bBhHf']:
        return lib.sadd, N.single
    elif dtype.char in ['F']:
        return lib.cadd, N.csingle
    elif dtype.char in ['DG']:
        return lib.zadd, complex
    else:
        return lib.dadd, float
    return func, ntype

def add(a, b):
    requires = ['CONTIGUOUS', 'ALIGNED']
    a = N.asanyarray(a)
    func, dtype = select(a.dtype)
    a = N.require(a, dtype, requires)
    b = N.require(b, dtype, requires)
    c = N.empty_like(a)
    func(a,b,c,a.size)
    return c

def filter2d(a):
    a = N.require(a, float, ['ALIGNED'])
    b = N.zeros_like(a)
    lib.dfilter2d(a, b, a.ctypes.strides, a.ctypes.shape)
    return b
