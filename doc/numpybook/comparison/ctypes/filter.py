from __future__ import division, absolute_import, print_function

__all__ = ['filter2d']

import numpy as N
import os
import ctypes

_path = os.path.dirname('__file__')
lib = N.ctypeslib.load_library('code', _path)

lib.dfilter2d.restype = None
lib.dfilter2d.argtypes = [N.ctypeslib.ndpointer(float, ndim=2,
                                                flags='aligned'),
                          N.ctypeslib.ndpointer(float, ndim=2,
                                                flags='aligned, contiguous,'\
                                                'writeable'),
                          ctypes.POINTER(N.ctypeslib.c_intp),
                          ctypes.POINTER(N.ctypeslib.c_intp)]

def filter2d(a):
    a = N.require(a, float, ['ALIGNED'])
    b = N.zeros_like(a)
    lib.dfilter2d(a, b, a.ctypes.strides, a.ctypes.shape)
    return b
