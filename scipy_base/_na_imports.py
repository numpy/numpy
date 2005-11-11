"""Imports from numarray for numerix, the numarray/Numeric interchangeability
module.  These array functions are used when numarray is chosen.
"""
from ppimport import ppimport, ppimport_attr
import numarray as _na
from numarray.numeric import *
import numarray.ufunc as fastumath

import _compiled_base
from _compiled_base import arraymap, _unique, _insert

try:
    from numarray.ieeespecial import isinf, isnan, isfinite
except ImportError,msg:
    isinf = isnan = isfinite = None
    print msg

from numarray.ieeespecial import \
     plus_inf as PINF, \
     minus_inf as NINF, \
     inf, \
     inf as infty, \
     inf as Infinity, \
     nan, \
     nan as NAN, \
     nan as Nan

try:
    from numarray.ieeespecial import \
         plus_zero as PZERO, \
         minus_zero as NZERO
except ImportError,msg:
    print msg

import numarray.linear_algebra as LinearAlgebra
import numarray.linear_algebra.mlab as MLab
import numarray.random_array as RandomArray
from numarray.fft import fft
try:
    from numarray.matrix import Matrix
except ImportError,msg:
    Matrix = None
    print msg
from numarray.linear_algebra import inverse, eigenvectors
from numarray.convolve import convolve, cross_correlate
from numarray.arrayprint import array2string

# LinearAlgebra = ppimport("numarray.linear_algebra")
# MLab = ppimport("numarray.mlab")
# inverse = ppimport_from("numarray.linear_algebra.inverse")
# eigenvectors = ppimport_from("numarray.linear_algebra.eigenvectors")
# convolve = ppimport_from("numarray.convolve.convolve")
# fft = ppimport_from("numarray.fft.fft")
# Matrix = ppimport_from("numarray.matrix.Matrix")
# RandomArray = ppimport("numarray.random_array")

class _TypeNamespace:
    """Numeric compatible type aliases for use with extension functions."""
    Int8          = typecode[Int8]
    UInt8         = typecode[UInt8]
    Int16         = typecode[Int16]
    UInt16        = typecode[UInt16]
    Int32         = typecode[Int32]
    UInt32        = typecode[UInt32]  
    Float32       = typecode[Float32]
    Float64       = typecode[Float64]
    Complex32     = typecode[Complex32]
    Complex64     = typecode[Complex64]

nx = _TypeNamespace()

def alter_numeric():
    pass

def restore_numeric():
    pass

conj = conjugate

UnsignedInt8 = UInt8
UnsignedInt16 = UInt16
UnsignedInt32 = UInt32

ArrayType = arraytype

class UfuncType(object):
    """numarray ufuncs work differently than Numeric ufuncs and
    have no single UfuncType... TBD"""
    pass


def zeros(shape, typecode='l', savespace=0):
    """scipy version of numarray.zeros() which supports creation of object
    arrays as well as numerical arrays.
    """
    if typecode == 'O':
        import numarray.objects as obj
        z = obj.ObjectArray(shape=shape)
        z[:] = 0
    else:
        z = _na.zeros(shape=shape, type=typecode)
    return z

def asscalar(a):
    """Returns Python scalar value corresponding to 'a' for rank-0 arrays
    or the unaltered array for non-rank-0."""
    return a[()]

# _Error.setMode(dividebyzero="ignore", invalid="ignore")
Error.setMode(all="ignore")

NX_VERSION = 'numarray %s' % _na.__version__
NX_INCLUDE = '"numarray/arrayobject.h"'
NX_ARRAYPKG = "numarray"

# Must appear after all public definititions
__all__ = []
for k in globals().keys():
    if k[0] != "_":
        __all__.append(k)
__all__.append("_insert")
__all__.append("_unique")


