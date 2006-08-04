# Compatibility module containing deprecated names

__all__ = ['NewAxis',
           'UFuncType', 'UfuncType', 'ArrayType', 'arraytype',
           'LittleEndian', 'Bool',
           'Character', 'UnsignedInt8', 'UnsignedInt16', 'UnsignedInt', 'UInt',
           'UInt8','UInt16','UInt32', 'UnsignedInt32', 'UnsignedInteger',
           # UnsignedInt64 and Unsigned128 added below if possible
           # same for Int64 and Int128, Float128, and Complex128
           'Int8', 'Int16', 'Int32', 
           'Int0', 'Int', 'Float0', 'Float', 'Complex0', 'Complex',
           'PyObject', 'Float32', 'Float64', 'Float16', 'Float8',
           'Complex32', 'Complex64', 'Complex8', 'Complex16',
           'sarray', 'arrayrange', 'cross_correlate',
           'matrixmultiply', 'outerproduct', 'innerproduct',
           'cross_product', 'array_constructor',
           'DumpArray', 'LoadArray', 'multiarray', 'divide_safe',
           # from cPickle
           'dump', 'dumps'
          ]


import numpy.core.multiarray as mu
import numpy.core.umath as um
import numpy.core.numerictypes as nt
from numpy.core.numeric import asarray, array, asanyarray, \
                               correlate, outer, concatenate, cross
from numpy.core.umath import sign, absolute, multiply
import numpy.core.numeric as _nx
import sys
_dt_ = nt.sctype2char

import types

from cPickle import dump, dumps

multiarray = mu

def sarray(a, dtype=None, copy=False):
    return array(a, dtype, copy)


#Use this to add a new axis to an array
#compatibility only
NewAxis = None

#deprecated
UFuncType = type(um.sin)
UfuncType = type(um.sin)
ArrayType = mu.ndarray
arraytype = mu.ndarray

LittleEndian = (sys.byteorder == 'little')

# backward compatible names from old Precision.py

Character = 'c'
UnsignedInt8 = _dt_(nt.uint8)
UInt8 = UnsignedInt8
UnsignedInt16 = _dt_(nt.uint16)
UInt16 = UnsignedInt16
UnsignedInt32 = _dt_(nt.uint32)
UInt32 = UnsignedInt32
UnsignedInt = _dt_(nt.uint)
UInt = UnsignedInt

try:
    UnsignedInt64 = _dt_(nt.uint64)
except AttributeError:
    pass
else:
    UInt64 = UnsignedInt64
    __all__ += ['UnsignedInt64', 'UInt64']
try:
    UnsignedInt128 = _dt_(nt.uint128)
except AttributeError:
    pass
else:
    UInt128 = UnsignedInt128
    __all__ += ['UnsignedInt128','UInt128']

Int8 = _dt_(nt.int8)
Int16 = _dt_(nt.int16)
Int32 = _dt_(nt.int32)

try:
    Int64 = _dt_(nt.int64)
except AttributeError:
    pass
else:
    __all__ += ['Int64']

try:
    Int128 = _dt_(nt.int128)
except AttributeError:
    pass
else:
    __all__ += ['Int128']

Bool = _dt_(bool)
Int0 = _dt_(int)
Int = _dt_(int)
Float0 = _dt_(float)
Float = _dt_(float)
Complex0 = _dt_(complex)
Complex = _dt_(complex)
PyObject = _dt_(nt.object_)
Float32 = _dt_(nt.float32)
Float64 = _dt_(nt.float64)

Float16='f'
Float8='f'
UnsignedInteger='L'
Complex8='F'
Complex16='F'

try:
    Float128 = _dt_(nt.float128)
except AttributeError:
    pass
else:
    __all__ += ['Float128']

Complex32 = _dt_(nt.complex64)
Complex64 = _dt_(nt.complex128)

try:
    Complex128 = _dt_(nt.complex256)
except AttributeError:
    pass
else:
    __all__ += ['Complex128']


from numpy import deprecate 

# backward compatibility
arrayrange = deprecate(mu.arange, 'arrayrange', 'arange')
cross_correlate = deprecate(correlate, 'cross_correlate', 'correlate')
cross_product = deprecate(cross, 'cross_product', 'cross')
divide_safe = deprecate(um.divide, 'divide_safe', 'divide')

# deprecated names
matrixmultiply = deprecate(mu.dot, 'matrixmultiply', 'dot')
outerproduct = deprecate(outer, 'outerproduct', 'outer')
innerproduct = deprecate(mu.inner, 'innerproduct', 'inner')


def DumpArray(m, fp):
    m.dump(fp)

def LoadArray(fp):
    import cPickle
    return cPickle.load(fp)

def array_constructor(shape, typecode, thestr, Endian=LittleEndian):
    if typecode == "O":
        x = array(thestr, "O")
    else:
        x = mu.fromstring(thestr, typecode)
    x.shape = shape
    if LittleEndian != Endian:
        return x.byteswap(TRUE)
    else:
        return x


