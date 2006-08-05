# Compatibility module containing deprecated names

__all__ = ['NewAxis',
           'UFuncType', 'UfuncType', 'ArrayType', 'arraytype',
           'LittleEndian',
           'sarray', 'arrayrange', 'cross_correlate',
           'matrixmultiply', 'outerproduct', 'innerproduct',
           'cross_product', 'array_constructor', 'pickle_array',
           'DumpArray', 'LoadArray', 'multiarray', 'divide_safe',
           # from cPickle
           'dump', 'dumps'
          ]

import numpy.core.multiarray as multiarray
import numpy.core.umath as um
from numpy.core.numeric import array, correlate, outer, cross
from numpy.core.umath import sign, absolute, multiply
import sys

import types

from cPickle import dump, dumps

def sarray(a, dtype=None, copy=False):
    return array(a, dtype, copy)

mu = multiarray

#Use this to add a new axis to an array
#compatibility only
NewAxis = None

#deprecated
UFuncType = type(um.sin)
UfuncType = type(um.sin)
ArrayType = mu.ndarray
arraytype = mu.ndarray

LittleEndian = (sys.byteorder == 'little')

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

def pickle_array(a):
    if a.dtype.hasobject:
        return (array_constructor,
                a.shape, a.dtype.char, a.tolist(), LittleEndian)
    else:
        return (array_constructor,
                (a.shape, a.dtype.char, a.tostring(), LittleEndian))
    
