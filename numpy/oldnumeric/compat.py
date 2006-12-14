# Compatibility module containing deprecated names

__all__ = ['NewAxis',
           'UFuncType', 'UfuncType', 'ArrayType', 'arraytype',
           'LittleEndian', 'arrayrange', 'matrixmultiply', 
           'array_constructor', 'pickle_array',
           'DumpArray', 'LoadArray', 'multiarray',
           # from cPickle
           'dump', 'dumps'
          ]

import numpy.core.multiarray as multiarray
import numpy.core.umath as um
from numpy.core.numeric import array, correlate, outer, cross
from numpy.core.umath import sign, absolute, multiply
import functions
import sys

import types

from cPickle import dump, dumps

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
arrayrange = deprecate(functions.arange, 'arrayrange', 'arange')

# deprecated names
matrixmultiply = deprecate(mu.dot, 'matrixmultiply', 'dot')

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
        return x.byteswap(True)
    else:
        return x

def pickle_array(a):
    if a.dtype.hasobject:
        return (array_constructor,
                a.shape, a.dtype.char, a.tolist(), LittleEndian)
    else:
        return (array_constructor,
                (a.shape, a.dtype.char, a.tostring(), LittleEndian))
    
