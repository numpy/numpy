# Compatibility module containing deprecated names

__all__ = ['NewAxis',
           'UFuncType', 'UfuncType', 'ArrayType', 'arraytype',
           'LittleEndian', 'arrayrange', 'matrixmultiply',
           'array_constructor', 'pickle_array',
           'DumpArray', 'LoadArray', 'multiarray',
           # from cPickle
           'dump', 'dumps', 'load', 'loads',
           'Unpickler', 'Pickler'
          ]

import numpy.core.multiarray as multiarray
import numpy.core.umath as um
from numpy.core.numeric import array
import functions
import sys

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

def loads(astr):
    import cPickle
    arr = cPickle.loads(astr.replace('Numeric', 'numpy.oldnumeric'))
    return arr

def load(fp):
    return loads(fp.read())

def _LoadArray(fp):
    import typeconv
    ln = fp.readline().split()
    if ln[0][0] == 'A': ln[0] = ln[0][1:]
    typecode = ln[0][0]
    endian = ln[0][1]
    itemsize = int(ln[0][2:])
    shape = [int(x) for x in ln[1:]]
    sz = itemsize
    for val in shape:
        sz *= val
    dstr = fp.read(sz)
    m = mu.fromstring(dstr, typeconv.convtypecode(typecode))
    m.shape = shape

    if (LittleEndian and endian == 'B') or (not LittleEndian and endian == 'L'):
        return m.byteswap(True)
    else:
        return m

import pickle, copy
if sys.version_info[0] >= 3:
    class Unpickler(pickle.Unpickler):
        # XXX: should we implement this? It's not completely straightforward
        #      to do.
        def __init__(self, *a, **kw):
            raise NotImplementedError(
                "numpy.oldnumeric.Unpickler is not supported on Python 3")
else:
    class Unpickler(pickle.Unpickler):
        def load_array(self):
            self.stack.append(_LoadArray(self))

        dispatch = copy.copy(pickle.Unpickler.dispatch)
        dispatch['A'] = load_array

class Pickler(pickle.Pickler):
    def __init__(self, *args, **kwds):
        raise NotImplementedError("Don't pickle new arrays with this")
    def save_array(self, object):
        raise NotImplementedError("Don't pickle new arrays with this")
