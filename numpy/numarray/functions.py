from __future__ import division, absolute_import, print_function

# missing Numarray defined names (in from numarray import *)

__all__ = ['asarray', 'ones', 'zeros', 'array', 'where']
__all__ += ['vdot', 'dot', 'matrixmultiply', 'ravel', 'indices',
            'arange', 'concatenate', 'all', 'allclose', 'alltrue', 'and_',
            'any', 'argmax', 'argmin', 'argsort', 'around', 'array_equal',
            'array_equiv', 'arrayrange', 'array_str', 'array_repr',
            'array2list', 'average', 'choose', 'CLIP', 'RAISE', 'WRAP',
            'clip', 'compress', 'copy', 'copy_reg',
            'diagonal', 'divide_remainder', 'e', 'explicit_type', 'pi',
            'flush_caches', 'fromfile', 'os', 'sys', 'STRICT',
            'SLOPPY', 'WARN', 'EarlyEOFError', 'SizeMismatchError',
            'SizeMismatchWarning', 'FileSeekWarning', 'fromstring',
            'fromfunction', 'fromlist', 'getShape', 'getTypeObject',
            'identity', 'info', 'innerproduct', 'inputarray',
            'isBigEndian', 'kroneckerproduct', 'lexsort', 'math',
            'operator', 'outerproduct', 'put', 'putmask', 'rank',
            'repeat', 'reshape', 'resize', 'round', 'searchsorted',
            'shape', 'size', 'sometrue', 'sort', 'swapaxes', 'take',
            'tcode', 'tname', 'tensormultiply', 'trace', 'transpose',
            'types', 'value', 'cumsum', 'cumproduct', 'nonzero', 'newobj',
            'togglebyteorder'
            ]

import copy
import types
import os
import sys
import math
import operator

import numpy as np
from numpy import dot as matrixmultiply, dot, vdot, ravel, concatenate, all,\
     allclose, any, argsort, array_equal, array_equiv,\
     array_str, array_repr, CLIP, RAISE, WRAP, clip, concatenate, \
     diagonal, e, pi, inner as innerproduct, nonzero, \
     outer as outerproduct, kron as kroneckerproduct, lexsort, putmask, rank, \
     resize, searchsorted, shape, size, sort, swapaxes, trace, transpose
from numpy.compat import long
from .numerictypes import typefrom

if sys.version_info[0] >= 3:
    import copyreg as copy_reg
else:
    import copy_reg

isBigEndian = sys.byteorder != 'little'
value = tcode = 'f'
tname = 'Float32'

#  If dtype is not None, then it is used
#  If type is not None, then it is used
#  If typecode is not None then it is used
#  If use_default is True, then the default
#   data-type is returned if all are None
def type2dtype(typecode, type, dtype, use_default=True):
    if dtype is None:
        if type is None:
            if use_default or typecode is not None:
                dtype = np.dtype(typecode)
        else:
            dtype = np.dtype(type)
    if use_default and dtype is None:
        dtype = np.dtype('int')
    return dtype

def fromfunction(shape, dimensions, type=None, typecode=None, dtype=None):
    dtype = type2dtype(typecode, type, dtype, 1)
    return np.fromfunction(shape, dimensions, dtype=dtype)
def ones(shape, type=None, typecode=None, dtype=None):
    dtype = type2dtype(typecode, type, dtype, 1)
    return np.ones(shape, dtype)

def zeros(shape, type=None, typecode=None, dtype=None):
    dtype = type2dtype(typecode, type, dtype, 1)
    return np.zeros(shape, dtype)

def where(condition, x=None, y=None, out=None):
    if x is None and y is None:
        arr = np.where(condition)
    else:
        arr = np.where(condition, x, y)
    if out is not None:
        out[...] = arr
        return out
    return arr

def indices(shape, type=None):
    return np.indices(shape, type)

def arange(a1, a2=None, stride=1, type=None, shape=None,
           typecode=None, dtype=None):
    dtype = type2dtype(typecode, type, dtype, 0)
    return np.arange(a1, a2, stride, dtype)

arrayrange = arange

def alltrue(x, axis=0):
    return np.alltrue(x, axis)

def and_(a, b):
    """Same as a & b
    """
    return a & b

def divide_remainder(a, b):
    a, b = asarray(a), asarray(b)
    return (a/b, a%b)

def around(array, digits=0, output=None):
    ret = np.around(array, digits, output)
    if output is None:
        return ret
    return

def array2list(arr):
    return arr.tolist()


def choose(selector, population, outarr=None, clipmode=RAISE):
    a = np.asarray(selector)
    ret = a.choose(population, out=outarr, mode=clipmode)
    if outarr is None:
        return ret
    return

def compress(condition, a, axis=0):
    return np.compress(condition, a, axis)

# only returns a view
def explicit_type(a):
    x = a.view()
    return x

# stub
def flush_caches():
    pass


class EarlyEOFError(Exception):
    "Raised in fromfile() if EOF unexpectedly occurs."
    pass

class SizeMismatchError(Exception):
    "Raised in fromfile() if file size does not match shape."
    pass

class SizeMismatchWarning(Warning):
    "Issued in fromfile() if file size does not match shape."
    pass

class FileSeekWarning(Warning):
    "Issued in fromfile() if there is unused data and seek() fails"
    pass


STRICT, SLOPPY, WARN = list(range(3))

_BLOCKSIZE=1024

# taken and adapted directly from numarray
def fromfile(infile, type=None, shape=None, sizing=STRICT,
             typecode=None, dtype=None):
    if isinstance(infile, (str, unicode)):
        infile = open(infile, 'rb')
    dtype = type2dtype(typecode, type, dtype, True)
    if shape is None:
        shape = (-1,)
    if not isinstance(shape, tuple):
        shape = (shape,)

    if (list(shape).count(-1)>1):
        raise ValueError("At most one unspecified dimension in shape")

    if -1 not in shape:
        if sizing != STRICT:
            raise ValueError("sizing must be STRICT if size complete")
        arr = np.empty(shape, dtype)
        bytesleft=arr.nbytes
        bytesread=0
        while(bytesleft > _BLOCKSIZE):
            data = infile.read(_BLOCKSIZE)
            if len(data) != _BLOCKSIZE:
                raise EarlyEOFError("Unexpected EOF reading data for size complete array")
            arr.data[bytesread:bytesread+_BLOCKSIZE]=data
            bytesread += _BLOCKSIZE
            bytesleft -= _BLOCKSIZE
        if bytesleft > 0:
            data = infile.read(bytesleft)
            if len(data) != bytesleft:
                raise EarlyEOFError("Unexpected EOF reading data for size complete array")
            arr.data[bytesread:bytesread+bytesleft]=data
        return arr


    ##shape is incompletely specified
    ##read until EOF
    ##implementation 1: naively use memory blocks
    ##problematic because memory allocation can be double what is
    ##necessary (!)

    ##the most common case, namely reading in data from an unchanging
    ##file whose size may be determined before allocation, should be
    ##quick -- only one allocation will be needed.

    recsize = int(dtype.itemsize * np.product([i for i in shape if i != -1]))
    blocksize = max(_BLOCKSIZE//recsize, 1)*recsize

    ##try to estimate file size
    try:
        curpos=infile.tell()
        infile.seek(0, 2)
        endpos=infile.tell()
        infile.seek(curpos)
    except (AttributeError, IOError):
        initsize=blocksize
    else:
        initsize=max(1, (endpos-curpos)//recsize)*recsize

    buf = np.newbuffer(initsize)

    bytesread=0
    while True:
        data=infile.read(blocksize)
        if len(data) != blocksize: ##eof
            break
        ##do we have space?
        if len(buf) < bytesread+blocksize:
            buf=_resizebuf(buf, len(buf)+blocksize)
            ## or rather a=resizebuf(a,2*len(a)) ?
        assert len(buf) >= bytesread+blocksize
        buf[bytesread:bytesread+blocksize]=data
        bytesread += blocksize

    if len(data) % recsize != 0:
        if sizing == STRICT:
            raise SizeMismatchError("Filesize does not match specified shape")
        if sizing == WARN:
            _warnings.warn("Filesize does not match specified shape",
                           SizeMismatchWarning)
        try:
            infile.seek(-(len(data) % recsize), 1)
        except AttributeError:
            _warnings.warn("Could not rewind (no seek support)",
                           FileSeekWarning)
        except IOError:
            _warnings.warn("Could not rewind (IOError in seek)",
                           FileSeekWarning)
    datasize = (len(data)//recsize) * recsize
    if len(buf) != bytesread+datasize:
        buf=_resizebuf(buf, bytesread+datasize)
    buf[bytesread:bytesread+datasize]=data[:datasize]
    ##deduce shape from len(buf)
    shape = list(shape)
    uidx = shape.index(-1)
    shape[uidx]=len(buf) // recsize

    a = np.ndarray(shape=shape, dtype=type, buffer=buf)
    if a.dtype.char == '?':
        np.not_equal(a, 0, a)
    return a


# this function is referenced in the code above but not defined.  adding
# it back. - phensley
def _resizebuf(buf, newsize):
    "Return a copy of BUF of size NEWSIZE."
    newbuf = np.newbuffer(newsize)
    if newsize > len(buf):
        newbuf[:len(buf)]=buf
    else:
        newbuf[:]=buf[:len(newbuf)]
    return newbuf


def fromstring(datastring, type=None, shape=None, typecode=None, dtype=None):
    dtype = type2dtype(typecode, type, dtype, True)
    if shape is None:
        count = -1
    else:
        count = np.product(shape)
    res = np.fromstring(datastring, dtype=dtype, count=count)
    if shape is not None:
        res.shape = shape
    return res


# check_overflow is ignored
def fromlist(seq, type=None, shape=None, check_overflow=0, typecode=None, dtype=None):
    dtype = type2dtype(typecode, type, dtype, False)
    return np.array(seq, dtype)

def array(sequence=None, typecode=None, copy=1, savespace=0,
          type=None, shape=None, dtype=None):
    dtype = type2dtype(typecode, type, dtype, 0)
    if sequence is None:
        if shape is None:
            return None
        if dtype is None:
            dtype = 'l'
        return np.empty(shape, dtype)
    if isinstance(sequence, file):
        return fromfile(sequence, dtype=dtype, shape=shape)
    if isinstance(sequence, str):
        return fromstring(sequence, dtype=dtype, shape=shape)
    if isinstance(sequence, buffer):
        arr = np.frombuffer(sequence, dtype=dtype)
    else:
        arr = np.array(sequence, dtype, copy=copy)
    if shape is not None:
        arr.shape = shape
    return arr

def asarray(seq, type=None, typecode=None, dtype=None):
    if isinstance(seq, np.ndarray) and type is None and \
           typecode is None and dtype is None:
        return seq
    return array(seq, type=type, typecode=typecode, copy=0, dtype=dtype)

inputarray = asarray


def getTypeObject(sequence, type):
    if type is not None:
        return type
    try:
        return typefrom(np.array(sequence))
    except:
        raise TypeError("Can't determine a reasonable type from sequence")

def getShape(shape, *args):
    try:
        if shape is () and not args:
            return ()
        if len(args) > 0:
            shape = (shape, ) + args
        else:
            shape = tuple(shape)
        dummy = np.array(shape)
        if not issubclass(dummy.dtype.type, np.integer):
            raise TypeError
        if len(dummy) > np.MAXDIMS:
            raise TypeError
    except:
        raise TypeError("Shape must be a sequence of integers")
    return shape


def identity(n, type=None, typecode=None, dtype=None):
    dtype = type2dtype(typecode, type, dtype, True)
    return np.identity(n, dtype)

def info(obj, output=sys.stdout, numpy=0):
    if numpy:
        bp = lambda x: x
    else:
        bp = lambda x: int(x)
    cls = getattr(obj, '__class__', type(obj))
    if numpy:
        nm = getattr(cls, '__name__', cls)
    else:
        nm = cls
    print("class: ", nm, file=output)
    print("shape: ", obj.shape, file=output)
    strides = obj.strides
    print("strides: ", strides, file=output)
    if not numpy:
        print("byteoffset: 0", file=output)
        if len(strides) > 0:
            bs = obj.strides[0]
        else:
            bs = obj.itemsize
        print("bytestride: ", bs, file=output)
    print("itemsize: ", obj.itemsize, file=output)
    print("aligned: ", bp(obj.flags.aligned), file=output)
    print("contiguous: ", bp(obj.flags.contiguous), file=output)
    if numpy:
        print("fortran: ", obj.flags.fortran, file=output)
    if not numpy:
        print("buffer: ", repr(obj.data), file=output)
    if not numpy:
        extra = " (DEBUG ONLY)"
        tic = "'"
    else:
        extra = ""
        tic = ""
    print("data pointer: %s%s" % (hex(obj.ctypes._as_parameter_.value), extra), file=output)
    print("byteorder: ", end=' ', file=output)
    endian = obj.dtype.byteorder
    if endian in ['|', '=']:
        print("%s%s%s" % (tic, sys.byteorder, tic), file=output)
        byteswap = False
    elif endian == '>':
        print("%sbig%s" % (tic, tic), file=output)
        byteswap = sys.byteorder != "big"
    else:
        print("%slittle%s" % (tic, tic), file=output)
        byteswap = sys.byteorder != "little"
    print("byteswap: ", bp(byteswap), file=output)
    if not numpy:
        print("type: ", typefrom(obj).name, file=output)
    else:
        print("type: %s" % obj.dtype, file=output)

#clipmode is ignored if axis is not 0 and array is not 1d
def put(array, indices, values, axis=0, clipmode=RAISE):
    if not isinstance(array, np.ndarray):
        raise TypeError("put only works on subclass of ndarray")
    work = asarray(array)
    if axis == 0:
        if array.ndim == 1:
            work.put(indices, values, clipmode)
        else:
            work[indices] = values
    elif isinstance(axis, (int, long, np.integer)):
        work = work.swapaxes(0, axis)
        work[indices] = values
        work = work.swapaxes(0, axis)
    else:
        def_axes = list(range(work.ndim))
        for x in axis:
            def_axes.remove(x)
        axis = list(axis)+def_axes
        work = work.transpose(axis)
        work[indices] = values
        work = work.transpose(axis)

def repeat(array, repeats, axis=0):
    return np.repeat(array, repeats, axis)


def reshape(array, shape, *args):
    if len(args) > 0:
        shape = (shape,) + args
    return np.reshape(array, shape)


import warnings as _warnings
def round(*args, **keys):
    _warnings.warn("round() is deprecated. Switch to around()",
                   DeprecationWarning)
    return around(*args, **keys)

def sometrue(array, axis=0):
    return np.sometrue(array, axis)

#clipmode is ignored if axis is not an integer
def take(array, indices, axis=0, outarr=None, clipmode=RAISE):
    array = np.asarray(array)
    if isinstance(axis, (int, long, np.integer)):
        res = array.take(indices, axis, outarr, clipmode)
        if outarr is None:
            return res
        return
    else:
        def_axes = list(range(array.ndim))
        for x in axis:
            def_axes.remove(x)
        axis = list(axis) + def_axes
        work = array.transpose(axis)
        res = work[indices]
        if outarr is None:
            return res
        outarr[...] = res
        return

def tensormultiply(a1, a2):
    a1, a2 = np.asarray(a1), np.asarray(a2)
    if (a1.shape[-1] != a2.shape[0]):
        raise ValueError("Unmatched dimensions")
    shape = a1.shape[:-1] + a2.shape[1:]
    return np.reshape(dot(np.reshape(a1, (-1, a1.shape[-1])),
                          np.reshape(a2, (a2.shape[0], -1))),
                      shape)

def cumsum(a1, axis=0, out=None, type=None, dim=0):
    return np.asarray(a1).cumsum(axis, dtype=type, out=out)

def cumproduct(a1, axis=0, out=None, type=None, dim=0):
    return np.asarray(a1).cumprod(axis, dtype=type, out=out)

def argmax(x, axis=-1):
    return np.argmax(x, axis)

def argmin(x, axis=-1):
    return np.argmin(x, axis)

def newobj(self, type):
    if type is None:
        return np.empty_like(self)
    else:
        return np.empty(self.shape, type)

def togglebyteorder(self):
    self.dtype=self.dtype.newbyteorder()

def average(a, axis=0, weights=None, returned=0):
    return np.average(a, axis, weights, returned)
