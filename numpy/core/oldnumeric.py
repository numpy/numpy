# Compatibility module containing deprecated names

__all__ = ['asarray', 'array', 'concatenate',
           'NewAxis',
           'UFuncType', 'UfuncType', 'ArrayType', 'arraytype',
           'LittleEndian', 'Bool', 
           'Character', 'UnsignedInt8', 'UnsignedInt16', 'UnsignedInt',
           'UInt8','UInt16','UInt32',
           # UnsignedInt64 and Unsigned128 added below if possible
           # same for Int64 and Int128, Float128, and Complex128
           'Int8', 'Int16', 'Int32',
           'Int0', 'Int', 'Float0', 'Float', 'Complex0', 'Complex',
           'PyObject', 'Float32', 'Float64',
           'Complex32', 'Complex64',
           'typecodes', 'sarray', 'arrayrange', 'cross_correlate',
           'matrixmultiply', 'outerproduct', 'innerproduct',
           # from cPickle
           'dump', 'dumps',
           # functions that are now methods
           'take', 'reshape', 'choose', 'repeat', 'put', 'putmask',
           'swapaxes', 'transpose', 'sort', 'argsort', 'argmax', 'argmin',
           'searchsorted', 'alen', 
           'resize', 'diagonal', 'trace', 'ravel', 'nonzero', 'shape',
           'compress', 'clip', 'sum', 'product', 'prod', 'sometrue', 'alltrue',
           'any', 'all', 'cumsum', 'cumproduct', 'cumprod', 'ptp', 'ndim',
           'rank', 'size', 'around', 'round_', 'mean', 'std', 'var', 'squeeze',
           'amax', 'amin','bsum'
          ]

import multiarray as mu
import umath as um
import numerictypes as nt
from numeric import asarray, array, correlate, outer, concatenate
from umath import sign, absolute, multiply
import numeric as _nx
import sys
_dt_ = nt.dtype2char

#Use this to add a new axis to an array
#compatibility only
NewAxis = None

#deprecated
UFuncType = type(um.sin)
UfuncType = type(um.sin)
ArrayType = mu.ndarray
arraytype = mu.ndarray

LittleEndian = (sys.byteorder == 'little')

# save away Python sum
bsum = sum

# backward compatible names from old Precision.py

Character = 'S1'
UnsignedInt8 = _dt_(nt.uint8)
UInt8 = UnsignedInt8
UnsignedInt16 = _dt_(nt.uint16)
UInt16 = UnsignedInt16
UnsignedInt32 = _dt_(nt.uint32)
UInt32 = UnsignedInt32
UnsignedInt = _dt_(nt.uint)

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

typecodes = {'Character':'S1',
             'Integer':'bhilqp',
             'UnsignedInteger':'BHILQP',
             'Float':'fdg',
             'Complex':'FDG',
             'AllInteger':'bBhHiIlLqQpP',
             'AllFloat':'fdgFDG',
             'All':'?bhilqpBHILQPfdgFDGSUVO'}

def sarray(a, dtype=None, copy=False):
    return array(a, dtype, copy)

# backward compatibility
arrayrange = mu.arange
cross_correlate = correlate

# deprecated names
matrixmultiply = mu.dot
outerproduct = outer
innerproduct = mu.inner

from cPickle import dump, dumps

# functions that are now methods

def take(a, indices, axis=0):
    a = asarray(a)
    return a.take(indices, axis)

def reshape(a, newshape):
    """Change the shape of a to newshape.  Return a new view object.
    """
    return asarray(a).reshape(newshape)

def choose(a, choices):
    a = asarray(a)
    return a.choose(choices)

def repeat(a, repeats, axis=0):
    """repeat elements of a repeats times along axis
       repeats is a sequence of length a.shape[axis]
       telling how many times to repeat each element.
       If repeats is an integer, it is interpreted as
       a tuple of length a.shape[axis] containing repeats.
       The argument a can be anything array(a) will accept.
    """
    a = array(a, copy=False)
    return a.repeat(repeats, axis)

def put (a, ind, v):
    """put(a, ind, v) results in a[n] = v[n] for all n in ind
       If v is shorter than mask it will be repeated as necessary.
       In particular v can be a scalar or length 1 array.
       The routine put is the equivalent of the following (although the loop
       is in C for speed):

           ind = array(indices, copy=False)
           v = array(values, copy=False).astype(a, a.dtype)
           for i in ind: a.flat[i] = v[i]
       a must be a contiguous Numeric array.
    """
    return a.put(v,ind)

def putmask (a, mask, v):
    """putmask(a, mask, v) results in a = v for all places mask is true.
       If v is shorter than mask it will be repeated as necessary.
       In particular v can be a scalar or length 1 array.
    """
    return a.putmask(v, mask)

def swapaxes(a, axis1, axis2):
    """swapaxes(a, axis1, axis2) returns array a with axis1 and axis2
    interchanged.
    """
    a = array(a, copy=False)
    return a.swapaxes(axis1, axis2)

def transpose(a, axes=None):
    """transpose(a, axes=None) returns array with dimensions permuted
    according to axes.  If axes is None (default) returns array with
    dimensions reversed.
    """
    a = array(a,copy=False)
    return a.transpose(axes)

def sort(a, axis=-1):
    """sort(a,axis=-1) returns array with elements sorted along given axis.
    """
    a = array(a, copy=True)
    a.sort(axis)
    return a

def argsort(a, axis=-1):
    """argsort(a,axis=-1) return the indices into a of the sorted array
    along the given axis, so that take(a,result,axis) is the sorted array.
    """
    a = array(a, copy=False)
    return a.argsort(axis)

def argmax(a, axis=-1):
    """argmax(a,axis=-1) returns the indices to the maximum value of the
    1-D arrays along the given axis.
    """
    a = array(a, copy=False)
    return a.argmax(axis)

def argmin(a, axis=-1):
    """argmin(a,axis=-1) returns the indices to the minimum value of the
    1-D arrays along the given axis.
    """
    a = array(a,copy=False)
    return a.argmin(axis)

def searchsorted(a, v):
    """searchsorted(a, v)
    """
    a = array(a,copy=False)
    return a.searchsorted(v)

def resize(a, new_shape):
    """resize(a,new_shape) returns a new array with the specified shape.
    The original array's total size can be any size. It
    fills the new array with repeated copies of a.

    Note that a.resize(new_shape) will fill array with 0's
    beyond current definition of a.
    """

    if isinstance(new_shape, (int, nt.integer)):
        new_shape = (new_shape,)
    a = ravel(a)
    Na = len(a)
    if not Na: return mu.zeros(new_shape, a.dtypechar)
    total_size = um.multiply.reduce(new_shape)
    n_copies = int(total_size / Na)
    extra = total_size % Na

    if total_size == 0:
        return a[:0]

    if extra != 0:
        n_copies = n_copies+1
        extra = Na-extra

    a = concatenate( (a,)*n_copies)
    if extra > 0:
        a = a[:-extra]

    return reshape(a, new_shape)

def squeeze(a):
    "Returns a with any ones from the shape of a removed"
    return asarray(a).squeeze()

def diagonal(a, offset=0, axis1=0, axis2=1):
    """diagonal(a, offset=0, axis1=0, axis2=1) returns the given diagonals
    defined by the last two dimensions of the array.
    """
    return asarray(a).diagonal(offset, axis1, axis2)

def trace(a, offset=0, axis1=0, axis2=1, dtype=None):
    """trace(a,offset=0, axis1=0, axis2=1) returns the sum along diagonals
    (defined by the last two dimenions) of the array.
    """
    return asarray(a).trace(offset, axis1, axis2, dtype)

def ravel(m):
    """ravel(m) returns a 1d array corresponding to all the elements of it's
    argument.
    """
    return asarray(m).ravel()

def nonzero(a):
    """nonzero(a) returns the indices of the elements of a which are not zero,
    a must be 1d
    """
    return asarray(a).nonzero()

def shape(a):
    """shape(a) returns the shape of a (as a function call which
       also works on nested sequences).
    """
    return asarray(a).shape

def compress(condition, m, axis=-1):
    """compress(condition, x, axis=-1) = those elements of x corresponding 
    to those elements of condition that are "true".  condition must be the
    same size as the given dimension of x."""
    return asarray(m).compress(condition, axis)

def clip(m, m_min, m_max):
    """clip(m, m_min, m_max) = every entry in m that is less than m_min is
    replaced by m_min, and every entry greater than m_max is replaced by
    m_max.
    """
    return asarray(m).clip(m_min, m_max)

def sum(x, axis=0, dtype=None):
    """Sum the array over the given axis.  The optional dtype argument
    is the data type for intermediate calculations.

    The default is to upcast (promote) smaller integer types to the
    platform-dependent Int.  For example, on 32-bit platforms:

        x.dtype                         default sum() dtype
        ---------------------------------------------------
        bool, Int8, Int16, Int32        Int32

    Examples:
    >>> sum([0.5, 1.5])
    2.0
    >>> sum([0.5, 1.5], dtype=Int32)
    1
    >>> sum([[0, 1], [0, 5]])
    array([0, 6])
    >>> sum([[0, 1], [0, 5]], axis=1)
    array([1, 5])
    """
    return asarray(x).sum(axis, dtype)

def product (x, axis=0, dtype=None):
    """Product of the array elements over the given axis."""
    return asarray(x).prod(axis, dtype)

def sometrue (x, axis=0):
    """Perform a logical_or over the given axis."""
    return asarray(x).any(axis)

def alltrue (x, axis=0):
    """Perform a logical_and over the given axis."""
    return asarray(x).all(axis)

def any(x,axis=None):
    """Return true if any elements of x are true:  sometrue(ravel(x))
    """
    return ravel(x).any(axis)

def all(x,axis=None):
    """Return true if all elements of x are true:  alltrue(ravel(x))
    """
    return ravel(x).all(axis)

def cumsum (x, axis=0, dtype=None):
    """Sum the array over the given axis."""
    return asarray(x).cumsum(axis, dtype)

def cumproduct (x, axis=0, dtype=None):
    """Sum the array over the given axis."""
    return asarray(x).cumprod(axis, dtype)

def ptp(a, axis=0):
    """Return maximum - minimum along the the given dimension
    """
    return asarray(a).ptp(axis)

def amax(a, axis=0):
    """Return the maximum of 'a' along dimension axis.
    """
    return asarray(a).max(axis)

def amin(a, axis=0):
    """Return the minimum of a along dimension axis.
    """
    return asarray(a).min(axis)

def alen(a):
    """Return the length of a Python object interpreted as an array
    """
    return len(asarray(a))

def prod(a, axis=0):
    """Return the product of the elements along the given axis
    """
    return asarray(a).prod(axis)

def cumprod(a, axis=0):
    """Return the cumulative product of the elments along the given axis
    """
    return asarray(a).cumprod(axis)

def ndim(a):
    try:
        return a.ndim
    except AttributeError:
        return asarray(a).ndim

def rank (a):
    """Get the rank of sequence a (the number of dimensions, not a matrix rank)
       The rank of a scalar is zero.
    """
    try:
        return a.ndim
    except AttributeError:
        return asarray(a).ndim

def size (a, axis=None):
    "Get the number of elements in sequence a, or along a certain axis."
    if axis is None:
        try:
            return a.size
        except AttributeError:
            return asarray(a).size
    else:
        try:
            return a.shape[axis]
        except AttributeError:
            return asarray(a).shape[axis]

def round_(a, decimals=0):
    """Round 'a' to the given number of decimal places.  Rounding
    behaviour is equivalent to Python.

    Return 'a' if the array is not floating point.  Round both the real
    and imaginary parts separately if the array is complex.
    """
    a = asarray(a)
    if not issubclass(a.dtype, _nx.inexact):
        return a
    if issubclass(a.dtype, _nx.complexfloating):
        return round_(a.real, decimals) + 1j*round_(a.imag, decimals)
    if decimals is not 0:
        decimals = asarray(decimals)
    s = sign(a)
    if decimals is not 0:
        a = absolute(multiply(a, 10.**decimals))
    else:
        a = absolute(a)
    rem = a-asarray(a).astype(_nx.intp)
    a = _nx.where(_nx.less(rem, 0.5), _nx.floor(a), _nx.ceil(a))
    # convert back
    if decimals is not 0:
        return multiply(a, s/(10.**decimals))
    else:
        return multiply(a, s)

around = round_

def mean(a, axis=0, dtype=None):
    return asarray(a).mean(axis, dtype)

def std(a, axis=0, dtype=None):
    return asarray(a).std(axis, dtype)

def var(a, axis=0, dtype=None):
    return asarray(a).var(axis, dtype)
