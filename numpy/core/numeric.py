__all__ = ['newaxis', 'ndarray', 'flatiter', 'ufunc',
           'arange', 'array', 'zeros', 'empty', 'broadcast', 'dtype',
           'fromstring', 'fromfile', 'frombuffer','newbuffer',
           'getbuffer', 'int_asbuffer', 'where', 'argwhere',
           'concatenate', 'fastCopyAndTranspose', 'lexsort',
           'set_numeric_ops', 'can_cast',
           'asarray', 'asanyarray', 'ascontiguousarray', 'asfortranarray',
           'isfortran', 'empty_like', 'zeros_like',
           'correlate', 'convolve', 'inner', 'dot', 'outer', 'vdot',
           'alterdot', 'restoredot', 'roll', 'rollaxis', 'cross', 'tensordot',
           'array2string', 'get_printoptions', 'set_printoptions',
           'array_repr', 'array_str', 'set_string_function',
           'little_endian', 'require',
           'fromiter', 'array_equal', 'array_equiv',
           'indices', 'fromfunction', 'loadtxt', 'savetxt',
           'load', 'loads', 'isscalar', 'binary_repr', 'base_repr',
           'ones', 'identity', 'allclose', 'compare_chararrays', 'putmask',
           'seterr', 'geterr', 'setbufsize', 'getbufsize',
           'seterrcall', 'geterrcall', 'errstate', 'flatnonzero',
           'Inf', 'inf', 'infty', 'Infinity',
           'nan', 'NaN', 'False_', 'True_', 'bitwise_not',
           'CLIP', 'RAISE', 'WRAP', 'MAXDIMS', 'BUFSIZE', 'ALLOW_THREADS']

import sys
import multiarray
import umath
from umath import *
import numerictypes
from numerictypes import *

bitwise_not = invert

CLIP = multiarray.CLIP
WRAP = multiarray.WRAP
RAISE = multiarray.RAISE
MAXDIMS = multiarray.MAXDIMS
ALLOW_THREADS = multiarray.ALLOW_THREADS
BUFSIZE = multiarray.BUFSIZE


# from Fernando Perez's IPython
def zeros_like(a):
    """Return an array of zeros of the shape and typecode of a.

    If you don't explicitly need the array to be zeroed, you should instead
    use empty_like(), which is faster as it only allocates memory."""
    try:
        return zeros(a.shape, a.dtype, a.flags.fnc)
    except AttributeError:
        try:
            wrap = a.__array_wrap__
        except AttributeError:
            wrap = None
        a = asarray(a)
        res = zeros(a.shape, a.dtype)
        if wrap:
            res = wrap(res)
        return res

def empty_like(a):
    """Return an empty (uninitialized) array of the shape and typecode of a.

    Note that this does NOT initialize the returned array.  If you require
    your array to be initialized, you should use zeros_like().

    """
    try:
        return empty(a.shape, a.dtype, a.flags.fnc)
    except AttributeError:
        try:
            wrap = a.__array_wrap__
        except AttributeError:
            wrap = None
        a = asarray(a)
        res = empty(a.shape, a.dtype)
        if wrap:
            res = wrap(res)
        return res

# end Fernando's utilities


def extend_all(module):
    adict = {}
    for a in __all__:
        adict[a] = 1
    try:
        mall = getattr(module, '__all__')
    except AttributeError:
        mall = [k for k in module.__dict__.keys() if not k.startswith('_')]
    for a in mall:
        if a not in adict:
            __all__.append(a)

extend_all(umath)
extend_all(numerictypes)

newaxis = None

ndarray = multiarray.ndarray
flatiter = multiarray.flatiter
broadcast = multiarray.broadcast
dtype = multiarray.dtype
ufunc = type(sin)

arange = multiarray.arange
array = multiarray.array
zeros = multiarray.zeros
empty = multiarray.empty
fromstring = multiarray.fromstring
fromiter = multiarray.fromiter
fromfile = multiarray.fromfile
frombuffer = multiarray.frombuffer
newbuffer = multiarray.newbuffer
getbuffer = multiarray.getbuffer
int_asbuffer = multiarray.int_asbuffer
where = multiarray.where
concatenate = multiarray.concatenate
fastCopyAndTranspose = multiarray._fastCopyAndTranspose
set_numeric_ops = multiarray.set_numeric_ops
can_cast = multiarray.can_cast
lexsort = multiarray.lexsort
compare_chararrays = multiarray.compare_chararrays
putmask = multiarray.putmask

def asarray(a, dtype=None, order=None):
    """Returns a as an array.

    Unlike array(), no copy is performed if a is already an array. Subclasses
    are converted to base class ndarray.
    """
    return array(a, dtype, copy=False, order=order)

def asanyarray(a, dtype=None, order=None):
    """Returns a as an array, but will pass subclasses through.
    """
    return array(a, dtype, copy=False, order=order, subok=1)

def ascontiguousarray(a, dtype=None):
    """Return 'a' as an array contiguous in memory (C order).
    """
    return array(a, dtype, copy=False, order='C', ndmin=1)

def asfortranarray(a, dtype=None):
    """Return 'a' as an array laid out in Fortran-order in memory.
    """
    return array(a, dtype, copy=False, order='F', ndmin=1)

def require(a, dtype=None, requirements=None):
    if requirements is None:
        requirements = []
    else:
        requirements = [x.upper() for x in requirements]

    if not requirements:
        return asanyarray(a, dtype=dtype)

    if 'ENSUREARRAY' in requirements or 'E' in requirements:
        subok = 0
    else:
        subok = 1

    arr = array(a, dtype=dtype, copy=False, subok=subok)

    copychar = 'A'
    if 'FORTRAN' in requirements or \
       'F_CONTIGUOUS' in requirements or \
       'F' in requirements:
        copychar = 'F'
    elif 'CONTIGUOUS' in requirements or \
         'C_CONTIGUOUS' in requirements or \
         'C' in requirements:
        copychar = 'C'

    for prop in requirements:
        if not arr.flags[prop]:
            arr = arr.copy(copychar)
            break
    return arr

def isfortran(a):
    """Returns True if 'a' is arranged in Fortran-order in memory with a.ndim > 1
    """
    return a.flags.fnc

def argwhere(a):
    """Return a 2-d array of shape N x a.ndim where each row
    is a sequence of indices into a.  This sequence must be
    converted to a tuple in order to be used to index into a.
    """
    return asarray(a.nonzero()).T

def flatnonzero(a):
    """Return indicies that are not-zero in flattened version of a

    Equivalent to a.ravel().nonzero()[0]
    """
    return a.ravel().nonzero()[0]

_mode_from_name_dict = {'v': 0,
                        's' : 1,
                        'f' : 2}

def _mode_from_name(mode):
    if isinstance(mode, type("")):
        return _mode_from_name_dict[mode.lower()[0]]
    return mode

def correlate(a,v,mode='valid'):
    """Return the discrete, linear correlation of 1-D sequences a and v; mode
    can be 'valid', 'same', or 'full' to specify the size of the resulting
    sequence
    """
    mode = _mode_from_name(mode)
    return multiarray.correlate(a,v,mode)


def convolve(a,v,mode='full'):
    """Returns the discrete, linear convolution of 1-D sequences a and v; mode
    can be 'valid', 'same', or 'full' to specify size of the resulting sequence.
    """
    a,v = array(a,ndmin=1),array(v,ndmin=1)
    if (len(v) > len(a)):
        a, v = v, a
    assert len(a) > 0, 'a cannot be empty'
    assert len(v) > 0, 'v cannot be empty'
    mode = _mode_from_name(mode)
    return multiarray.correlate(a,asarray(v)[::-1],mode)

inner = multiarray.inner
dot = multiarray.dot

def outer(a,b):
    """Returns the outer product of two vectors.

    result[i,j] = a[i]*b[j] when a and b are vectors.
    Will accept any arguments that can be made into vectors.
    """
    a = asarray(a)
    b = asarray(b)
    return a.ravel()[:,newaxis]*b.ravel()[newaxis,:]

def vdot(a, b):
    """Returns the dot product of 2 vectors (or anything that can be made into
    a vector).

    Note: this is not the same as `dot`, as it takes the conjugate of its first
    argument if complex and always returns a scalar."""
    return dot(asarray(a).ravel().conj(), asarray(b).ravel())

# try to import blas optimized dot if available
try:
    # importing this changes the dot function for basic 4 types
    # to blas-optimized versions.
    from _dotblas import dot, vdot, inner, alterdot, restoredot
except ImportError:
    def alterdot():
        pass
    def restoredot():
        pass


def tensordot(a, b, axes=2):
    """tensordot returns the product for any (ndim >= 1) arrays.

    r_{xxx, yyy} = \sum_k a_{xxx,k} b_{k,yyy} where

    the axes to be summed over are given by the axes argument.
    the first element of the sequence determines the axis or axes
    in arr1 to sum over, and the second element in axes argument sequence
    determines the axis or axes in arr2 to sum over.

    When there is more than one axis to sum over, the corresponding
    arguments to axes should be sequences of the same length with the first
    axis to sum over given first in both sequences, the second axis second,
    and so forth.

    If the axes argument is an integer, N, then the last N dimensions of a
    and first N dimensions of b are summed over.
    """
    try:
        iter(axes)
    except:
        axes_a = range(-axes,0)
        axes_b = range(0,axes)
    else:
        axes_a, axes_b = axes
    try:
        na = len(axes_a)
        axes_a = list(axes_a)
    except TypeError:
        axes_a = [axes_a]
        na = 1
    try:
        nb = len(axes_b)
        axes_b = list(axes_b)
    except TypeError:
        axes_b = [axes_b]
        nb = 1

    a, b = asarray(a), asarray(b)
    as_ = a.shape
    nda = len(a.shape)
    bs = b.shape
    ndb = len(b.shape)
    equal = 1
    if (na != nb): equal = 0
    else:
        for k in xrange(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = 0
                break
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    if not equal:
        raise ValueError, "shape-mismatch for sum"

    # Move the axes to sum over to the end of "a"
    # and to the front of "b"
    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = notin + axes_a
    N2 = 1
    for axis in axes_a:
        N2 *= as_[axis]
    newshape_a = (-1, N2)
    olda = [as_[axis] for axis in notin]

    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = axes_b + notin
    N2 = 1
    for axis in axes_b:
        N2 *= bs[axis]
    newshape_b = (N2, -1)
    oldb = [bs[axis] for axis in notin]

    at = a.transpose(newaxes_a).reshape(newshape_a)
    bt = b.transpose(newaxes_b).reshape(newshape_b)
    res = dot(at, bt)
    return res.reshape(olda + oldb)

def roll(a, shift, axis=None):
    """Roll the elements in the array by 'shift' positions along
    the given axis.
    """
    a = asanyarray(a)
    if axis is None:
        n = a.size
        reshape=1
    else:
        n = a.shape[axis]
        reshape=0
    shift %= n
    indexes = concatenate((arange(n-shift,n),arange(n-shift)))
    res = a.take(indexes, axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res

def rollaxis(a, axis, start=0):
    """Return transposed array so that axis is rolled before start.

    if a.shape is (3,4,5,6)
    rollaxis(a, 3, 1).shape is (3,6,4,5)
    rollaxis(a, 2, 0).shape is (5,3,4,6)
    rollaxis(a, 1, 3).shape is (3,5,4,6)
    rollaxis(a, 1, 4).shape is (3,5,6,4)
    """
    n = a.ndim
    if axis < 0:
        axis += n
    if start < 0:
        start += n
    msg = 'rollaxis: %s (%d) must be >=0 and < %d'
    if not (0 <= axis < n):
        raise ValueError, msg % ('axis', axis, n)
    if not (0 <= start < n+1):
        raise ValueError, msg % ('start', start, n+1)
    if (axis < start): # it's been removed
        start -= 1
    if axis==start:
        return a
    axes = range(0,n)
    axes.remove(axis)
    axes.insert(start, axis)
    return a.transpose(axes)

# fix hack in scipy which imports this function
def _move_axis_to_0(a, axis):
    return rollaxis(a, axis, 0)

def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    """Return the cross product of two (arrays of) vectors.

    The cross product is performed over the last axis of a and b by default,
    and can handle axes with dimensions 2 and 3. For a dimension of 2,
    the z-component of the equivalent three-dimensional cross product is
    returned.
    """
    if axis is not None:
        axisa,axisb,axisc=(axis,)*3
    a = asarray(a).swapaxes(axisa, 0)
    b = asarray(b).swapaxes(axisb, 0)
    msg = "incompatible dimensions for cross product\n"\
          "(dimension must be 2 or 3)"
    if (a.shape[0] not in [2,3]) or (b.shape[0] not in [2,3]):
        raise ValueError(msg)
    if a.shape[0] == 2:
        if (b.shape[0] == 2):
            cp = a[0]*b[1] - a[1]*b[0]
            if cp.ndim == 0:
                return cp
            else:
                return cp.swapaxes(0, axisc)
        else:
            x = a[1]*b[2]
            y = -a[0]*b[2]
            z = a[0]*b[1] - a[1]*b[0]
    elif a.shape[0] == 3:
        if (b.shape[0] == 3):
            x = a[1]*b[2] - a[2]*b[1]
            y = a[2]*b[0] - a[0]*b[2]
            z = a[0]*b[1] - a[1]*b[0]
        else:
            x = -a[2]*b[1]
            y = a[2]*b[0]
            z = a[0]*b[1] - a[1]*b[0]
    cp = array([x,y,z])
    if cp.ndim == 1:
        return cp
    else:
        return cp.swapaxes(0,axisc)


#Use numarray's printing function
from arrayprint import array2string, get_printoptions, set_printoptions

_typelessdata = [int_, float_, complex_]
if issubclass(intc, int):
    _typelessdata.append(intc)

if issubclass(longlong, int):
    _typelessdata.append(longlong)

def array_repr(arr, max_line_width=None, precision=None, suppress_small=None):
    if arr.size > 0 or arr.shape==(0,):
        lst = array2string(arr, max_line_width, precision, suppress_small,
                           ', ', "array(")
    else: # show zero-length shape unless it is (0,)
        lst = "[], shape=%s" % (repr(arr.shape),)
    typeless = arr.dtype.type in _typelessdata

    if arr.__class__ is not ndarray:
        cName= arr.__class__.__name__
    else:
        cName = "array"
    if typeless and arr.size:
        return cName + "(%s)" % lst
    else:
        typename=arr.dtype.name
        lf = ''
        if issubclass(arr.dtype.type, flexible):
            if arr.dtype.names:
                typename = "%s" % str(arr.dtype)
            else:
                typename = "'%s'" % str(arr.dtype)
            lf = '\n'+' '*len("array(")
        return cName + "(%s, %sdtype=%s)" % (lst, lf, typename)

def array_str(a, max_line_width=None, precision=None, suppress_small=None):
    return array2string(a, max_line_width, precision, suppress_small, ' ', "", str)

set_string_function = multiarray.set_string_function
set_string_function(array_str, 0)
set_string_function(array_repr, 1)

little_endian = (sys.byteorder == 'little')


def indices(dimensions, dtype=int):
    """Returns an array representing a grid of indices with row-only, and
    column-only variation.
    """
    dimensions = tuple(dimensions)
    N = len(dimensions)
    if N == 0:
        return array([],dtype=dtype)
    res = empty((N,)+dimensions, dtype=dtype)
    for i, dim in enumerate(dimensions):
        tmp = arange(dim,dtype=dtype)
        tmp.shape = (1,)*i + (dim,)+(1,)*(N-i-1)
        newdim = dimensions[:i] + (1,)+ dimensions[i+1:]
        val = zeros(newdim, dtype)
        add(tmp, val, res[i])
    return res

def fromfunction(function, shape, **kwargs):
    """Returns an array constructed by calling a function on a tuple of number
    grids.

    The function should accept as many arguments as the length of shape and
    work on array inputs.  The shape argument is a sequence of numbers
    indicating the length of the desired output for each axis.

    The function can also accept keyword arguments (except dtype), which will
    be passed through fromfunction to the function itself.  The dtype argument
    (default float) determines the data-type of the index grid passed to the
    function.
    """
    dtype = kwargs.pop('dtype', float)
    args = indices(shape, dtype=dtype)
    return function(*args,**kwargs)

def isscalar(num):
    """Returns True if the type of num is a scalar type.
    """
    if isinstance(num, generic):
        return True
    else:
        return type(num) in ScalarType

_lkup = {
    '0':'0000',
    '1':'0001',
    '2':'0010',
    '3':'0011',
    '4':'0100',
    '5':'0101',
    '6':'0110',
    '7':'0111',
    '8':'1000',
    '9':'1001',
    'a':'1010',
    'b':'1011',
    'c':'1100',
    'd':'1101',
    'e':'1110',
    'f':'1111',
    'A':'1010',
    'B':'1011',
    'C':'1100',
    'D':'1101',
    'E':'1110',
    'F':'1111',
    'L':''}

def binary_repr(num, width=None):
    """Return the binary representation of the input number as a string.

    This is equivalent to using base_repr with base 2, but about 25x
    faster.

    For negative numbers, if width is not given, a - sign is added to the
    front. If width is given, the two's complement of the number is
    returned, with respect to that width.
    """
    sign = ''
    if num < 0:
        if width is None:
            sign = '-'
            num = -num
        else:
            # replace num with its 2-complement
            num = 2**width + num
    elif num == 0:
        return '0'*(width or 1)
    ostr = hex(num)
    bin = ''.join([_lkup[ch] for ch in ostr[2:]])
    bin = bin.lstrip('0')
    if width is not None:
        bin = bin.zfill(width)
    return sign + bin

def base_repr (number, base=2, padding=0):
    """Return the representation of a number in the given base.

    Base can't be larger than 36.
    """
    if number < 0:
        raise ValueError("negative numbers not handled in base_repr")
    if base > 36:
        raise ValueError("bases greater than 36 not handled in base_repr")

    chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    import math
    lnb = math.log(base)
    res = padding*chars[0]
    if number == 0:
        return res + chars[0]
    exponent = int (math.log (number)/lnb)
    while(exponent >= 0):
        term = long(base)**exponent
        lead_digit = int(number / term)
        res += chars[lead_digit]
        number -= term*lead_digit
        exponent -= 1
    return res

from cPickle import load, loads
_cload = load
_file = file

def load(file):
    """Wrapper around cPickle.load which accepts either a file-like object or
    a filename.
    """
    if isinstance(file, type("")):
        file = _file(file,"rb")
    return _cload(file)

# Adapted from matplotlib

def _getconv(dtype):
    typ = dtype.type
    if issubclass(typ, bool_):
        return lambda x: bool(int(x))
    if issubclass(typ, integer):
        return int
    elif issubclass(typ, floating):
        return float
    elif issubclass(typ, complex):
        return complex
    else:
        return str


def _string_like(obj):
    try: obj + ''
    except (TypeError, ValueError): return 0
    return 1

def loadtxt(fname, dtype=float, comments='#', delimiter=None, converters=None,
            skiprows=0, usecols=None, unpack=False):
    """
    Load ASCII data from fname into an array and return the array.

    The data must be regular, same number of values in every row

    fname can be a filename or a file handle.  Support for gzipped files is
    automatic, if the filename ends in .gz

    See scipy.loadmat to read and write matfiles.

    Example usage:

      X = loadtxt('test.dat')  # data in two columns
      t = X[:,0]
      y = X[:,1]

    Alternatively, you can do the same with "unpack"; see below

      X = loadtxt('test.dat')    # a matrix of data
      x = loadtxt('test.dat')    # a single column of data


    dtype - the data-type of the resulting array.  If this is a
    record data-type, the the resulting array will be 1-d and each row will
    be interpreted as an element of the array. The number of columns
    used must match the number of fields in the data-type in this case.

    comments - the character used to indicate the start of a comment
    in the file

    delimiter is a string-like character used to seperate values in the
    file. If delimiter is unspecified or none, any whitespace string is
    a separator.

    converters, if not None, is a dictionary mapping column number to
    a function that will convert that column to a float.  Eg, if
    column 0 is a date string: converters={0:datestr2num}

    skiprows is the number of rows from the top to skip

    usecols, if not None, is a sequence of integer column indexes to
    extract where 0 is the first column, eg usecols=(1,4,5) to extract
    just the 2nd, 5th and 6th columns

    unpack, if True, will transpose the matrix allowing you to unpack
    into named arguments on the left hand side

        t,y = load('test.dat', unpack=True) # for  two column data
        x,y,z = load('somefile.dat', usecols=(3,5,7), unpack=True)

    """

    if _string_like(fname):
        if fname.endswith('.gz'):
            import gzip
            fh = gzip.open(fname)
        else:
            fh = file(fname)
    elif hasattr(fname, 'seek'):
        fh = fname
    else:
        raise ValueError('fname must be a string or file handle')
    X = []

    dtype = multiarray.dtype(dtype)
    defconv = _getconv(dtype)
    converterseq = None
    if converters is None:
        converters = {}
        if dtype.names is not None:
            converterseq = [_getconv(dtype.fields[name][0]) \
                            for name in dtype.names]

    for i,line in enumerate(fh):
        if i<skiprows: continue
        line = line[:line.find(comments)].strip()
        if not len(line): continue
        vals = line.split(delimiter)
        if converterseq is None:
            converterseq = [converters.get(j,defconv) \
                            for j in xrange(len(vals))]
        if usecols is not None:
            row = [converterseq[j](vals[j]) for j in usecols]
        else:
            row = [converterseq[j](val) for j,val in enumerate(vals)]
        if dtype.names is not None:
            row = tuple(row)
        X.append(row)

    X = array(X, dtype)
    r,c = X.shape
    if r==1 or c==1:
        X.shape = max([r,c]),
    if unpack: return X.T
    else:  return X


# adjust so that fmt can change across columns if desired.

def savetxt(fname, X, fmt='%.18e',delimiter=' '):
    """
    Save the data in X to file fname using fmt string to convert the
    data to strings

    fname can be a filename or a file handle.  If the filename ends in .gz,
    the file is automatically saved in compressed gzip format.  The load()
    command understands gzipped files transparently.

    Example usage:

    save('test.out', X)         # X is an array
    save('test1.out', (x,y,z))  # x,y,z equal sized 1D arrays
    save('test2.out', x)        # x is 1D
    save('test3.out', x, fmt='%1.4e')  # use exponential notation

    delimiter is used to separate the fields, eg delimiter ',' for
    comma-separated values
    """

    if _string_like(fname):
        if fname.endswith('.gz'):
            import gzip
            fh = gzip.open(fname,'wb')
        else:
            fh = file(fname,'w')
    elif hasattr(fname, 'seek'):
        fh = fname
    else:
        raise ValueError('fname must be a string or file handle')


    X = asarray(X)
    origShape = None
    if len(X.shape)==1:
        origShape = X.shape
        X.shape = len(X), 1
    for row in X:
        fh.write(delimiter.join([fmt%val for val in row]) + '\n')

    if origShape is not None:
        X.shape = origShape







# These are all essentially abbreviations
# These might wind up in a special abbreviations module

def _maketup(descr, val):
    dt = dtype(descr)
    # Place val in all scalar tuples:
    fields = dt.fields
    if fields is None:
        return val
    else:
        res = [_maketup(fields[name][0],val) for name in dt.names]
        return tuple(res)

def ones(shape, dtype=None, order='C'):
    """Returns an array of the given dimensions which is initialized to all
    ones.
    """
    a = empty(shape, dtype, order)
    try:
        a.fill(1)
        # Above is faster now after addition of fast loops.
        #a = zeros(shape, dtype, order)
        #a+=1
    except TypeError:
        obj = _maketup(dtype, 1)
        a.fill(obj)
    return a

def identity(n, dtype=None):
    """Returns the identity 2-d array of shape n x n.

    identity(n)[i,j] == 1 for all i == j
                     == 0 for all i != j
    """
    a = array([1]+n*[0],dtype=dtype)
    b = empty((n,n),dtype=dtype)

    # Note that this assignment depends on the convention that since the a
    # array is shorter than the flattened b array, then the a array will
    # be repeated until it is the appropriate size. Given a's construction,
    # this nicely sets the diagonal to all ones.
    b.flat = a
    return b

def allclose(a, b, rtol=1.e-5, atol=1.e-8):
    """Returns True if all components of a and b are equal subject to given
    tolerances.

    The relative error rtol must be positive and << 1.0
    The absolute error atol usually comes into play for those elements of b that
    are very small or zero; it says how small a must be also.
    """
    x = array(a, copy=False)
    y = array(b, copy=False)
    xinf = isinf(x)
    if not all(xinf == isinf(y)):
        return False
    if not any(xinf):
        return all(less_equal(absolute(x-y), atol + rtol * absolute(y)))
    if not all(x[xinf] == y[xinf]):
        return False
    x = x[~xinf]
    y = y[~xinf]
    return all(less_equal(absolute(x-y), atol + rtol * absolute(y)))

def array_equal(a1, a2):
    try:
        a1, a2 = asarray(a1), asarray(a2)
    except:
        return 0
    if a1.shape != a2.shape:
        return 0
    return logical_and.reduce(equal(a1,a2).ravel())

def array_equiv(a1, a2):
    try:
        a1, a2 = asarray(a1), asarray(a2)
    except:
        return 0
    try:
        return logical_and.reduce(equal(a1,a2).ravel())
    except ValueError:
        return 0


_errdict = {"ignore":ERR_IGNORE,
            "warn":ERR_WARN,
            "raise":ERR_RAISE,
            "call":ERR_CALL,
            "print":ERR_PRINT,
            "log":ERR_LOG}

_errdict_rev = {}
for key in _errdict.keys():
    _errdict_rev[_errdict[key]] = key
del key

def seterr(all=None, divide=None, over=None, under=None, invalid=None):
    """Set how floating-point errors are handled.

    Valid values for each type of error are the strings
    "ignore", "warn", "raise", and "call". Returns the old settings.
    If 'all' is specified, values that are not otherwise specified
    will be set to 'all', otherwise they will retain their old
    values.

    Note that operations on integer scalar types (such as int16) are
    handled like floating point, and are affected by these settings.

    Example:

    >>> seterr(over='raise') # doctest: +SKIP
    {'over': 'ignore', 'divide': 'ignore', 'invalid': 'ignore', 'under': 'ignore'}

    >>> seterr(all='warn', over='raise') # doctest: +SKIP
    {'over': 'raise', 'divide': 'ignore', 'invalid': 'ignore', 'under': 'ignore'}

    >>> int16(32000) * int16(3) # doctest: +SKIP
    Traceback (most recent call last):
          File "<stdin>", line 1, in ?
    FloatingPointError: overflow encountered in short_scalars
    >>> seterr(all='ignore') # doctest: +SKIP
    {'over': 'ignore', 'divide': 'ignore', 'invalid': 'ignore', 'under': 'ignore'}

    """

    pyvals = umath.geterrobj()
    old = geterr()

    if divide is None: divide = all or old['divide']
    if over is None: over = all or old['over']
    if under is None: under = all or old['under']
    if invalid is None: invalid = all or old['invalid']

    maskvalue = ((_errdict[divide] << SHIFT_DIVIDEBYZERO) +
                 (_errdict[over] << SHIFT_OVERFLOW ) +
                 (_errdict[under] << SHIFT_UNDERFLOW) +
                 (_errdict[invalid] << SHIFT_INVALID))

    pyvals[1] = maskvalue
    umath.seterrobj(pyvals)
    return old


def geterr():
    """Get the current way of handling floating-point errors.

    Returns a dictionary with entries "divide", "over", "under", and
    "invalid", whose values are from the strings
    "ignore", "print", "log", "warn", "raise", and "call".
    """
    maskvalue = umath.geterrobj()[1]
    mask = 7
    res = {}
    val = (maskvalue >> SHIFT_DIVIDEBYZERO) & mask
    res['divide'] = _errdict_rev[val]
    val = (maskvalue >> SHIFT_OVERFLOW) & mask
    res['over'] = _errdict_rev[val]
    val = (maskvalue >> SHIFT_UNDERFLOW) & mask
    res['under'] = _errdict_rev[val]
    val = (maskvalue >> SHIFT_INVALID) & mask
    res['invalid'] = _errdict_rev[val]
    return res

def setbufsize(size):
    """Set the size of the buffer used in ufuncs.
    """
    if size > 10e6:
        raise ValueError, "Buffer size, %s, is too big." % size
    if size < 5:
        raise ValueError, "Buffer size, %s, is too small." %size
    if size % 16 != 0:
        raise ValueError, "Buffer size, %s, is not a multiple of 16." %size

    pyvals = umath.geterrobj()
    old = getbufsize()
    pyvals[0] = size
    umath.seterrobj(pyvals)
    return old

def getbufsize():
    """Return the size of the buffer used in ufuncs.
    """
    return umath.geterrobj()[0]

def seterrcall(func):
    """Set the callback function used when a floating-point error handler
    is set to 'call' or the object with a write method for use when
    the floating-point error handler is set to 'log'

    'func' should be a function that takes two arguments. The first is
    type of error ("divide", "over", "under", or "invalid"), and the second
    is the status flag (= divide + 2*over + 4*under + 8*invalid).

    Returns the old handler.
    """
    if func is not None and not callable(func):
        if not hasattr(func, 'write') or not callable(func.write):
            raise ValueError, "Only callable can be used as callback"
    pyvals = umath.geterrobj()
    old = geterrcall()
    pyvals[2] = func
    umath.seterrobj(pyvals)
    return old

def geterrcall():
    """Return the current callback function used on floating-point errors.
    """
    return umath.geterrobj()[2]

class _unspecified(object):
    pass
_Unspecified = _unspecified()

class errstate(object):
    """with errstate(**state): --> operations in following block use given state.

    # Set error handling to known state.
    >>> _ = seterr(invalid='raise', divide='raise', over='raise', under='ignore')

    |>> a = -arange(3)
    |>> with errstate(invalid='ignore'):
    ...     print sqrt(a)
    [ 0.     -1.#IND -1.#IND]
    |>> print sqrt(a.astype(complex))
    [ 0. +0.00000000e+00j  0. +1.00000000e+00j  0. +1.41421356e+00j]
    |>> print sqrt(a)
    Traceback (most recent call last):
     ...
    FloatingPointError: invalid encountered in sqrt
    |>> with errstate(divide='ignore'):
    ...     print a/0
    [0 0 0]
    |>> print a/0
    Traceback (most recent call last):
        ...
    FloatingPointError: divide by zero encountered in divide

    """
    # Note that we don't want to run the above doctests because they will fail
    # without a from __future__ import with_statement
    def __init__(self, **kwargs):
        self.call = kwargs.pop('call',_Unspecified)
        self.kwargs = kwargs
    def __enter__(self):
        self.oldstate = seterr(**self.kwargs)
        if self.call is not _Unspecified:
            self.oldcall = seterrcall(self.call)
    def __exit__(self, *exc_info):
        seterr(**self.oldstate)
        if self.call is not _Unspecified:
            seterrcall(self.oldcall)

def _setdef():
    defval = [UFUNC_BUFSIZE_DEFAULT, ERR_DEFAULT2, None]
    umath.seterrobj(defval)

# set the default values
_setdef()

Inf = inf = infty = Infinity = PINF
nan = NaN = NAN
False_ = bool_(False)
True_ = bool_(True)

import fromnumeric
from fromnumeric import *
extend_all(fromnumeric)
