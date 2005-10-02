
import sys
import types, math

import multiarray
import umath
from umath import *
from numerictypes import *
from _compiled_base import _insert

newaxis = None

arange = multiarray.arange
array = multiarray.array
zeros = multiarray.zeros
empty = multiarray.empty
fromstring = multiarray.fromstring
fromfile = multiarray.fromfile
frombuffer = multiarray.frombuffer
where = multiarray.where
concatenate = multiarray.concatenate
fastCopyAndTranspose = multiarray._fastCopyAndTranspose
register_dtype = multiarray.register_dtype
can_cast = multiarray.can_cast

def asarray(a, dtype=None):
    """asarray(a,dtype=None) returns a as a NumPy array.  Unlike array(),
    no copy is performed if a is already an array.
    """
    return array(a, dtype, copy=0)

def ensure_array(a, dtype=None):
    """ensure_array(a, dtype=None) always returns an actual ndarray object.
    No copy is performed if a is already an array.
    Meant primarily for debugging. 
    """
    # exact check
    while 1:
        if type(a) is ndarray:
            return a
        try:
            if dtype is None:
                return a.__array__()
            else:
                return a.__array__(dtype)
        except AttributeError:
            a = array(a,dtype,copy=1)  # copy irrelevant
            dtype = None

def isfortran(a):
    flags = a.flags
    return flags['FORTRAN'] and a.ndim > 1


# from Fernando Perez's IPython
def zeros_like(a):
    """Return an array of zeros of the shape and typecode of a.

    If you don't explicitly need the array to be zeroed, you should instead
    use empty_like(), which is faster as it only allocates memory."""

    a = asarray(a)
    return zeros(a.shape,a.dtype,a.flags['FORTRAN'] and a.ndim > 1)

def empty_like(a):
    """Return an empty (uninitialized) array of the shape and typecode of a.

    Note that this does NOT initialize the returned array.  If you require
    your array to be initialized, you should use zeros_like().

    """
    asarray(a)
    return empty(a.shape,a.dtype,a.flags['FORTRAN'] and a.ndim > 1)
# end Fernando's utilities

_mode_from_name_dict = {'v': 0,
                        's' : 1,
                        'f' : 2}

def _mode_from_name(mode):
    if isinstance(mode, type("")):
        return _mode_from_name_dict[mode.lower()[0]]
    return mode
        
def correlate(a,v,mode='valid'):
    mode = _mode_from_name(mode)
    return multiarray.correlate(a,v,mode)


def convolve(a,v,mode='full'):
    """Returns the discrete, linear convolution of 1-D
    sequences a and v; mode can be 0 (valid), 1 (same), or 2 (full)
    to specify size of the resulting sequence.
    """
    if (len(v) > len(a)):
        a, v = v, a
    mode = _mode_from_name(mode)
    return correlate(a,asarray(v)[::-1],mode)

ndarray = multiarray.ndarray
ndbigarray = multiarray.ndbigarray
ufunc = type(sin)

inner = multiarray.inner
dot = multiarray.dot

def outer(a,b):
   """outer(a,b) returns the outer product of two vectors.
      result(i,j) = a(i)*b(j) when a and b are vectors
      Will accept any arguments that can be made into vectors.
   """
   a = asarray(a)
   b = asarray(b)
   return a.ravel()[:,newaxis]*b.ravel()[newaxis,:]

def vdot(a, b):
    """Returns the dot product of 2 vectors (or anything that can be made into
       a vector). NB: this is not the same as `dot`, as it takes the conjugate
       of its first argument if complex and always returns a scalar."""
    return dot(asarray(a).ravel().conj(), asarray(b).ravel())

# try to import blas optimized dot if available
try:
    # importing this changes the dot function for basic 4 types
    # to blas-optimized versions.
    from scipy.lib._dotblas import dot, vdot, inner, alterdot, restoredot
except ImportError:
    def alterdot():
        pass
    def restoredot():
        pass


def _move_axis_to_0(a, axis):
    if axis == 0:
        return a
    n = a.ndim
    if axis < 0:
        axis += n
    axes = range(1, axis+1) + [0,] + range(axis+1, n)
    return a.transpose(axes)

def cross(a, b, axisa=-1, axisb=-1, axisc=-1):
    """Return the cross product of two (arrays of) vectors.

    The cross product is performed over the last axis of a and b by default,
    and can handle axes with dimensions 2 and 3. For a dimension of 2,
    the z-component of the equivalent three-dimensional cross product is
    returned.
    """
    a = _move_axis_to_0(asarray(a), axisa)
    b = _move_axis_to_0(asarray(b), axisb)
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
                return cp.swapaxes(0,axisc)
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

def array_repr(arr, max_line_width=None, precision=None, suppress_small=None):
    if arr.size > 0 or arr.shape==(0,):
        lst = array2string(arr, max_line_width, precision, suppress_small,
                           ', ', "array(")
    else: # show zero-length shape unless it is (0,)
        lst = "[], shape=%s" % (repr(arr.shape),)
    typeless = arr.dtype in _typelessdata

    if arr.__class__ is not ndarray:
        cName= arr.__class__.__name__
    else:
        cName = "array"
    if typeless and arr.size:
        return cName + "(%s)" % lst
    else:
        typename=arr.dtype.__name__[:-8]
        return cName + "(%s, dtype=%s)" % (lst, typename)

def array_str(a, max_line_width = None, precision = None, suppress_small = None):
    return array2string(a, max_line_width, precision, suppress_small, ' ', "")

set_string_function = multiarray.set_string_function
set_string_function(array_str, 0)
set_string_function(array_repr, 1)


little_endian = (sys.byteorder == 'little')

def indices(dimensions, dtype=intp):
    """indices(dimensions,dtype=intp) returns an array representing a grid
    of indices with row-only, and column-only variation.
    """
    tmp = ones(dimensions, dtype)
    lst = []
    for i in range(len(dimensions)):
        lst.append( add.accumulate(tmp, i, )-1 )
    return array(lst)

def fromfunction(function, dimensions, **kwargs):
    """fromfunction(function, dimensions) returns an array constructed by
    calling function on a tuple of number grids.  The function should
    accept as many arguments as there are dimensions which is a list of
    numbers indicating the length of the desired output for each axis.

    The function can also accept keyword arguments which will be
    passed in as well. 
    """
    args = indices(dimensions)
    return function(*args,**kwargs)
    

from cPickle import load, loads
_cload = load
_file = file

def load(file):
    if isinstance(file, type("")):
        file = _file(file,"rb")
    return _cload(file)


# These are all essentially abbreviations
# These might wind up in a special abbreviations module

def ones(shape, dtype=intp, fortran=0):
    """ones(shape, dtype=intp) returns an array of the given
    dimensions which is initialized to all ones. 
    """
    a=zeros(shape, dtype, fortran)
    a+=1
    ### a[...]=1  -- slower
    return a
 
def identity(n,dtype=intp):
    """identity(n) returns the identity matrix of shape n x n.
    """
    a = array([1]+n*[0],dtype=dtype)
    b = empty((n,n),dtype=dtype)
    b.flat = a
    return b

def allclose (a, b, rtol=1.e-5, atol=1.e-8):
    """ allclose(a,b,rtol=1.e-5,atol=1.e-8)
        Returns true if all components of a and b are equal
        subject to given tolerances.
        The relative error rtol must be positive and << 1.0
        The absolute error atol comes into play for those elements
        of y that are very small or zero; it says how small x must be also.
    """
    x = array(a, copy=0)
    y = array(b, copy=0)
    d = less(absolute(x-y), atol + rtol * absolute(y))
    return alltrue(ravel(d))
            

_errdict = {"ignore":ERR_IGNORE,
            "warn":ERR_WARN,
            "raise":ERR_RAISE,
            "call":ERR_CALL}

_errdict_rev = {}
for key in _errdict.keys():
    _errdict_rev[_errdict[key]] = key
del key

def seterr(divide="ignore", over="ignore", under="ignore", invalid="ignore", where=0):
    maskvalue = (_errdict[divide] << SHIFT_DIVIDEBYZERO) + \
                (_errdict[over] << SHIFT_OVERFLOW ) + \
                (_errdict[under] << SHIFT_UNDERFLOW) + \
                (_errdict[invalid] << SHIFT_INVALID)
    frame = sys._getframe().f_back
    try:
        where = where.lower()
    except AttributeError:
        pass
    if not where or where[0] == 'l':
        frame.f_locals[UFUNC_ERRMASK_NAME] = maskvalue
    elif where == 1 or where[0] == 'g':
        frame.f_globals[UFUNC_ERRMASK_NAME] = maskvalue
    elif where == 2 or where[0] == 'b':
        frame.f_builtins[UFUNC_ERRMASK_NAME] = maskvalue
    return

    frame.f_locals[UFUNC_ERRMASK_NAME] = maskvalue
    return

def geterr():
    frame = sys._getframe().f_back
    try:
        maskvalue = frame.f_locals[UFUNC_ERRMASK_NAME]
    except KeyError:
        maskvalue = ERR_DEFAULT

    mask = 3
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

def setbufsize(size, where=0):
    if size > 10e6:
        raise ValueError, "Very big buffers.. %s" % size
    frame = sys._getframe().f_back
    try:
        wh = where.lower()
    except AttributeError:
        pass
    if not where or where[0] == 'l':
        frame.f_locals[UFUNC_BUFSIZE_NAME] = size
    elif where == 1 or where[0] == 'g':
        frame.f_globals[UFUNC_BUFSIZE_NAME] = size
    elif where == 2 or where[0] == 'b':
        frame.f_builtins[UFUNC_BUFSIZE_NAME] = size
    return

def getbufsize(size):
    frame = sys._getframe().f_back
    try:
        retval = frame.f_locals[UFUNC_BUFSIZE_NAME]
    except KeyError:
        retval = frame.f_globals[UFUNC_BUFSIZE_NAME]
    except KeyError:
        retval = frame.f_builtins[UFUNC_BUFSIZE_NAME]
    except KeyError:
        retvalue = UFUNC_BUFSIZE_DEFAULT

    return retval


# Set the UFUNC_BUFSIZE_NAME to something
# Set the UFUNC_ERRMASK_NAME to something
seterr(where='builtin')
setbufsize(UFUNC_BUFSIZE_DEFAULT,where='builtin')

inf = PINF
nan = NAN
from oldnumeric import *


