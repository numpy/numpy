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
           'indices', 'fromfunction',
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

ndarray = multiarray.ndarray
flatiter = multiarray.flatiter
broadcast = multiarray.broadcast
dtype = multiarray.dtype
ufunc = type(sin)


# originally from Fernando Perez's IPython
def zeros_like(a):
    """
    Returns an array of zeros with the same shape and type as a given array.

    Equivalent to ``a.copy().fill(0)``.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` defines the parameters of
        the returned array.

    Returns
    -------
    out : ndarray
        Array of zeros with same shape and type as `a`.

    See Also
    --------
    numpy.ones_like : Return an array of ones with shape and type of input.
    numpy.empty_like : Return an empty array with shape and type of input.
    numpy.zeros : Return a new array setting values to zero.
    numpy.ones : Return a new array setting values to one.
    numpy.empty : Return a new uninitialized array.

    Examples
    --------
    >>> x = np.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.zeros_like(x)
    array([[0, 0, 0],
           [0, 0, 0]])

    """
    if isinstance(a, ndarray):
        res = ndarray.__new__(type(a), a.shape, a.dtype, order=a.flags.fnc)
        res.fill(0)
        return res
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
    """
    Create a new array with the same shape and type as another.

    Parameters
    ----------
    a : ndarray
        Returned array will have same shape and type as `a`.

    See Also
    --------
    zeros_like, ones_like, zeros, ones, empty

    Notes
    -----
    This function does *not* initialize the returned array; to do that use
    `zeros_like` or `ones_like` instead.

    Examples
    --------
    >>> a = np.array([[1,2,3],[4,5,6]])
    >>> np.empty_like(a)
    >>> np.empty_like(a)
    array([[-1073741821, -1067702173,       65538],    #random data
           [      25670,    23454291,       71800]])

    """
    if isinstance(a, ndarray):
        res = ndarray.__new__(type(a), a.shape, a.dtype, order=a.flags.fnc)
        return res
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
    """
    Convert the input to an array.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array.  This
        includes lists, lists of tuples, tuples, tuples of tuples, tuples
        of lists and ndarrays.
    dtype : data-type, optional
        By default, the data-type is inferred from the input data.
    order : {'C', 'F'}, optional
        Whether to use row-major ('C') or column-major ('FORTRAN') memory
        representation.  Defaults to 'C'.

    Returns
    -------
    out : ndarray
        Array interpretation of `a`.  No copy is performed if the input
        is already an ndarray.  If `a` is a subclass of ndarray, a base
        class ndarray is returned.

    See Also
    --------
    asanyarray : Similar function which passes through subclasses.
    ascontiguousarray : Convert input to a contiguous array.
    asfarray : Convert input to a floating point ndarray.
    asfortranarray : Convert input to an ndarray with column-major
                     memory order.
    asarray_chkfinite : Similar function which checks input for NaNs and Infs.
    fromiter : Create an array from an iterator.
    fromfunction : Construct an array by executing a function on grid
                   positions.

    Examples
    --------
    Convert a list into an array:

    >>> a = [1, 2]
    >>> np.asarray(a)
    array([1, 2])

    Existing arrays are not copied:

    >>> a = np.array([1, 2])
    >>> np.asarray(a) is a
    True

    """
    return array(a, dtype, copy=False, order=order)

def asanyarray(a, dtype=None, order=None):
    """
    Convert the input to a ndarray, but pass ndarray subclasses through.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array.  This
        includes scalars, lists, lists of tuples, tuples, tuples of tuples,
        tuples of lists and ndarrays.
    dtype : data-type, optional
        By default, the data-type is inferred from the input data.
    order : {'C', 'F'}, optional
        Whether to use row-major ('C') or column-major ('F') memory
        representation.  Defaults to 'C'.

    Returns
    -------
    out : ndarray or an ndarray subclass
        Array interpretation of `a`.  If `a` is an ndarray or a subclass
        of ndarray, it is returned as-is and no copy is performed.

    See Also
    --------
    asarray : Similar function which always returns ndarrays.
    ascontiguousarray : Convert input to a contiguous array.
    asfarray : Convert input to a floating point ndarray.
    asfortranarray : Convert input to an ndarray with column-major
                     memory order.
    asarray_chkfinite : Similar function which checks input for NaNs and Infs.
    fromiter : Create an array from an iterator.
    fromfunction : Construct an array by executing a function on grid
                   positions.

    Examples
    --------
    Convert a list into an array:

    >>> a = [1, 2]
    >>> np.asanyarray(a)
    array([1, 2])

    Instances of `ndarray` subclasses are passed through as-is:

    >>> a = np.matrix([1, 2])
    >>> np.asanyarray(a) is a
    True

    """
    return array(a, dtype, copy=False, order=order, subok=True)

def ascontiguousarray(a, dtype=None):
    """
    Return a contiguous array in memory (C order).

    Parameters
    ----------
    a : array_like
        Input array.
    dtype : str or dtype object, optional
        Data-type of returned array.

    Returns
    -------
    out : ndarray
        Contiguous array of same shape and content as `a`, with type `dtype`
        if specified.

    See Also
    --------
    asfortranarray : Convert input to an ndarray with column-major
                     memory order.
    require : Return an ndarray that satisfies requirements.
    ndarray.flags : Information about the memory layout of the array.

    Examples
    --------
    >>> x = np.arange(6).reshape(2,3)
    >>> np.ascontiguousarray(x, dtype=np.float32)
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.]], dtype=float32)
    >>> x.flags['C_CONTIGUOUS']
    True

    """
    return array(a, dtype, copy=False, order='C', ndmin=1)

def asfortranarray(a, dtype=None):
    """
    Return an array laid out in Fortran order in memory.

    Parameters
    ----------
    a : array_like
        Input array.
    dtype : str or dtype object, optional
        By default, the data-type is inferred from the input data.

    Returns
    -------
    out : ndarray
        The input `a` in Fortran, or column-major, order.

    See Also
    --------
    ascontiguousarray : Convert input to a contiguous (C order) array.
    asanyarray : Convert input to an ndarray with either row or
        column-major memory order.
    require : Return an ndarray that satisfies requirements.
    ndarray.flags : Information about the memory layout of the array.

    Examples
    --------
    >>> x = np.arange(6).reshape(2,3)
    >>> y = np.asfortranarray(x)
    >>> x.flags['F_CONTIGUOUS']
    False
    >>> y.flags['F_CONTIGUOUS']
    True

    """
    return array(a, dtype, copy=False, order='F', ndmin=1)

def require(a, dtype=None, requirements=None):
    """
    Return an ndarray of the provided type that satisfies requirements.

    This function is useful to be sure that an array with the correct flags
    is returned for passing to compiled code (perhaps through ctypes).

    Parameters
    ----------
    a : array_like
       The object to be converted to a type-and-requirement satisfying array
    dtype : data-type
       The required data-type (None is the default data-type -- float64)
    requirements : list of strings
       The requirements list can be any of the following

       * 'ENSUREARRAY' ('E')  - ensure that  a base-class ndarray
       * 'F_CONTIGUOUS' ('F') - ensure a Fortran-contiguous array
       * 'C_CONTIGUOUS' ('C') - ensure a C-contiguous array
       * 'ALIGNED' ('A')      - ensure a data-type aligned array
       * 'WRITEABLE' ('W')    - ensure a writeable array
       * 'OWNDATA' ('O')      - ensure an array that owns its own data

    Notes
    -----
    The returned array will be guaranteed to have the listed requirements
    by making a copy if needed.

    """
    if requirements is None:
        requirements = []
    else:
        requirements = [x.upper() for x in requirements]

    if not requirements:
        return asanyarray(a, dtype=dtype)

    if 'ENSUREARRAY' in requirements or 'E' in requirements:
        subok = False
    else:
        subok = True

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
    """
    Returns True if array is arranged in Fortran-order in memory
    and dimension > 1.

    Parameters
    ----------
    a : ndarray
        Input array.


    Examples
    --------

    np.array allows to specify whether the array is written in C-contiguous
    order (last index varies the fastest), or FORTRAN-contiguous order in
    memory (first index varies the fastest).

    >>> a = np.array([[1, 2, 3], [4, 5, 6]], order='C')
    >>> a
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> np.isfortran(a)
    False

    >>> b = np.array([[1, 2, 3], [4, 5, 6]], order='FORTRAN')
    >>> b
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> np.isfortran(b)
    True


    The transpose of a C-ordered array is a FORTRAN-ordered array.

    >>> a = np.array([[1, 2, 3], [4, 5, 6]], order='C')
    >>> a
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> np.isfortran(a)
    False
    >>> b = a.T
    >>> b
    array([[1, 4],
           [2, 5],
           [3, 6]])
    >>> np.isfortran(b)
    True

    1-D arrays always evaluate as False.

    >>> np.isfortran(np.array([1, 2], order='FORTRAN'))
    False

    """
    return a.flags.fnc

def argwhere(a):
    """
    Find the indices of array elements that are non-zero, grouped by element.

    Parameters
    ----------
    a : array_like
        Input data.

    Returns
    -------
    index_array : ndarray
        Indices of elements that are non-zero. Indices are grouped by element.

    See Also
    --------
    where, nonzero

    Notes
    -----
    ``np.argwhere(a)`` is the same as ``np.transpose(np.nonzero(a))``.

    The output of ``argwhere`` is not suitable for indexing arrays.
    For this purpose use ``where(a)`` instead.

    Examples
    --------
    >>> x = np.arange(6).reshape(2,3)
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.argwhere(x>1)
    array([[0, 2],
           [1, 0],
           [1, 1],
           [1, 2]])

    """
    return asarray(a.nonzero()).T

def flatnonzero(a):
    """
    Return indices that are non-zero in the flattened version of a.

    This is equivalent to a.ravel().nonzero()[0].

    Parameters
    ----------
    a : ndarray
        Input array.

    Returns
    -------
    res : ndarray
        Output array, containing the indices of the elements of `a.ravel()`
        that are non-zero.

    See Also
    --------
    nonzero : Return the indices of the non-zero elements of the input array.
    ravel : Return a 1-D array containing the elements of the input array.

    Examples
    --------
    >>> x = np.arange(-2, 3)
    >>> x
    array([-2, -1,  0,  1,  2])
    >>> np.flatnonzero(x)
    array([0, 1, 3, 4])

    Use the indices of the non-zero elements as an index array to extract
    these elements:

    >>> x.ravel()[np.flatnonzero(x)]
    array([-2, -1,  1,  2])

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
    """
    Discrete, linear correlation of two 1-dimensional sequences.

    This function is equivalent to

    >>> np.convolve(a, v[::-1], mode=mode)

    where ``v[::-1]`` is the reverse of `v`.

    Parameters
    ----------
    a, v : array_like
        Input sequences.
    mode : {'valid', 'same', 'full'}, optional
        Refer to the `convolve` docstring.  Note that the default
        is `valid`, unlike `convolve`, which uses `full`.

    See Also
    --------
    convolve : Discrete, linear convolution of two
               one-dimensional sequences.

    """
    mode = _mode_from_name(mode)
    return multiarray.correlate(a,v,mode)


def convolve(a,v,mode='full'):
    """
    Returns the discrete, linear convolution of two one-dimensional sequences.

    The convolution operator is often seen in signal processing, where it
    models the effect of a linear time-invariant system on a signal [1]_.  In
    probability theory, the sum of two independent random variables is
    distributed according to the convolution of their individual
    distributions.

    Parameters
    ----------
    a : (N,) array_like
        First one-dimensional input array.
    v : (M,) array_like
        Second one-dimensional input array.
    mode : {'full', 'valid', 'same'}, optional
        'full':
          By default, mode is 'full'.  This returns the convolution
          at each point of overlap, with an output shape of (N+M-1,). At
          the end-points of the convolution, the signals do not overlap
          completely, and boundary effects may be seen.

        'same':
          Mode `same` returns output of length ``max(M, N)``.  Boundary
          effects are still visible.

        'valid':
          Mode `valid` returns output of length
          ``max(M, N) - min(M, N) + 1``.  The convolution product is only given
          for points where the signals overlap completely.  Values outside
          the signal boundary have no effect.

    Returns
    -------
    out : ndarray
        Discrete, linear convolution of `a` and `v`.

    See Also
    --------
    scipy.signal.fftconv : Convolve two arrays using the Fast Fourier
                           Transform.
    scipy.linalg.toeplitz : Used to construct the convolution operator.

    Notes
    -----
    The discrete convolution operation is defined as

    .. math:: (f * g)[n] = \\sum_{m = -\\infty}^{\\infty} f[m] f[n - m]

    It can be shown that a convolution :math:`x(t) * y(t)` in time/space
    is equivalent to the multiplication :math:`X(f) Y(f)` in the Fourier
    domain, after appropriate padding (padding is necessary to prevent
    circular convolution).  Since multiplication is more efficient (faster)
    than convolution, the function `scipy.signal.fftconvolve` exploits the
    FFT to calculate the convolution of large data-sets.

    References
    ----------
    .. [1] Wikipedia, "Convolution", http://en.wikipedia.org/wiki/Convolution.

    Examples
    --------
    Note how the convolution operator flips the second array
    before "sliding" the two across one another:

    >>> np.convolve([1, 2, 3], [0, 1, 0.5])
    array([ 0. ,  1. ,  2.5,  4. ,  1.5])

    Only return the middle values of the convolution.
    Contains boundary effects, where zeros are taken
    into account:

    >>> np.convolve([1,2,3],[0,1,0.5], 'same')
    array([ 1. ,  2.5,  4. ])

    The two arrays are of the same length, so there
    is only one position where they completely overlap:

    >>> np.convolve([1,2,3],[0,1,0.5], 'valid')
    array([ 2.5])

    """
    a,v = array(a, ndmin=1),array(v, ndmin=1)
    if (len(v) > len(a)):
        a, v = v, a
    if len(a) == 0 :
        raise ValueError('a cannot be empty')
    if len(v) == 0 :
        raise ValueError('v cannot be empty')
    mode = _mode_from_name(mode)
    return multiarray.correlate(a, v[::-1], mode)

def outer(a,b):
    """
    Returns the outer product of two vectors.

    Given two vectors, ``[a0, a1, ..., aM]`` and ``[b0, b1, ..., bN]``,
    the outer product becomes::

      [[a0*b0  a0*b1 ... a0*bN ]
       [a1*b0    .
       [ ...          .
       [aM*b0            aM*bN ]]

    Parameters
    ----------
    a : array_like, shaped (M,)
        First input vector.  If either of the input vectors are not
        1-dimensional, they are flattened.
    b : array_like, shaped (N,)
        Second input vector.

    Returns
    -------
    out : ndarray, shaped (M, N)
        ``out[i, j] = a[i] * b[j]``

    Notes
    -----
    The outer product of vectors is a special case of the Kronecker product.

    Examples
    --------
    >>> x = np.array(['a', 'b', 'c'], dtype=object)

    >>> np.outer(x, [1, 2, 3])
    array([[a, aa, aaa],
           [b, bb, bbb],
           [c, cc, ccc]], dtype=object)

    """
    a = asarray(a)
    b = asarray(b)
    return a.ravel()[:,newaxis]*b.ravel()[newaxis,:]

# try to import blas optimized dot if available
try:
    # importing this changes the dot function for basic 4 types
    # to blas-optimized versions.
    from _dotblas import dot, vdot, inner, alterdot, restoredot
except ImportError:
    # docstrings are in add_newdocs.py
    inner = multiarray.inner
    dot = multiarray.dot
    def vdot(a, b):
        return dot(asarray(a).ravel().conj(), asarray(b).ravel())
    def alterdot():
        pass
    def restoredot():
        pass

def tensordot(a, b, axes=2):
    """
    Returns the tensor dot product for (ndim >= 1) arrays along an axes.

    The first element of the sequence determines the axis or axes
    in `a` to sum over, and the second element in `axes` argument sequence
    determines the axis or axes in `b` to sum over.

    Parameters
    ----------
    a : array_like
        Input array.
    b : array_like
        Input array.
    axes : shape tuple
        Axes to be summed over.

    See Also
    --------
    dot

    Notes
    -----
    r_{xxx, yyy} = \\sum_k a_{xxx,k} b_{k,yyy}

    When there is more than one axis to sum over, the corresponding
    arguments to axes should be sequences of the same length with the first
    axis to sum over given first in both sequences, the second axis second,
    and so forth.

    If the `axes` argument is an integer, N, then the last N dimensions of `a`
    and first N dimensions of `b` are summed over.

    Examples
    --------
    >>> a = np.arange(60.).reshape(3,4,5)
    >>> b = np.arange(24.).reshape(4,3,2)
    >>> c = np.tensordot(a,b, axes=([1,0],[0,1]))
    >>> c.shape
    (5, 2)
    >>> c
    array([[ 4400.,  4730.],
           [ 4532.,  4874.],
           [ 4664.,  5018.],
           [ 4796.,  5162.],
           [ 4928.,  5306.]])

    >>> # A slower but equivalent way of computing the same...
    >>> c = np.zeros((5,2))
    >>> for i in range(5):
    ...   for j in range(2):
    ...     for k in range(3):
    ...       for n in range(4):
    ...         c[i,j] += a[k,n,i] * b[n,k,j]

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
    equal = True
    if (na != nb): equal = False
    else:
        for k in xrange(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
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
    """
    Roll array elements along a given axis.

    Elements that roll beyond the last position are re-introduced at
    the first.

    Parameters
    ----------
    a : array_like
        Input array.
    shift : int
        The number of places by which elements are shifted.
    axis : int, optional
        The axis along which elements are shifted.  By default, the array
        is flattened before shifting, after which the original
        shape is restored.

    Returns
    -------
    res : ndarray
        Output array, with the same shape as `a`.

    See Also
    --------
    rollaxis : Roll the specified axis backwards, until it lies in a
               given position.

    Examples
    --------
    >>> x = np.arange(10)
    >>> np.roll(x, 2)
    array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])

    >>> x2 = np.reshape(x, (2,5))
    >>> x2
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])
    >>> np.roll(x2, 1)
    array([[9, 0, 1, 2, 3],
           [4, 5, 6, 7, 8]])
    >>> np.roll(x2, 1, axis=0)
    array([[5, 6, 7, 8, 9],
           [0, 1, 2, 3, 4]])
    >>> np.roll(x2, 1, axis=1)
    array([[4, 0, 1, 2, 3],
           [9, 5, 6, 7, 8]])

    """
    a = asanyarray(a)
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    shift %= n
    indexes = concatenate((arange(n-shift,n),arange(n-shift)))
    res = a.take(indexes, axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res

def rollaxis(a, axis, start=0):
    """
    Roll the specified axis backwards, until it lies in a given position.

    Parameters
    ----------
    a : ndarray
        Input array.
    axis : int
        The axis to roll backwards.  The positions of the other axes do not
        change relative to one another.
    start : int, optional
        The axis is rolled until it lies before this position.

    Returns
    -------
    res : ndarray
        Output array.

    See Also
    --------
    roll : Roll the elements of an array by a number of positions along a
           given axis.

    Examples
    --------
    >>> a = np.ones((3,4,5,6))
    >>> np.rollaxis(a, 3, 1).shape
    (3, 6, 4, 5)
    >>> np.rollaxis(a, 2).shape
    (5, 3, 4, 6)
    >>> np.rollaxis(a, 1, 4).shape
    (3, 5, 6, 4)

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
    """
    Return the cross product of two (arrays of) vectors.

    The cross product of `a` and `b` in :math:`R^3` is a vector perpendicular
    to both `a` and `b`.  If `a` and `b` are arrays of vectors, the vectors
    are defined by the last axis of `a` and `b` by default, and these axes
    can have dimensions 2 or 3.  Where the dimension of either `a` or `b` is
    2, the third component of the input vector is assumed to be zero and the
    cross product calculated accordingly.  In cases where both input vectors
    have dimension 2, the z-component of the cross product is returned.

    Parameters
    ----------
    a : array_like
        Components of the first vector(s).
    b : array_like
        Components of the second vector(s).
    axisa : int, optional
        Axis of `a` that defines the vector(s).  By default, the last axis.
    axisb : int, optional
        Axis of `b` that defines the vector(s).  By default, the last axis.
    axisc : int, optional
        Axis of `c` containing the cross product vector(s).  By default, the
        last axis.
    axis : int, optional
        If defined, the axis of `a`, `b` and `c` that defines the vector(s)
        and cross product(s).  Overrides `axisa`, `axisb` and `axisc`.

    Returns
    -------
    c : ndarray
        Vector cross product(s).

    Raises
    ------
    ValueError
        When the dimension of the vector(s) in `a` and/or `b` does not
        equal 2 or 3.

    See Also
    --------
    inner : Inner product
    outer : Outer product.
    ix_ : Construct index arrays.

    Examples
    --------
    Vector cross-product.

    >>> x = [1, 2, 3]
    >>> y = [4, 5, 6]
    >>> np.cross(x, y)
    array([-3,  6, -3])

    One vector with dimension 2.

    >>> x = [1, 2]
    >>> y = [4, 5, 6]
    >>> np.cross(x, y)
    array([12, -6, -3])

    Equivalently:

    >>> x = [1, 2, 0]
    >>> y = [4, 5, 6]
    >>> np.cross(x, y)
    array([12, -6, -3])

    Both vectors with dimension 2.

    >>> x = [1,2]
    >>> y = [4,5]
    >>> np.cross(x, y)
    -3

    Multiple vector cross-products. Note that the direction of the cross
    product vector is defined by the `right-hand rule`.

    >>> x = np.array([[1,2,3], [4,5,6]])
    >>> y = np.array([[4,5,6], [1,2,3]])
    >>> np.cross(x, y)
    array([[-3,  6, -3],
           [ 3, -6,  3]])

    The orientation of `c` can be changed using the `axisc` keyword.

    >>> np.cross(x, y, axisc=0)
    array([[-3,  3],
           [ 6, -6],
           [-3,  3]])

    Change the vector definition of `x` and `y` using `axisa` and `axisb`.

    >>> x = np.array([[1,2,3], [4,5,6], [7, 8, 9]])
    >>> y = np.array([[7, 8, 9], [4,5,6], [1,2,3]])
    >>> np.cross(x, y)
    array([[ -6,  12,  -6],
           [  0,   0,   0],
           [  6, -12,   6]])
    >>> np.cross(x, y, axisa=0, axisb=0)
    array([[-24,  48, -24],
           [-30,  60, -30],
           [-36,  72, -36]])

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
    """
    Return the string representation of an array.

    Parameters
    ----------
    arr : ndarray
      Input array.
    max_line_width : int
      The maximum number of columns the string should span. Newline
      characters splits the string appropriately after array elements.
    precision : int
      Floating point precision.
    suppress_small : bool
      Represent very small numbers as zero.

    Returns
    -------
    string : str
      The string representation of an array.


    Examples
    --------
    >>> np.array_repr(np.array([1,2]))
    'array([1, 2])'
    >>> np.array_repr(np.ma.array([0.]))
    'MaskedArray([ 0.])'
    >>> np.array_repr(np.array([], np.int32))
    'array([], dtype=int32)'

    """
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
    """
    Return a string representation of an array.

    Parameters
    ----------
    a : ndarray
        Input array.
    max_line_width : int, optional
        Inserts newlines if text is longer than `max_line_width`.
    precision : int, optional
        If `a` is float, `precision` sets floating point precision.
    suppress_small : boolean, optional
        Represent very small numbers as zero.

    See Also
    --------
    array2string, array_repr

    Examples
    --------
    >>> np.array_str(np.arange(3))
    >>> '[0 1 2]'

    """
    return array2string(a, max_line_width, precision, suppress_small, ' ', "", str)

set_string_function = multiarray.set_string_function
set_string_function(array_str, 0)
set_string_function(array_repr, 1)

little_endian = (sys.byteorder == 'little')


def indices(dimensions, dtype=int):
    """
    Return an array representing the indices of a grid.

    Compute an array where the subarrays contain index values 0,1,...
    varying only along the corresponding axis.

    Parameters
    ----------
    dimensions : sequence of ints
        The shape of the grid.
    dtype : optional
        Data_type of the result.

    Returns
    -------
    grid : ndarray
        The array of grid indices,
        ``grid.shape = (len(dimensions),) + tuple(dimensions)``.

    See Also
    --------
    mgrid, meshgrid

    Notes
    -----
    The output shape is obtained by prepending the number of dimensions
    in front of the tuple of dimensions, i.e. if `dimensions` is a tuple
    ``(r0, ..., rN-1)`` of length ``N``, the output shape is
    ``(N,r0,...,rN-1)``.

    The subarrays ``grid[k]`` contains the N-D array of indices along the
    ``k-th`` axis. Explicitly::

        grid[k,i0,i1,...,iN-1] = ik

    Examples
    --------
    >>> grid = np.indices((2,3))
    >>> grid.shape
    (2,2,3)
    >>> grid[0]        # row indices
    array([[0, 0, 0],
           [1, 1, 1]])
    >>> grid[1]        # column indices
    array([[0, 1, 2],
           [0, 1, 2]])

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
    """
    Construct an array by executing a function over each coordinate.

    The resulting array therefore has a value ``fn(x, y, z)`` at
    coordinate ``(x, y, z)``.

    Parameters
    ----------
    fn : callable
        The function is called with N parameters, each of which
        represents the coordinates of the array varying along a
        specific axis.  For example, if `shape` were ``(2, 2)``, then
        the parameters would be two arrays, ``[[0, 0], [1, 1]]`` and
        ``[[0, 1], [0, 1]]``.  `fn` must be capable of operating on
        arrays, and should return a scalar value.
    shape : (N,) tuple of ints
        Shape of the output array, which also determines the shape of
        the coordinate arrays passed to `fn`.
    dtype : data-type, optional
        Data-type of the coordinate arrays passed to `fn`.  By default,
        `dtype` is float.

    See Also
    --------
    indices, meshgrid

    Notes
    -----
    Keywords other than `shape` and `dtype` are passed to the function.

    Examples
    --------
    >>> np.fromfunction(lambda i, j: i == j, (3, 3), dtype=int)
    array([[ True, False, False],
           [False,  True, False],
           [False, False,  True]], dtype=bool)

    >>> np.fromfunction(lambda i, j: i + j, (3, 3), dtype=int)
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4]])

    """
    dtype = kwargs.pop('dtype', float)
    args = indices(shape, dtype=dtype)
    return function(*args,**kwargs)

def isscalar(num):
    """
    Returns True if the type of `num` is a scalar type.

    Parameters
    ----------
    num : any
        Input argument, can be of any type and shape.

    Returns
    -------
    val : bool
        True if `num` is a scalar type, False if it is not.

    Examples
    --------
    >>> np.isscalar(3.1)
    True
    >>> np.isscalar([3.1])
    False
    >>> np.isscalar(False)
    True

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
    """
    Return the binary representation of the input number as a string.

    For negative numbers, if width is not given, a minus sign is added to the
    front. If width is given, the two's complement of the number is
    returned, with respect to that width.

    In a two's-complement system negative numbers are represented by the two's
    complement of the absolute value. This is the most common method of
    representing signed integers on computers [1]_. A N-bit two's-complement
    system can represent every integer in the range
    :math:`-2^{N-1}` to :math:`+2^{N-1}-1`.

    Parameters
    ----------
    num : int
        Only an integer decimal number can be used.
    width : int, optional
        The length of the returned string if `num` is positive, the length of
        the two's complement if `num` is negative.

    Returns
    -------
    bin : str
        Binary representation of `num` or two's complement of `num`.

    See Also
    --------
    base_repr: Return a string representation of a number in the given base
               system.

    Notes
    -----
    `binary_repr` is equivalent to using `base_repr` with base 2, but about 25x
    faster.

    References
    ----------
    .. [1] Wikipedia, "Two's complement",
        http://en.wikipedia.org/wiki/Two's_complement

    Examples
    --------
    >>> np.binary_repr(3)
    '11'
    >>> np.binary_repr(-3)
    '-11'
    >>> np.binary_repr(3, width=4)
    '0011'

    The two's complement is returned when the input number is negative and
    width is specified:

    >>> np.binary_repr(-3, width=4)
    '1101'

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
    """
    Return a string representation of a number in the given base system.

    Parameters
    ----------
    number : scalar
        The value to convert. Only positive values are handled.
    base : int
        Convert `number` to the `base` number system. The valid range is 2-36,
        the default value is 2.
    padding : int, optional
        Number of zeros padded on the left.

    Returns
    -------
    out : str
        String representation of `number` in `base` system.

    See Also
    --------
    binary_repr : Faster version of `base_repr` for base 2 that also handles
        negative numbers.

    Examples
    --------
    >>> np.base_repr(3, 5)
    '3'
    >>> np.base_repr(6, 5)
    '11'
    >>> np.base_repr(7, base=5, padding=3)
    '00012'

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
    """
    Return a new array of given shape and type, filled with ones.

    Please refer to the documentation for `zeros`.

    See Also
    --------
    zeros

    Examples
    --------
    >>> np.ones(5)
    array([ 1.,  1.,  1.,  1.,  1.])

    >>> np.ones((5,), dtype=np.int)
    array([1, 1, 1, 1, 1])

    >>> np.ones((2, 1))
    array([[ 1.],
           [ 1.]])

    >>> s = (2,2)
    >>> np.ones(s)
    array([[ 1.,  1.],
           [ 1.,  1.]])

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
    """
    Return the identity array.

    The identity array is a square array with ones on
    the main diagonal.

    Parameters
    ----------
    n : int
        Number of rows (and columns) in `n` x `n` output.
    dtype : data-type, optional
        Data-type of the output.  Defaults to ``float``.

    Returns
    -------
    out : ndarray
        `n` x `n` array with its main diagonal set to one,
        and all other elements 0.

    Examples
    --------
    >>> np.identity(3)
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])

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
    """
    Returns True if two arrays are element-wise equal within a tolerance.

    The tolerance values are positive, typically very small numbers.  The
    relative difference (`rtol` * abs(`b`)) and the absolute difference
    `atol` are added together to compare against the absolute difference
    between `a` and `b`.

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    rtol : float
        The relative tolerance parameter (see Notes).
    atol : float
        The absolute tolerance parameter (see Notes).

    Returns
    -------
    y : bool
        Returns True if the two arrays are equal within the given
        tolerance; False otherwise. If either array contains NaN, then
        False is returned.

    See Also
    --------
    all, any, alltrue, sometrue

    Notes
    -----
    If the following equation is element-wise True, then allclose returns
    True.

     absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))

    Examples
    --------
    >>> np.allclose([1e10,1e-7], [1.00001e10,1e-8])
    False
    >>> np.allclose([1e10,1e-8], [1.00001e10,1e-9])
    True
    >>> np.allclose([1e10,1e-8], [1.0001e10,1e-9])
    False
    >>> np.allclose([1.0, np.nan], [1.0, np.nan])
    False

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
    """
    True if two arrays have the same shape and elements, False otherwise.

    Parameters
    ----------
    a1, a2 : array_like
        Input arrays.

    Returns
    -------
    b : bool
        Returns True if the arrays are equal.

    See Also
    --------
    allclose: Returns True if two arrays are element-wise equal within a
              tolerance.
    array_equiv: Returns True if input arrays are shape consistent and all
                 elements equal.

    Examples
    --------
    >>> np.array_equal([1, 2], [1, 2])
    True
    >>> np.array_equal(np.array([1, 2]), np.array([1, 2]))
    True
    >>> np.array_equal([1, 2], [1, 2, 3])
    False
    >>> np.array_equal([1, 2], [1, 4])
    False

    """
    try:
        a1, a2 = asarray(a1), asarray(a2)
    except:
        return False
    if a1.shape != a2.shape:
        return False
    return bool(logical_and.reduce(equal(a1,a2).ravel()))

def array_equiv(a1, a2):
    """
    Returns True if input arrays are shape consistent and all elements equal.

    Shape consistent means they are either the same shape, or one input array
    can be broadcasted to create the same shape as the other one.

    Parameters
    ----------
    a1, a2 : array_like
        Input arrays.

    Returns
    -------
    out : bool
        True if equivalent, False otherwise.

    Examples
    --------
    >>> np.array_equiv([1, 2], [1, 2])
    >>> True
    >>> np.array_equiv([1, 2], [1, 3])
    >>> False

    Showing the shape equivalence:

    >>> np.array_equiv([1, 2], [[1, 2], [1, 2]])
    >>> True
    >>> np.array_equiv([1, 2], [[1, 2, 1, 2], [1, 2, 1, 2]])
    >>> False

    >>> np.array_equiv([1, 2], [[1, 2], [1, 3]])
    >>> False

    """
    try:
        a1, a2 = asarray(a1), asarray(a2)
    except:
        return False
    try:
        return bool(logical_and.reduce(equal(a1,a2).ravel()))
    except ValueError:
        return False


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
    """
    Set how floating-point errors are handled.

    Note that operations on integer scalar types (such as `int16`) are
    handled like floating point, and are affected by these settings.

    Parameters
    ----------
    all : {'ignore', 'warn', 'raise', 'call'}, optional
        Set treatment for all types of floating-point errors at once:

        - ignore: Take no action when the exception occurs
        - warn: Print a `RuntimeWarning` (via the Python `warnings` module)
        - raise: Raise a `FloatingPointError`
        - call: Call a function specified using the `seterrcall` function.

        The default is not to change the current behavior.
    divide : {'ignore', 'warn', 'raise', 'call'}, optional
        Treatment for division by zero.
    over : {'ignore', 'warn', 'raise', 'call'}, optional
        Treatment for floating-point overflow.
    under : {'ignore', 'warn', 'raise', 'call'}, optional
        Treatment for floating-point underflow.
    invalid : {'ignore', 'warn', 'raise', 'call'}, optional
        Treatment for invalid floating-point operation.

    Returns
    -------
    old_settings : dict
        Dictionary containing the old settings.

    See also
    --------
    seterrcall : set a callback function for the 'call' mode.
    geterr, geterrcall

    Notes
    -----
    The floating-point exceptions are defined in the IEEE 754 standard [1]:

    - Division by zero: infinite result obtained from finite numbers.
    - Overflow: result too large to be expressed.
    - Underflow: result so close to zero that some precision
      was lost.
    - Invalid operation: result is not an expressible number, typically
      indicates that a NaN was produced.

    .. [1] http://en.wikipedia.org/wiki/IEEE_754

    Examples
    --------

    Set mode:

    >>> seterr(over='raise') # doctest: +SKIP
    {'over': 'ignore', 'divide': 'ignore', 'invalid': 'ignore',
     'under': 'ignore'}

    >>> old_settings = seterr(all='warn', over='raise') # doctest: +SKIP

    >>> int16(32000) * int16(3) # doctest: +SKIP
    Traceback (most recent call last):
          File "<stdin>", line 1, in ?
    FloatingPointError: overflow encountered in short_scalars
    >>> seterr(all='ignore') # doctest: +SKIP
    {'over': 'ignore', 'divide': 'ignore', 'invalid': 'ignore',
     'under': 'ignore'}

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
    """
    Set the size of the buffer used in ufuncs.

    Parameters
    ----------
    size : int
        Size of buffer.

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
    """
    Set the floating-point error callback function or log object.

    There are two ways to capture floating-point error messages.  The first
    is to set the error-handler to 'call', using `seterr`.  Then, set
    the function to call using this function.

    The second is to set the error-handler to `log`, using `seterr`.
    Floating-point errors then trigger a call to the 'write' method of
    the provided object.

    Parameters
    ----------
    log_func_or_obj : callable f(err, flag) or object with write method
        Function to call upon floating-point errors ('call'-mode) or
        object whose 'write' method is used to log such message ('log'-mode).

        The call function takes two arguments. The first is the
        type of error (one of "divide", "over", "under", or "invalid"),
        and the second is the status flag.  The flag is a byte, whose
        least-significant bits indicate the status::

          [0 0 0 0 invalid over under invalid]

        In other words, ``flags = divide + 2*over + 4*under + 8*invalid``.

        If an object is provided, it's write method should take one argument,
        a string.

    Returns
    -------
    h : callable or log instance
        The old error handler.

    Examples
    --------
    Callback upon error:

    >>> def err_handler(type, flag):
        print "Floating point error (%s), with flag %s" % (type, flag)
    ...

    >>> saved_handler = np.seterrcall(err_handler)
    >>> save_err = np.seterr(all='call')

    >>> np.array([1,2,3])/0.0
    Floating point error (divide by zero), with flag 1
    array([ Inf,  Inf,  Inf])

    >>> np.seterrcall(saved_handler)
    >>> np.seterr(**save_err)

    Log error message:

    >>> class Log(object):
            def write(self, msg):
                print "LOG: %s" % msg
    ...

    >>> log = Log()
    >>> saved_handler = np.seterrcall(log)
    >>> save_err = np.seterr(all='log')

    >>> np.array([1,2,3])/0.0
    LOG: Warning: divide by zero encountered in divide

    >>> np.seterrcall(saved_handler)
    >>> np.seterr(**save_err)

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
    >>> _ = np.seterr(invalid='raise', divide='raise', over='raise',
    ...               under='ignore')

    >>> a = -np.arange(3)
    >>> with np.errstate(invalid='ignore'): # doctest: +SKIP
    ...     print np.sqrt(a)                # with statement requires Python 2.5
    [ 0.     -1.#IND -1.#IND]
    >>> print np.sqrt(a.astype(complex))
    [ 0.+0.j          0.+1.j          0.+1.41421356j]
    >>> print np.sqrt(a)
    Traceback (most recent call last):
     ...
    FloatingPointError: invalid value encountered in sqrt
    >>> with np.errstate(divide='ignore'):  # doctest: +SKIP
    ...     print a/0
    [0 0 0]
    >>> print a/0
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
