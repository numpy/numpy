# Module containing non-deprecated functions borrowed from Numeric.
__docformat__ = "restructuredtext en"

# functions that are now methods
__all__ = ['take', 'reshape', 'choose', 'repeat', 'put',
           'swapaxes', 'transpose', 'sort', 'argsort', 'argmax', 'argmin',
           'searchsorted', 'alen',
           'resize', 'diagonal', 'trace', 'ravel', 'nonzero', 'shape',
           'compress', 'clip', 'sum', 'product', 'prod', 'sometrue', 'alltrue',
           'any', 'all', 'cumsum', 'cumproduct', 'cumprod', 'ptp', 'ndim',
           'rank', 'size', 'around', 'round_', 'mean', 'std', 'var', 'squeeze',
           'amax', 'amin',
          ]

import multiarray as mu
import umath as um
import numerictypes as nt
from numeric import asarray, array, asanyarray, concatenate
_dt_ = nt.sctype2char

import types

try:
    _gentype = types.GeneratorType
except AttributeError:
    _gentype = types.NoneType

# save away Python sum
_sum_ = sum

# functions that are now methods
def _wrapit(obj, method, *args, **kwds):
    try:
        wrap = obj.__array_wrap__
    except AttributeError:
        wrap = None
    result = getattr(asarray(obj),method)(*args, **kwds)
    if wrap and isinstance(result, mu.ndarray):
        if not isinstance(result, mu.ndarray):
            result = asarray(result)
        result = wrap(result)
    return result


def take(a, indices, axis=None, out=None, mode='raise'):
    """Return an array with values pulled from the given array at the given
    indices.

    This function does the same thing as "fancy" indexing; however, it can be
    easier to use if you need to specify a given axis.

    :Parameters:
      - `a` : array
        The source array
      - `indices` : int array
        The indices of the values to extract.
      - `axis` : None or int, optional (default=None)
        The axis over which to select values. None signifies that the operation
        should be performed over the flattened array.
      - `out` : array, optional
        If provided, the result will be inserted into this array. It should be
        of the appropriate shape and dtype.
      - `mode` : one of 'raise', 'wrap', or 'clip', optional (default='raise')
        Specifies how out-of-bounds indices will behave.
        - 'raise' : raise an error
        - 'wrap' : wrap around
        - 'clip' : clip to the range

    :Returns:
      - `subarray` : array

    :See also:
      numpy.ndarray.take() is the equivalent method.
    """
    try:
        take = a.take
    except AttributeError:
        return _wrapit(a, 'take', indices, axis, out, mode)
    return take(indices, axis, out, mode)


# not deprecated --- copy if necessary, view otherwise
def reshape(a, newshape, order='C'):
    """Return an array that uses the data of the given array, but with a new
    shape.

    :Parameters:
      - `a` : array
      - `newshape` : shape tuple or int
        The new shape should be compatible with the original shape. If an
        integer, then the result will be a 1D array of that length.
      - `order` : 'C' or 'FORTRAN', optional (default='C')
        Whether the array data should be viewed as in C (row-major) order or
        FORTRAN (column-major) order.

    :Returns:
      - `reshaped_array` : array
        This will be a new view object if possible; otherwise, it will return
        a copy.

    :See also:
      numpy.ndarray.reshape() is the equivalent method.
    """
    try:
        reshape = a.reshape
    except AttributeError:
        return _wrapit(a, 'reshape', newshape, order=order)
    return reshape(newshape, order=order)


def choose(a, choices, out=None, mode='raise'):
    """Use an index array to construct a new array from a set of choices.

    Given an array of integers in {0, 1, ..., n-1} and a set of n choice arrays,
    this function will create a new array that merges each of the choice arrays.
    Where a value in `a` is i, then the new array will have the value that
    choices[i] contains in the same place.

    :Parameters:
      - `a` : int array
        This array must contain integers in [0, n-1], where n is the number of
        choices.
      - `choices` : sequence of arrays
        Each of the choice arrays should have the same shape as the index array.
      - `out` : array, optional
        If provided, the result will be inserted into this array. It should be
        of the appropriate shape and dtype
      - `mode` : one of 'raise', 'wrap', or 'clip', optional (default='raise')
        Specifies how out-of-bounds indices will behave.
        - 'raise' : raise an error
        - 'wrap' : wrap around
        - 'clip' : clip to the range

    :Returns:
      - `merged_array` : array

    :See also:
      numpy.ndarray.choose() is the equivalent method.

    :Example:
      >>> choices = [[0, 1, 2, 3], [10, 11, 12, 13],
      ...   [20, 21, 22, 23], [30, 31, 32, 33]]
      >>> choose([2, 3, 1, 0], choices)
      array([20, 31, 12,  3])
      >>> choose([2, 4, 1, 0], choices, mode='clip')
      array([20, 31, 12,  3])
      >>> choose([2, 4, 1, 0], choices, mode='wrap')
      array([20,  1, 12,  3])

    """
    try:
        choose = a.choose
    except AttributeError:
        return _wrapit(a, 'choose', choices, out=out, mode=mode)
    return choose(choices, out=out, mode=mode)


def repeat(a, repeats, axis=None):
    """Repeat elements of an array.

    :Parameters:
      - `a` : array
      - `repeats` : int or int array
        The number of repetitions for each element. If a plain integer, then it
        is applied to all elements. If an array, it needs to be of the same
        length as the chosen axis.
      - `axis` : None or int, optional (default=None)
        The axis along which to repeat values. If None, then this function will
        operated on the flattened array `a` and return a similarly flat result.

    :Returns:
      - `repeated_array` : array

    :See also:
      numpy.ndarray.repeat() is the equivalent method.

    :Example:
      >>> repeat([0, 1, 2], 2)
      array([0, 0, 1, 1, 2, 2])
      >>> repeat([0, 1, 2], [2, 3, 4])
      array([0, 0, 1, 1, 1, 2, 2, 2, 2])

    """
    try:
        repeat = a.repeat
    except AttributeError:
        return _wrapit(a, 'repeat', repeats, axis)
    return repeat(repeats, axis)


def put (a, ind, v, mode='raise'):
    """put(a, ind, v) results in a[n] = v[n] for all n in ind
       If v is shorter than mask it will be repeated as necessary.
       In particular v can be a scalar or length 1 array.
       The routine put is the equivalent of the following (although the loop
       is in C for speed):

           ind = array(indices, copy=False)
           v = array(values, copy=False).astype(a.dtype)
           for i in ind: a.flat[i] = v[i]
       a must be a contiguous numpy array.
    """
    return a.put(ind, v, mode)


def swapaxes(a, axis1, axis2):
    """swapaxes(a, axis1, axis2) returns array a with axis1 and axis2
    interchanged.
    """
    try:
        swapaxes = a.swapaxes
    except AttributeError:
        return _wrapit(a, 'swapaxes', axis1, axis2)
    return swapaxes(axis1, axis2)


def transpose(a, axes=None):
    """transpose(a, axes=None) returns a view of the array with
    dimensions permuted according to axes.  If axes is None
    (default) returns array with dimensions reversed.
    """
    try:
        transpose = a.transpose
    except AttributeError:
        return _wrapit(a, 'transpose', axes)
    return transpose(axes)


def sort(a, axis=-1, kind='quicksort', order=None):
    """Return copy of 'a' sorted along the given axis.

    *Description*

    Perform an inplace sort along the given axis using the algorithm specified
    by the kind keyword.

    *Parameters*:

        a : array type
            Array to be sorted.

        axis : integer
            Axis to be sorted along. None indicates that the flattened array
            should be used. Default is -1.

        kind : string
            Sorting algorithm to use. Possible values are 'quicksort',
            'mergesort', or 'heapsort'. Default is 'quicksort'.

        order : list type or None
            When a is an array with fields defined, this argument specifies
            which fields to compare first, second, etc.  Not all fields need be
            specified.

    *Returns*:

        sorted_array : type is unchanged.

    *SeeAlso*:

        argsort
            Indirect sort
        lexsort
            Indirect stable sort on multiple keys
        searchsorted
            Find keys in sorted array

    *Notes*

        The various sorts are characterized by average speed, worst case
        performance, need for work space, and whether they are stable. A stable
        sort keeps items with the same key in the same relative order. The
        three available algorithms have the following properties:

        +-----------+-------+-------------+------------+-------+
        |    kind   | speed |  worst case | work space | stable|
        +===========+=======+=============+============+=======+
        | quicksort |   1   | O(n^2)      |     0      |   no  |
        +-----------+-------+-------------+------------+-------+
        | mergesort |   2   | O(n*log(n)) |    ~n/2    |   yes |
        +-----------+-------+-------------+------------+-------+
        | heapsort  |   3   | O(n*log(n)) |     0      |   no  |
        +-----------+-------+-------------+------------+-------+

        All the sort algorithms make temporary copies of the data when the sort
        is not along the last axis. Consequently, sorts along the last axis are
        faster and use less space than sorts along other axis.

    """
    if axis is None:
        a = asanyarray(a).flatten()
        axis = 0
    else:
        a = asanyarray(a).copy()
    a.sort(axis, kind, order)
    return a


def argsort(a, axis=-1, kind='quicksort', order=None):
    """Returns array of indices that index 'a' in sorted order.

    *Description*

    Perform an indirect sort along the given axis using the algorithm specified
    by the kind keyword. It returns an array of indices of the same shape as
    a that index data along the given axis in sorted order.

    *Parameters*:

        a : array type
            Array containing values that the returned indices should sort.

        axis : integer
            Axis to be indirectly sorted. None indicates that the flattened
            array should be used. Default is -1.

        kind : string
            Sorting algorithm to use. Possible values are 'quicksort',
            'mergesort', or 'heapsort'. Default is 'quicksort'.

        order : list type or None
            When a is an array with fields defined, this argument specifies
            which fields to compare first, second, etc.  Not all fields need be
            specified.

    *Returns*:

        indices : integer array
            Array of indices that sort 'a' along the specified axis.

    *SeeAlso*:

        lexsort
            Indirect stable sort with multiple keys
        sort
            Inplace sort

    *Notes*

        The various sorts are characterized by average speed, worst case
        performance, need for work space, and whether they are stable. A stable
        sort keeps items with the same key in the same relative order. The
        three available algorithms have the following properties:

        +-----------+-------+-------------+------------+-------+
        |    kind   | speed |  worst case | work space | stable|
        +===========+=======+=============+============+=======+
        | quicksort |   1   | O(n^2)      |     0      |   no  |
        +-----------+-------+-------------+------------+-------+
        | mergesort |   2   | O(n*log(n)) |    ~n/2    |   yes |
        +-----------+-------+-------------+------------+-------+
        | heapsort  |   3   | O(n*log(n)) |     0      |   no  |
        +-----------+-------+-------------+------------+-------+

        All the sort algorithms make temporary copies of the data when the sort
        is not along the last axis. Consequently, sorts along the last axis are
        faster and use less space than sorts along other axis.

    """
    try:
        argsort = a.argsort
    except AttributeError:
        return _wrapit(a, 'argsort', axis, kind, order)
    return argsort(axis, kind, order)


def argmax(a, axis=None):
    """argmax(a,axis=None) returns the indices to the maximum value of the
    1-D arrays along the given axis.
    """
    try:
        argmax = a.argmax
    except AttributeError:
        return _wrapit(a, 'argmax', axis)
    return argmax(axis)


def argmin(a, axis=None):
    """argmin(a,axis=None) returns the indices to the minimum value of the
    1-D arrays along the given axis.
    """
    try:
        argmin = a.argmin
    except AttributeError:
        return _wrapit(a, 'argmin', axis)
    return argmin(axis)


def searchsorted(a, v, side='left'):
    """Returns indices where keys in v should be inserted to maintain order.

    *Description*

        Find the indices into a sorted array such that if the corresponding
        keys in v were inserted before the indices the order of a would be
        preserved.  If side='left', then the first such index is returned. If
        side='right', then the last such index is returned. If there is no such
        index because the key is out of bounds, then the length of a is
        returned, i.e., the key would need to be appended. The returned index
        array has the same shape as v.

    *Parameters*:

        a : array
            1-d array sorted in ascending order.

        v : array or list type
            Array of keys to be searched for in a.

        side : string
            Possible values are : 'left', 'right'. Default is 'left'. Return
            the first or last index where the key could be inserted.

    *Returns*:

        indices : integer array
            Array of insertion points with the same shape as v.

    *SeeAlso*:

        sort
            Inplace sort
        histogram
            Produce histogram from 1-d data


    *Notes*

        The array a must be 1-d and is assumed to be sorted in ascending order.
        Searchsorted uses binary search to find the required insertion points.

    """
    try:
        searchsorted = a.searchsorted
    except AttributeError:
        return _wrapit(a, 'searchsorted', v, side)
    return searchsorted(v, side)


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
    if not Na: return mu.zeros(new_shape, a.dtype.char)
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
    try:
        squeeze = a.squeeze
    except AttributeError:
        return _wrapit(a, 'squeeze')
    return squeeze()


def diagonal(a, offset=0, axis1=0, axis2=1):
    """Return specified diagonals. Uses first two indices by default.

    *Description*

    If a is 2-d, returns the diagonal of self with the given offset, i.e., the
    collection of elements of the form a[i,i+offset]. If a is n-d with n > 2,
    then the axes specified by axis1 and axis2 are used to determine the 2-d
    subarray whose diagonal is returned. The shape of the resulting array can be
    determined by removing axis1 and axis2 and appending an index to the right
    equal to the size of the resulting diagonals.

    *Parameters*:

        offset : integer
            Offset of the diagonal from the main diagonal. Can be both positive
            and negative. Defaults to main diagonal.

        axis1 : integer
            Axis to be used as the first axis of the 2-d subarrays from which
            the diagonals should be taken. Defaults to first axis.

        axis2 : integer
            Axis to be used as the second axis of the 2-d subarrays from which
            the diagonals should be taken. Defaults to second axis.

    *Returns*:

        array_of_diagonals : type of original array
            If a is 2-d, then a 1-d array containing the diagonal is returned.
            If a is n-d, n > 2, then an array of diagonals is returned.

    *SeeAlso*:

        diag :
            matlab workalike for 1-d and 2-d arrays
        diagflat :
            creates diagonal arrays
        trace :
            sum along diagonals

    *Examples*:

        >>> a = arange(4).reshape(2,2)
        >>> a
        array([[0, 1],
               [2, 3]])
        >>> a.diagonal()
        array([0, 3])
        >>> a.diagonal(1)
        array([1])

        >>> a = arange(8).reshape(2,2,2)
        >>> a
        array([[[0, 1],
                [2, 3]],
               [[4, 5],
                [6, 7]]])
        >>> a.diagonal(0,-2,-1)
        array([[0, 3],
               [4, 7]])

    """
    return asarray(a).diagonal(offset, axis1, axis2)


def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    """trace(a,offset=0, axis1=0, axis2=1) returns the sum along diagonals
    (defined by the last two dimenions) of the array.
    """
    return asarray(a).trace(offset, axis1, axis2, dtype, out)

def ravel(m,order='C'):
    """ravel(m) returns a 1d array corresponding to all the elements of it's
    argument.  The new array is a view of m if possible, otherwise it is
    a copy.
    """
    a = asarray(m)
    return a.ravel(order)

def nonzero(a):
    """nonzero(a) returns the indices of the elements of a which are not zero
    """
    try:
        nonzero = a.nonzero
    except AttributeError:
        res = _wrapit(a, 'nonzero')
    else:
        res = nonzero()
    return res

def shape(a):
    """shape(a) returns the shape of a (as a function call which
       also works on nested sequences).
    """
    try:
        result = a.shape
    except AttributeError:
        result = asarray(a).shape
    return result

def compress(condition, m, axis=None, out=None):
    """compress(condition, x, axis=None) = those elements of x corresponding
    to those elements of condition that are "true".  condition must be the
    same size as the given dimension of x."""
    try:
        compress = m.compress
    except AttributeError:
        return _wrapit(m, 'compress', condition, axis, out)
    return compress(condition, axis, out)

def clip(m, m_min, m_max):
    """clip(m, m_min, m_max) = every entry in m that is less than m_min is
    replaced by m_min, and every entry greater than m_max is replaced by
    m_max.
    """
    try:
        clip = m.clip
    except AttributeError:
        return _wrapit(m, 'clip', m_min, m_max)
    return clip(m_min, m_max)

def sum(x, axis=None, dtype=None, out=None):
    """Sum the array over the given axis.  The optional dtype argument
    is the data type for intermediate calculations.

    The default is to upcast (promote) smaller integer types to the
    platform-dependent Int.  For example, on 32-bit platforms:

        x.dtype                         default sum() dtype
        ---------------------------------------------------
        bool, int8, int16, int32        int32

    Examples:
    >>> N.sum([0.5, 1.5])
    2.0
    >>> N.sum([0.5, 1.5], dtype=N.int32)
    1
    >>> N.sum([[0, 1], [0, 5]])
    6
    >>> N.sum([[0, 1], [0, 5]], axis=1)
    array([1, 5])
    """
    if isinstance(x, _gentype):
        res = _sum_(x)
        if out is not None:
            out[...] = res
            return out
        return res
    try:
        sum = x.sum
    except AttributeError:
        return _wrapit(x, 'sum', axis, dtype, out)
    return sum(axis, dtype, out)

def product (x, axis=None, dtype=None, out=None):
    """Product of the array elements over the given axis."""
    try:
        prod = x.prod
    except AttributeError:
        return _wrapit(x, 'prod', axis, dtype, out)
    return prod(axis, dtype, out)

def sometrue (x, axis=None, out=None):
    """Perform a logical_or over the given axis."""
    try:
        any = x.any
    except AttributeError:
        return _wrapit(x, 'any', axis, out)
    return any(axis, out)

def alltrue (x, axis=None, out=None):
    """Perform a logical_and over the given axis."""
    try:
        all = x.all
    except AttributeError:
        return _wrapit(x, 'all', axis, out)
    return all(axis, out)

def any(x,axis=None, out=None):
    """Return true if any elements of x are true:
    """
    try:
        any = x.any
    except AttributeError:
        return _wrapit(x, 'any', axis, out)
    return any(axis, out)

def all(x,axis=None, out=None):
    """Return true if all elements of x are true:
    """
    try:
        all = x.all
    except AttributeError:
        return _wrapit(x, 'all', axis, out)
    return all(axis, out)

def cumsum (x, axis=None, dtype=None, out=None):
    """Sum the array over the given axis."""
    try:
        cumsum = x.cumsum
    except AttributeError:
        return _wrapit(x, 'cumsum', axis, dtype, out)
    return cumsum(axis, dtype, out)

def cumproduct (x, axis=None, dtype=None, out=None):
    """Sum the array over the given axis."""
    try:
        cumprod = x.cumprod
    except AttributeError:
        return _wrapit(x, 'cumprod', axis, dtype, out)
    return cumprod(axis, dtype, out)

def ptp(a, axis=None, out=None):
    """Return maximum - minimum along the the given dimension
    """
    try:
        ptp = a.ptp
    except AttributeError:
        return _wrapit(a, 'ptp', axis, out)
    return ptp(axis, out)

def amax(a, axis=None, out=None):
    """Return the maximum of 'a' along dimension axis.
    """
    try:
        amax = a.max
    except AttributeError:
        return _wrapit(a, 'max', axis, out)
    return amax(axis, out)

def amin(a, axis=None, out=None):
    """Return the minimum of a along dimension axis.
    """
    try:
        amin = a.min
    except AttributeError:
        return _wrapit(a, 'min', axis, out)
    return amin(axis, out)

def alen(a):
    """Return the length of a Python object interpreted as an array
    of at least 1 dimension.
    """
    try:
        return len(a)
    except TypeError:
        return len(array(a,ndmin=1))

def prod(a, axis=None, dtype=None, out=None):
    """Return the product of the elements along the given axis
    """
    try:
        prod = a.prod
    except AttributeError:
        return _wrapit(a, 'prod', axis, dtype, out)
    return prod(axis, dtype, out)

def cumprod(a, axis=None, dtype=None, out=None):
    """Return the cumulative product of the elments along the given axis
    """
    try:
        cumprod = a.cumprod
    except AttributeError:
        return _wrapit(a, 'cumprod', axis, dtype, out)
    return cumprod(axis, dtype, out)

def ndim(a):
    try:
        return a.ndim
    except AttributeError:
        return asarray(a).ndim

def rank(a):
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

def round_(a, decimals=0, out=None):
    """Returns reference to result. Copies a and rounds to 'decimals' places.

    Keyword arguments:
        decimals -- number of decimal places to round to (default 0).
        out -- existing array to use for output (default copy of a).

    Returns:
        Reference to out, where None specifies a copy of the original array a.

    Round to the specified number of decimals. When 'decimals' is negative it
    specifies the number of positions to the left of the decimal point. The
    real and imaginary parts of complex numbers are rounded separately.
    Nothing is done if the array is not of float type and 'decimals' is greater
    than or equal to 0.

    The keyword 'out' may be used to specify a different array to hold the
    result rather than the default 'a'. If the type of the array specified by
    'out' differs from that of 'a', the result is cast to the new type,
    otherwise the original type is kept. Floats round to floats by default.

    Numpy rounds to even. Thus 1.5 and 2.5 round to 2.0, -0.5 and 0.5 round to
    0.0, etc. Results may also be surprising due to the inexact representation
    of decimal fractions in IEEE floating point and the errors introduced in
    scaling the numbers when 'decimals' is something other than 0.

    The function around is an alias for round_.

    """
    try:
        round = a.round
    except AttributeError:
        return _wrapit(a, 'round', decimals, out)
    return round(decimals, out)

around = round_

def mean(a, axis=None, dtype=None, out=None):
    """Compute the mean along the specified axis.

    *Description*

        Returns the average of the array elements.  The average is taken over
        the flattened array by default, otherwise over the specified axis.

    *Parameters*:

        axis : integer
            Axis along which the means are computed. The default is
            to compute the standard deviation of the flattened array.

        dtype : type
            Type to use in computing the means. For arrays of
            integer type the default is float32, for arrays of float types it
            is the same as the array type.

        out : ndarray
            Alternative output array in which to place the result. It must have
            the same shape as the expected output but the type will be cast if
            necessary.

    *Returns*:

        mean : The return type varies, see above.
            A new array holding the result is returned unless out is specified,
            in which case a reference to out is returned.

    *SeeAlso*:

        var
            Variance
        std
            Standard deviation

    *Notes*

        The mean is the sum of the elements along the axis divided by the
        number of elements.

    """
    try:
        mean = a.mean
    except AttributeError:
        return _wrapit(a, 'mean', axis, dtype, out)
    return mean(axis, dtype, out)


def std(a, axis=None, dtype=None, out=None):
    """Compute the standard deviation along the specified axis.

    *Description*

        Returns the standard deviation of the array elements, a measure of the
        spread of a distribution. The standard deviation is computed for the
        flattened array by default, otherwise over the specified axis.

    *Parameters*:

        axis : integer
            Axis along which the standard deviation is computed. The default is
            to compute the standard deviation of the flattened array.

        dtype : type
            Type to use in computing the standard deviation. For arrays of
            integer type the default is float32, for arrays of float types it
            is the same as the array type.

        out : ndarray
            Alternative output array in which to place the result. It must have
            the same shape as the expected output but the type will be cast if
            necessary.

    *Returns*:

        standard_deviation : The return type varies, see above.
            A new array holding the result is returned unless out is specified,
            in which case a reference to out is returned.

    *SeeAlso*:

        var
            Variance
        mean
            Average

    *Notes*

        The standard deviation is the square root of the average of the squared
        deviations from the mean, i.e. var = sqrt(mean((x - x.mean())**2)).
        The computed standard deviation is biased, i.e., the mean is computed
        by dividing by the number of elements, N, rather than by N-1.

    """
    try:
        std = a.std
    except AttributeError:
        return _wrapit(a, 'std', axis, dtype, out)
    return std(axis, dtype, out)


def var(a, axis=None, dtype=None, out=None):
    """Compute the variance along the specified axis.

    *Description*

        Returns the variance of the array elements, a measure of the spread of
        a distribution.  The variance is computed for the flattened array by
        default, otherwise over the specified axis.

    *Parameters*:

        axis : integer
            Axis along which the variance is computed. The default is to
            compute the variance of the flattened array.

        dtype : type
            Type to use in computing the variance. For arrays of integer type
            the default is float32, for arrays of float types it is the same as
            the array type.

        out : ndarray
            Alternative output array in which to place the result. It must have
            the same shape as the expected output but the type will be cast if
            necessary.

    *Returns*:

        variance : depends, see above
            A new array holding the result is returned unless out is specified,
            in which case a reference to out is returned.

    *SeeAlso*:

        std
            Standard deviation
        mean
            Average

    *Notes*

        The variance is the average of the squared deviations from the mean,
        i.e.  var = mean((x - x.mean())**2).  The computed variance is biased,
        i.e., the mean is computed by dividing by the number of elements, N,
        rather than by N-1.

    """
    try:
        var = a.var
    except AttributeError:
        return _wrapit(a, 'var', axis, dtype, out)
    return var(axis, dtype, out)
