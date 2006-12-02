# Module containing non-deprecated functions borrowed from Numeric.

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
from numeric import asarray, array, asanyarray, correlate, outer, concatenate
from umath import sign, absolute, multiply
import numeric as _nx
import sys
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
    try:
        take = a.take
    except AttributeError:
        return _wrapit(a, 'take', indices, axis, out, mode)
    return take(indices, axis, out, mode)

# not deprecated --- copy if necessary, view otherwise
def reshape(a, newshape, order='C'):
    """Change the shape of a to newshape.
    Return a new view object if possible otherwise return a copy.
    """
    try:
        reshape = a.reshape
    except AttributeError:
        return _wrapit(a, 'reshape', newshape, order=order)
    return reshape(newshape, order=order)

def choose(a, choices, out=None, mode='raise'):
    try:
        choose = a.choose
    except AttributeError:
        return _wrapit(a, 'choose', choices, out=out, mode=mode)
    return choose(choices, out=out, mode=mode)

def repeat(a, repeats, axis=None):
    """repeat elements of a repeats times along axis
       repeats is a sequence of length a.shape[axis]
       telling how many times to repeat each element.
       If repeats is an integer, it is interpreted as
       a tuple of length a.shape[axis] containing repeats.
       The argument a can be anything array(a) will accept.
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
    """Returns copy of 'a' sorted along the given axis.

    Keyword arguments:

    axis  -- axis to be sorted (default -1).  Can be None
             to indicate that a flattened and sorted array should
             be returned (the array method does not support this). 
    kind  -- sorting algorithm (default 'quicksort')
             Possible values: 'quicksort', 'mergesort', or 'heapsort'.
    order -- For an array with fields defined, this argument allows
             specification of which fields to compare first, second,
             etc.  Not all fields need be specified.
            

    Returns: None.

    This method sorts 'a' in place along the given axis using the algorithm
    specified by the kind keyword.

    The various sorts may characterized by average speed, worst case
    performance, need for work space, and whether they are stable. A stable
    sort keeps items with the same key in the same relative order and is most
    useful when used with argsort where the key might differ from the items
    being sorted. The three available algorithms have the following properties:

    |------------------------------------------------------|
    |    kind   | speed |  worst case | work space | stable|
    |------------------------------------------------------|
    |'quicksort'|   1   | O(n^2)      |     0      |   no  |
    |'mergesort'|   2   | O(n*log(n)) |    ~n/2    |   yes |
    |'heapsort' |   3   | O(n*log(n)) |     0      |   no  |
    |------------------------------------------------------|

    All the sort algorithms make temporary copies of the data when the sort is
    not along the last axis. Consequently, sorts along the last axis are faster
    and use less space than sorts along other axis.

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

    Keyword arguments:

    axis  -- axis to be indirectly sorted (default -1)
             Can be None to indicate return indices into the
             flattened array. 
    kind  -- sorting algorithm (default 'quicksort')
             Possible values: 'quicksort', 'mergesort', or 'heapsort'
    order -- For an array with fields defined, this argument allows
             specification of which fields to compare first, second,
             etc.  Not all fields need be specified.

    Returns: array of indices that sort 'a' along the specified axis.

    This method executes an indirect sort along the given axis using the
    algorithm specified by the kind keyword. It returns an array of indices of
    the same shape as 'a' that index data along the given axis in sorted order.

    The various sorts are characterized by average speed, worst case
    performance, need for work space, and whether they are stable. A stable
    sort keeps items with the same key in the same relative order. The three
    available algorithms have the following properties:

    |------------------------------------------------------|
    |    kind   | speed |  worst case | work space | stable|
    |------------------------------------------------------|
    |'quicksort'|   1   | O(n^2)      |     0      |   no  |
    |'mergesort'|   2   | O(n*log(n)) |    ~n/2    |   yes |
    |'heapsort' |   3   | O(n*log(n)) |     0      |   no  |
    |------------------------------------------------------|

    All the sort algorithms make temporary copies of the data when the sort is not
    along the last axis. Consequently, sorts along the last axis are faster and use
    less space than sorts along other axis.

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
    """-> index array. Inserting v[i] before a[index[i]] maintains a in order.

    Required arguments:
        a -- sorted 1-D array to be searched.
        v -- array of keys to be searched for in a.

    Keyword arguments:
        side -- {'left', 'right'}, default('left').

    Returns:
        array of indices with the same shape as a.

    The array to be searched must be 1-D and is assumed to be sorted in
    ascending order.

    The function call

        searchsorted(a, v, side='left')

    returns an index array with the same shape as v such that for each value i
    in the index and the corresponding key in v the following holds:

           a[j] < key <= a[i] for all j < i,

    If such an index does not exist, a.size() is used. Consequently, i is the
    index of the first item in 'a' that is >= key. If the key were to be
    inserted into a in the slot before the index i, then the order of a would
    be preserved and i would be the smallest index with that property.

    The function call

        searchsorted(a, v, side='right')

    returns an index array with the same shape as v such that for each value i
    in the index and the corresponding key in v the following holds:

           a[j] <= key < a[i] for all j < i,

    If such an index does not exist, a.size() is used. Consequently, i is the
    index of the first item in 'a' that is > key. If the key were to be
    inserted into a in the slot before the index i, then the order of a would
    be preserved and i would be the largest index with that property.

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
    """diagonal(a, offset=0, axis1=0, axis2=1) returns the given diagonals
    defined by the last two dimensions of the array.
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
    """mean(a, axis=None, dtype=None)
    Return the arithmetic mean.

    The mean is the sum of the elements divided by the number of elements.

    See also: average
    """
    try:
        mean = a.mean
    except AttributeError:
        return _wrapit(a, 'mean', axis, dtype, out)
    return mean(axis, dtype, out)

def std(a, axis=None, dtype=None, out=None):
    """std(sample, axis=None, dtype=None)
    Return the standard deviation, a measure of the spread of a distribution.

    The standard deviation is the square root of the average of the squared
    deviations from the mean, i.e. std = sqrt(mean((x - x.mean())**2)).

    See also: var
    """
    try:
        std = a.std
    except AttributeError:
        return _wrapit(a, 'std', axis, dtype, out)
    return std(axis, dtype, out)

def var(a, axis=None, dtype=None, out=None):
    """var(sample, axis=None, dtype=None)
    Return the variance, a measure of the spread of a distribution.

    The variance is the average of the squared deviations from the mean,
    i.e. var = mean((x - x.mean())**2).

    See also: std
    """
    try:
        var = a.var
    except AttributeError:
        return _wrapit(a, 'var', axis, dtype, out)
    return var(axis, dtype, out)
