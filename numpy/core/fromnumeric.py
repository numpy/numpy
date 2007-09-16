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
    """Return an array formed from the elements of a at the given indices.

    This function does the same thing as "fancy" indexing; however, it can
    be easier to use if you need to specify a given axis.

    *Parameters*:

        a : array
            The source array
        indices : int array
            The indices of the values to extract.
        axis : {None, int}, optional
            The axis over which to select values. None signifies that the
            operation should be performed over the flattened array.
        out : {None, array}, optional
            If provided, the result will be inserted into this array. It should
            be of the appropriate shape and dtype.
        mode : {'raise', 'wrap', 'clip'}, optional
            Specifies how out-of-bounds indices will behave.
            'raise' -- raise an error
            'wrap' -- wrap around
            'clip' -- clip to the range

    *Returns*:

        subarray : array
            The returned array has the same type as a.

    *See Also*:

       `ndarray.take` : equivalent method

    """
    try:
        take = a.take
    except AttributeError:
        return _wrapit(a, 'take', indices, axis, out, mode)
    return take(indices, axis, out, mode)


# not deprecated --- copy if necessary, view otherwise
def reshape(a, newshape, order='C'):
    """Returns an array containing the data of a, but with a new shape.

    *Parameters*:

        a : array
            Array to be reshaped.
        newshape : shape tuple or int
           The new shape should be compatible with the original shape. If an
           integer, then the result will be a 1D array of that length.
        order : {'C', 'FORTRAN'}, optional
            Determines whether the array data should be viewed as in C
            (row-major) order or FORTRAN (column-major) order.

    *Returns*:

        reshaped_array : array
            This will be a new view object if possible; otherwise, it will
            return a copy.

    *See Also*:

        `ndarray.reshape` : Equivalent method.

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

    *Parameters*:

        a : int array
            This array must contain integers in [0, n-1], where n is the number
            of choices.
        choices : sequence of arrays
            Each of the choice arrays should have the same shape as the index
            array.
        out : array, optional
            If provided, the result will be inserted into this array. It should
            be of the appropriate shape and dtype
        mode : {'raise', 'wrap', 'clip'}, optional
            Specifies how out-of-bounds indices will behave.
            'raise' : raise an error
            'wrap' : wrap around
            'clip' : clip to the range

    *Returns*:

        merged_array : array

    *See Also*:

        `ndarray.choose` : equivalent method

    *Examples*

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

    *Parameters*:

        a : {array_like}
            Blah.
        repeats : {integer, integer_array}
            The number of repetitions for each element. If a plain integer, then
            it is applied to all elements. If an array, it needs to be of the
            same length as the chosen axis.
        axis : {None, integer}, optional
            The axis along which to repeat values. If None, then this function
            will operated on the flattened array `a` and return a similarly flat
            result.

    *Returns*:

        repeated_array : array

    *See Also*:

        `ndarray.repeat` : equivalent method

    *Examples*

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
    """Set a[n] = v[n] for all n in ind.

    If v is shorter than mask it will be repeated as necessary.  In particular v
    can be a scalar or length 1 array.  The routine put is the equivalent of the
    following (although the loop is in C for speed):

        ind = array(indices, copy=False)
        v = array(values, copy=False).astype(a.dtype)
        for i in ind: a.flat[i] = v[i]

    a must be a contiguous numpy array.

    """
    return a.put(ind, v, mode)


def swapaxes(a, axis1, axis2):
    """Return array a with axis1 and axis2 interchanged.

    Blah, Blah.

    """
    try:
        swapaxes = a.swapaxes
    except AttributeError:
        return _wrapit(a, 'swapaxes', axis1, axis2)
    return swapaxes(axis1, axis2)


def transpose(a, axes=None):
    """Return a view of the array with dimensions permuted.

    Permutes axis according to list axes.  If axes is None (default) returns
    array with dimensions reversed.

    """
    try:
        transpose = a.transpose
    except AttributeError:
        return _wrapit(a, 'transpose', axes)
    return transpose(axes)


def sort(a, axis=-1, kind='quicksort', order=None):
    """Return copy of 'a' sorted along the given axis.

    Perform an inplace sort along the given axis using the algorithm
    specified by the kind keyword.

    *Parameters*:

        a : array
            Array to be sorted.
        axis : {None, int} optional
            Axis along which to sort. None indicates that the flattened
            array should be used.
        kind : {'quicksort', 'mergesort', 'heapsort'}, optional
            Sorting algorithm to use.
        order : {None, list type}, optional
            When a is an array with fields defined, this argument specifies
            which fields to compare first, second, etc.  Not all fields need be
            specified.

    *Returns*:

        sorted_array : array of same type as a

    *See Also*:

        `argsort` : Indirect sort.

        `lexsort` : Indirect stable sort on multiple keys.

        `searchsorted` : Find keys in sorted array.

    *Notes*

        The various sorts are characterized by average speed, worst case
        performance, need for work space, and whether they are stable. A
        stable sort keeps items with the same key in the same relative
        order. The three available algorithms have the following
        properties:

        +-----------+-------+-------------+------------+-------+
        |    kind   | speed |  worst case | work space | stable|
        +===========+=======+=============+============+=======+
        | quicksort |   1   | O(n^2)      |     0      |   no  |
        +-----------+-------+-------------+------------+-------+
        | mergesort |   2   | O(n*log(n)) |    ~n/2    |   yes |
        +-----------+-------+-------------+------------+-------+
        | heapsort  |   3   | O(n*log(n)) |     0      |   no  |
        +-----------+-------+-------------+------------+-------+

        All the sort algorithms make temporary copies of the data when
        the sort is not along the last axis. Consequently, sorts along
        the last axis are faster and use less space than sorts along
        other axis.

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

    Perform an indirect sort along the given axis using the algorithm specified
    by the kind keyword. It returns an array of indices of the same shape as a
    that index data along the given axis in sorted order.

    *Parameters*:

        a : array
            Array to be sorted.
        axis : {None, int} optional
            Axis along which to sort. None indicates that the flattened
            array should be used.
        kind : {'quicksort', 'mergesort', 'heapsort'}, optional
            Sorting algorithm to use.
        order : {None, list type}, optional
            When a is an array with fields defined, this argument specifies
            which fields to compare first, second, etc.  Not all fields need be
            specified.

    *Returns*:

        index_array : {integer_array}
            Array of indices that sort 'a' along the specified axis.

    *See Also*:

        `lexsort` : Indirect stable sort with multiple keys.

        `sort` : Inplace sort.

    *Notes*

        The various sorts are characterized by average speed, worst case
        performance, need for work space, and whether they are stable. A
        stable sort keeps items with the same key in the same relative
        order. The three available algorithms have the following
        properties:

        +-----------+-------+-------------+------------+-------+
        |    kind   | speed |  worst case | work space | stable|
        +===========+=======+=============+============+=======+
        | quicksort |   1   | O(n^2)      |     0      |   no  |
        +-----------+-------+-------------+------------+-------+
        | mergesort |   2   | O(n*log(n)) |    ~n/2    |   yes |
        +-----------+-------+-------------+------------+-------+
        | heapsort  |   3   | O(n*log(n)) |     0      |   no  |
        +-----------+-------+-------------+------------+-------+

        All the sort algorithms make temporary copies of the data when
        the sort is not along the last axis. Consequently, sorts along
        the last axis are faster and use less space than sorts along
        other axis.

    """
    try:
        argsort = a.argsort
    except AttributeError:
        return _wrapit(a, 'argsort', axis, kind, order)
    return argsort(axis, kind, order)


def argmax(a, axis=None):
    """Returns array of indices of the maximum values of along the given axis.

    *Parameters*:

        a : {array_like}
            Array to look in.
        axis : {None, integer}
            If None, the index is into the flattened array, otherwise along
            the specified axis

    *Returns*:

        index_array : {integer_array}

    *Examples*

        >>> a = arange(6).reshape(2,3)
        >>> argmax(a)
        5
        >>> argmax(a,0)
        array([1, 1, 1])
        >>> argmax(a,1)
        array([2, 2])

    """
    try:
        argmax = a.argmax
    except AttributeError:
        return _wrapit(a, 'argmax', axis)
    return argmax(axis)


def argmin(a, axis=None):
    """Return array of indices to the minimum values along the given axis.

    *Parameters*:

        a : {array_like}
            Array to look in.
        axis : {None, integer}
            If None, the index is into the flattened array, otherwise along
            the specified axis

    *Returns*:

        index_array : {integer_array}

    *Examples*

        >>> a = arange(6).reshape(2,3)
        >>> argmin(a)
        0
        >>> argmin(a,0)
        array([0, 0, 0])
        >>> argmin(a,1)
        array([0, 0])

    """
    try:
        argmin = a.argmin
    except AttributeError:
        return _wrapit(a, 'argmin', axis)
    return argmin(axis)


def searchsorted(a, v, side='left'):
    """Return indices where keys in v should be inserted to maintain order.

    Find the indices into a sorted array such that if the corresponding keys in
    v were inserted before the indices the order of a would be preserved.  If
    side='left', then the first such index is returned. If side='right', then
    the last such index is returned. If there is no such index because the key
    is out of bounds, then the length of a is returned, i.e., the key would need
    to be appended. The returned index array has the same shape as v.

    *Parameters*:

        a : 1-d array
            Array must be sorted in ascending order.
        v : array or list type
            Array of keys to be searched for in a.
        side : {'left', 'right'}, optional
            If 'left', the index of the first location where the key could be
            inserted is found, if 'right', the index of the last such element is
            returned. If the is no such element, then either 0 or N is returned,
            where N is the size of the array.

    *Returns*:

        indices : integer array
            Array of insertion points with the same shape as v.

    *See Also*:

        `sort` : Inplace sort.

        `histogram` : Produce histogram from 1-d data.

    *Notes*

        The array a must be 1-d and is assumed to be sorted in ascending
        order.  Searchsorted uses binary search to find the required
        insertion points.

    *Examples*

        >>> searchsorted([1,2,3,4,5],[6,4,0])
        array([5, 3, 0])

    """
    try:
        searchsorted = a.searchsorted
    except AttributeError:
        return _wrapit(a, 'searchsorted', v, side)
    return searchsorted(v, side)


def resize(a, new_shape):
    """Return a new array with the specified shape.

    The original array's total size can be any size.  The new array is
    filled with repeated copies of a.

    Note that a.resize(new_shape) will fill the array with 0's beyond
    current definition of a.

    *Parameters*:

        a : {array_like}
            Array to be reshaped.

        new_shape : {tuple}
            Shape of reshaped array.

    *Returns*:

        reshaped_array : {array}
            The new array is formed from the data in the old array, repeated if
            necessary to fill out the required number of elements, with the new
            shape.

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
    """Remove single-dimensional entries from the shape of a.

    *Examples*

        >>> x = array([[[1,1,1],[2,2,2],[3,3,3]]])
        >>> x
        array([[[1, 1, 1],
              [2, 2, 2],
              [3, 3, 3]]])
        >>> x.shape
        (1, 3, 3)
        >>> squeeze(x).shape
        (3, 3)

    """
    try:
        squeeze = a.squeeze
    except AttributeError:
        return _wrapit(a, 'squeeze')
    return squeeze()


def diagonal(a, offset=0, axis1=0, axis2=1):
    """Return specified diagonals.

    If a is 2-d, returns the diagonal of self with the given offset, i.e., the
    collection of elements of the form a[i,i+offset]. If a has more than two
    dimensions, then the axes specified by axis1 and axis2 are used to determine
    the 2-d subarray whose diagonal is returned. The shape of the resulting
    array can be determined by removing axis1 and axis2 and appending an index
    to the right equal to the size of the resulting diagonals.

    *Parameters*:

        a : {array_like}
            Array from whis the diagonals are taken.
        offset : {0, integer}, optional
            Offset of the diagonal from the main diagonal. Can be both positive
            and negative. Defaults to main diagonal.
        axis1 : {0, integer}, optional
            Axis to be used as the first axis of the 2-d subarrays from which
            the diagonals should be taken. Defaults to first axis.
        axis2 : {1, integer}, optional
            Axis to be used as the second axis of the 2-d subarrays from which
            the diagonals should be taken. Defaults to second axis.

    *Returns*:

        array_of_diagonals : array of same type as a
            If a is 2-d, a 1-d array containing the diagonal is
            returned.  If a has larger dimensions, then an array of
            diagonals is returned.

    *See Also*:

        `diag` : Matlab workalike for 1-d and 2-d arrays.

        `diagflat` : Create diagonal arrays.

        `trace` : Sum along diagonals.

    *Examples*

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
    """Return the sum along diagonals of the array.

    If a is 2-d, returns the sum along the diagonal of self with the given offset, i.e., the
    collection of elements of the form a[i,i+offset]. If a has more than two
    dimensions, then the axes specified by axis1 and axis2 are used to determine
    the 2-d subarray whose trace is returned. The shape of the resulting
    array can be determined by removing axis1 and axis2 and appending an index
    to the right equal to the size of the resulting diagonals. Arrays of integer
    type are summed

    *Parameters*:

        a : {array_like}
            Array from whis the diagonals are taken.
        offset : {0, integer}, optional
            Offset of the diagonal from the main diagonal. Can be both positive
            and negative. Defaults to main diagonal.
        axis1 : {0, integer}, optional
            Axis to be used as the first axis of the 2-d subarrays from which
            the diagonals should be taken. Defaults to first axis.
        axis2 : {1, integer}, optional
            Axis to be used as the second axis of the 2-d subarrays from which
            the diagonals should be taken. Defaults to second axis.
        dtype : {None, dtype}, optional
            Determines the type of the returned array and of the accumulator
            where the elements are summed. If dtype has the value None and a is
            of integer type of precision less than the default integer
            precision, then the default integer precision is used. Otherwise,
            the precision is the same as that of a.
        out : {None, array}, optional
            Array into which the sum can be placed. It's type is preserved and
            it must be of the right shape to hold the output.

    *Returns*:

        sum_along_diagonals : array
            If a is 2-d, a 0-d array containing the diagonal is
            returned.  If a has larger dimensions, then an array of
            diagonals is returned.

    *Examples*

        >>> trace(eye(3))
        3.0
        >>> a = arange(8).reshape((2,2,2))
        >>> trace(a)
        array([6, 8])

    """
    return asarray(a).trace(offset, axis1, axis2, dtype, out)

def ravel(a, order='C'):
    """Return a 1d array containing the elements of a.

    Returns the elements of a as a 1d array. The elements in the new array
    are taken in the order specified by the order keyword. The new array is
    a view of a if possible, otherwise it is a copy.

    *Parameters*:

        a : {array_like}

        order : {'C','F'}, optional
            If order is 'C' the elements are taken in row major order. If order
            is 'F' they are taken in column major order.

    *Returns*:

        1d_array : {array}

    *See Also*:

        `ndarray.flat` : 1d iterator over the array.

        `ndarray.flatten` : 1d array copy of the elements of a in C order.

    *Examples*

        >>> x = array([[1,2,3],[4,5,6]])
        >>> x
        array([[1, 2, 3],
              [4, 5, 6]])
        >>> ravel(x)
        array([1, 2, 3, 4, 5, 6])

    """
    return asarray(a).ravel(order)


def nonzero(a):
    """Return the indices of the elements of a which are not zero.

    *Parameters*:

        a : {array_like}

    *Returns*:

        tuple_of_arrays : {tuple}

    *Examples*

        >>> eye(3)[nonzero(eye(3))]
        array([ 1.,  1.,  1.])
        >>> nonzero(eye(3))
        (array([0, 1, 2]), array([0, 1, 2]))
        >>> eye(3)[nonzero(eye(3))]
        array([ 1.,  1.,  1.])

    """
    try:
        nonzero = a.nonzero
    except AttributeError:
        res = _wrapit(a, 'nonzero')
    else:
        res = nonzero()
    return res


def shape(a):
    """Return the shape of a.

    *Parameters*:

        a : {array_like}
            Array whose shape is desired. If a is not an array, a conversion is
            attempted.

    *Returns*:

        tuple_of_integers :
            The elements of the tuple are the length of the corresponding array
            dimension.

    *Examples*

        >>> shape(eye(3))
        (3, 3)
        >>> shape([[1,2]])
        (1, 2)

    """
    try:
        result = a.shape
    except AttributeError:
        result = asarray(a).shape
    return result


def compress(condition, a, axis=None, out=None):
    """Return a where condition is true.

    Equivalent to a[condition].

    """
    try:
        compress = a.compress
    except AttributeError:
        return _wrapit(a, 'compress', condition, axis, out)
    return compress(condition, axis, out)


def clip(a, a_min, a_max):
    """Limit the values of a to [a_min, a_max].  Equivalent to

    a[a < a_min] = a_min
    a[a > a_max] = a_max

    """
    try:
        clip = a.clip
    except AttributeError:
        return _wrapit(a, 'clip', a_min, a_max)
    return clip(a_min, a_max)


def sum(a, axis=None, dtype=None, out=None):
    """Sum the array over the given axis.

    *Parameters*:

        a : {array_type}
            Array containing elements whose sum is desired. If a is not an array, a
            conversion is attempted.
        axis : {None, integer}
            Axis over which the sum is taken. If None is used, then the sum is
            over all the array elements.
        dtype : {None, dtype}, optional
            Determines the type of the returned array and of the accumulator
            where the elements are summed. If dtype has the value None and the
            type of a is an integer type of precision less than the default
            platform integer, then the default platform integer precision is
            used.  Otherwise, the dtype is the same as that of a.
        out : {None, array}, optional
            Array into which the sum can be placed. It's type is preserved and
            it must be of the right shape to hold the output.

    *Returns*:

        sum_along_axis : {array, scalar}, see dtype parameter above.
            Returns an array whose shape is the same as a with the specified
            axis removed. Returns a 0d array when a is 1d or dtype=None.
            Returns a reference to the specified output array if specified.

    *See Also*:

        `ndarray.sum` : equivalent method

    *Examples*

        >>> sum([0.5, 1.5])
        2.0
        >>> sum([0.5, 1.5], dtype=N.int32)
        1
        >>> sum([[0, 1], [0, 5]])
        6
        >>> sum([[0, 1], [0, 5]], axis=1)
        array([1, 5])

    """
    if isinstance(a, _gentype):
        res = _sum_(a)
        if out is not None:
            out[...] = res
            return out
        return res
    try:
        sum = a.sum
    except AttributeError:
        return _wrapit(a, 'sum', axis, dtype, out)
    return sum(axis, dtype, out)


def product (a, axis=None, dtype=None, out=None):
    """Product of the array elements over the given axis.

    *Parameters*:

        a : {array_like}
            Array containing elements whose product is desired. If a is not an array, a
            conversion is attempted.
        axis : {None, integer}
            Axis over which the product is taken. If None is used, then the
            product is over all the array elements.
        dtype : {None, dtype}, optional
            Determines the type of the returned array and of the accumulator
            where the elements are multiplied. If dtype has the value None and
            the type of a is an integer type of precision less than the default
            platform integer, then the default platform integer precision is
            used.  Otherwise, the dtype is the same as that of a.
        out : {None, array}, optional
            Alternative output array in which to place the result. It must have
            the same shape as the expected output but the type will be cast if
            necessary.

    *Returns*:

        product_along_axis : {array, scalar}, see dtype parameter above.
            Returns an array whose shape is the same as a with the specified
            axis removed. Returns a 0d array when a is 1d or dtype=None.
            Returns a reference to the specified output array if specified.

    *See Also*:

        `ndarray.prod` : equivalent method

    *Examples*

        >>> product([1.,2.])
        2.0
        >>> product([1.,2.], dtype=int32)
        2
        >>> product([[1.,2.],[3.,4.]])
        24.0
        >>> product([[1.,2.],[3.,4.]], axis=1)
        array([  2.,  12.])


    """
    try:
        prod = a.prod
    except AttributeError:
        return _wrapit(a, 'prod', axis, dtype, out)
    return prod(axis, dtype, out)


def sometrue (a, axis=None, out=None):
    """Perform a logical_or over the given axis.

    *See Also*:

        `ndarray.any` : equivalent method

    """
    try:
        any = a.any
    except AttributeError:
        return _wrapit(a, 'any', axis, out)
    return any(axis, out)


def alltrue (a, axis=None, out=None):
    """Perform a logical_and over the given axis.

    *See Also*:

        `ndarray.all` : equivalent method

        `all` : equivalent function

    """
    try:
        all = a.all
    except AttributeError:
        return _wrapit(a, 'all', axis, out)
    return all(axis, out)


def any(a,axis=None, out=None):
    """Return true if any elements of x are true.

    *See Also*:

        `ndarray.any` : equivalent method

    """
    try:
        any = a.any
    except AttributeError:
        return _wrapit(a, 'any', axis, out)
    return any(axis, out)


def all(a,axis=None, out=None):
    """Return true if all elements of x are true:

    *See Also*:

        `ndarray.all` : equivalent method

        `alltrue` : equivalent function

    """
    try:
        all = a.all
    except AttributeError:
        return _wrapit(a, 'all', axis, out)
    return all(axis, out)


def cumsum (a, axis=None, dtype=None, out=None):
    """Sum the array over the given axis.

    Blah, Blah.

    """
    try:
        cumsum = a.cumsum
    except AttributeError:
        return _wrapit(a, 'cumsum', axis, dtype, out)
    return cumsum(axis, dtype, out)


def cumproduct (a, axis=None, dtype=None, out=None):
    """Return the cumulative product over the given axis.

    Blah, Blah.

    """
    try:
        cumprod = a.cumprod
    except AttributeError:
        return _wrapit(a, 'cumprod', axis, dtype, out)
    return cumprod(axis, dtype, out)


def ptp(a, axis=None, out=None):
    """Return maximum - minimum along the the given dimension.

    Blah, Blah.

    """
    try:
        ptp = a.ptp
    except AttributeError:
        return _wrapit(a, 'ptp', axis, out)
    return ptp(axis, out)


def amax(a, axis=None, out=None):
    """Return the maximum of 'a' along dimension axis.

    Blah, Blah.

    """
    try:
        amax = a.max
    except AttributeError:
        return _wrapit(a, 'max', axis, out)
    return amax(axis, out)


def amin(a, axis=None, out=None):
    """Return the minimum of a along dimension axis.

    Blah, Blah.

    """
    try:
        amin = a.min
    except AttributeError:
        return _wrapit(a, 'min', axis, out)
    return amin(axis, out)


def alen(a):
    """Return the length of a Python object interpreted as an array
    of at least 1 dimension.

    Blah, Blah.

    """
    try:
        return len(a)
    except TypeError:
        return len(array(a,ndmin=1))


def prod(a, axis=None, dtype=None, out=None):
    """Return the product of the elements along the given axis.

    Blah, Blah.

    """
    try:
        prod = a.prod
    except AttributeError:
        return _wrapit(a, 'prod', axis, dtype, out)
    return prod(axis, dtype, out)


def cumprod(a, axis=None, dtype=None, out=None):
    """Return the cumulative product of the elements along the given axis.

    Blah, Blah.

    """
    try:
        cumprod = a.cumprod
    except AttributeError:
        return _wrapit(a, 'cumprod', axis, dtype, out)
    return cumprod(axis, dtype, out)


def ndim(a):
    """Return the number of dimensions of a.

    If a is not already an array, a conversion is attempted. Scalars are zero
    dimensional.

    *Parameters*:

        a : {array_like}
            Array whose number of dimensions are desired. If a is not an array, a
            conversion is attempted.

    *Returns*:

        number_of_dimensions : {integer}
            Returns the number of dimensions.

    *See Also*:

        `rank` : equivalent function.

        `ndarray.ndim` : equivalent method

        `shape` : dimensions of array

        `ndarray.shape` : dimensions of array

    *Examples*

        >>> ndim([[1,2,3],[4,5,6]])
        2
        >>> ndim(array([[1,2,3],[4,5,6]]))
        2
        >>> ndim(1)
        0

    """
    try:
        return a.ndim
    except AttributeError:
        return asarray(a).ndim


def rank(a):
    """Return the number of dimensions of a.

    In old Numeric, rank was the term used for the number of dimensions. If a is
    not already an array, a conversion is attempted. Scalars are zero
    dimensional.

    *Parameters*:

        a : {array_like}
            Array whose number of dimensions is desired. If a is not an array, a
            conversion is attempted.

    *Returns*:

        number_of_dimensions : {integer}
            Returns the number of dimensions.

    *See Also*:

        `ndim` : equivalent function

        `ndarray.ndim` : equivalent method

        `shape` : dimensions of array

        `ndarray.shape` : dimensions of array

    *Examples*

        >>> rank([[1,2,3],[4,5,6]])
        2
        >>> rank(array([[1,2,3],[4,5,6]]))
        2
        >>> rank(1)
        0

    """
    try:
        return a.ndim
    except AttributeError:
        return asarray(a).ndim


def size(a, axis=None):
    """Return the number of elements along given axis.

    *Parameters*:

        a : {array_like}
            Array whose axis size is desired. If a is not an array, a conversion
            is attempted.
        axis : {None, integer}, optional
            Axis along which the elements are counted. None means all elements
            in the array.

    *Returns*:

        element_count : {integer}
            Count of elements along specified axis.

    *See Also*:

        `shape` : dimensions of array

        `ndarray.shape` : dimensions of array

        `ndarray.size` : number of elements in array

    *Examples*

        >>> a = array([[1,2,3],[4,5,6]])
        >>> size(a)
        6
        >>> size(a,1)
        3
        >>> size(a,0)
        2

    """
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


def around(a, decimals=0, out=None):
    """Round a to the given number of decimals.

    The real and imaginary parts of complex numbers are rounded separately. The
    result of rounding a float is a float so the type must be cast if integers
    are desired.  Nothing is done if the input is an integer array and the
    decimals parameter has a value >= 0.

    *Parameters*:

        a : {array_like}
            Array containing numbers whose rounded values are desired. If a is
            not an array, a conversion is attempted.
        decimals : {0, int}, optional
            Number of decimal places to round to. When decimals is negative it
            specifies the number of positions to the left of the decimal point.
        out : {None, array}, optional
            Alternative output array in which to place the result. It must have
            the same shape as the expected output but the type will be cast if
            necessary. Numpy rounds floats to floats by default.

        *Returns*:

        rounded_array : {array}
            If out=None, returns a new array of the same type as a containing
            the rounded values, otherwise a reference to the output array is
            returned.

    *See Also*:

        `round_` : equivalent function

        `ndarray.round` : equivalent method

    *Notes*

        Numpy rounds to even. Thus 1.5 and 2.5 round to 2.0, -0.5 and 0.5 round
        to 0.0, etc. Results may also be surprising due to the inexact
        representation of decimal fractions in IEEE floating point and the
        errors introduced when scaling by powers of ten.

    *Examples*

        >>> around([.5, 1.5, 2.5, 3.5, 4.5])
        array([ 0.,  2.,  2.,  4.,  4.])
        >>> around([1,2,3,11], decimals=1)
        array([ 1,  2,  3, 11])
        >>> around([1,2,3,11], decimals=-1)
        array([ 0,  0,  0, 10])

    """
    try:
        round = a.round
    except AttributeError:
        return _wrapit(a, 'round', decimals, out)
    return round(decimals, out)


def round_(a, decimals=0, out=None):
    """Round a to the given number of decimals.

    The real and imaginary parts of complex numbers are rounded separately. The
    result of rounding a float is a float so the type must be cast if integers
    are desired.  Nothing is done if the input is an integer array and the
    decimals parameter has a value >= 0.

    *Parameters*:

        a : {array_like}
            Array containing numbers whose rounded values are desired. If a is
            not an array, a conversion is attempted.
        decimals : {0, integer}, optional
            Number of decimal places to round to. When decimals is negative it
            specifies the number of positions to the left of the decimal point.
        out : {None, array}, optional
            Alternative output array in which to place the result. It must have
            the same shape as the expected output but the type will be cast if
            necessary.

    *Returns*:

        rounded_array : {array}
            If out=None, returns a new array of the same type as a containing
            the rounded values, otherwise a reference to the output array is
            returned.

    *See Also*:

        `around` : equivalent function

        `ndarray.round` : equivalent method

    *Notes*

        Numpy rounds to even. Thus 1.5 and 2.5 round to 2.0, -0.5 and 0.5 round
        to 0.0, etc. Results may also be surprising due to the inexact
        representation of decimal fractions in IEEE floating point and the
        errors introduced when scaling by powers of ten.

    *Examples*

        >>> round_([.5, 1.5, 2.5, 3.5, 4.5])
        array([ 0.,  2.,  2.,  4.,  4.])
        >>> round_([1,2,3,11], decimals=1)
        array([ 1,  2,  3, 11])
        >>> round_([1,2,3,11], decimals=-1)
        array([ 0,  0,  0, 10])

    """
    try:
        round = a.round
    except AttributeError:
        return _wrapit(a, 'round', decimals, out)
    return round(decimals, out)


def mean(a, axis=None, dtype=None, out=None):
    """Compute the mean along the specified axis.

    Returns the average of the array elements.  The average is taken
    over the flattened array by default, otherwise over the specified
    axis. The dtype returned for integer type arrays is float

    *Parameters*:

        a : {array_like}
            Array containing numbers whose mean is desired. If a is not an
            array, a conversion is attempted.
        axis : {None, integer}, optional
            Axis along which the means are computed. The default is to compute
            the standard deviation of the flattened array.
        dtype : {None, dtype}, optional
            Type to use in computing the means. For arrays of integer type the
            default is float32, for arrays of float types it is the same as the
            array type.
        out : {None, array}, optional
            Alternative output array in which to place the result. It must have
            the same shape as the expected output but the type will be cast if
            necessary.

    *Returns*:

        mean : {array, scalar}, see dtype parameter above
            If out=None, returns a new array containing the mean values,
            otherwise a reference to the output array is returned.

    *See Also*:

        `var` : Variance

        `std` : Standard deviation

    *Notes*

        The mean is the sum of the elements along the axis divided by the number
        of elements.

    *Examples*

        >>> a = array([[1,2],[3,4]])
        >>> mean(a)
        2.5
        >>> mean(a,0)
        array([ 2.,  3.])
        >>> mean(a,1)
        array([ 1.5,  3.5])

    """
    try:
        mean = a.mean
    except AttributeError:
        return _wrapit(a, 'mean', axis, dtype, out)
    return mean(axis, dtype, out)


def std(a, axis=None, dtype=None, out=None):
    """Compute the standard deviation along the specified axis.

    Returns the standard deviation of the array elements, a measure of the
    spread of a distribution. The standard deviation is computed for the
    flattened array by default, otherwise over the specified axis.

    *Parameters*:

        a : {array_like}
            Array containing numbers whose standard deviation is desired. If a
            is not an array, a conversion is attempted.
        axis : {None, integer}, optional
            Axis along which the standard deviation is computed. The default is
            to compute the standard deviation of the flattened array.
        dtype : {None, dtype}, optional
            Type to use in computing the standard deviation. For arrays of
            integer type the default is float32, for arrays of float types it is
            the same as the array type.
        out : {None, array}, optional
            Alternative output array in which to place the result. It must have
            the same shape as the expected output but the type will be cast if
            necessary.

    *Returns*:

        standard_deviation : {array, scalar}, see dtype parameter above.
            If out=None, returns a new array containing the standard deviation,
            otherwise a reference to the output array is returned.

    *See Also*:

        `var` : Variance

        `mean` : Average

    *Notes*

        The standard deviation is the square root of the average of the squared
        deviations from the mean, i.e. var = sqrt(mean((x - x.mean())**2)).  The
        computed standard deviation is biased, i.e., the mean is computed by
        dividing by the number of elements, N, rather than by N-1.

    *Examples*

        >>> a = array([[1,2],[3,4]])
        >>> std(a)
        1.1180339887498949
        >>> std(a,0)
        array([ 1.,  1.])
        >>> std(a,1)
        array([ 0.5,  0.5])

    """
    try:
        std = a.std
    except AttributeError:
        return _wrapit(a, 'std', axis, dtype, out)
    return std(axis, dtype, out)


def var(a, axis=None, dtype=None, out=None):
    """Compute the variance along the specified axis.

    Returns the variance of the array elements, a measure of the spread of a
    distribution. The variance is computed for the flattened array by default,
    otherwise over the specified axis.

    *Parameters*:

        a : {array_like}
            Array containing numbers whose variance is desired. If a is not an
            array, a conversion is attempted.
        axis : {None, integer}, optional
            Axis along which the variance is computed. The default is to compute
            the variance of the flattened array.
        dtype : {None, dtype}, optional
            Type to use in computing the variance. For arrays of integer type
            the default is float32, for arrays of float types it is the same as
            the array type.
        out : {None, array}, optional
            Alternative output array in which to place the result. It must have
            the same shape as the expected output but the type will be cast if
            necessary.

    *Returns*:

        variance : {array, scalar}, see dtype parameter above
            If out=None, returns a new array containing the variance, otherwise
            a reference to the output array is returned.

    *See Also*:

        `std` : Standard deviation

        `mean` : Average

    *Notes*

        The variance is the average of the squared deviations from the mean,
        i.e.  var = mean((x - x.mean())**2).  The computed variance is biased,
        i.e., the mean is computed by dividing by the number of elements, N,
        rather than by N-1.

    *Examples*

        >>> a = array([[1,2],[3,4]])
        >>> var(a)
        1.25
        >>> var(a,0)
        array([ 1.,  1.])
        >>> var(a,1)
        array([ 0.25,  0.25])

    """
    try:
        var = a.var
    except AttributeError:
        return _wrapit(a, 'var', axis, dtype, out)
    return var(axis, dtype, out)
