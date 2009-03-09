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
    if wrap:
        if not isinstance(result, mu.ndarray):
            result = asarray(result)
        result = wrap(result)
    return result


def take(a, indices, axis=None, out=None, mode='raise'):
    """
    Take elements from an array along an axis.

    This function does the same thing as "fancy" indexing (indexing arrays
    using arrays); however, it can be easier to use if you need elements
    along a given axis.

    Parameters
    ----------
    a : array_like
        The source array.
    indices : array_like, int
        The indices of the values to extract.
    axis : int, optional
        The axis over which to select values.  By default, the
        flattened input array is used.
    out : ndarray, optional
        If provided, the result will be placed in this array. It should
        be of the appropriate shape and dtype.
    mode : {'raise', 'wrap', 'clip'}, optional
        Specifies how out-of-bounds indices will behave.
        'raise' -- raise an error
        'wrap' -- wrap around
        'clip' -- clip to the range

    Returns
    -------
    subarray : ndarray
        The returned array has the same type as `a`.

    See Also
    --------
    ndarray.take : equivalent method

    Examples
    --------
    >>> a = [4, 3, 5, 7, 6, 8]
    >>> indices = [0, 1, 4]
    >>> np.take(a, indices)
    array([4, 3, 6])

    In this example if `a` is a ndarray, "fancy" indexing can be used.
    >>> a = np.array(a)
    >>> a[indices]
    array([4, 3, 6])

    """
    try:
        take = a.take
    except AttributeError:
        return _wrapit(a, 'take', indices, axis, out, mode)
    return take(indices, axis, out, mode)


# not deprecated --- copy if necessary, view otherwise
def reshape(a, newshape, order='C'):
    """
    Gives a new shape to an array without changing its data.

    Parameters
    ----------
    a : array_like
        Array to be reshaped.
    newshape : {tuple, int}
        The new shape should be compatible with the original shape. If
        an integer, then the result will be a 1-D array of that length.
        One shape dimension can be -1. In this case, the value is inferred
        from the length of the array and remaining dimensions.
    order : {'C', 'F'}, optional
        Determines whether the array data should be viewed as in C
        (row-major) order or FORTRAN (column-major) order.

    Returns
    -------
    reshaped_array : ndarray
        This will be a new view object if possible; otherwise, it will
        be a copy.

    See Also
    --------
    ndarray.reshape : Equivalent method.

    Examples
    --------
    >>> a = np.array([[1,2,3], [4,5,6]])
    >>> np.reshape(a, 6)
    array([1, 2, 3, 4, 5, 6])
    >>> np.reshape(a, 6, order='F')
    array([1, 4, 2, 5, 3, 6])
    >>> np.reshape(a, (3,-1))       # the unspecified value is inferred to be 2
    array([[1, 2],
           [3, 4],
           [5, 6]])

    """
    try:
        reshape = a.reshape
    except AttributeError:
        return _wrapit(a, 'reshape', newshape, order=order)
    return reshape(newshape, order=order)


def choose(a, choices, out=None, mode='raise'):
    """
    Use an index array to construct a new array from a set of choices.

    Given an array of integers and a set of n choice arrays, this function
    will create a new array that merges each of the choice arrays.  Where a
    value in `a` is i, then the new array will have the value that
    choices[i] contains in the same place.

    Parameters
    ----------
    a : int array
        This array must contain integers in [0, n-1], where n is the number
        of choices.
    choices : sequence of arrays
        Choice arrays. The index array and all of the choices should be
        broadcastable to the same shape.
    out : array, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype
    mode : {'raise', 'wrap', 'clip'}, optional
        Specifies how out-of-bounds indices will behave:

          * 'raise' : raise an error
          * 'wrap' : wrap around
          * 'clip' : clip to the range

    Returns
    -------
    merged_array : array
        The merged results.

    See Also
    --------
    ndarray.choose : equivalent method

    Examples
    --------

    >>> choices = [[0, 1, 2, 3], [10, 11, 12, 13],
    ...   [20, 21, 22, 23], [30, 31, 32, 33]]
    >>> np.choose([2, 3, 1, 0], choices)
    array([20, 31, 12,  3])
    >>> np.choose([2, 4, 1, 0], choices, mode='clip')
    array([20, 31, 12,  3])
    >>> np.choose([2, 4, 1, 0], choices, mode='wrap')
    array([20,  1, 12,  3])

    """
    try:
        choose = a.choose
    except AttributeError:
        return _wrapit(a, 'choose', choices, out=out, mode=mode)
    return choose(choices, out=out, mode=mode)


def repeat(a, repeats, axis=None):
    """
    Repeat elements of an array.

    Parameters
    ----------
    a : array_like
        Input array.
    repeats : {int, array of ints}
        The number of repetitions for each element.  `repeats` is broadcasted
        to fit the shape of the given axis.
    axis : int, optional
        The axis along which to repeat values.  By default, use the
        flattened input array, and return a flat output array.

    Returns
    -------
    repeated_array : ndarray
        Output array which has the same shape as `a`, except along
        the given axis.

    See Also
    --------
    tile : Tile an array.

    Examples
    --------
    >>> x = np.array([[1,2],[3,4]])
    >>> np.repeat(x, 2)
    array([1, 1, 2, 2, 3, 3, 4, 4])
    >>> np.repeat(x, 3, axis=1)
    array([[1, 1, 1, 2, 2, 2],
           [3, 3, 3, 4, 4, 4]])
    >>> np.repeat(x, [1, 2], axis=0)
    array([[1, 2],
           [3, 4],
           [3, 4]])

    """
    try:
        repeat = a.repeat
    except AttributeError:
        return _wrapit(a, 'repeat', repeats, axis)
    return repeat(repeats, axis)


def put(a, ind, v, mode='raise'):
    """
    Changes specific elements of one array by replacing from another array.

    Set `a`.flat[n] = `v`\\[n] for all n in `ind`.  If `v` is shorter than
    `ind`, it will repeat which is different than `a[ind]` = `v`.

    Parameters
    ----------
    a : array_like (contiguous)
        Target array.
    ind : array_like
        Target indices, interpreted as integers.
    v : array_like
        Values to place in `a` at target indices.
    mode : {'raise', 'wrap', 'clip'}, optional
        Specifies how out-of-bounds indices will behave.

    * 'raise' -- raise an error
    * 'wrap' -- wrap around
    * 'clip' -- clip to the range

    Notes
    -----
    If `v` is shorter than `mask` it will be repeated as necessary.  In
    particular `v` can be a scalar or length 1 array.  The routine put
    is the equivalent of the following (although the loop is in C for
    speed):
    ::

        ind = array(indices, copy=False)
        v = array(values, copy=False).astype(a.dtype)
        for i in ind: a.flat[i] = v[i]

    Examples
    --------
    >>> x = np.arange(5)
    >>> np.put(x,[0,2,4],[-1,-2,-3])
    >>> print x
    [-1  1 -2  3 -3]

    """
    return a.put(ind, v, mode)


def swapaxes(a, axis1, axis2):
    """
    Interchange two axes of an array.

    Parameters
    ----------
    a : array_like
        Input array.
    axis1 : int
        First axis.
    axis2 : int
        Second axis.

    Returns
    -------
    a_swapped : ndarray
        If `a` is an ndarray, then a view of `a` is returned; otherwise
        a new array is created.

    Examples
    --------
    >>> x = np.array([[1,2,3]])
    >>> np.swapaxes(x,0,1)
    array([[1],
           [2],
           [3]])

    >>> x = np.array([[[0,1],[2,3]],[[4,5],[6,7]]])
    >>> x
    array([[[0, 1],
            [2, 3]],
    <BLANKLINE>
           [[4, 5],
            [6, 7]]])

    >>> np.swapaxes(x,0,2)
    array([[[0, 4],
            [2, 6]],
    <BLANKLINE>
           [[1, 5],
            [3, 7]]])

    """
    try:
        swapaxes = a.swapaxes
    except AttributeError:
        return _wrapit(a, 'swapaxes', axis1, axis2)
    return swapaxes(axis1, axis2)


def transpose(a, axes=None):
    """
    Permute the dimensions of an array.

    Parameters
    ----------
    a : array_like
        Input array.
    axes : list of ints, optional
        By default, reverse the dimensions, otherwise permute the axes
        according to the values given.

    Returns
    -------
    p : ndarray
        `a` with its axes permuted.  A view is returned whenever
        possible.

    See Also
    --------
    rollaxis

    Examples
    --------
    >>> x = np.arange(4).reshape((2,2))
    >>> x
    array([[0, 1],
           [2, 3]])

    >>> np.transpose(x)
    array([[0, 2],
           [1, 3]])

    >>> x = np.ones((1, 2, 3))
    >>> np.transpose(x, (1, 0, 2)).shape
    (2, 1, 3)

    """
    try:
        transpose = a.transpose
    except AttributeError:
        return _wrapit(a, 'transpose', axes)
    return transpose(axes)


def sort(a, axis=-1, kind='quicksort', order=None):
    """
    Return a sorted copy of an array.

    Parameters
    ----------
    a : array_like
        Array to be sorted.
    axis : int or None, optional
        Axis along which to sort. If None, the array is flattened before
        sorting. The default is -1, which sorts along the last axis.
    kind : {'quicksort', 'mergesort', 'heapsort'}, optional
        Sorting algorithm. Default is 'quicksort'.
    order : list, optional
        When `a` is a structured array, this argument specifies which fields
        to compare first, second, and so on.  This list does not need to
        include all of the fields.

    Returns
    -------
    sorted_array : ndarray
        Array of the same type and shape as `a`.

    See Also
    --------
    ndarray.sort : Method to sort an array in-place.
    argsort : Indirect sort.
    lexsort : Indirect stable sort on multiple keys.
    searchsorted : Find elements in a sorted array.

    Notes
    -----
    The various sorting algorithms are characterized by their average speed,
    worst case performance, work space size, and whether they are stable. A
    stable sort keeps items with the same key in the same relative
    order. The three available algorithms have the following
    properties:

    =========== ======= ============= ============ =======
       kind      speed   worst case    work space  stable
    =========== ======= ============= ============ =======
    'quicksort'    1     O(n^2)            0          no
    'mergesort'    2     O(n*log(n))      ~n/2        yes
    'heapsort'     3     O(n*log(n))       0          no
    =========== ======= ============= ============ =======

    All the sort algorithms make temporary copies of the data when
    sorting along any but the last axis.  Consequently, sorting along
    the last axis is faster and uses less space than sorting along
    any other axis.

    Examples
    --------
    >>> a = np.array([[1,4],[3,1]])
    >>> np.sort(a)                # sort along the last axis
    array([[1, 4],
           [1, 3]])
    >>> np.sort(a, axis=None)     # sort the flattened array
    array([1, 1, 3, 4])
    >>> np.sort(a, axis=0)        # sort along the first axis
    array([[1, 1],
           [3, 4]])

    Use the `order` keyword to specify a field to use when sorting a
    structured array:

    >>> dtype = [('name', 'S10'), ('height', float), ('age', int)]
    >>> values = [('Arthur', 1.8, 41), ('Lancelot', 1.9, 38),
    ...           ('Galahad', 1.7, 38)]
    >>> a = np.array(values, dtype=dtype)       # create a structured array
    >>> np.sort(a, order='height')                        # doctest: +SKIP
    array([('Galahad', 1.7, 38), ('Arthur', 1.8, 41),
           ('Lancelot', 1.8999999999999999, 38)],
          dtype=[('name', '|S10'), ('height', '<f8'), ('age', '<i4')])

    Sort by age, then height if ages are equal:

    >>> np.sort(a, order=['age', 'height'])               # doctest: +SKIP
    array([('Galahad', 1.7, 38), ('Lancelot', 1.8999999999999999, 38),
           ('Arthur', 1.8, 41)],
          dtype=[('name', '|S10'), ('height', '<f8'), ('age', '<i4')])

    """
    if axis is None:
        a = asanyarray(a).flatten()
        axis = 0
    else:
        a = asanyarray(a).copy()
    a.sort(axis, kind, order)
    return a


def argsort(a, axis=-1, kind='quicksort', order=None):
    """
    Returns the indices that would sort an array.

    Perform an indirect sort along the given axis using the algorithm specified
    by the `kind` keyword. It returns an array of indices of the same shape as
    `a` that index data along the given axis in sorted order.

    Parameters
    ----------
    a : array_like
        Array to sort.
    axis : int, optional
        Axis along which to sort.  If not given, the flattened array is used.
    kind : {'quicksort', 'mergesort', 'heapsort'}, optional
        Sorting algorithm.
    order : list, optional
        When `a` is an array with fields defined, this argument specifies
        which fields to compare first, second, etc.  Not all fields need be
        specified.

    Returns
    -------
    index_array : ndarray, int
        Array of indices that sort `a` along the specified axis.
        In other words, ``a[index_array]`` yields a sorted `a`.

    See Also
    --------
    sort : Describes sorting algorithms used.
    lexsort : Indirect stable sort with multiple keys.
    ndarray.sort : Inplace sort.

    Notes
    -----
    See `sort` for notes on the different sorting algorithms.

    Examples
    --------
    One dimensional array:

    >>> x = np.array([3, 1, 2])
    >>> np.argsort(x)
    array([1, 2, 0])

    Two-dimensional array:

    >>> x = np.array([[0, 3], [2, 2]])
    >>> x
    array([[0, 3],
           [2, 2]])

    >>> np.argsort(x, axis=0)
    array([[0, 1],
           [1, 0]])

    >>> np.argsort(x, axis=1)
    array([[0, 1],
           [0, 1]])

    Sorting with keys:

    >>> x = np.array([(1, 0), (0, 1)], dtype=[('x', '<i4'), ('y', '<i4')])
    >>> x
    array([(1, 0), (0, 1)],
          dtype=[('x', '<i4'), ('y', '<i4')])

    >>> np.argsort(x, order=('x','y'))
    array([1, 0])

    >>> np.argsort(x, order=('y','x'))
    array([0, 1])

    """
    try:
        argsort = a.argsort
    except AttributeError:
        return _wrapit(a, 'argsort', axis, kind, order)
    return argsort(axis, kind, order)


def argmax(a, axis=None):
    """
    Indices of the maximum values along an axis.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise
        along the specified axis.

    Returns
    -------
    index_array : ndarray, int
        Array of indices into the array.  It has the same shape as `a`,
        except with `axis` removed.

    See Also
    --------
    argmin : Indices of the minimum values along an axis.
    amax : The maximum value along a given axis.
    unravel_index : Convert a flat index into an index tuple.

    Examples
    --------
    >>> a = np.arange(6).reshape(2,3)
    >>> np.argmax(a)
    5
    >>> np.argmax(a, axis=0)
    array([1, 1, 1])
    >>> np.argmax(a, axis=1)
    array([2, 2])

    """
    try:
        argmax = a.argmax
    except AttributeError:
        return _wrapit(a, 'argmax', axis)
    return argmax(axis)


def argmin(a, axis=None):
    """
    Return the indices of the minimum values along an axis.

    See Also
    --------
    argmax : Similar function.  Please refer to `numpy.argmax` for detailed
        documentation.

    """
    try:
        argmin = a.argmin
    except AttributeError:
        return _wrapit(a, 'argmin', axis)
    return argmin(axis)


def searchsorted(a, v, side='left'):
    """
    Find indices where elements should be inserted to maintain order.

    Find the indices into a sorted array `a` such that, if the corresponding
    elements in `v` were inserted before the indices, the order of `a` would
    be preserved.

    Parameters
    ----------
    a : 1-D array_like of shape (N,)
        Input array, sorted in ascending order.
    v : array_like
        Values to insert into `a`.
    side : {'left', 'right'}, optional
        If 'left', the index of the first suitable location found is given.  If
        'right', return the last such index.  If there is no suitable
        index, return either 0 or N (where N is the length of `a`).

    Returns
    -------
    indices : array of ints
        Array of insertion points with the same shape as `v`.

    See Also
    --------
    sort : In-place sort.
    histogram : Produce histogram from 1-D data.

    Notes
    -----
    Binary search is used to find the required insertion points.

    Examples
    --------
    >>> np.searchsorted([1,2,3,4,5], 3)
    2
    >>> np.searchsorted([1,2,3,4,5], 3, side='right')
    3
    >>> np.searchsorted([1,2,3,4,5], [-10, 10, 2, 3])
    array([0, 5, 1, 2])

    """
    try:
        searchsorted = a.searchsorted
    except AttributeError:
        return _wrapit(a, 'searchsorted', v, side)
    return searchsorted(v, side)


def resize(a, new_shape):
    """
    Return a new array with the specified shape.

    If the new array is larger than the original array, then the new array
    is filled with repeated copied of `a`. Note that this behavior is different
    from a.resize(new_shape) which fills with zeros instead of repeated
    copies of `a`.

    Parameters
    ----------
    a : array_like
        Array to be resized.

    new_shape : {tuple, int}
        Shape of resized array.

    Returns
    -------
    reshaped_array : ndarray
        The new array is formed from the data in the old array, repeated if
        necessary to fill out the required number of elements.

    See Also
    --------
    ndarray.resize : resize an array in-place.

    Examples
    --------
    >>> a=np.array([[0,1],[2,3]])
    >>> np.resize(a,(1,4))
    array([[0, 1, 2, 3]])
    >>> np.resize(a,(2,4))
    array([[0, 1, 2, 3],
           [0, 1, 2, 3]])

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
    """
    Remove single-dimensional entries from the shape of an array.

    Parameters
    ----------
    a : array_like
        Input data.

    Returns
    -------
    squeezed : ndarray
        The input array, but with with all dimensions of length 1
        removed.  Whenever possible, a view on `a` is returned.

    Examples
    --------
    >>> x = np.array([[[0], [1], [2]]])
    >>> x.shape
    (1, 3, 1)
    >>> np.squeeze(x).shape
    (3,)

    """
    try:
        squeeze = a.squeeze
    except AttributeError:
        return _wrapit(a, 'squeeze')
    return squeeze()


def diagonal(a, offset=0, axis1=0, axis2=1):
    """
    Return specified diagonals.

    If `a` is 2-D, returns the diagonal of `a` with the given offset,
    i.e., the collection of elements of the form `a[i,i+offset]`.
    If `a` has more than two dimensions, then the axes specified
    by `axis1` and `axis2` are used to determine the 2-D subarray
    whose diagonal is returned. The shape of the resulting array
    can be determined by removing `axis1` and `axis2` and appending
    an index to the right equal to the size of the resulting diagonals.

    Parameters
    ----------
    a : array_like
        Array from which the diagonals are taken.
    offset : int, optional
        Offset of the diagonal from the main diagonal. Can be both positive
        and negative. Defaults to main diagonal (0).
    axis1 : int, optional
        Axis to be used as the first axis of the 2-D subarrays from which
        the diagonals should be taken. Defaults to first axis (0).
    axis2 : int, optional
        Axis to be used as the second axis of the 2-D subarrays from which
        the diagonals should be taken. Defaults to second axis (1).

    Returns
    -------
    array_of_diagonals : ndarray
        If `a` is 2-D, a 1-D array containing the diagonal is
        returned.  If `a` has larger dimensions, then an array of
        diagonals is returned.

    Raises
    ------
    ValueError
        If the dimension of `a` is less than 2.

    See Also
    --------
    diag : Matlab workalike for 1-D and 2-D arrays.
    diagflat : Create diagonal arrays.
    trace : Sum along diagonals.

    Examples
    --------
    >>> a = np.arange(4).reshape(2,2)
    >>> a
    array([[0, 1],
           [2, 3]])
    >>> a.diagonal()
    array([0, 3])
    >>> a.diagonal(1)
    array([1])

    >>> a = np.arange(8).reshape(2,2,2)
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
    """
    Return the sum along diagonals of the array.

    If a is 2-d, returns the sum along the diagonal of self with the given
    offset, i.e., the collection of elements of the form a[i,i+offset]. If
    a has more than two dimensions, then the axes specified by axis1 and
    axis2 are used to determine the 2-d subarray whose trace is returned.
    The shape of the resulting array can be determined by removing axis1
    and axis2 and appending an index to the right equal to the size of the
    resulting diagonals.

    Parameters
    ----------
    a : array_like
        Array from whis the diagonals are taken.
    offset : integer, optional
        Offset of the diagonal from the main diagonal. Can be both positive
        and negative. Defaults to main diagonal.
    axis1 : integer, optional
        Axis to be used as the first axis of the 2-d subarrays from which
        the diagonals should be taken. Defaults to first axis.
    axis2 : integer, optional
        Axis to be used as the second axis of the 2-d subarrays from which
        the diagonals should be taken. Defaults to second axis.
    dtype : dtype, optional
        Determines the type of the returned array and of the accumulator
        where the elements are summed. If dtype has the value None and a is
        of integer type of precision less than the default integer
        precision, then the default integer precision is used. Otherwise,
        the precision is the same as that of a.
    out : array, optional
        Array into which the sum can be placed. Its type is preserved and
        it must be of the right shape to hold the output.

    Returns
    -------
    sum_along_diagonals : ndarray
        If a is 2-d, a 0-d array containing the diagonal is
        returned.  If a has larger dimensions, then an array of
        diagonals is returned.

    Examples
    --------
    >>> np.trace(np.eye(3))
    3.0
    >>> a = np.arange(8).reshape((2,2,2))
    >>> np.trace(a)
    array([6, 8])

    """
    return asarray(a).trace(offset, axis1, axis2, dtype, out)

def ravel(a, order='C'):
    """
    Return a flattened array.

    A 1-d array, containing the elements of the input, is returned.  A copy is
    made only if needed.

    Parameters
    ----------
    a : array_like
        Input array.  The elements in `a` are read in the order specified by
        `order`, and packed as a 1-dimensional array.
    order : {'C','F'}, optional
        The elements of `a` are read in this order.  It can be either
        'C' for row-major order, or `F` for column-major order.
        By default, row-major order is used.

    Returns
    -------
    1d_array : ndarray
        Output of the same dtype as `a`, and of shape ``(a.size(),)`` (or
        ``(np.prod(a.shape),)``).

    See Also
    --------
    ndarray.flat : 1-D iterator over an array.
    ndarray.flatten : 1-D array copy of the elements of an array
                      in row-major order.

    Notes
    -----
    In row-major order, the row index varies the slowest, and the column
    index the quickest.  This can be generalised to multiple dimensions,
    where row-major order implies that the index along the first axis
    varies slowest, and the index along the last quickest.  The opposite holds
    for Fortran-, or column-major, mode.

    Examples
    --------
    If an array is in C-order (default), then `ravel` is equivalent
    to ``reshape(-1)``:

    >>> x = np.array([[1, 2, 3], [4, 5, 6]])
    >>> print x.reshape(-1)
    [1  2  3  4  5  6]

    >>> print np.ravel(x)
    [1  2  3  4  5  6]

    When flattening using Fortran-order, however, we see

    >>> print np.ravel(x, order='F')
    [1 4 2 5 3 6]

    """
    return asarray(a).ravel(order)


def nonzero(a):
    """
    Return the indices of the elements that are non-zero.

    Returns a tuple of arrays, one for each dimension of `a`, containing
    the indices of the non-zero elements in that dimension. The
    corresponding non-zero values can be obtained with::

        a[nonzero(a)]

    To group the indices by element, rather than dimension, use::

        transpose(nonzero(a))

    The result of this is always a 2-D array, with a row for
    each non-zero element.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    tuple_of_arrays : tuple
        Indices of elements that are non-zero.

    See Also
    --------
    flatnonzero :
        Return indices that are non-zero in the flattened version of the input
        array.
    ndarray.nonzero :
        Equivalent ndarray method.

    Examples
    --------
    >>> x = np.eye(3)
    >>> x
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> np.nonzero(x)
    (array([0, 1, 2]), array([0, 1, 2]))

    >>> x[np.nonzero(x)]
    array([ 1.,  1.,  1.])
    >>> np.transpose(np.nonzero(x))
    array([[0, 0],
           [1, 1],
           [2, 2]])

    """
    try:
        nonzero = a.nonzero
    except AttributeError:
        res = _wrapit(a, 'nonzero')
    else:
        res = nonzero()
    return res


def shape(a):
    """
    Return the shape of an array.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    shape : tuple
        The elements of the tuple give the lengths of the corresponding array
        dimensions.

    See Also
    --------
    alen,
    ndarray.shape : array method

    Examples
    --------
    >>> np.shape(np.eye(3))
    (3, 3)
    >>> np.shape([[1,2]])
    (1, 2)
    >>> np.shape([0])
    (1,)
    >>> np.shape(0)
    ()

    >>> a = np.array([(1,2),(3,4)], dtype=[('x', 'i4'), ('y', 'i4')])
    >>> np.shape(a)
    (2,)
    >>> a.shape
    (2,)

    """
    try:
        result = a.shape
    except AttributeError:
        result = asarray(a).shape
    return result


def compress(condition, a, axis=None, out=None):
    """
    Return selected slices of an array along given axis.

    Parameters
    ----------
    condition : array_like
        Boolean 1-D array selecting which entries to return. If len(condition)
        is less than the size of `a` along the given axis, then output is
        truncated to the length of the condition array.
    a : array_like
        Array from which to extract a part.
    axis : int, optional
        Axis along which to take slices. If None (default), work on the
        flattened array.
    out : ndarray, optional
        Output array.  Its type is preserved and it must be of the right
        shape to hold the output.

    Returns
    -------
    compressed_array : ndarray
        A copy of `a` without the slices along axis for which `condition`
        is false.

    See Also
    --------
    ndarray.compress: Equivalent method.

    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.compress([0, 1], a, axis=0)
    array([[3, 4]])
    >>> np.compress([1], a, axis=1)
    array([[1],
           [3]])
    >>> np.compress([0,1,1], a)
    array([2, 3])

    """
    try:
        compress = a.compress
    except AttributeError:
        return _wrapit(a, 'compress', condition, axis, out)
    return compress(condition, axis, out)


def clip(a, a_min, a_max, out=None):
    """
    Clip (limit) the values in an array.

    Given an interval, values outside the interval are clipped to
    the interval edges.  For example, if an interval of ``[0, 1]``
    is specified, values smaller than 0 become 0, and values larger
    than 1 become 1.

    Parameters
    ----------
    a : array_like
        Array containing elements to clip.
    a_min : scalar or array_like
        Minimum value.
    a_max : scalar or array_like
        Maximum value.  If `a_min` or `a_max` are array_like, then they will
        be broadcasted to the shape of `a`.
    out : ndarray, optional
        The results will be placed in this array. It may be the input
        array for in-place clipping.  `out` must be of the right shape
        to hold the output.  Its type is preserved.

    Returns
    -------
    clipped_array : ndarray
        An array with the elements of `a`, but where values
        < `a_min` are replaced with `a_min`, and those > `a_max`
        with `a_max`.

    Examples
    --------
    >>> a = np.arange(10)
    >>> np.clip(a, 1, 8)
    array([1, 1, 2, 3, 4, 5, 6, 7, 8, 8])
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> np.clip(a, 3, 6, out=a)
    array([3, 3, 3, 3, 4, 5, 6, 6, 6, 6])
    >>> a
    array([3, 3, 3, 3, 4, 5, 6, 6, 6, 6])
    >>> np.clip(a, [3,4,1,1,1,4,4,4,4,4], 8)
    array([3, 4, 2, 3, 4, 5, 6, 7, 8, 8])

    """
    try:
        clip = a.clip
    except AttributeError:
        return _wrapit(a, 'clip', a_min, a_max, out)
    return clip(a_min, a_max, out)


def sum(a, axis=None, dtype=None, out=None):
    """
    Return the sum of array elements over a given axis.

    Parameters
    ----------
    a : array_like
        Elements to sum.
    axis : integer, optional
        Axis over which the sum is taken. By default `axis` is None,
        and all elements are summed.
    dtype : dtype, optional
        The type of the returned array and of the accumulator in which
        the elements are summed.  By default, the dtype of `a` is used.
        An exception is when `a` has an integer type with less precision
        than the default platform integer.  In that case, the default
        platform integer is used instead.
    out : ndarray, optional
        Array into which the output is placed.  By default, a new array is
        created.  If `out` is given, it must be of the appropriate shape
        (the shape of `a` with `axis` removed, i.e.,
        ``numpy.delete(a.shape, axis)``).  Its type is preserved.

    Returns
    -------
    sum_along_axis : ndarray
        An array with the same shape as `a`, with the specified
        axis removed.   If `a` is a 0-d array, or if `axis` is None, a scalar
        is returned.  If an output array is specified, a reference to
        `out` is returned.

    See Also
    --------
    ndarray.sum : equivalent method

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.

    Examples
    --------
    >>> np.sum([0.5, 1.5])
    2.0
    >>> np.sum([0.5, 1.5], dtype=np.int32)
    1
    >>> np.sum([[0, 1], [0, 5]])
    6
    >>> np.sum([[0, 1], [0, 5]], axis=1)
    array([1, 5])

    If the accumulator is too small, overflow occurs:

    >>> np.ones(128, dtype=np.int8).sum(dtype=np.int8)
    -128

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
    """
    Return the product of array elements over a given axis.

    See Also
    --------
    prod : equivalent function; see for details.

    """
    try:
        prod = a.prod
    except AttributeError:
        return _wrapit(a, 'prod', axis, dtype, out)
    return prod(axis, dtype, out)


def sometrue(a, axis=None, out=None):
    """
    Check whether some values are true.

    Refer to `any` for full documentation.

    See Also
    --------
    any : equivalent function

    """
    try:
        any = a.any
    except AttributeError:
        return _wrapit(a, 'any', axis, out)
    return any(axis, out)


def alltrue (a, axis=None, out=None):
    """
    Check if all elements of input array are true.

    See Also
    --------
    numpy.all : Equivalent function; see for details.

    """
    try:
        all = a.all
    except AttributeError:
        return _wrapit(a, 'all', axis, out)
    return all(axis, out)


def any(a,axis=None, out=None):
    """
    Test whether any elements of an array evaluate to True along an axis.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        Axis over which to perform the operation.
        If None, use a flattened input array and return a bool.
    out : ndarray, optional
        Array into which the result is placed. Its type is preserved
        and it must be of the right shape to hold the output.

    Returns
    -------
    out : ndarray
        A logical OR is performed along `axis`, and the result placed
        in `out`.  If `out` was not specified, a new output array is created.

    See Also
    --------
    ndarray.any : equivalent method

    Notes
    -----
    Since NaN is not equal to zero, NaN evaluates to True.

    Examples
    --------
    >>> np.any([[True, False], [True, True]])
    True

    >>> np.any([[True, False], [False, False]], axis=0)
    array([ True, False], dtype=bool)

    >>> np.any([-1, 0, 5])
    True

    >>> np.any(np.nan)
    True

    """
    try:
        any = a.any
    except AttributeError:
        return _wrapit(a, 'any', axis, out)
    return any(axis, out)


def all(a,axis=None, out=None):
    """
    Returns True if all elements evaluate to True.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        Axis over which to perform the operation.
        If None, use a flattened input array and return a bool.
    out : ndarray, optional
        Array into which the result is placed. Its type is preserved
        and it must be of the right shape to hold the output.

    Returns
    -------
    out : ndarray, bool
        A logical AND is performed along `axis`, and the result placed
        in `out`.  If `out` was not specified, a new output array is created.

    See Also
    --------
    ndarray.all : equivalent method

    Notes
    -----
    Since NaN is not equal to zero, NaN evaluates to True.

    Examples
    --------
    >>> np.all([[True,False],[True,True]])
    False

    >>> np.all([[True,False],[True,True]], axis=0)
    array([ True, False], dtype=bool)

    >>> np.all([-1, 4, 5])
    True

    >>> np.all([1.0, np.nan])
    True

    """
    try:
        all = a.all
    except AttributeError:
        return _wrapit(a, 'all', axis, out)
    return all(axis, out)


def cumsum (a, axis=None, dtype=None, out=None):
    """
    Return the cumulative sum of the elements along a given axis.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : int, optional
        Axis along which the cumulative sum is computed. The default
        (`axis` = `None`) is to compute the cumsum over the flattened
        array. `axis` may be negative, in which case it counts from the
        last to the first axis.
    dtype : dtype, optional
        Type of the returned array and of the accumulator in which the
        elements are summed.  If `dtype` is not specified, it defaults
        to the dtype of `a`, unless `a` has an integer dtype with a
        precision less than that of the default platform integer.  In
        that case, the default platform integer is used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type will be cast if necessary.

    Returns
    -------
    cumsum : ndarray.
        A new array holding the result is returned unless `out` is
        specified, in which case a reference to `out` is returned.

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.

    Examples
    --------
    >>> a = np.array([[1,2,3],[4,5,6]])
    >>> np.cumsum(a)
    array([ 1,  3,  6, 10, 15, 21])
    >>> np.cumsum(a,dtype=float)     # specifies type of output value(s)
    array([  1.,   3.,   6.,  10.,  15.,  21.])
    >>> np.cumsum(a,axis=0)      # sum over rows for each of the 3 columns
    array([[1, 2, 3],
           [5, 7, 9]])
    >>> np.cumsum(a,axis=1)      # sum over columns for each of the 2 rows
    array([[ 1,  3,  6],
           [ 4,  9, 15]])

    """
    try:
        cumsum = a.cumsum
    except AttributeError:
        return _wrapit(a, 'cumsum', axis, dtype, out)
    return cumsum(axis, dtype, out)


def cumproduct(a, axis=None, dtype=None, out=None):
    """
    Return the cumulative product over the given axis.


    See Also
    --------
    cumprod : equivalent function; see for details.

    """
    try:
        cumprod = a.cumprod
    except AttributeError:
        return _wrapit(a, 'cumprod', axis, dtype, out)
    return cumprod(axis, dtype, out)


def ptp(a, axis=None, out=None):
    """
    Range of values (maximum - minimum) along an axis.

    The name of the function comes from the acronym for 'peak to peak'.

    Parameters
    ----------
    a : array_like
        Input values.
    axis : int, optional
        Axis along which to find the peaks.  By default, flatten the
        array.
    out : array_like
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type of the output values will be cast if necessary.

    Returns
    -------
    ptp : ndarray
        A new array holding the result, unless `out` was
        specified, in which case a reference to `out` is returned.

    Examples
    --------
    >>> x = np.arange(4).reshape((2,2))
    >>> x
    array([[0, 1],
           [2, 3]])

    >>> np.ptp(x, axis=0)
    array([2, 2])

    >>> np.ptp(x, axis=1)
    array([1, 1])

    """
    try:
        ptp = a.ptp
    except AttributeError:
        return _wrapit(a, 'ptp', axis, out)
    return ptp(axis, out)


def amax(a, axis=None, out=None):
    """
    Return the maximum along an axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : int, optional
        Axis along which to operate.  By default flattened input is used.
    out : ndarray, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.

    Returns
    -------
    amax : ndarray
        A new array or a scalar with the result, or a reference to `out`
        if it was specified.

    Examples
    --------
    >>> a = np.arange(4).reshape((2,2))
    >>> a
    array([[0, 1],
           [2, 3]])
    >>> np.amax(a, axis=0)
    array([2, 3])
    >>> np.amax(a, axis=1)
    array([1, 3])

    Notes
    -----
    NaN values are propagated, that is if at least one item is nan, the
    corresponding max value will be nan as well. To ignore NaN values (matlab
    behavior), please use nanmax.

    See Also
    --------
    nanmax: nan values are ignored instead of being propagated
    fmax: same behavior as the C99 fmax function
	
    """
    try:
        amax = a.max
    except AttributeError:
        return _wrapit(a, 'max', axis, out)
    return amax(axis, out)


def amin(a, axis=None, out=None):
    """
    Return the minimum along an axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : int, optional
        Axis along which to operate.  By default a flattened input is used.
    out : ndarray, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.

    Returns
    -------
    amin : ndarray
        A new array or a scalar with the result, or a reference to `out` if it
        was specified.

    Examples
    --------
    >>> a = np.arange(4).reshape((2,2))
    >>> a
    array([[0, 1],
           [2, 3]])
    >>> np.amin(a)           # Minimum of the flattened array
    0
    >>> np.amin(a, axis=0)         # Minima along the first axis
    array([0, 1])
    >>> np.amin(a, axis=1)         # Minima along the second axis
    array([0, 2])

    Notes
    -----
    NaN values are propagated, that is if at least one item is nan, the
    corresponding min value will be nan as well. To ignore NaN values (matlab
    behavior), please use nanmin.

    See Also
    --------
    nanmin: nan values are ignored instead of being propagated
    fmin: same behavior as the C99 fmin function
    """
    try:
        amin = a.min
    except AttributeError:
        return _wrapit(a, 'min', axis, out)
    return amin(axis, out)


def alen(a):
    """
    Return the length of the first dimension of the input array.

    Parameters
    ----------
    a : array_like
       Input array.

    Returns
    -------
    alen : int
       Length of the first dimension of `a`.

    See Also
    --------
    shape

    Examples
    --------
    >>> a = np.zeros((7,4,5))
    >>> a.shape[0]
    7
    >>> np.alen(a)
    7

    """
    try:
        return len(a)
    except TypeError:
        return len(array(a,ndmin=1))


def prod(a, axis=None, dtype=None, out=None):
    """
    Return the product of array elements over a given axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : int, optional
        Axis over which the product is taken.  By default, the product
        of all elements is calculated.
    dtype : data-type, optional
        The data-type of the returned array, as well as of the accumulator
        in which the elements are multiplied.  By default, if `a` is of
        integer type, `dtype` is the default platform integer. (Note: if
        the type of `a` is unsigned, then so is `dtype`.)  Otherwise,
        the dtype is the same as that of `a`.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output, but the type of the
        output values will be cast if necessary.

    Returns
    -------
    product_along_axis : ndarray, see `dtype` parameter above.
        An array shaped as `a` but with the specified axis removed.
        Returns a reference to `out` if specified.

    See Also
    --------
    ndarray.prod : equivalent method

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.  That means that, on a 32-bit platform:

    >>> x = np.array([536870910, 536870910, 536870910, 536870910])
    >>> np.prod(x) #random
    16

    Examples
    --------
    By default, calculate the product of all elements:

    >>> np.prod([1.,2.])
    2.0

    Even when the input array is two-dimensional:

    >>> np.prod([[1.,2.],[3.,4.]])
    24.0

    But we can also specify the axis over which to multiply:

    >>> np.prod([[1.,2.],[3.,4.]], axis=1)
    array([  2.,  12.])

    If the type of `x` is unsigned, then the output type is
    the unsigned platform integer:

    >>> x = np.array([1, 2, 3], dtype=np.uint8)
    >>> np.prod(x).dtype == np.uint
    True

    If `x` is of a signed integer type, then the output type
    is the default platform integer:

    >>> x = np.array([1, 2, 3], dtype=np.int8)
    >>> np.prod(x).dtype == np.int
    True

    """
    try:
        prod = a.prod
    except AttributeError:
        return _wrapit(a, 'prod', axis, dtype, out)
    return prod(axis, dtype, out)


def cumprod(a, axis=None, dtype=None, out=None):
    """
    Return the cumulative product of elements along a given axis.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        Axis along which the cumulative product is computed.  By default the
        input is flattened.
    dtype : dtype, optional
        Type of the returned array, as well as of the accumulator in which
        the elements are multiplied.  If dtype is not specified, it defaults
        to the dtype of `a`, unless `a` has an integer dtype with a precision
        less than that of the default platform integer.  In that case, the
        default platform integer is used instead.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type of the resulting values will be cast if necessary.

    Returns
    -------
    cumprod : ndarray
        A new array holding the result is returned unless `out` is
        specified, in which case a reference to out is returned.

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.

    Examples
    --------
    >>> a = np.array([1,2,3])
    >>> np.cumprod(a) # intermediate results 1, 1*2
    ...               # total product 1*2*3 = 6
    array([1, 2, 6])
    >>> a = np.array([[1, 2, 3], [4, 5, 6]])
    >>> np.cumprod(a, dtype=float) # specify type of output
    array([   1.,    2.,    6.,   24.,  120.,  720.])

    The cumulative product for each column (i.e., over the rows of)
    `a`:

    >>> np.cumprod(a, axis=0)
    array([[ 1,  2,  3],
           [ 4, 10, 18]])

    The cumulative product for each row (i.e. over the columns of)
    `a`:

    >>> np.cumprod(a,axis=1)
    array([[  1,   2,   6],
           [  4,  20, 120]])

    """
    try:
        cumprod = a.cumprod
    except AttributeError:
        return _wrapit(a, 'cumprod', axis, dtype, out)
    return cumprod(axis, dtype, out)


def ndim(a):
    """
    Return the number of dimensions of an array.

    Parameters
    ----------
    a : array_like
        Input array.  If it is not already an ndarray, a conversion is
        attempted.

    Returns
    -------
    number_of_dimensions : int
        The number of dimensions in `a`.  Scalars are zero-dimensional.

    See Also
    --------
    ndarray.ndim : equivalent method
    shape : dimensions of array
    ndarray.shape : dimensions of array

    Examples
    --------
    >>> np.ndim([[1,2,3],[4,5,6]])
    2
    >>> np.ndim(np.array([[1,2,3],[4,5,6]]))
    2
    >>> np.ndim(1)
    0

    """
    try:
        return a.ndim
    except AttributeError:
        return asarray(a).ndim


def rank(a):
    """
    Return the number of dimensions of an array.

    If `a` is not already an array, a conversion is attempted.
    Scalars are zero dimensional.

    Parameters
    ----------
    a : array_like
        Array whose number of dimensions is desired. If `a` is not an array,
        a conversion is attempted.

    Returns
    -------
    number_of_dimensions : int
        The number of dimensions in the array.

    See Also
    --------
    ndim : equivalent function
    ndarray.ndim : equivalent property
    shape : dimensions of array
    ndarray.shape : dimensions of array

    Notes
    -----
    In the old Numeric package, `rank` was the term used for the number of
    dimensions, but in Numpy `ndim` is used instead.

    Examples
    --------
    >>> np.rank([1,2,3])
    1
    >>> np.rank(np.array([[1,2,3],[4,5,6]]))
    2
    >>> np.rank(1)
    0

    """
    try:
        return a.ndim
    except AttributeError:
        return asarray(a).ndim


def size(a, axis=None):
    """
    Return the number of elements along a given axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : int, optional
        Axis along which the elements are counted.  By default, give
        the total number of elements.

    Returns
    -------
    element_count : int
        Number of elements along the specified axis.

    See Also
    --------
    shape : dimensions of array
    ndarray.shape : dimensions of array
    ndarray.size : number of elements in array

    Examples
    --------
    >>> a = np.array([[1,2,3],[4,5,6]])
    >>> np.size(a)
    6
    >>> np.size(a,1)
    3
    >>> np.size(a,0)
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
    """
    Evenly round to the given number of decimals.

    Parameters
    ----------
    a : array_like
        Input data.
    decimals : int, optional
        Number of decimal places to round to (default: 0).  If
        decimals is negative, it specifies the number of positions to
        the left of the decimal point.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output, but the type of the output
        values will be cast if necessary.

    Returns
    -------
    rounded_array : ndarray
        An array of the same type as `a`, containing the rounded values.
        Unless `out` was specified, a new array is created.  A reference to
        the result is returned.

        The real and imaginary parts of complex numbers are rounded
        separately.  The result of rounding a float is a float.

    See Also
    --------
    ndarray.round : equivalent method

    Notes
    -----
    For values exactly halfway between rounded decimal values, Numpy
    rounds to the nearest even value. Thus 1.5 and 2.5 round to 2.0,
    -0.5 and 0.5 round to 0.0, etc. Results may also be surprising due
    to the inexact representation of decimal fractions in the IEEE
    floating point standard [1]_ and errors introduced when scaling
    by powers of ten.

    References
    ----------
    .. [1] "Lecture Notes on the Status of  IEEE 754", William Kahan,
           http://www.cs.berkeley.edu/~wkahan/ieee754status/IEEE754.PDF
    .. [2] "How Futile are Mindless Assessments of
           Roundoff in Floating-Point Computation?", William Kahan,
           http://www.cs.berkeley.edu/~wkahan/Mindless.pdf

    Examples
    --------
    >>> np.around([.5, 1.5, 2.5, 3.5, 4.5])
    array([ 0.,  2.,  2.,  4.,  4.])
    >>> np.around([1,2,3,11], decimals=1)
    array([ 1,  2,  3, 11])
    >>> np.around([1,2,3,11], decimals=-1)
    array([ 0,  0,  0, 10])

    """
    try:
        round = a.round
    except AttributeError:
        return _wrapit(a, 'round', decimals, out)
    return round(decimals, out)


def round_(a, decimals=0, out=None):
    """
    Round an array to the given number of decimals.

    Refer to `around` for full documentation.

    See Also
    --------
    around : equivalent function

    """
    try:
        round = a.round
    except AttributeError:
        return _wrapit(a, 'round', decimals, out)
    return round(decimals, out)


def mean(a, axis=None, dtype=None, out=None):
    """
    Compute the arithmetic mean along the specified axis.

    Returns the average of the array elements.  The average is taken
    over the flattened array by default, otherwise over the specified
    axis. float64 intermediate and return values are used for integer
    inputs.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose mean is desired. If `a` is not an
        array, a conversion is attempted.
    axis : int, optional
        Axis along which the means are computed. The default is to compute
        the mean of the flattened array.
    dtype : dtype, optional
        Type to use in computing the mean. For integer inputs, the default
        is float64; for floating point, inputs it is the same as the input
        dtype.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type will be cast if
        necessary.

    Returns
    -------
    mean : ndarray, see dtype parameter above
        If `out=None`, returns a new array containing the mean values,
        otherwise a reference to the output array is returned.

    See Also
    --------
    average : Weighted average

    Notes
    -----
    The arithmetic mean is the sum of the elements along the axis divided
    by the number of elements.

    Examples
    --------
    >>> a = np.array([[1,2],[3,4]])
    >>> np.mean(a)
    2.5
    >>> np.mean(a,0)
    array([ 2.,  3.])
    >>> np.mean(a,1)
    array([ 1.5,  3.5])

    """
    try:
        mean = a.mean
    except AttributeError:
        return _wrapit(a, 'mean', axis, dtype, out)
    return mean(axis, dtype, out)


def std(a, axis=None, dtype=None, out=None, ddof=0):
    """
    Compute the standard deviation along the specified axis.

    Returns the standard deviation, a measure of the spread of a distribution,
    of the array elements. The standard deviation is computed for the
    flattened array by default, otherwise over the specified axis.

    Parameters
    ----------
    a : array_like
        Calculate the standard deviation of these values.
    axis : int, optional
        Axis along which the standard deviation is computed. The default is
        to compute the standard deviation of the flattened array.
    dtype : dtype, optional
        Type to use in computing the standard deviation. For arrays of
        integer type the default is float64, for arrays of float types it is
        the same as the array type.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the calculated
        values) will be cast if necessary.
    ddof : int, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
        By default `ddof` is zero (biased estimate).

    Returns
    -------
    standard_deviation : {ndarray, scalar}; see dtype parameter above.
        If `out` is None, return a new array containing the standard deviation,
        otherwise return a reference to the output array.

    See Also
    --------
    numpy.var : Variance
    numpy.mean : Average

    Notes
    -----
    The standard deviation is the square root of the average of the squared
    deviations from the mean, i.e., ``var = sqrt(mean(abs(x - x.mean())**2))``.

    The mean is normally calculated as ``x.sum() / N``, where
    ``N = len(x)``.  If, however, `ddof` is specified, the divisor ``N - ddof``
    is used instead.

    Note that, for complex numbers, std takes the absolute
    value before squaring, so that the result is always real and nonnegative.

    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.std(a)
    1.1180339887498949
    >>> np.std(a, 0)
    array([ 1.,  1.])
    >>> np.std(a, 1)
    array([ 0.5,  0.5])

    """
    try:
        std = a.std
    except AttributeError:
        return _wrapit(a, 'std', axis, dtype, out, ddof)
    return std(axis, dtype, out, ddof)


def var(a, axis=None, dtype=None, out=None, ddof=0):
    """
    Compute the variance along the specified axis.

    Returns the variance of the array elements, a measure of the spread of a
    distribution. The variance is computed for the flattened array by default,
    otherwise over the specified axis.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose variance is desired. If `a` is not an
        array, a conversion is attempted.
    axis : int, optional
        Axis along which the variance is computed. The default is to compute
        the variance of the flattened array.
    dtype : dtype, optional
        Type to use in computing the variance. For arrays of integer type
        the default is float32; for arrays of float types it is the same as
        the array type.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type is cast if
        necessary.
    ddof : int, optional
        "Delta Degrees of Freedom": the divisor used in calculation is
        ``N - ddof``, where ``N`` represents the number of elements. By
        default `ddof` is zero.

    Returns
    -------
    variance : ndarray, see dtype parameter above
        If out=None, returns a new array containing the variance; otherwise
        a reference to the output array is returned.

    See Also
    --------
    std : Standard deviation
    mean : Average

    Notes
    -----
    The variance is the average of the squared deviations from the mean,
    i.e.,  ``var = mean(abs(x - x.mean())**2)``.

    The mean is normally calculated as ``x.sum() / N``, where ``N = len(x)``.
    If, however, `ddof` is specified, the divisor ``N - ddof`` is used
    instead. In standard statistical practice, ``ddof=1`` provides an
    unbiased estimator of the variance of the infinite population. ``ddof=0``
    provides a maximum likelihood estimate of the variance for normally
    distributed variables.

    Note that for complex numbers, the absolute value is taken before
    squaring, so that the result is always real and nonnegative.

    Examples
    --------
    >>> a = np.array([[1,2],[3,4]])
    >>> np.var(a)
    1.25
    >>> np.var(a,0)
    array([ 1.,  1.])
    >>> np.var(a,1)
    array([ 0.25,  0.25])

    """
    try:
        var = a.var
    except AttributeError:
        return _wrapit(a, 'var', axis, dtype, out, ddof)
    return var(axis, dtype, out, ddof)
