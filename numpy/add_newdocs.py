# This is only meant to add docs to objects defined in C-extension modules.
# The purpose is to allow easier editing of the docstrings without
# requiring a re-compile.

# NOTE: Many of the methods of ndarray have corresponding functions.
#       If you update these docstrings, please keep also the ones in
#       core/fromnumeric.py, core/defmatrix.py up-to-date.

from lib import add_newdoc

###############################################################################
#
# flatiter
#
# flatiter needs a toplevel description
#
###############################################################################

add_newdoc('numpy.core', 'flatiter',
    """
    Flat iterator object to iterate over arrays.

    A `flatiter` iterator is returned by ``x.flat`` for any array `x`.
    It allows iterating over the array as if it were a 1-D array,
    either in a for-loop or by calling its `next` method.

    Iteration is done in C-contiguous style, with the last index varying the
    fastest. The iterator can also be indexed using basic slicing or
    advanced indexing.

    See Also
    --------
    ndarray.flat : Return a flat iterator over an array.
    ndarray.flatten : Returns a flattened copy of an array.

    Notes
    -----
    A `flatiter` iterator can not be constructed directly from Python code
    by calling the `flatiter` constructor.

    Examples
    --------
    >>> x = np.arange(6).reshape(2, 3)
    >>> fl = x.flat
    >>> type(fl)
    <type 'numpy.flatiter'>
    >>> for item in fl:
    ...     print item
    ...
    0
    1
    2
    3
    4
    5

    >>> fl[2:4]
    array([2, 3])

    """)

# flatiter attributes

add_newdoc('numpy.core', 'flatiter', ('base',
    """documentation needed

    """))



add_newdoc('numpy.core', 'flatiter', ('coords',
    """An N-d tuple of current coordinates.

    """))



add_newdoc('numpy.core', 'flatiter', ('index',
    """documentation needed

    """))

# flatiter functions

add_newdoc('numpy.core', 'flatiter', ('__array__',
    """__array__(type=None) Get array from iterator

    """))


add_newdoc('numpy.core', 'flatiter', ('copy',
    """copy() Get a copy of the iterator as a 1-d array

    """))


###############################################################################
#
# broadcast
#
###############################################################################

add_newdoc('numpy.core', 'broadcast',
    """
    Produce an object that mimics broadcasting.

    Parameters
    ----------
    in1, in2, ... : array_like
        Input parameters.

    Returns
    -------
    b : broadcast object
        Broadcast the input parameters against one another, and
        return an object that encapsulates the result.
        Amongst others, it has ``shape`` and ``nd`` properties, and
        may be used as an iterator.

    Examples
    --------
    Manually adding two vectors, using broadcasting:

    >>> x = np.array([[1], [2], [3]])
    >>> y = np.array([4, 5, 6])
    >>> b = np.broadcast(x, y)

    >>> out = np.empty(b.shape)
    >>> out.flat = [u+v for (u,v) in b]
    >>> out
    array([[ 5.,  6.,  7.],
           [ 6.,  7.,  8.],
           [ 7.,  8.,  9.]])

    Compare against built-in broadcasting:

    >>> x + y
    array([[5, 6, 7],
           [6, 7, 8],
           [7, 8, 9]])

    """)

# attributes

add_newdoc('numpy.core', 'broadcast', ('index',
    """current index in broadcasted result

    """))

add_newdoc('numpy.core', 'broadcast', ('iters',
    """tuple of individual iterators

    """))

add_newdoc('numpy.core', 'broadcast', ('nd',
    """number of dimensions of broadcasted result

    """))

add_newdoc('numpy.core', 'broadcast', ('numiter',
    """number of iterators

    """))

add_newdoc('numpy.core', 'broadcast', ('shape',
    """shape of broadcasted result

    """))

add_newdoc('numpy.core', 'broadcast', ('size',
    """total size of broadcasted result

    """))


###############################################################################
#
# numpy functions
#
###############################################################################

add_newdoc('numpy.core.multiarray', 'array',
    """
    array(object, dtype=None, copy=True, order=None, subok=False, ndmin=True)

    Create an array.

    Parameters
    ----------
    object : array_like
        An array, any object exposing the array interface, an
        object whose __array__ method returns an array, or any
        (nested) sequence.
    dtype : data-type, optional
        The desired data-type for the array.  If not given, then
        the type will be determined as the minimum type required
        to hold the objects in the sequence.  This argument can only
        be used to 'upcast' the array.  For downcasting, use the
        .astype(t) method.
    copy : bool, optional
        If true (default), then the object is copied.  Otherwise, a copy
        will only be made if __array__ returns a copy, if obj is a
        nested sequence, or if a copy is needed to satisfy any of the other
        requirements (`dtype`, `order`, etc.).
    order : {'C', 'F', 'A'}, optional
        Specify the order of the array.  If order is 'C' (default), then the
        array will be in C-contiguous order (last-index varies the
        fastest).  If order is 'F', then the returned array
        will be in Fortran-contiguous order (first-index varies the
        fastest).  If order is 'A', then the returned array may
        be in any order (either C-, Fortran-contiguous, or even
        discontiguous).
    subok : bool, optional
        If True, then sub-classes will be passed-through, otherwise
        the returned array will be forced to be a base-class array (default).
    ndmin : int, optional
        Specifies the minimum number of dimensions that the resulting
        array should have.  Ones will be pre-pended to the shape as
        needed to meet this requirement.

    Examples
    --------
    >>> np.array([1, 2, 3])
    array([1, 2, 3])

    Upcasting:

    >>> np.array([1, 2, 3.0])
    array([ 1.,  2.,  3.])

    More than one dimension:

    >>> np.array([[1, 2], [3, 4]])
    array([[1, 2],
           [3, 4]])

    Minimum dimensions 2:

    >>> np.array([1, 2, 3], ndmin=2)
    array([[1, 2, 3]])

    Type provided:

    >>> np.array([1, 2, 3], dtype=complex)
    array([ 1.+0.j,  2.+0.j,  3.+0.j])

    Data-type consisting of more than one element:

    >>> x = np.array([(1,2),(3,4)],dtype=[('a','<i4'),('b','<i4')])
    >>> x['a']
    array([1, 3])

    Creating an array from sub-classes:

    >>> np.array(np.mat('1 2; 3 4'))
    array([[1, 2],
           [3, 4]])

    >>> np.array(np.mat('1 2; 3 4'), subok=True)
    matrix([[1, 2],
            [3, 4]])

    """)

add_newdoc('numpy.core.multiarray', 'empty',
    """
    empty(shape, dtype=float, order='C')

    Return a new array of given shape and type, without initializing entries.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty array
    dtype : data-type, optional
        Desired output data-type.
    order : {'C', 'F'}, optional
        Whether to store multi-dimensional data in C (row-major) or
        Fortran (column-major) order in memory.

    See Also
    --------
    empty_like, zeros, ones

    Notes
    -----
    `empty`, unlike `zeros`, does not set the array values to zero,
    and may therefore be marginally faster.  On the other hand, it requires
    the user to manually set all the values in the array, and should be
    used with caution.

    Examples
    --------
    >>> np.empty([2, 2])
    array([[ -9.74499359e+001,   6.69583040e-309],  #random data
           [  2.13182611e-314,   3.06959433e-309]])

    >>> np.empty([2, 2], dtype=int)
    array([[-1073741821, -1067949133],  #random data
           [  496041986,    19249760]])

    """)


add_newdoc('numpy.core.multiarray','scalar',
    """scalar(dtype,obj)

    Return a new scalar array of the given type initialized with
    obj. Mainly for pickle support.  The dtype must be a valid data-type
    descriptor.  If dtype corresponds to an OBJECT descriptor, then obj
    can be any object, otherwise obj must be a string. If obj is not given
    it will be interpreted as None for object type and zeros for all other
    types.

    """)

add_newdoc('numpy.core.multiarray', 'zeros',
    """
    zeros(shape, dtype=float, order='C')

    Return a new array of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : {tuple of ints, int}
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory.

    Returns
    -------
    out : ndarray
        Array of zeros with the given shape, dtype, and order.

    See Also
    --------
    numpy.zeros_like : Return an array of zeros with shape and type of input.
    numpy.ones_like : Return an array of ones with shape and type of input.
    numpy.empty_like : Return an empty array with shape and type of input.
    numpy.ones : Return a new array setting values to one.
    numpy.empty : Return a new uninitialized array.

    Examples
    --------
    >>> np.zeros(5)
    array([ 0.,  0.,  0.,  0.,  0.])

    >>> np.zeros((5,), dtype=numpy.int)
    array([0, 0, 0, 0, 0])

    >>> np.zeros((2, 1))
    array([[ 0.],
           [ 0.]])

    >>> s = (2,2)
    >>> np.zeros(s)
    array([[ 0.,  0.],
           [ 0.,  0.]])

    >>> np.zeros((2,), dtype=[('x', 'i4'), ('y', 'i4')])
    array([(0, 0), (0, 0)],
          dtype=[('x', '<i4'), ('y', '<i4')])

    """)

add_newdoc('numpy.core.multiarray','set_typeDict',
    """set_typeDict(dict)

    Set the internal dictionary that can look up an array type using a
    registered code.

    """)

add_newdoc('numpy.core.multiarray', 'fromstring',
    """
    fromstring(string, dtype=float, count=-1, sep='')

    Return a new 1d array initialized from raw binary or text data in
    string.

    Parameters
    ----------
    string : str
        A string containing the data.
    dtype : dtype, optional
        The data type of the array. For binary input data, the data must be
        in exactly this format.
    count : int, optional
        Read this number of `dtype` elements from the data. If this is
        negative, then the size will be determined from the length of the
        data.
    sep : str, optional
        If provided and not empty, then the data will be interpreted as
        ASCII text with decimal numbers. This argument is interpreted as the
        string separating numbers in the data. Extra whitespace between
        elements is also ignored.

    Returns
    -------
    arr : array
        The constructed array.

    Raises
    ------
    ValueError
        If the string is not the correct size to satisfy the requested
        `dtype` and `count`.

    Examples
    --------
    >>> np.fromstring('\\x01\\x02', dtype=np.uint8)
    array([1, 2], dtype=uint8)
    >>> np.fromstring('1 2', dtype=int, sep=' ')
    array([1, 2])
    >>> np.fromstring('1, 2', dtype=int, sep=',')
    array([1, 2])
    >>> np.fromstring('\\x01\\x02\\x03\\x04\\x05', dtype=np.uint8, count=3)
    array([1, 2, 3], dtype=uint8)

    Invalid inputs:

    >>> np.fromstring('\\x01\\x02\\x03\\x04\\x05', dtype=np.int32)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: string size must be a multiple of element size
    >>> np.fromstring('\\x01\\x02', dtype=np.uint8, count=3)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: string is smaller than requested size

    """)

add_newdoc('numpy.core.multiarray', 'fromiter',
    """
    fromiter(iterable, dtype, count=-1)

    Create a new 1-dimensional array from an iterable object.

    Parameters
    ----------
    iterable : iterable object
        An iterable object providing data for the array.
    dtype : data-type
        The data type of the returned array.
    count : int, optional
        The number of items to read from iterable. The default is -1,
        which means all data is read.

    Returns
    -------
    out : ndarray
        The output array.

    Notes
    -----
    Specify ``count`` to improve performance.  It allows
    ``fromiter`` to pre-allocate the output array, instead of
    resizing it on demand.

    Examples
    --------
    >>> iterable = (x*x for x in range(5))
    >>> np.fromiter(iterable, np.float)
    array([  0.,   1.,   4.,   9.,  16.])

    """)

add_newdoc('numpy.core.multiarray', 'fromfile',
    """
    fromfile(file, dtype=float, count=-1, sep='')

    Construct an array from data in a text or binary file.

    A highly efficient way of reading binary data with a known data-type,
    as well as parsing simply formatted text files.  Data written using the
    `tofile` method can be read using this function.

    Parameters
    ----------
    file : file or string
        Open file object or filename.
    dtype : data-type
        Data type of the returned array.
        For binary files, it is used to determine the size and byte-order
        of the items in the file.
    count : int
        Number of items to read. ``-1`` means all items (i.e., the complete
        file).
    sep : string
        Separator between items if file is a text file.
        Empty ("") separator means the file should be treated as binary.
        Spaces (" ") in the separator match zero or more whitespace characters.
        A separator consisting only of spaces must match at least one
        whitespace.

    See also
    --------
    load, save
    ndarray.tofile
    loadtxt : More flexible way of loading data from a text file.

    Notes
    -----
    Do not rely on the combination of `tofile` and `fromfile` for
    data storage, as the binary files generated are are not platform
    independent.  In particular, no byte-order or data-type information is
    saved.  Data can be stored in the platform independent ``.npy`` format
    using `save` and `load` instead.

    Examples
    --------
    Construct an ndarray:

    >>> dt = np.dtype([('time', [('min', int), ('sec', int)]),
    ...                ('temp', float)])
    >>> x = np.zeros((1,), dtype=dt)
    >>> x['time']['min'] = 10; x['temp'] = 98.25
    >>> x
    array([((10, 0), 98.25)],
          dtype=[('time', [('min', '<i4'), ('sec', '<i4')]), ('temp', '<f8')])

    Save the raw data to disk:

    >>> import os
    >>> fname = os.tmpnam()
    >>> x.tofile(fname)

    Read the raw data from disk:

    >>> np.fromfile(fname, dtype=dt)
    array([((10, 0), 98.25)],
          dtype=[('time', [('min', '<i4'), ('sec', '<i4')]), ('temp', '<f8')])

    The recommended way to store and load data:

    >>> np.save(fname, x)
    >>> np.load(fname + '.npy')
    array([((10, 0), 98.25)],
          dtype=[('time', [('min', '<i4'), ('sec', '<i4')]), ('temp', '<f8')])

    """)

add_newdoc('numpy.core.multiarray', 'frombuffer',
    """
    frombuffer(buffer, dtype=float, count=-1, offset=0)

    Interpret a buffer as a 1-dimensional array.

    Parameters
    ----------
    buffer
        An object that exposes the buffer interface.
    dtype : data-type, optional
        Data type of the returned array.
    count : int, optional
        Number of items to read. ``-1`` means all data in the buffer.
    offset : int, optional
        Start reading the buffer from this offset.

    Notes
    -----
    If the buffer has data that is not in machine byte-order, this
    should be specified as part of the data-type, e.g.::

      >>> dt = np.dtype(int)
      >>> dt = dt.newbyteorder('>')
      >>> np.frombuffer(buf, dtype=dt)

    The data of the resulting array will not be byteswapped,
    but will be interpreted correctly.

    Examples
    --------
    >>> s = 'hello world'
    >>> np.frombuffer(s, dtype='S1', count=5, offset=6)
    array(['w', 'o', 'r', 'l', 'd'],
          dtype='|S1')

    """)

add_newdoc('numpy.core.multiarray', 'concatenate',
    """
    concatenate((a1, a2, ...), axis=0)

    Join a sequence of arrays together.

    Parameters
    ----------
    a1, a2, ... : sequence of ndarrays
        The arrays must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default).
    axis : int, optional
        The axis along which the arrays will be joined.  Default is 0.

    Returns
    -------
    res : ndarray
        The concatenated array.

    See Also
    --------
    array_split : Split an array into multiple sub-arrays of equal or
                  near-equal size.
    split : Split array into a list of multiple sub-arrays of equal size.
    hsplit : Split array into multiple sub-arrays horizontally (column wise)
    vsplit : Split array into multiple sub-arrays vertically (row wise)
    dsplit : Split array into multiple sub-arrays along the 3rd axis (depth).
    hstack : Stack arrays in sequence horizontally (column wise)
    vstack : Stack arrays in sequence vertically (row wise)
    dstack : Stack arrays in sequence depth wise (along third dimension)

    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> b = np.array([[5, 6]])
    >>> np.concatenate((a, b), axis=0)
    array([[1, 2],
           [3, 4],
           [5, 6]])
    >>> np.concatenate((a, b.T), axis=1)
    array([[1, 2, 5],
           [3, 4, 6]])

    """)

add_newdoc('numpy.core', 'inner',
    """
    inner(a, b)

    Inner product of two arrays.

    Ordinary inner product of vectors for 1-D arrays (without complex
    conjugation), in higher dimensions a sum product over the last axes.

    Parameters
    ----------
    a, b : array_like
        If `a` and `b` are nonscalar, their last dimensions of must match.

    Returns
    -------
    out : ndarray
        `out.shape = a.shape[:-1] + b.shape[:-1]`

    Raises
    ------
    ValueError
        If the last dimension of `a` and `b` has different size.

    See Also
    --------
    tensordot : Sum products over arbitrary axes.
    dot : Generalised matrix product, using second last dimension of `b`.

    Notes
    -----
    For vectors (1-D arrays) it computes the ordinary inner-product::

        np.inner(a, b) = sum(a[:]*b[:])

    More generally, if `ndim(a) = r > 0` and `ndim(b) = s > 0`::

        np.inner(a, b) = np.tensordot(a, b, axes=(-1,-1))

    or explicitly::

        np.inner(a, b)[i0,...,ir-1,j0,...,js-1]
             = sum(a[i0,...,ir-1,:]*b[j0,...,js-1,:])

    In addition `a` or `b` may be scalars, in which case::

       np.inner(a,b) = a*b

    Examples
    --------
    Ordinary inner product for vectors:

    >>> a = np.array([1,2,3])
    >>> b = np.array([0,1,0])
    >>> np.inner(a, b)
    2

    A multidimensional example:

    >>> a = np.arange(24).reshape((2,3,4))
    >>> b = np.arange(4)
    >>> np.inner(a, b)
    array([[ 14,  38,  62],
           [ 86, 110, 134]])

    An example where `b` is a scalar:

    >>> np.inner(np.eye(2), 7)
    array([[ 7.,  0.],
           [ 0.,  7.]])

    """)

add_newdoc('numpy.core','fastCopyAndTranspose',
    """_fastCopyAndTranspose(a)""")

add_newdoc('numpy.core.multiarray','correlate',
    """cross_correlate(a,v, mode=0)""")

add_newdoc('numpy.core.multiarray', 'arange',
    """
    arange([start,] stop[, step,], dtype=None)

    Return evenly spaced values within a given interval.

    Values are generated within the half-open interval ``[start, stop)``
    (in other words, the interval including `start` but excluding `stop`).
    For integer arguments the function is equivalent to the Python built-in
    `range <http://docs.python.org/lib/built-in-funcs.html>`_ function,
    but returns a ndarray rather than a list.

    Parameters
    ----------
    start : number, optional
        Start of interval.  The interval includes this value.  The default
        start value is 0.
    stop : number
        End of interval.  The interval does not include this value.
    step : number, optional
        Spacing between values.  For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.  The default
        step size is 1.  If `step` is specified, `start` must also be given.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.

    Returns
    -------
    out : ndarray
        Array of evenly spaced values.

        For floating point arguments, the length of the result is
        ``ceil((stop - start)/step)``.  Because of floating point overflow,
        this rule may result in the last element of `out` being greater
        than `stop`.

    See Also
    --------
    linspace : Evenly spaced numbers with careful handling of endpoints.
    ogrid: Arrays of evenly spaced numbers in N-dimensions
    mgrid: Grid-shaped arrays of evenly spaced numbers in N-dimensions

    Examples
    --------
    >>> np.arange(3)
    array([0, 1, 2])
    >>> np.arange(3.0)
    array([ 0.,  1.,  2.])
    >>> np.arange(3,7)
    array([3, 4, 5, 6])
    >>> np.arange(3,7,2)
    array([3, 5])

    """)

add_newdoc('numpy.core.multiarray','_get_ndarray_c_version',
    """_get_ndarray_c_version()

    Return the compile time NDARRAY_VERSION number.

    """)

add_newdoc('numpy.core.multiarray','_reconstruct',
    """_reconstruct(subtype, shape, dtype)

    Construct an empty array. Used by Pickles.

    """)


add_newdoc('numpy.core.multiarray', 'set_string_function',
    """
    set_string_function(f, repr=1)

    Set a Python function to be used when pretty printing arrays.

    Parameters
    ----------
    f : function or None
        Function to be used to pretty print arrays. The function should expect
        a single array argument and return a string of the representation of
        the array. If None, the function is reset to the default NumPy function
        to print arrays.
    repr : bool, optional
        If True (default), the function for pretty printing (``__repr__``)
        is set, if False the function that returns the default string
        representation (``__str__``) is set.

    See Also
    --------
    set_printoptions, get_printoptions

    Examples
    --------
    >>> def pprint(arr):
    ...     return 'HA! - What are you going to do now?'
    ...
    >>> np.set_string_function(pprint)
    >>> a = np.arange(10)
    >>> a
    HA! - What are you going to do now?
    >>> print a
    [0 1 2 3 4 5 6 7 8 9]

    We can reset the function to the default:

    >>> np.set_string_function(None)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'l')

    `repr` affects either pretty printing or normal string representation.
    Note that ``__repr__`` is still affected by setting ``__str__``
    because the width of each array element in the returned string becomes
    equal to the length of the result of ``__str__()``.

    >>> x = np.arange(4)
    >>> np.set_string_function(lambda x:'random', repr=False)
    >>> x.__str__()
    'random'
    >>> x.__repr__()
    'array([     0,      1,      2,      3])'

    """)

add_newdoc('numpy.core.multiarray', 'set_numeric_ops',
    """
    set_numeric_ops(op1=func1, op2=func2, ...)

    Set numerical operators for array objects.

    Parameters
    ----------
    op1, op2, ... : callable
        Each ``op = func`` pair describes an operator to be replaced.
        For example, ``add = lambda x, y: np.add(x, y) % 5`` would replace
        addition by modulus 5 addition.

    Returns
    -------
    saved_ops : list of callables
        A list of all operators, stored before making replacements.

    Notes
    -----
    .. WARNING::
       Use with care!  Incorrect usage may lead to memory errors.

    A function replacing an operator cannot make use of that operator.
    For example, when replacing add, you may not use ``+``.  Instead,
    directly call ufuncs:

    >>> def add_mod5(x, y):
    ...     return np.add(x, y) % 5
    ...
    >>> old_funcs = np.set_numeric_ops(add=add_mod5)

    >>> x = np.arange(12).reshape((3, 4))
    >>> x + x
    array([[0, 2, 4, 1],
           [3, 0, 2, 4],
           [1, 3, 0, 2]])

    >>> ignore = np.set_numeric_ops(**old_funcs) # restore operators

    """)

add_newdoc('numpy.core.multiarray', 'where',
    """
    where(condition, [x, y])

    Return elements, either from `x` or `y`, depending on `condition`.

    If only `condition` is given, return ``condition.nonzero()``.

    Parameters
    ----------
    condition : array_like, bool
        When True, yield `x`, otherwise yield `y`.
    x, y : array_like, optional
        Values from which to choose.

    Returns
    -------
    out : ndarray or tuple of ndarrays
        If both `x` and `y` are specified, the output array, shaped like
        `condition`, contains elements of `x` where `condition` is True,
        and elements from `y` elsewhere.

        If only `condition` is given, return the tuple
        ``condition.nonzero()``, the indices where `condition` is True.

    See Also
    --------
    nonzero, choose

    Notes
    -----
    If `x` and `y` are given and input arrays are 1-D, `where` is
    equivalent to::

        [xv if c else yv for (c,xv,yv) in zip(condition,x,y)]

    Examples
    --------
    >>> x = np.arange(9.).reshape(3, 3)
    >>> np.where( x > 5 )
    (array([2, 2, 2]), array([0, 1, 2]))
    >>> x[np.where( x > 3.0 )]               # Note: result is 1D.
    array([ 4.,  5.,  6.,  7.,  8.])
    >>> np.where(x < 5, x, -1)               # Note: broadcasting.
    array([[ 0.,  1.,  2.],
           [ 3.,  4., -1.],
           [-1., -1., -1.]])

    >>> np.where([[True, False], [True, True]],
    ...          [[1, 2], [3, 4]],
    ...          [[9, 8], [7, 6]])
    array([[1, 8],
           [3, 4]])

    >>> np.where([[0, 1], [1, 0]])
    (array([0, 1]), array([1, 0]))

    """)


add_newdoc('numpy.core.multiarray', 'lexsort',
    """
    lexsort(keys, axis=-1)

    Perform an indirect sort using a sequence of keys.

    Given multiple sorting keys, which can be interpreted as columns in a
    spreadsheet, lexsort returns an array of integer indices that describes
    the sort order by multiple columns. The last key in the sequence is used
    for the primary sort order, the second-to-last key for the secondary sort
    order, and so on. The keys argument must be a sequence of objects that
    can be converted to arrays of the same shape. If a 2D array is provided
    for the keys argument, it's rows are interpreted as the sorting keys and
    sorting is according to the last row, second last row etc.

    Parameters
    ----------
    keys : (k,N) array or tuple containing k (N,)-shaped sequences
        The `k` different "columns" to be sorted.  The last column (or row if
        `keys` is a 2D array) is the primary sort key.
    axis : int, optional
        Axis to be indirectly sorted.  By default, sort over the last axis.

    Returns
    -------
    indices : (N,) ndarray of ints
        Array of indices that sort the keys along the specified axis.

    See Also
    --------
    argsort : Indirect sort.
    ndarray.sort : In-place sort.
    sort : Return a sorted copy of an array.

    Examples
    --------
    Sort names: first by surname, then by name.

    >>> surnames =    ('Hertz',    'Galilei', 'Hertz')
    >>> first_names = ('Heinrich', 'Galileo', 'Gustav')
    >>> ind = np.lexsort((first_names, surnames))
    >>> ind
    array([1, 2, 0])

    >>> [surnames[i] + ", " + first_names[i] for i in ind]
    ['Galilei, Galileo', 'Hertz, Gustav', 'Hertz, Heinrich']

    Sort two columns of numbers:

    >>> a = [1,5,1,4,3,4,4] # First column
    >>> b = [9,4,0,4,0,2,1] # Second column
    >>> ind = np.lexsort((b,a)) # Sort by a, then by b
    >>> print ind
    [2 0 4 6 5 3 1]

    >>> [(a[i],b[i]) for i in ind]
    [(1, 0), (1, 9), (3, 0), (4, 1), (4, 2), (4, 4), (5, 4)]

    Note that sorting is first according to the elements of ``a``.
    Secondary sorting is according to the elements of ``b``.

    A normal ``argsort`` would have yielded:

    >>> [(a[i],b[i]) for i in np.argsort(a)]
    [(1, 9), (1, 0), (3, 0), (4, 4), (4, 2), (4, 1), (5, 4)]

    Structured arrays are sorted lexically by ``argsort``:

    >>> x = np.array([(1,9), (5,4), (1,0), (4,4), (3,0), (4,2), (4,1)],
    ...              dtype=np.dtype([('x', int), ('y', int)]))

    >>> np.argsort(x) # or np.argsort(x, order=('x', 'y'))
    array([2, 0, 4, 6, 5, 3, 1])

    """)

add_newdoc('numpy.core.multiarray', 'can_cast',
    """
    can_cast(from=d1, to=d2)

    Returns True if cast between data types can occur without losing precision.

    Parameters
    ----------
    from: data type code
        Data type code to cast from.
    to: data type code
        Data type code to cast to.

    Returns
    -------
    out : bool
        True if cast can occur without losing precision.

    """)

add_newdoc('numpy.core.multiarray','newbuffer',
    """newbuffer(size)

    Return a new uninitialized buffer object of size bytes

    """)

add_newdoc('numpy.core.multiarray', 'getbuffer',
    """
    getbuffer(obj [,offset[, size]])

    Create a buffer object from the given object referencing a slice of
    length size starting at offset.

    Default is the entire buffer. A read-write buffer is attempted followed
    by a read-only buffer.

    Parameters
    ----------
    obj : object

    offset : int, optional

    size : int, optional

    Returns
    -------
    buffer_obj : buffer

    Examples
    --------
    >>> buf = np.getbuffer(np.ones(5), 1, 3)
    >>> len(buf)
    3
    >>> buf[0]
    '\\x00'
    >>> buf
    <read-write buffer for 0x8af1e70, size 3, offset 1 at 0x8ba4ec0>

    """)

add_newdoc('numpy.core', 'dot',
    """
    dot(a, b)

    Dot product of two arrays.

    For 2-D arrays it is equivalent to matrix multiplication, and for 1-D
    arrays to inner product of vectors (without complex conjugation). For
    N dimensions it is a sum product over the last axis of `a` and
    the second-to-last of `b`::

        dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])

    Parameters
    ----------
    a : array_like
        First argument.
    b : array_like
        Second argument.

    Returns
    -------
    output : ndarray
        Returns the dot product of `a` and `b`.  If `a` and `b` are both
        scalars or both 1-D arrays then a scalar is returned; otherwise
        an array is returned.

    Raises
    ------
    ValueError
        If the last dimension of `a` is not the same size as
        the second-to-last dimension of `b`.

    See Also
    --------
    vdot : Complex-conjugating dot product.
    tensordot : Sum products over arbitrary axes.

    Examples
    --------
    >>> np.dot(3, 4)
    12

    Neither argument is complex-conjugated:

    >>> np.dot([2j, 3j], [2j, 3j])
    (-13+0j)

    For 2-D arrays it's the matrix product:

    >>> a = [[1, 0], [0, 1]]
    >>> b = [[4, 1], [2, 2]]
    >>> np.dot(a, b)
    array([[4, 1],
           [2, 2]])

    >>> a = np.arange(3*4*5*6).reshape((3,4,5,6))
    >>> b = np.arange(3*4*5*6)[::-1].reshape((5,4,6,3))
    >>> np.dot(a, b)[2,3,2,1,2,2]
    499128
    >>> sum(a[2,3,2,:] * b[1,2,:,2])
    499128

    """)

add_newdoc('numpy.core', 'alterdot',
    """
    Change `dot`, `vdot`, and `innerproduct` to use accelerated BLAS functions.

    Typically, as a user of Numpy, you do not explicitly call this function. If
    Numpy is built with an accelerated BLAS, this function is automatically
    called when Numpy is imported.

    When Numpy is built with an accelerated BLAS like ATLAS, these functions
    are replaced to make use of the faster implementations.  The faster
    implementations only affect float32, float64, complex64, and complex128
    arrays. Furthermore, the BLAS API only includes matrix-matrix,
    matrix-vector, and vector-vector products. Products of arrays with larger
    dimensionalities use the built in functions and are not accelerated.

    See Also
    --------
    restoredot : `restoredot` undoes the effects of `alterdot`.

    """)

add_newdoc('numpy.core', 'restoredot',
    """
    Restore `dot`, `vdot`, and `innerproduct` to the default non-BLAS
    implementations.

    Typically, the user will only need to call this when troubleshooting and
    installation problem, reproducing the conditions of a build without an
    accelerated BLAS, or when being very careful about benchmarking linear
    algebra operations.

    See Also
    --------
    alterdot : `restoredot` undoes the effects of `alterdot`.

    """)

add_newdoc('numpy.core', 'vdot',
    """
    Return the dot product of two vectors.

    The vdot(`a`, `b`) function handles complex numbers differently than
    dot(`a`, `b`).  If the first argument is complex the complex conjugate
    of the first argument is used for the calculation of the dot product.

    For 2-D arrays it is equivalent to matrix multiplication, and for 1-D
    arrays to inner product of vectors (with complex conjugation of `a`).
    For N dimensions it is a sum product over the last axis of `a` and
    the second-to-last of `b`::

        dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])

    Parameters
    ----------
    a : array_like
        If `a` is complex the complex conjugate is taken before calculation
        of the dot product.
    b : array_like
        Second argument to the dot product.

    Returns
    -------
    output : ndarray
        Returns dot product of `a` and `b`.  Can be an int, float, or
        complex depending on the types of `a` and `b`.

    See Also
    --------
    dot : Return the dot product without using the complex conjugate of the
          first argument.

    Notes
    -----
    The dot product is the summation of element wise multiplication.

    .. math::
     a \\cdot b = \\sum_{i=1}^n a_i^*b_i = a_1^*b_1+a_2^*b_2+\\cdots+a_n^*b_n

    Examples
    --------
    >>> a = np.array([1+2j,3+4j])
    >>> b = np.array([5+6j,7+8j])
    >>> np.vdot(a, b)
    (70-8j)
    >>> np.vdot(b, a)
    (70+8j)
    >>> a = np.array([[1, 4], [5, 6]])
    >>> b = np.array([[4, 1], [2, 2]])
    >>> np.vdot(a, b)
    30
    >>> np.vdot(b, a)
    30

    """)


##############################################################################
#
# Documentation for ndarray attributes and methods
#
##############################################################################


##############################################################################
#
# ndarray object
#
##############################################################################


add_newdoc('numpy.core.multiarray', 'ndarray',
    """
    ndarray(shape, dtype=float, buffer=None, offset=0,
            strides=None, order=None)

    An array object represents a multidimensional, homogeneous array
    of fixed-size items.  An associated data-type object
    describes the format of each element in the array (its byte-order,
    how many bytes it occupies in memory, whether it is an integer or
    a floating point number, etc.).

    Arrays should be constructed using `array`, `zeros` or `empty` (refer to
    the ``See Also`` section below).  The parameters given here describe
    a low-level method for instantiating an array (`ndarray(...)`).

    For more information, refer to the `numpy` module and examine the
    the methods and attributes of an array.

    Attributes
    ----------
    T : ndarray
        Transponent of the array.
    data : buffer
        Array data in memory.
    dtype : data type
        Data type, describing the format of the elements in the array.
    flags : dict
        Dictionary containing information related to memory use, e.g.,
        'C_CONTIGUOUS', 'OWNDATA', 'WRITEABLE', and others.
    flat : ndarray
        Return flattened version of the array as an iterator.  The iterator
        allows assignments, e.g., ``x.flat = 3``.
    imag : ndarray
        Imaginary part of the array.
    real : ndarray
        Real part of the array.
    size : int
        Number of elements in the array.
    itemsize : int
        The size of each element in memory (in bytes).
    nbytes : int
        The total number of bytes required to store the array data,
        i.e., ``itemsize * size``.
    ndim : int
        The number of dimensions that the array has.
    shape : tuple of ints
        Shape of the array.
    strides : tuple of ints
        The step-size required to move from one element to the next in memory.
        For example, a contiguous ``(3, 4)`` array of type ``int16`` in C-order
        has strides ``(8, 2)``.  This implies that to move from element to
        element in memory requires jumps of 2 bytes.  To move from row-to-row,
        one needs to jump 6 bytes at a time (``2 * 4``).
    ctypes : ctypes object
        Class containing properties of the array needed for interaction
        with ctypes.
    base : ndarray
        If the array is a view on another array, that array is
        its `base` (unless that array is also a view).  The `base` array
        is where the array data is ultimately stored.

    Parameters
    ----------
    shape : tuple of ints
        Shape of created array.
    dtype : data type, optional
        Any object that can be interpreted a numpy data type.
    buffer : object exposing buffer interface, optional
        Used to fill the array with data.
    offset : int, optional
        Offset of array data in buffer.
    strides : tuple of ints, optional
        Strides of data in memory.
    order : {'C', 'F'}, optional
        Row-major or column-major order.

    See Also
    --------
    array : Construct an array.
    zeros : Create an array and fill its allocated memory with zeros.
    empty : Create an array, but leave its allocated memory unchanged.
    dtype : Create a data type.

    Notes
    -----
    There are two modes of creating an array using __new__:

    1. If `buffer` is None, then only `shape`, `dtype`, and `order`
       are used.
    2. If `buffer` is an object exporting the buffer interface, then
       all keywords are interpreted.

    No __init__ method is needed because the array is fully initialized
    after the __new__ method.

    Examples
    --------
    These examples illustrate the low-level `ndarray` constructor.  Refer
    to the `See Also` section for easier ways of constructing an ndarray.

    First mode, `buffer` is None:

    >>> np.ndarray(shape=(2,2), dtype=float, order='F')
    array([[ -1.13698227e+002,   4.25087011e-303],
           [  2.88528414e-306,   3.27025015e-309]])

    Second mode:

    >>> np.ndarray((2,), buffer=np.array([1,2,3]),
    ...            offset=np.int_().itemsize,
    ...            dtype=int) # offset = 1*itemsize, i.e. skip first element
    array([2, 3])

    """)


##############################################################################
#
# ndarray attributes
#
##############################################################################


add_newdoc('numpy.core.multiarray', 'ndarray', ('__array_interface__',
    """Array protocol: Python side."""))


add_newdoc('numpy.core.multiarray', 'ndarray', ('__array_finalize__',
    """None."""))


add_newdoc('numpy.core.multiarray', 'ndarray', ('__array_priority__',
    """Array priority."""))


add_newdoc('numpy.core.multiarray', 'ndarray', ('__array_struct__',
    """Array protocol: C-struct side."""))


add_newdoc('numpy.core.multiarray', 'ndarray', ('_as_parameter_',
    """Allow the array to be interpreted as a ctypes object by returning the
    data-memory location as an integer

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('base',
    """
    Base object if memory is from some other object.

    Examples
    --------

    Base of an array owning its memory is None:

    >>> x = np.array([1,2,3,4])
    >>> x.base is None
    True

    Slicing creates a view, and the memory is shared with x:

    >>> y = x[2:]
    >>> y.base is x
    True

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('ctypes',
    """A ctypes interface object.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('data',
    """Buffer object pointing to the start of the data.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('dtype',
    """Data-type for the array.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('imag',
    """
    The imaginary part of the array.

    Examples
    --------
    >>> x = np.sqrt([1+0j, 0+1j])
    >>> x.imag
    array([ 0.        ,  0.70710678])
    >>> x.imag.dtype
    dtype('float64')

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('itemsize',
    """
    Length of one element in bytes.

    Examples
    --------
    >>> x = np.array([1,2,3], dtype=np.float64)
    >>> x.itemsize
    8
    >>> x = np.array([1,2,3], dtype=np.complex128)
    >>> x.itemsize
    16

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('flags',
    """
    Information about the memory layout of the array.

    Attributes
    ----------
    C_CONTIGUOUS (C)
        The data is in a single, C-style contiguous segment.
    F_CONTIGUOUS (F)
        The data is in a single, Fortran-style contiguous segment.
    OWNDATA (O)
        The array owns the memory it uses or borrows it from another object.
    WRITEABLE (W)
        The data area can be written to.
    ALIGNED (A)
        The data and strides are aligned appropriately for the hardware.
    UPDATEIFCOPY (U)
        This array is a copy of some other array. When this array is
        deallocated, the base array will be updated with the contents of
        this array.

    FNC
        F_CONTIGUOUS and not C_CONTIGUOUS.
    FORC
        F_CONTIGUOUS or C_CONTIGUOUS (one-segment test).
    BEHAVED (B)
        ALIGNED and WRITEABLE.
    CARRAY (CA)
        BEHAVED and C_CONTIGUOUS.
    FARRAY (FA)
        BEHAVED and F_CONTIGUOUS and not C_CONTIGUOUS.

    Notes
    -----
    The `flags` object can be also accessed dictionary-like, and using
    lowercased attribute names. Short flag names are only supported in
    dictionary access.

    Only the UPDATEIFCOPY, WRITEABLE, and ALIGNED flags can be changed by
    the user, via assigning to ``flags['FLAGNAME']`` or `ndarray.setflags`.
    The array flags cannot be set arbitrarily:

    - UPDATEIFCOPY can only be set ``False``.
    - ALIGNED can only be set ``True`` if the data is truly aligned.
    - WRITEABLE can only be set ``True`` if the array owns its own memory
      or the ultimate owner of the memory exposes a writeable buffer
      interface or is a string.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('flat',
    """
    A 1-D flat iterator over the array.

    This is a `flatiter` instance, which acts similarly to a Python iterator.

    See Also
    --------
    flatten : Return a copy of the array collapsed into one dimension.
    flatiter

    Examples
    --------
    >>> x = np.arange(1, 7).reshape(2, 3)
    >>> x
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> x.flat[3]
    4
    >>> x.T
    array([[1, 4],
           [2, 5],
           [3, 6]])
    >>> x.T.flat[3]
    5

    >>> type(x.flat)
    <type 'numpy.flatiter'>

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('nbytes',
    """
    Number of bytes in the array.

    Examples
    --------
    >>> x = np.zeros((3,5,2), dtype=np.complex128)
    >>> x.nbytes
    480
    >>> np.prod(x.shape) * x.itemsize
    480

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('ndim',
    """
    Number of array dimensions.

    Examples
    --------

    >>> x = np.array([1,2,3])
    >>> x.ndim
    1
    >>> y = np.zeros((2,3,4))
    >>> y.ndim
    3

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('real',
    """
    The real part of the array.

    Examples
    --------
    >>> x = np.sqrt([1+0j, 0+1j])
    >>> x.real
    array([ 1.        ,  0.70710678])
    >>> x.real.dtype
    dtype('float64')

    See Also
    --------
    numpy.real : equivalent function

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('shape',
    """
    Tuple of array dimensions.

    Examples
    --------
    >>> x = np.array([1,2,3,4])
    >>> x.shape
    (4,)
    >>> y = np.zeros((4,5,6))
    >>> y.shape
    (4, 5, 6)
    >>> y.shape = (2, 5, 2, 3, 2)
    >>> y.shape
    (2, 5, 2, 3, 2)

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('size',
    """
    Number of elements in the array.

    Examples
    --------
    >>> x = np.zeros((3,5,2), dtype=np.complex128)
    >>> x.size
    30

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('strides',
    """
    Tuple of bytes to step in each dimension.

    The byte offset of element ``(i[0], i[1], ..., i[n])`` in an array `a`
    is::

        offset = sum(np.array(i) * a.strides)

    Examples
    --------
    >>> x = np.reshape(np.arange(5*6*7*8), (5,6,7,8)).transpose(2,3,1,0)
    >>> x.strides
    (32, 4, 224, 1344)
    >>> i = np.array([3,5,2,2])
    >>> offset = sum(i * x.strides)
    >>> x[3,5,2,2]
    813
    >>> offset / x.itemsize
    813

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('T',
    """
    Same as self.transpose() except self is returned for self.ndim < 2.

    Examples
    --------
    >>> x = np.array([[1.,2.],[3.,4.]])
    >>> x.T
    array([[ 1.,  3.],
           [ 2.,  4.]])

    """))


##############################################################################
#
# ndarray methods
#
##############################################################################


add_newdoc('numpy.core.multiarray', 'ndarray', ('__array__',
    """ a.__array__(|dtype) -> reference if type unchanged, copy otherwise.

    Returns either a new reference to self if dtype is not given or a new array
    of provided data type if dtype is different from the current dtype of the
    array.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('__array_prepare__',
    """a.__array_prepare__(obj) -> Object of same type as ndarray object obj.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('__array_wrap__',
    """a.__array_wrap__(obj) -> Object of same type as ndarray object a.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('__copy__',
    """a.__copy__([order])

    Return a copy of the array.

    Parameters
    ----------
    order : {'C', 'F', 'A'}, optional
        If order is 'C' (False) then the result is contiguous (default).
        If order is 'Fortran' (True) then the result has fortran order.
        If order is 'Any' (None) then the result has fortran order
        only if the array already is in fortran order.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('__deepcopy__',
    """a.__deepcopy__() -> Deep copy of array.

    Used if copy.deepcopy is called on an array.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('__reduce__',
    """a.__reduce__()

    For pickling.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('__setstate__',
    """a.__setstate__(version, shape, dtype, isfortran, rawdata)

    For unpickling.

    Parameters
    ----------
    version : int
        optional pickle version. If omitted defaults to 0.
    shape : tuple
    dtype : data-type
    isFortran : bool
    rawdata : string or list
        a binary string with the data (or a list if 'a' is an object array)

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('all',
    """
    a.all(axis=None, out=None)

    Returns True if all elements evaluate to True.

    Refer to `numpy.all` for full documentation.

    See Also
    --------
    numpy.all : equivalent function

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('any',
    """
    a.any(axis=None, out=None)

    Check if any of the elements of `a` are true.

    Refer to `numpy.any` for full documentation.

    See Also
    --------
    numpy.any : equivalent function

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('argmax',
    """
    a.argmax(axis=None, out=None)

    Return indices of the maximum values along the given axis of `a`.

    Parameters
    ----------
    axis : int, optional
        Axis along which to operate.  By default flattened input is used.
    out : ndarray, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.

    Returns
    -------
    index_array : ndarray
        An array of indices or single index value, or a reference to `out`
        if it was specified.

    Examples
    --------
    >>> a = np.arange(6).reshape(2,3)
    >>> a.argmax()
    5
    >>> a.argmax(0)
    array([1, 1, 1])
    >>> a.argmax(1)
    array([2, 2])

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('argmin',
    """
    a.argmin(axis=None, out=None)

    Return indices of the minimum values along the given axis of `a`.

    Refer to `numpy.ndarray.argmax` for detailed documentation.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('argsort',
    """
    a.argsort(axis=-1, kind='quicksort', order=None)

    Returns the indices that would sort this array.

    Refer to `numpy.argsort` for full documentation.

    See Also
    --------
    numpy.argsort : equivalent function

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('astype',
    """
    a.astype(t)

    Copy of the array, cast to a specified type.

    Parameters
    ----------
    t : string or dtype
        Typecode or data-type to which the array is cast.

    Examples
    --------
    >>> x = np.array([1, 2, 2.5])
    >>> x
    array([ 1. ,  2. ,  2.5])

    >>> x.astype(int)
    array([1, 2, 2])

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('byteswap',
    """
    a.byteswap(inplace)

    Swap the bytes of the array elements

    Toggle between low-endian and big-endian data representation by
    returning a byteswapped array, optionally swapped in-place.

    Parameters
    ----------
    inplace: bool, optional
        If ``True``, swap bytes in-place, default is ``False``.

    Returns
    -------
    out: ndarray
        The byteswapped array. If `inplace` is ``True``, this is
        a view to self.

    Examples
    --------
    >>> A = np.array([1, 256, 8755], dtype=np.int16)
    >>> map(hex, A)
    ['0x1', '0x100', '0x2233']
    >>> A.byteswap(True)
    array([  256,     1, 13090], dtype=int16)
    >>> map(hex, A)
    ['0x100', '0x1', '0x3322']

    Arrays of strings are not swapped

    >>> A = np.array(['ceg', 'fac'])
    >>> A.byteswap()
    array(['ceg', 'fac'],
          dtype='|S3')

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('choose',
    """
    a.choose(choices, out=None, mode='raise')

    Use an index array to construct a new array from a set of choices.

    Refer to `numpy.choose` for full documentation.

    See Also
    --------
    numpy.choose : equivalent function

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('clip',
    """
    a.clip(a_min, a_max, out=None)

    Return an array whose values are limited to ``[a_min, a_max]``.

    Refer to `numpy.clip` for full documentation.

    See Also
    --------
    numpy.clip : equivalent function

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('compress',
    """
    a.compress(condition, axis=None, out=None)

    Return selected slices of this array along given axis.

    Refer to `numpy.compress` for full documentation.

    See Also
    --------
    numpy.compress : equivalent function

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('conj',
    """a.conj()

    Return an array with all complex-valued elements conjugated.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('conjugate',
    """a.conjugate()

    Return an array with all complex-valued elements conjugated.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('copy',
    """
    a.copy(order='C')

    Return a copy of the array.

    Parameters
    ----------
    order : {'C', 'F', 'A'}, optional
        By default, the result is stored in C-contiguous (row-major) order in
        memory.  If `order` is `F`, the result has 'Fortran' (column-major)
        order.  If order is 'A' ('Any'), then the result has the same order
        as the input.

    Examples
    --------
    >>> x = np.array([[1,2,3],[4,5,6]], order='F')

    >>> y = x.copy()

    >>> x.fill(0)

    >>> x
    array([[0, 0, 0],
           [0, 0, 0]])

    >>> y
    array([[1, 2, 3],
           [4, 5, 6]])

    >>> y.flags['C_CONTIGUOUS']
    True

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('cumprod',
    """
    a.cumprod(axis=None, dtype=None, out=None)

    Return the cumulative product of the elements along the given axis.

    Refer to `numpy.cumprod` for full documentation.

    See Also
    --------
    numpy.cumprod : equivalent function

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('cumsum',
    """
    a.cumsum(axis=None, dtype=None, out=None)

    Return the cumulative sum of the elements along the given axis.

    Refer to `numpy.cumsum` for full documentation.

    See Also
    --------
    numpy.cumsum : equivalent function

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('diagonal',
    """
    a.diagonal(offset=0, axis1=0, axis2=1)

    Return specified diagonals.

    Refer to `numpy.diagonal` for full documentation.

    See Also
    --------
    numpy.diagonal : equivalent function

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('dump',
    """a.dump(file)

    Dump a pickle of the array to the specified file.
    The array can be read back with pickle.load or numpy.load.

    Parameters
    ----------
    file : str
        A string naming the dump file.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('dumps',
    """a.dumps()

    Returns the pickle of the array as a string.
    pickle.loads or numpy.loads will convert the string back to an array.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('fill',
    """
    a.fill(value)

    Fill the array with a scalar value.

    Parameters
    ----------
    a : ndarray
        Input array
    value : scalar
        All elements of `a` will be assigned this value.

    Examples
    --------
    >>> a = np.array([1, 2])
    >>> a.fill(0)
    >>> a
    array([0, 0])
    >>> a = np.empty(2)
    >>> a.fill(1)
    >>> a
    array([ 1.,  1.])

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('flatten',
    """
    a.flatten(order='C')

    Return a copy of the array collapsed into one dimension.

    Parameters
    ----------
    order : {'C', 'F'}, optional
        Whether to flatten in C (row-major) or Fortran (column-major) order.
        The default is 'C'.

    Returns
    -------
    y : ndarray
        A copy of the input array, flattened to one dimension.

    See Also
    --------
    ravel : Return a flattened array.
    flat : A 1-D flat iterator over the array.

    Examples
    --------
    >>> a = np.array([[1,2], [3,4]])
    >>> a.flatten()
    array([1, 2, 3, 4])
    >>> a.flatten('F')
    array([1, 3, 2, 4])

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('getfield',
    """a.getfield(dtype, offset)

    Returns a field of the given array as a certain type. A field is a view of
    the array data with each itemsize determined by the given type and the
    offset into the current array.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('item',
    """a.item()

    Copy the first element of array to a standard Python scalar and return
    it. The array must be of size one.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('max',
    """
    a.max(axis=None, out=None)

    Return the maximum along a given axis.

    Refer to `numpy.amax` for full documentation.

    See Also
    --------
    numpy.amax : equivalent function

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('mean',
    """
    a.mean(axis=None, dtype=None, out=None)

    Returns the average of the array elements along given axis.

    Refer to `numpy.mean` for full documentation.

    See Also
    --------
    numpy.mean : equivalent function

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('min',
    """
    a.min(axis=None, out=None)

    Return the minimum along a given axis.

    Refer to `numpy.amin` for full documentation.

    See Also
    --------
    numpy.amin : equivalent function

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('newbyteorder',
    """
    arr.newbyteorder(new_order='S')

    Return the array with the same data viewed with a different byte order.

    Equivalent to::

        arr.view(arr.dtype.newbytorder(new_order))

    Changes are also made in all fields and sub-arrays of the array data
    type.



    Parameters
    ----------
    new_order : string, optional
        Byte order to force; a value from the byte order specifications
        above. `new_order` codes can be any of::

         * 'S' - swap dtype from current to opposite endian
         * {'<', 'L'} - little endian
         * {'>', 'B'} - big endian
         * {'=', 'N'} - native order
         * {'|', 'I'} - ignore (no change to byte order)

        The default value ('S') results in swapping the current
        byte order. The code does a case-insensitive check on the first
        letter of `new_order` for the alternatives above.  For example,
        any of 'B' or 'b' or 'biggish' are valid to specify big-endian.


    Returns
    -------
    new_arr : array
        New array object with the dtype reflecting given change to the
        byte order.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('nonzero',
    """
    a.nonzero()

    Return the indices of the elements that are non-zero.

    Refer to `numpy.nonzero` for full documentation.

    See Also
    --------
    numpy.nonzero : equivalent function

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('prod',
    """
    a.prod(axis=None, dtype=None, out=None)

    Return the product of the array elements over the given axis

    Refer to `numpy.prod` for full documentation.

    See Also
    --------
    numpy.prod : equivalent function

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('ptp',
    """
    a.ptp(axis=None, out=None)

    Peak to peak (maximum - minimum) value along a given axis.

    Refer to `numpy.ptp` for full documentation.

    See Also
    --------
    numpy.ptp : equivalent function

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('put',
    """
    a.put(indices, values, mode='raise')

    Set a.flat[n] = values[n] for all n in indices.

    Refer to `numpy.put` for full documentation.

    See Also
    --------
    numpy.put : equivalent function

    """))


add_newdoc('numpy.core.multiarray', 'putmask',
    """
    putmask(a, mask, values)

    Changes elements of an array based on conditional and input values.

    Sets ``a.flat[n] = values[n]`` for each n where ``mask.flat[n]==True``.

    If `values` is not the same size as `a` and `mask` then it will repeat.
    This gives behavior different from ``a[mask] = values``.

    Parameters
    ----------
    a : array_like
        Target array.
    mask : array_like
        Boolean mask array. It has to be the same shape as `a`.
    values : array_like
        Values to put into `a` where `mask` is True. If `values` is smaller
        than `a` it will be repeated.

    See Also
    --------
    place, put, take

    Examples
    --------
    >>> x = np.arange(6).reshape(2, 3)
    >>> np.putmask(x, x>2, x**2)
    >>> x
    array([[ 0,  1,  2],
           [ 9, 16, 25]])

    If `values` is smaller than `a` it is repeated:

    >>> x = np.arange(5)
    >>> np.putmask(x, x>1, [-33, -44])
    >>> x
    array([  0,   1, -33, -44, -33])

    """)


add_newdoc('numpy.core.multiarray', 'ndarray', ('ravel',
    """
    a.ravel([order])

    Return a flattened array.

    Refer to `numpy.ravel` for full documentation.

    See Also
    --------
    numpy.ravel : equivalent function

    ndarray.flat : a flat iterator on the array.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('repeat',
    """
    a.repeat(repeats, axis=None)

    Repeat elements of an array.

    Refer to `numpy.repeat` for full documentation.

    See Also
    --------
    numpy.repeat : equivalent function

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('reshape',
    """
    a.reshape(shape, order='C')

    Returns an array containing the same data with a new shape.

    Refer to `numpy.reshape` for full documentation.

    See Also
    --------
    numpy.reshape : equivalent function

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('resize',
    """
    a.resize(new_shape, refcheck=True, order=False)

    Change shape and size of array in-place.

    Parameters
    ----------
    a : ndarray
        Input array.
    new_shape : {tuple, int}
        Shape of resized array.
    refcheck : bool, optional
        If False, memory referencing will not be checked. Default is True.
    order : bool, optional
        <needs an explanation>. Default if False.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `a` does not own its own data, or references or views to it exist.

    Examples
    --------
    Shrinking an array: array is flattened in C-order, resized, and reshaped:

    >>> a = np.array([[0,1],[2,3]])
    >>> a.resize((2,1))
    >>> a
    array([[0],
           [1]])

    Enlarging an array: as above, but missing entries are filled with zeros:

    >>> b = np.array([[0,1],[2,3]])
    >>> b.resize((2,3))
    >>> b
    array([[0, 1, 2],
           [3, 0, 0]])

    Referencing an array prevents resizing:

    >>> c = a
    >>> a.resize((1,1))
    Traceback (most recent call last):
    ...
    ValueError: cannot resize an array that has been referenced ...

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('round',
    """
    a.round(decimals=0, out=None)

    Return an array rounded a to the given number of decimals.

    Refer to `numpy.around` for full documentation.

    See Also
    --------
    numpy.around : equivalent function

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('searchsorted',
    """
    a.searchsorted(v, side='left')

    Find indices where elements of v should be inserted in a to maintain order.

    For full documentation, see `numpy.searchsorted`

    See Also
    --------
    numpy.searchsorted : equivalent function

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('setfield',
    """m.setfield(value, dtype, offset) -> None.
    places val into field of the given array defined by the data type and offset.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('setflags',
    """a.setflags(write=None, align=None, uic=None)

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('sort',
    """
    a.sort(axis=-1, kind='quicksort', order=None)

    Sort an array, in-place.

    Parameters
    ----------
    axis : int, optional
        Axis along which to sort. Default is -1, which means sort along the
        last axis.
    kind : {'quicksort', 'mergesort', 'heapsort'}, optional
        Sorting algorithm. Default is 'quicksort'.
    order : list, optional
        When `a` is an array with fields defined, this argument specifies
        which fields to compare first, second, etc.  Not all fields need be
        specified.

    See Also
    --------
    numpy.sort : Return a sorted copy of an array.
    argsort : Indirect sort.
    lexsort : Indirect stable sort on multiple keys.
    searchsorted : Find elements in sorted array.

    Notes
    -----
    See ``sort`` for notes on the different sorting algorithms.

    Examples
    --------
    >>> a = np.array([[1,4], [3,1]])
    >>> a.sort(axis=1)
    >>> a
    array([[1, 4],
           [1, 3]])
    >>> a.sort(axis=0)
    >>> a
    array([[1, 3],
           [1, 4]])

    Use the `order` keyword to specify a field to use when sorting a
    structured array:

    >>> a = np.array([('a', 2), ('c', 1)], dtype=[('x', 'S1'), ('y', int)])
    >>> a.sort(order='y')
    >>> a
    array([('c', 1), ('a', 2)],
          dtype=[('x', '|S1'), ('y', '<i4')])

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('squeeze',
    """
    a.squeeze()

    Remove single-dimensional entries from the shape of `a`.

    Refer to `numpy.squeeze` for full documentation.

    See Also
    --------
    numpy.squeeze : equivalent function

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('std',
    """
    a.std(axis=None, dtype=None, out=None, ddof=0)

    Returns the standard deviation of the array elements along given axis.

    Refer to `numpy.std` for full documentation.

    See Also
    --------
    numpy.std : equivalent function

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('sum',
    """
    a.sum(axis=None, dtype=None, out=None)

    Return the sum of the array elements over the given axis.

    Refer to `numpy.sum` for full documentation.

    See Also
    --------
    numpy.sum : equivalent function

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('swapaxes',
    """
    a.swapaxes(axis1, axis2)

    Return a view of the array with `axis1` and `axis2` interchanged.

    Refer to `numpy.swapaxes` for full documentation.

    See Also
    --------
    numpy.swapaxes : equivalent function

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('take',
    """
    a.take(indices, axis=None, out=None, mode='raise')

    Return an array formed from the elements of a at the given indices.

    Refer to `numpy.take` for full documentation.

    See Also
    --------
    numpy.take : equivalent function

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('tofile',
    """
    a.tofile(fid, sep="", format="%s")

    Write array to a file as text or binary.

    Data is always written in 'C' order, independently of the order of `a`.
    The data produced by this method can be recovered by using the function
    fromfile().

    This is a convenience function for quick storage of array data.
    Information on endianess and precision is lost, so this method is not a
    good choice for files intended to archive data or transport data between
    machines with different endianess. Some of these problems can be overcome
    by outputting the data as text files at the expense of speed and file size.

    Parameters
    ----------
    fid : file or string
        An open file object or a string containing a filename.
    sep : string
        Separator between array items for text output.
        If "" (empty), a binary file is written, equivalently to
        file.write(a.tostring()).
    format : string
        Format string for text file output.
        Each entry in the array is formatted to text by converting it to the
        closest Python type, and using "format" % item.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('tolist',
    """
    a.tolist()

    Return the array as a possibly nested list.

    Return a copy of the array data as a (nested) Python list.
    Data items are converted to the nearest compatible Python type.

    Parameters
    ----------
    none

    Returns
    -------
    y : list
        The possibly nested list of array elements.

    Notes
    -----
    The array may be recreated, ``a = np.array(a.tolist())``.

    Examples
    --------
    >>> a = np.array([1, 2])
    >>> a.tolist()
    [1, 2]
    >>> a = np.array([[1, 2], [3, 4]])
    >>> list(a)
    [array([1, 2]), array([3, 4])]
    >>> a.tolist()
    [[1, 2], [3, 4]]

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('tostring',
    """a.tostring(order='C')

    Construct a Python string containing the raw data bytes in the array.

    Parameters
    ----------
    order : {'C', 'F', None}
        Order of the data for multidimensional arrays:
        C, Fortran, or the same as for the original array.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('trace',
    """
    a.trace(offset=0, axis1=0, axis2=1, dtype=None, out=None)

    Return the sum along diagonals of the array.

    Refer to `numpy.trace` for full documentation.

    See Also
    --------
    numpy.trace : equivalent function

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('transpose',
    """
    a.transpose(*axes)

    Returns a view of 'a' with axes transposed. If no axes are given,
    or None is passed, switches the order of the axes. For a 2-d
    array, this is the usual matrix transpose. If axes are given,
    they describe how the axes are permuted.

    See Also
    --------
    ndarray.T : array property returning the array transposed


    Examples
    --------
    >>> a = np.array([[1,2],[3,4]])
    >>> a
    array([[1, 2],
           [3, 4]])
    >>> a.transpose()
    array([[1, 3],
           [2, 4]])
    >>> a.transpose((1,0))
    array([[1, 3],
           [2, 4]])
    >>> a.transpose(1,0)
    array([[1, 3],
           [2, 4]])

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('var',
    """
    a.var(axis=None, dtype=None, out=None, ddof=0)

    Returns the variance of the array elements, along given axis.

    Refer to `numpy.var` for full documentation.

    See Also
    --------
    numpy.var : equivalent function

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('view',
    """
    a.view(dtype=None, type=None)

    New view of array with the same data.

    Parameters
    ----------
    dtype : data-type
        Data-type descriptor of the returned view, e.g. float32 or int16.
    type : python type
        Type of the returned view, e.g. ndarray or matrix.


    Notes
    -----

    `a.view()` is used two different ways.

    `a.view(some_dtype)` or `a.view(dtype=some_dtype)` constructs a view of
    the array's memory with a different dtype. This can cause a
    reinterpretation of the bytes of memory.

    `a.view(ndarray_subclass)`, or `a.view(type=ndarray_subclass)`, just
    returns an instance of ndarray_subclass that looks at the same array (same
    shape, dtype, etc.). This does not cause a reinterpretation of the memory.


    Examples
    --------
    >>> x = np.array([(1, 2)], dtype=[('a', np.int8), ('b', np.int8)])

    Viewing array data using a different type and dtype:

    >>> y = x.view(dtype=np.int16, type=np.matrix)
    >>> y
    matrix([[513]], dtype=int16)
    >>> print type(y)
    <class 'numpy.matrixlib.defmatrix.matrix'>

    Creating a view on a structured array so it can be used in calculations

    >>> x = np.array([(1, 2),(3,4)], dtype=[('a', np.int8), ('b', np.int8)])
    >>> xv = x.view(dtype=np.int8).reshape(-1,2)
    >>> xv
    array([[1, 2],
           [3, 4]], dtype=int8)
    >>> xv.mean(0)
    array([ 2.,  3.])

    Making changes to the view changes the underlying array

    >>> xv[0,1] = 20
    >>> print x
    [(1, 20) (3, 4)]

    Using a view to convert an array to a record array:

    >>> z = x.view(np.recarray)
    >>> z.a
    array([1], dtype=int8)

    Views share data:

    >>> x[0] = (9, 10)
    >>> z[0]
    (9, 10)

    """))


##############################################################################
#
# umath functions
#
##############################################################################

add_newdoc('numpy.core.umath', 'frexp',
    """
    Return normalized fraction and exponent of 2 of input array, element-wise.

    Returns (`out1`, `out2`) from equation ``x` = out1 * 2**out2``.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    (out1, out2) : tuple of ndarrays, (float, int)
        `out1` is a float array with values between -1 and 1.
        `out2` is an int array which represent the exponent of 2.

    See Also
    --------
    ldexp : Compute ``y = x1 * 2**x2``, the inverse of `frexp`.

    Notes
    -----
    Complex dtypes are not supported, they will raise a TypeError.

    Examples
    --------
    >>> x = np.arange(9)
    >>> y1, y2 = np.frexp(x)
    >>> y1
    array([ 0.   ,  0.5  ,  0.5  ,  0.75 ,  0.5  ,  0.625,  0.75 ,  0.875,
            0.5  ])
    >>> y2
    array([0, 1, 2, 2, 3, 3, 3, 3, 4])
    >>> y1 * 2**y2
    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.])

    """)

add_newdoc('numpy.core.umath', 'frompyfunc',
    """
    frompyfunc(func, nin, nout)

    Takes an arbitrary Python function and returns a ufunc.

    Parameters
    ----------
    func : Python function
        An arbitrary Python function.
    nin : int
        The number of input arguments.
    nout : int
        The number of objects returned by `func`.

    Returns
    -------
    out : ufunc
        Returns a Numpy universal function (``ufunc`` object).

    Notes
    -----
    The returned ufunc always returns PyObject arrays.

    """)

add_newdoc('numpy.core.umath', 'ldexp',
    """
    Compute y = x1 * 2**x2.

    Parameters
    ----------
    x1 : array_like
        The array of multipliers.
    x2 : array_like
        The array of exponents.

    Returns
    -------
    y : array_like
        The output array, the result of ``x1 * 2**x2``.

    See Also
    --------
    frexp : Return (y1, y2) from ``x = y1 * 2**y2``, the inverse of `ldexp`.

    Notes
    -----
    Complex dtypes are not supported, they will raise a TypeError.

    `ldexp` is useful as the inverse of `frexp`, if used by itself it is
    more clear to simply use the expression ``x1 * 2**x2``.

    Examples
    --------
    >>> np.ldexp(5, np.arange(4))
    array([  5.,  10.,  20.,  40.], dtype=float32)

    >>> x = np.arange(6)
    >>> np.ldexp(*np.frexp(x))
    array([ 0.,  1.,  2.,  3.,  4.,  5.])

    """)

add_newdoc('numpy.core.umath', 'geterrobj',
    """
    geterrobj()

    Return the current object that defines floating-point error handling.

    The error object contains all information that defines the error handling
    behavior in Numpy. `geterrobj` is used internally by the other
    functions that get and set error handling behavior (`geterr`, `seterr`,
    `geterrcall`, `seterrcall`).

    Returns
    -------
    errobj : list
        The error object, a list containing three elements:
        [internal numpy buffer size, error mask, error callback function].

        The error mask is a single integer that holds the treatment information
        on all four floating point errors. If we print it in base 8, we can see
        what treatment is set for "invalid", "under", "over", and "divide" (in
        that order). The printed string can be interpreted with

        * 0 : 'ignore'
        * 1 : 'warn'
        * 2 : 'raise'
        * 3 : 'call'
        * 4 : 'print'
        * 5 : 'log'

    See Also
    --------
    seterrobj, seterr, geterr, seterrcall, geterrcall
    getbufsize, setbufsize

    Notes
    -----
    For complete documentation of the types of floating-point exceptions and
    treatment options, see `seterr`.

    Examples
    --------
    >>> np.geterrobj()  # first get the defaults
    [10000, 0, None]

    >>> def err_handler(type, flag):
    ...     print "Floating point error (%s), with flag %s" % (type, flag)
    ...
    >>> old_bufsize = np.setbufsize(20000)
    >>> old_err = np.seterr(divide='raise')
    >>> old_handler = np.seterrcall(err_handler)
    >>> np.geterrobj()
    [20000, 2, <function err_handler at 0x91dcaac>]

    >>> old_err = np.seterr(all='ignore')
    >>> np.base_repr(np.geterrobj()[1], 8)
    '0'
    >>> old_err = np.seterr(divide='warn', over='log', under='call',
                            invalid='print')
    >>> np.base_repr(np.geterrobj()[1], 8)
    '4351'

    """)

add_newdoc('numpy.core.umath', 'seterrobj',
    """
    seterrobj(errobj)

    Set the object that defines floating-point error handling.

    The error object contains all information that defines the error handling
    behavior in Numpy. `seterrobj` is used internally by the other
    functions that set error handling behavior (`seterr`, `seterrcall`).

    Parameters
    ----------
    errobj : list
        The error object, a list containing three elements:
        [internal numpy buffer size, error mask, error callback function].

        The error mask is a single integer that holds the treatment information
        on all four floating point errors. If we print it in base 8, we can see
        what treatment is set for "invalid", "under", "over", and "divide" (in
        that order). The printed string can be interpreted with

        * 0 : 'ignore'
        * 1 : 'warn'
        * 2 : 'raise'
        * 3 : 'call'
        * 4 : 'print'
        * 5 : 'log'

    See Also
    --------
    geterrobj, seterr, geterr, seterrcall, geterrcall
    getbufsize, setbufsize

    Notes
    -----
    For complete documentation of the types of floating-point exceptions and
    treatment options, see `seterr`.

    Examples
    --------
    >>> old_errobj = np.geterrobj()  # first get the defaults
    >>> old_errobj
    [10000, 0, None]

    >>> def err_handler(type, flag):
    ...     print "Floating point error (%s), with flag %s" % (type, flag)
    ...
    >>> new_errobj = [20000, 12, err_handler]
    >>> np.seterrobj(new_errobj)
    >>> np.base_repr(12, 8)  # int for divide=4 ('print') and over=1 ('warn')
    '14'
    >>> np.geterr()
    {'over': 'warn', 'divide': 'print', 'invalid': 'ignore', 'under': 'ignore'}
    >>> np.geterrcall()
    <function err_handler at 0xb75e9304>

    """)


##############################################################################
#
# lib._compiled_base functions
#
##############################################################################

add_newdoc('numpy.lib._compiled_base', 'digitize',
    """
    digitize(x, bins)

    Return the indices of the bins to which each value in input array belongs.

    Each index ``i`` returned is such that ``bins[i-1] <= x < bins[i]`` if
    `bins` is monotonically increasing, or ``bins[i-1] > x >= bins[i]`` if
    `bins` is monotonically decreasing. If values in `x` are beyond the
    bounds of `bins`, 0 or ``len(bins)`` is returned as appropriate.

    Parameters
    ----------
    x : array_like
        Input array to be binned. It has to be 1-dimensional.
    bins : array_like
        Array of bins. It has to be 1-dimensional and monotonic.

    Returns
    -------
    out : ndarray of ints
        Output array of indices, of same shape as `x`.

    Raises
    ------
    ValueError
        If the input is not 1-dimensional, or if `bins` is not monotonic.
    TypeError
        If the type of the input is complex.

    See Also
    --------
    bincount, histogram, unique

    Notes
    -----
    If values in `x` are such that they fall outside the bin range,
    attempting to index `bins` with the indices that `digitize` returns
    will result in an IndexError.

    Examples
    --------
    >>> x = np.array([0.2, 6.4, 3.0, 1.6])
    >>> bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
    >>> inds = np.digitize(x, bins)
    >>> inds
    array([1, 4, 3, 2])
    >>> for n in range(x.size):
    ...   print bins[inds[n]-1], "<=", x[n], "<", bins[inds[n]]
    ...
    0.0 <= 0.2 < 1.0
    4.0 <= 6.4 < 10.0
    2.5 <= 3.0 < 4.0
    1.0 <= 1.6 < 2.5

    """)

add_newdoc('numpy.lib._compiled_base', 'bincount',
    """
    bincount(x, weights=None)

    Count number of occurrences of each value in array of non-negative ints.

    The number of bins (of size 1) is one larger than the largest value in
    `x`. Each bin gives the number of occurrences of its index value in `x`.
    If `weights` is specified the input array is weighted by it, i.e. if a
    value ``n`` is found at position ``i``, ``out[n] += weight[i]`` instead
    of ``out[n] += 1``.

    Parameters
    ----------
    x : array_like, 1 dimension, nonnegative ints
        Input array. The length of `x` is equal to ``np.amax(x)+1``.
    weights : array_like, optional
        Weights, array of the same shape as `x`.

    Returns
    -------
    out : ndarray of ints
        The result of binning the input array.

    Raises
    ------
    ValueError
        If the input is not 1-dimensional, or contains elements with negative
        values.
    TypeError
        If the type of the input is float or complex.

    See Also
    --------
    histogram, digitize, unique

    Examples
    --------
    >>> np.bincount(np.arange(5))
    array([1, 1, 1, 1, 1])
    >>> np.bincount(np.array([0, 1, 1, 3, 2, 1, 7]))
    array([1, 3, 1, 1, 0, 0, 0, 1])

    >>> x = np.array([0, 1, 1, 3, 2, 1, 7, 23])
    >>> np.bincount(x).size == np.amax(x)+1
    True

    >>> np.bincount(np.arange(5, dtype=np.float))
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: array cannot be safely cast to required type

    """)

add_newdoc('numpy.lib._compiled_base', 'add_docstring',
    """
    docstring(obj, docstring)

    Add a docstring to a built-in obj if possible.
    If the obj already has a docstring raise a RuntimeError
    If this routine does not know how to add a docstring to the object
    raise a TypeError
    """)

add_newdoc('numpy.lib._compiled_base', 'packbits',
    """
    packbits(myarray, axis=None)

    Packs the elements of a binary-valued array into bits in a uint8 array.

    The result is padded to full bytes by inserting zero bits at the end.

    Parameters
    ----------
    myarray : array_like
        An integer type array whose elements should be packed to bits.
    axis : int, optional
        The dimension over which bit-packing is done.
        ``None`` implies packing the flattened array.

    Returns
    -------
    packed : ndarray
        Array of type uint8 whose elements represent bits corresponding to the
        logical (0 or nonzero) value of the input elements. The shape of
        `packed` has the same number of dimensions as the input (unless `axis`
        is None, in which case the output is 1-D).

    See Also
    --------
    unpackbits: Unpacks elements of a uint8 array into a binary-valued output
                array.

    Examples
    --------
    >>> a = np.array([[[1,0,1],
    ...                [0,1,0]],
    ...               [[1,1,0],
    ...                [0,0,1]]])
    >>> b = np.packbits(a, axis=-1)
    >>> b
    array([[[160],[64]],[[192],[32]]], dtype=uint8)

    Note that in binary 160 = 1010 0000, 64 = 0100 0000, 192 = 1100 0000,
    and 32 = 0010 0000.

    """)

add_newdoc('numpy.lib._compiled_base', 'unpackbits',
    """
    unpackbits(myarray, axis=None)

    Unpacks elements of a uint8 array into a binary-valued output array.

    Each element of `myarray` represents a bit-field that should be unpacked
    into a binary-valued output array. The shape of the output array is either
    1-D (if `axis` is None) or the same shape as the input array with unpacking
    done along the axis specified.

    Parameters
    ----------
    myarray : ndarray, uint8 type
       Input array.
    axis : int, optional
       Unpacks along this axis.

    Returns
    -------
    unpacked : ndarray, uint8 type
       The elements are binary-valued (0 or 1).

    See Also
    --------
    packbits : Packs the elements of a binary-valued array into bits in a uint8
               array.

    Examples
    --------
    >>> a = np.array([[2], [7], [23]], dtype=np.uint8)
    >>> a
    array([[ 2],
           [ 7],
           [23]], dtype=uint8)
    >>> b = np.unpackbits(a, axis=1)
    >>> b
    array([[0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 1, 1, 1],
           [0, 0, 0, 1, 0, 1, 1, 1]], dtype=uint8)

    """)


##############################################################################
#
# Documentation for ufunc attributes and methods
#
##############################################################################


##############################################################################
#
# ufunc object
#
##############################################################################

add_newdoc('numpy.core', 'ufunc',
    """
    Functions that operate element by element on whole arrays.

    Unary ufuncs:
    =============

    op(X, out=None)
    Apply op to X elementwise

    Parameters
    ----------
    X : array_like
        Input array
    out : array_like
        An array to store the output. Must be the same shape as X.

    Returns
    -------
    r : array_like
        r will have the same shape as X; if out is provided, r will be
        equal to out.

    Binary ufuncs:
    ==============

    op(X, Y, out=None)
    Apply op to X and Y elementwise. May "broadcast" to make
    the shapes of X and Y congruent.

    The broadcasting rules are:
    * Dimensions of length 1 may be prepended to either array
    * Arrays may be repeated along dimensions of length 1

    Parameters
    ----------
    X : array_like
        First input array
    Y : array_like
        Second input array
    out : array-like
        An array to store the output. Must be the same shape as the
        output would have.

    Returns
    -------
    r : array-like
        The return value; if out is provided, r will be equal to out.

    """)


##############################################################################
#
# ufunc methods
#
##############################################################################

add_newdoc('numpy.core', 'ufunc', ('reduce',
    """
    reduce(array, axis=0, dtype=None, out=None)

    Reduce applies the operator to all elements of the array.

    For a one-dimensional array, reduce produces results equivalent to:
    ::

     r = op.identity
     for i in xrange(len(A)):
       r = op(r,A[i])
     return r

    For example, add.reduce() is equivalent to sum().

    Parameters
    ----------
    array : array_like
        The array to act on.
    axis : integer, optional
        The axis along which to apply the reduction.
    dtype : data-type-code, optional
        The type used to represent the intermediate results. Defaults
        to the data type of the output array if this is provided, or
        the data type of the input array if no output array is provided.
    out : array_like, optional
        A location into which the result is stored. If not provided a
        freshly-allocated array is returned.

    Returns
    -------
    r : ndarray
        The reduced values. If out was supplied, r is equal to out.

    Examples
    --------
    >>> np.multiply.reduce([2,3,5])
    30

    """))

add_newdoc('numpy.core', 'ufunc', ('accumulate',
    """
    accumulate(array, axis=None, dtype=None, out=None)

    Accumulate the result of applying the operator to all elements.

    For a one-dimensional array, accumulate produces results equivalent to:
    ::

     r = np.empty(len(A))
     t = op.identity
     for i in xrange(len(A)):
        t = op(t,A[i])
        r[i] = t
     return r

    For example, add.accumulate() is equivalent to cumsum().

    Parameters
    ----------
    array : array_like
        The array to act on.
    axis : int, optional
        The axis along which to apply the accumulation.
    dtype : data-type-code, optional
        The type used to represent the intermediate results. Defaults
        to the data type of the output array if this is provided, or
        the data type of the input array if no output array is provided.
    out : ndarray, optional
        A location into which the result is stored. If not provided a
        freshly-allocated array is returned.

    Returns
    -------
    r : ndarray
        The accumulated values. If `out` was supplied, `r` is equal to
        `out`.

    Examples
    --------
    >>> np.multiply.accumulate([2,3,5])
    array([2,6,30])

    """))

add_newdoc('numpy.core', 'ufunc', ('reduceat',
    """
    reduceat(self, array, indices, axis=None, dtype=None, out=None)

    Reduceat performs a reduce with specified slices over an axis.

    Computes op.reduce(`array[indices[i]:indices[i+1]]`)
    for i=0..end with an implicit `indices[i+1]` = len(`array`)
    assumed when i = end - 1.

    If `indices[i]` >= `indices[i + 1]`
    then the result is `array[indices[i]]` for that value.

    The function op.accumulate(`array`) is the same as
    op.reduceat(`array`, `indices`)[::2]
    where `indices` is range(len(`array`)-1) with a zero placed
    in every other sample:
    `indices` = zeros(len(`array`)*2 - 1)
    `indices[1::2]` = range(1, len(`array`))

    The output shape is based on the size of `indices`.

    Parameters
    ----------
    array : array_like
        The array to act on.
    indices : array_like
        Paired indices specifying slices to reduce.
    axis : int, optional
        The axis along which to apply the reduceat.
    dtype : data-type-code, optional
        The type used to represent the intermediate results. Defaults
        to the data type of the output array if this is provided, or
        the data type of the input array if no output array is provided.
    out : ndarray, optional
        A location into which the result is stored. If not provided a
        freshly-allocated array is returned.

    Returns
    -------
    r : array
        The reduced values. If `out` was supplied, `r` is equal to `out`.

    Examples
    --------
    To take the running sum of four successive values:

    >>> np.add.reduceat(np.arange(8),[0,4, 1,5, 2,6, 3,7])[::2]
    array([ 6, 10, 14, 18])

    """))

add_newdoc('numpy.core', 'ufunc', ('outer',
    """
    outer(A,B)

    Compute the result of applying op to all pairs (a,b)

    op.outer(A,B) is equivalent to
    op(A[:,:,...,:,newaxis,...,newaxis]*B[newaxis,...,newaxis,:,...,:])
    where A has B.ndim new axes appended and B has A.ndim new axes prepended.

    For A and B one-dimensional, this is equivalent to
    ::

     r = empty(len(A),len(B))
     for i in xrange(len(A)):
         for j in xrange(len(B)):
             r[i,j] = A[i]*B[j]

    If A and B are higher-dimensional, the result has dimension A.ndim+B.ndim

    Parameters
    ----------
    A : array_like
        First term
    B : array_like
        Second term

    Returns
    -------
    r : ndarray
        Output array

    Examples
    --------
    >>> np.multiply.outer([1,2,3],[4,5,6])
    array([[ 4,  5,  6],
           [ 8, 10, 12],
           [12, 15, 18]])

    """))


##############################################################################
#
# Documentation for dtype attributes and methods
#
##############################################################################

##############################################################################
#
# dtype object
#
##############################################################################

add_newdoc('numpy.core.multiarray', 'dtype',
    """
    dtype(obj, align=False, copy=False)

    Create a data type object.

    A numpy array is homogeneous, and contains elements described by a
    dtype object. A dtype object can be constructed from different
    combinations of fundamental numeric types.

    Parameters
    ----------
    obj
        Object to be converted to a data type object.
    align : bool, optional
        Add padding to the fields to match what a C compiler would output
        for a similar C-struct. Can be ``True`` only if `obj` is a dictionary
        or a comma-separated string.
    copy : bool, optional
        Make a new copy of the data-type object. If ``False``, the result
        may just be a reference to a built-in data-type object.

    Examples
    --------
    Using array-scalar type:

    >>> np.dtype(np.int16)
    dtype('int16')

    Record, one field name 'f1', containing int16:

    >>> np.dtype([('f1', np.int16)])
    dtype([('f1', '<i2')])

    Record, one field named 'f1', in itself containing a record with one field:

    >>> np.dtype([('f1', [('f1', np.int16)])])
    dtype([('f1', [('f1', '<i2')])])

    Record, two fields: the first field contains an unsigned int, the
    second an int32:

    >>> np.dtype([('f1', np.uint), ('f2', np.int32)])
    dtype([('f1', '<u4'), ('f2', '<i4')])

    Using array-protocol type strings:

    >>> np.dtype([('a','f8'),('b','S10')])
    dtype([('a', '<f8'), ('b', '|S10')])

    Using comma-separated field formats.  The shape is (2,3):

    >>> np.dtype("i4, (2,3)f8")
    dtype([('f0', '<i4'), ('f1', '<f8', (2, 3))])

    Using tuples.  ``int`` is a fixed type, 3 the field's shape.  ``void``
    is a flexible type, here of size 10:

    >>> np.dtype([('hello',(np.int,3)),('world',np.void,10)])
    dtype([('hello', '<i4', 3), ('world', '|V10')])

    Subdivide ``int16`` into 2 ``int8``'s, called x and y.  0 and 1 are
    the offsets in bytes:

    >>> np.dtype((np.int16, {'x':(np.int8,0), 'y':(np.int8,1)}))
    dtype(('<i2', [('x', '|i1'), ('y', '|i1')]))

    Using dictionaries.  Two fields named 'gender' and 'age':

    >>> np.dtype({'names':['gender','age'], 'formats':['S1',np.uint8]})
    dtype([('gender', '|S1'), ('age', '|u1')])

    Offsets in bytes, here 0 and 25:

    >>> np.dtype({'surname':('S25',0),'age':(np.uint8,25)})
    dtype([('surname', '|S25'), ('age', '|u1')])

    """)

##############################################################################
#
# dtype attributes
#
##############################################################################

add_newdoc('numpy.core.multiarray', 'dtype', ('alignment',
    """
    The required alignment (bytes) of this data-type according to the compiler.

    More information is available in the C-API section of the manual.

    """))

add_newdoc('numpy.core.multiarray', 'dtype', ('byteorder',
    """
    A character indicating the byte-order of this data-type object.

    One of:

    ===  ==============
    '='  native
    '<'  little-endian
    '>'  big-endian
    '|'  not applicable
    ===  ==============

    All built-in data-type objects have byteorder either '=' or '|'.

    Examples
    --------

    >>> dt = np.dtype('i2')
    >>> dt.byteorder
    '='
    >>> # endian is not relevant for 8 bit numbers
    >>> np.dtype('i1').byteorder
    '|'
    >>> # or ASCII strings
    >>> np.dtype('S2').byteorder
    '|'
    >>> # Even if specific code is given, and it is native
    >>> # '=' is the byteorder
    >>> import sys
    >>> sys_is_le = sys.byteorder == 'little'
    >>> native_code = sys_is_le and '<' or '>'
    >>> swapped_code = sys_is_le and '>' or '<'
    >>> dt = np.dtype(native_code + 'i2')
    >>> dt.byteorder
    '='
    >>> # Swapped code shows up as itself
    >>> dt = np.dtype(swapped_code + 'i2')
    >>> dt.byteorder == swapped_code
    True

    """))

add_newdoc('numpy.core.multiarray', 'dtype', ('char',
    """A unique character code for each of the 21 different built-in types."""))

add_newdoc('numpy.core.multiarray', 'dtype', ('descr',
    """
    Array-interface compliant full description of the data-type.

    The format is that required by the 'descr' key in the
    `__array_interface__` attribute.

    """))

add_newdoc('numpy.core.multiarray', 'dtype', ('fields',
    """
    Dictionary showing any named fields defined for this data type, or None.

    The dictionary is indexed by keys that are the names of the fields.
    Each entry in the dictionary is a tuple fully describing the field::

      (dtype, offset[, title])

    If present, the optional title can be any object (if it is string
    or unicode then it will also be a key in the fields dictionary,
    otherwise it's meta-data). Notice also, that the first two elements
    of the tuple can be passed directly as arguments to the `ndarray.getfield`
    and `ndarray.setfield` methods.

    See Also
    --------
    ndarray.getfield, ndarray.setfield

    """))

add_newdoc('numpy.core.multiarray', 'dtype', ('flags',
    """
    Bit-flags describing how this data type is to be interpreted.

    Bit-masks are in `numpy.core.multiarray` as the constants
    `ITEM_HASOBJECT`, `LIST_PICKLE`, `ITEM_IS_POINTER`, `NEEDS_INIT`,
    `NEEDS_PYAPI`, `USE_GETITEM`, `USE_SETITEM`. A full explanation
    of these flags is in C-API documentation; they are largely useful
    for user-defined data-types.

    """))

add_newdoc('numpy.core.multiarray', 'dtype', ('hasobject',
    """
    Boolean indicating whether this dtype contains any reference-counted
    objects in any fields or sub-dtypes.

    Recall that what is actually in the ndarray memory representing
    the Python object is the memory address of that object (a pointer).
    Special handling may be required, and this attribute is useful for
    distinguishing data types that may contain arbitrary Python objects
    and data-types that won't.

    """))

add_newdoc('numpy.core.multiarray', 'dtype', ('isbuiltin',
    """
    Integer indicating how this dtype relates to the built-in dtypes.

    Read-only.

    =  ========================================================================
    0  if this is a structured array type, with fields
    1  if this is a dtype compiled into numpy (such as ints, floats etc)
    2  if the dtype is for a user-defined numpy type
       A user-defined type uses the numpy C-API machinery to extend
       numpy to handle a new array type.  See the Guide to Numpy for
       details.
    =  ========================================================================

    Examples
    --------
    >>> dt = np.dtype('i2')
    >>> dt.isbuiltin
    1
    >>> dt = np.dtype('f8')
    >>> dt.isbuiltin
    1
    >>> dt = np.dtype([('field1', 'f8')])
    >>> dt.isbuiltin
    0

    """))

add_newdoc('numpy.core.multiarray', 'dtype', ('isnative',
    """
    Boolean indicating whether the byte order of this dtype is native
    to the platform.

    """))

add_newdoc('numpy.core.multiarray', 'dtype', ('itemsize',
    """
    The element size of this data-type object.

    For 18 of the 21 types this number is fixed by the data-type.
    For the flexible data-types, this number can be anything.

    """))

add_newdoc('numpy.core.multiarray', 'dtype', ('kind',
    """
    A character code (one of 'biufcSUV') identifying the general kind of data.

    """))

add_newdoc('numpy.core.multiarray', 'dtype', ('name',
    """
    A bit-width name for this data-type.

    Un-sized flexible data-type objects do not have this attribute.

    """))

add_newdoc('numpy.core.multiarray', 'dtype', ('names',
    """
    Ordered list of field names, or ``None`` if there are no fields.

    The names are ordered according to increasing byte offset.

    """))

add_newdoc('numpy.core.multiarray', 'dtype', ('num',
    """
    A unique number for each of the 21 different built-in types.

    These are roughly ordered from least-to-most precision.

    """))

add_newdoc('numpy.core.multiarray', 'dtype', ('shape',
    """
    Shape tuple of the sub-array if this data type describes a sub-array,
    and ``()`` otherwise.

    """))

add_newdoc('numpy.core.multiarray', 'dtype', ('str',
    """The array-protocol typestring of this data-type object."""))

add_newdoc('numpy.core.multiarray', 'dtype', ('subdtype',
    """
    Tuple ``(item_dtype, shape)`` if this `dtype` describes a sub-array, and
    None otherwise.

    The *shape* is the fixed shape of the sub-array described by this
    data type, and *item_dtype* the data type of the array.

    If a field whose dtype object has this attribute is retrieved,
    then the extra dimensions implied by *shape* are tacked on to
    the end of the retrieved array.

    """))

add_newdoc('numpy.core.multiarray', 'dtype', ('type',
    """The type object used to instantiate a scalar of this data-type."""))

##############################################################################
#
# dtype methods
#
##############################################################################

add_newdoc('numpy.core.multiarray', 'dtype', ('newbyteorder',
    """
    newbyteorder(new_order='S')

    Return a new dtype with a different byte order.

    Changes are also made in all fields and sub-arrays of the data type.

    Parameters
    ----------
    new_order : string, optional
        Byte order to force; a value from the byte order
        specifications below.  The default value ('S') results in
        swapping the current byte order.
        `new_order` codes can be any of::

         * 'S' - swap dtype from current to opposite endian
         * {'<', 'L'} - little endian
         * {'>', 'B'} - big endian
         * {'=', 'N'} - native order
         * {'|', 'I'} - ignore (no change to byte order)

        The code does a case-insensitive check on the first letter of
        `new_order` for these alternatives.  For example, any of '>'
        or 'B' or 'b' or 'brian' are valid to specify big-endian.

    Returns
    -------
    new_dtype : dtype
        New dtype object with the given change to the byte order.

    Notes
    -----
    Changes are also made in all fields and sub-arrays of the data type.

    Examples
    --------
    >>> import sys
    >>> sys_is_le = sys.byteorder == 'little'
    >>> native_code = sys_is_le and '<' or '>'
    >>> swapped_code = sys_is_le and '>' or '<'
    >>> native_dt = np.dtype(native_code+'i2')
    >>> swapped_dt = np.dtype(swapped_code+'i2')
    >>> native_dt.newbyteorder('S') == swapped_dt
    True
    >>> native_dt.newbyteorder() == swapped_dt
    True
    >>> native_dt == swapped_dt.newbyteorder('S')
    True
    >>> native_dt == swapped_dt.newbyteorder('=')
    True
    >>> native_dt == swapped_dt.newbyteorder('N')
    True
    >>> native_dt == native_dt.newbyteorder('|')
    True
    >>> np.dtype('<i2') == native_dt.newbyteorder('<')
    True
    >>> np.dtype('<i2') == native_dt.newbyteorder('L')
    True
    >>> np.dtype('>i2') == native_dt.newbyteorder('>')
    True
    >>> np.dtype('>i2') == native_dt.newbyteorder('B')
    True

    """))


##############################################################################
#
# nd_grid instances
#
##############################################################################

add_newdoc('numpy.lib.index_tricks', 'mgrid',
    """
    `nd_grid` instance which returns a dense multi-dimensional "meshgrid".

    An instance of `numpy.lib.index_tricks.nd_grid` which returns an dense
    (or fleshed out) mesh-grid when indexed, so that each returned argument
    has the same shape.  The dimensions and number of the output arrays are
    equal to the number of indexing dimensions.  If the step length is not a
    complex number, then the stop is not inclusive.

    However, if the step length is a **complex number** (e.g. 5j), then
    the integer part of its magnitude is interpreted as specifying the
    number of points to create between the start and stop values, where
    the stop value **is inclusive**.

    Returns
    ----------
    mesh-grid `ndarrays` all of the same dimensions

    See Also
    --------
    numpy.lib.index_tricks.nd_grid : class of `ogrid` and `mgrid` objects
    ogrid : like mgrid but returns open (not fleshed out) mesh grids
    r_ : array concatenator

    Examples
    --------
    >>> np.mgrid[0:5,0:5]
    array([[[0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4]],
           [[0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4]]])
    >>> np.mgrid[-1:1:5j]
    array([-1. , -0.5,  0. ,  0.5,  1. ])

    """)

add_newdoc('numpy.lib.index_tricks', 'ogrid',
    """
    `nd_grid` instance which returns an open multi-dimensional "meshgrid".

    An instance of `numpy.lib.index_tricks.nd_grid` which returns an open
    (i.e. not fleshed out) mesh-grid when indexed, so that only one dimension
    of each returned array is greater than 1.  The dimension and number of the
    output arrays are equal to the number of indexing dimensions.  If the step
    length is not a complex number, then the stop is not inclusive.

    However, if the step length is a **complex number** (e.g. 5j), then
    the integer part of its magnitude is interpreted as specifying the
    number of points to create between the start and stop values, where
    the stop value **is inclusive**.

    Returns
    ----------
    mesh-grid `ndarrays` with only one dimension :math:`\\neq 1`

    See Also
    --------
    np.lib.index_tricks.nd_grid : class of `ogrid` and `mgrid` objects
    mgrid : like `ogrid` but returns dense (or fleshed out) mesh grids
    r_ : array concatenator

    Examples
    --------
    >>> from numpy import ogrid
    >>> ogrid[-1:1:5j]
    array([-1. , -0.5,  0. ,  0.5,  1. ])
    >>> ogrid[0:5,0:5]
    [array([[0],
            [1],
            [2],
            [3],
            [4]]), array([[0, 1, 2, 3, 4]])]

    """)


##############################################################################
#
# Documentation for `generic` attributes and methods
#
##############################################################################

add_newdoc('numpy.core.numerictypes', 'generic',
    """
    """)

# Attributes

add_newdoc('numpy.core.numerictypes', 'generic', ('T',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('base',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('data',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('dtype',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('flags',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('flat',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('imag',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('itemsize',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('nbytes',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('ndim',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('real',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('shape',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('size',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('strides',
    """
    """))

# Methods

add_newdoc('numpy.core.numerictypes', 'generic', ('all',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('any',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('argmax',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('argmin',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('argsort',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('astype',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('byteswap',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('choose',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('clip',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('compress',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('conjugate',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('copy',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('cumprod',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('cumsum',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('diagonal',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('dump',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('dumps',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('fill',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('flatten',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('getfield',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('item',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('itemset',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('max',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('mean',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('min',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('newbyteorder',
    """
    newbyteorder(new_order='S')

    Return a new dtype with a different byte order.

    Changes are also made in all fields and sub-arrays of the data type.

    The `new_order` code can be any from the following:

    * {'<', 'L'} - little endian
    * {'>', 'B'} - big endian
    * {'=', 'N'} - native order
    * 'S' - swap dtype from current to opposite endian
    * {'|', 'I'} - ignore (no change to byte order)

    Parameters
    ----------
    new_order : string, optional
        Byte order to force; a value from the byte order specifications
        above.  The default value ('S') results in swapping the current
        byte order. The code does a case-insensitive check on the first
        letter of `new_order` for the alternatives above.  For example,
        any of 'B' or 'b' or 'biggish' are valid to specify big-endian.


    Returns
    -------
    new_dtype : dtype
        New dtype object with the given change to the byte order.

    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('nonzero',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('prod',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('ptp',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('put',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('ravel',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('repeat',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('reshape',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('resize',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('round',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('searchsorted',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('setfield',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('setflags',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('sort',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('squeeze',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('std',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('sum',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('swapaxes',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('take',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('tofile',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('tolist',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('tostring',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('trace',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('transpose',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('var',
    """
    """))

add_newdoc('numpy.core.numerictypes', 'generic', ('view',
    """
    """))


##############################################################################
#
# Documentation for other scalar classes
#
##############################################################################

add_newdoc('numpy.core.numerictypes', 'bool_',
    """Boolean. Character code ``?``.""")

add_newdoc('numpy.core.numerictypes', 'complex64',
    """
    """)

add_newdoc('numpy.core.numerictypes', 'complex128',
    """
    """)

add_newdoc('numpy.core.numerictypes', 'complex256',
    """
    """)

add_newdoc('numpy.core.numerictypes', 'float32',
    """32-bit floating-point number. Character code ``f``.""")

add_newdoc('numpy.core.numerictypes', 'float64',
    """64-bit floating-point number. Character code ``d``.""")

add_newdoc('numpy.core.numerictypes', 'float96',
    """
    """)

add_newdoc('numpy.core.numerictypes', 'float128',
    """
    """)

add_newdoc('numpy.core.numerictypes', 'int8',
    """8-bit integer. Character code ``b``.""")

add_newdoc('numpy.core.numerictypes', 'int16',
    """16-bit integer. Character code ``h``.""")

add_newdoc('numpy.core.numerictypes', 'int32',
    """32-bit integer. Character code ``i4``.""")

add_newdoc('numpy.core.numerictypes', 'int64',
    """64-bit integer. Character code ``i8``.""")

add_newdoc('numpy.core.numerictypes', 'object_',
    """
    """)
