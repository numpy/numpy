# This is only meant to add docs to objects defined in C-extension modules.
# The purpose is to allow easier editing of the docstrings without
# requiring a re-compile.

# NOTE: Many of the methods of ndarray have corresponding functions.
#       If you update these docstrings, please keep also the ones in
#       core/fromnumeric.py, core/defmatrix.py up-to-date.

from lib import add_newdoc

add_newdoc('numpy.core', 'dtype',
"""Create a data type.

A numpy array is homogeneous, and contains elements described by a
dtype.  A dtype can be constructed from different combinations of
fundamental numeric types, as illustrated below.

Examples
--------

Using array-scalar type:
>>> dtype(int16)
dtype('int16')

Record, one field name 'f1', containing int16:
>>> dtype([('f1', int16)])
dtype([('f1', '<i2')])

Record, one field named 'f1', in itself containing a record with one field:
>>> dtype([('f1', [('f1', int16)])])
dtype([('f1', [('f1', '<i2')])])

Record, two fields: the first field contains an unsigned int, the
second an int32:
>>> dtype([('f1', uint), ('f2', int32)])
dtype([('f1', '<u4'), ('f2', '<i4')])

Using array-protocol type strings:
>>> dtype([('a','f8'),('b','S10')])
dtype([('a', '<f8'), ('b', '|S10')])

Using comma-separated field formats.  The shape is (2,3):
>>> dtype("i4, (2,3)f8")
dtype([('f0', '<i4'), ('f1', '<f8', (2, 3))])

Using tuples.  ``int`` is a fixed type, 3 the field's shape.  ``void``
is a flexible type, here of size 10:
>>> dtype([('hello',(int,3)),('world',void,10)])
dtype([('hello', '<i4', 3), ('world', '|V10')])

Subdivide ``int16`` into 2 ``int8``'s, called x and y.  0 and 1 are
the offsets in bytes:
>>> dtype((int16, {'x':(int8,0), 'y':(int8,1)}))
dtype(('<i2', [('x', '|i1'), ('y', '|i1')]))

Using dictionaries.  Two fields named 'gender' and 'age':
>>> dtype({'names':['gender','age'], 'formats':['S1',uint8]})
dtype([('gender', '|S1'), ('age', '|u1')])

Offsets in bytes, here 0 and 25:
>>> dtype({'surname':('S25',0),'age':(uint8,25)})
dtype([('surname', '|S25'), ('age', '|u1')])

""")

add_newdoc('numpy.core','dtype',
           [('fields', "Fields of the data-type or None if no fields"),
            ('names', "Names of fields or None if no fields"),
            ('alignment', "Needed alignment for this data-type"),
            ('byteorder',
             "Little-endian (<), big-endian (>), native (=), or "\
             "not-applicable (|)"),
            ('char', "Letter typecode for this data-type"),
            ('type', "Type object associated with this data-type"),
            ('kind', "Character giving type-family of this data-type"),
            ('itemsize', "Size of each item"),
            ('hasobject', "Non-zero if Python objects are in "\
             "this data-type"),
            ('num', "Internally-used number for builtin base"),
            ('newbyteorder',
"""self.newbyteorder(endian)

Returns a copy of the dtype object with altered byteorders.
If `endian` is not given all byteorders are swapped.
Otherwise endian can be '>', '<', or '=' to force a particular
byteorder.  Data-types in all fields are also updated in the
new dtype object.
"""),
            ("__reduce__", "self.__reduce__() for pickling"),
            ("__setstate__", "self.__setstate__() for pickling"),
            ("subdtype", "A tuple of (descr, shape) or None"),
            ("descr", "The array_interface data-type descriptor."),
            ("str", "The array interface typestring."),
            ("name", "The name of the true data-type"),
            ("base", "The base data-type or self if no subdtype"),
            ("shape", "The shape of the subdtype or (1,)"),
            ("isbuiltin", "Is this a built-in data-type?"),
            ("isnative", "Is the byte-order of this data-type native?")
            ]
           )

###############################################################################
#
# flatiter
#
# flatiter needs a toplevel description
#
###############################################################################

# attributes
add_newdoc('numpy.core', 'flatiter', ('base',
    """documentation needed

    """))



add_newdoc('numpy.core', 'flatiter', ('coords',
    """An N-d tuple of current coordinates.

    """))



add_newdoc('numpy.core', 'flatiter', ('index',
    """documentation needed

    """))



# functions
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

add_newdoc('numpy.core.multiarray','array',
    """array(object, dtype=None, copy=1,order=None, subok=0,ndmin=0)

    Return an array from object with the specified data-type.

    Parameters
    ----------
    object : array-like
        an array, any object exposing the array interface, any
        object whose __array__ method returns an array, or any
        (nested) sequence.
    dtype : data-type
        The desired data-type for the array.  If not given, then
        the type will be determined as the minimum type required
        to hold the objects in the sequence.  This argument can only
        be used to 'upcast' the array.  For downcasting, use the
        .astype(t) method.
    copy : bool
        If true, then force a copy.  Otherwise a copy will only occur
        if __array__ returns a copy, obj is a nested sequence, or
        a copy is needed to satisfy any of the other requirements
    order : {'C', 'F', 'A' (None)}
        Specify the order of the array.  If order is 'C', then the
        array will be in C-contiguous order (last-index varies the
        fastest).  If order is 'FORTRAN', then the returned array
        will be in Fortran-contiguous order (first-index varies the
        fastest).  If order is None, then the returned array may
        be in either C-, or Fortran-contiguous order or even
        discontiguous.
    subok : bool
        If True, then sub-classes will be passed-through, otherwise
        the returned array will be forced to be a base-class array
    ndmin : int
        Specifies the minimum number of dimensions that the resulting
        array should have.  1's will be pre-pended to the shape as
        needed to meet this requirement.

    """)

add_newdoc('numpy.core.multiarray','empty',
    """empty(shape, dtype=float, order='C')

    Return a new array of given shape and type with all entries uninitialized.
    This can be faster than zeros.

    Parameters
    ----------
    shape : tuple of integers
        Shape of the new array
    dtype : data-type
        The desired data-type for the array.
    order : {'C', 'F'}
        Whether to store multidimensional data in C or Fortran order.

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

add_newdoc('numpy.core.multiarray','zeros',
    """zeros(shape, dtype=float, order='C')

    Return a new array of given shape and type, filled zeros.

    Parameters
    ----------
    shape : tuple of integers
        Shape of the new array
    dtype : data-type
        The desired data-type for the array.
    order : {'C', 'F'}
        Whether to store multidimensional data in C or Fortran order.

    """)

add_newdoc('numpy.core.multiarray','set_typeDict',
    """set_typeDict(dict)

    Set the internal dictionary that can look up an array type using a
    registered code.

    """)

add_newdoc('numpy.core.multiarray','fromstring',
    """fromstring(string, dtype=float, count=-1, sep='')

    Return a new 1d array initialized from the raw binary data in string.

    If count is positive, the new array will have count elements, otherwise its
    size is determined by the size of string.  If sep is not empty then the
    string is interpreted in ASCII mode and converted to the desired number type
    using sep as the separator between elements (extra whitespace is ignored).
    ASCII integer conversions are base-10; octal and hex are not supported.

    """)

add_newdoc('numpy.core.multiarray','fromiter',
    """fromiter(iterable, dtype, count=-1)

    Return a new 1d array initialized from iterable.

    Parameters
    ----------
    iterable
        Iterable object from which to obtain data
    dtype : data-type
        Data type of the returned array.
    count : int
        Number of items to read. -1 means all data in the iterable.

    Returns
    -------
    new_array : ndarray

    """)

add_newdoc('numpy.core.multiarray','fromfile',
    """fromfile(file=, dtype=float, count=-1, sep='')

    Return an array of the given data type from a text or binary file.

    Data written using the tofile() method can be conveniently recovered using
    this function.

    Parameters
    ----------
    file : file or string
        Open file object or string containing a file name.
    dtype : data-type
        Data type of the returned array.
        For binary files, it is also used to determine the size and order of
        the items in the file.
    count : int
        Number of items to read. -1 means all data in the whole file.
    sep : string
        Separator between items if file is a text file.
        Empty ("") separator means the file should be treated as binary.

    See also
    --------
    loadtxt : load data from text files

    Notes
    -----
    WARNING: This function should be used sparingly as the binary files are not
    platform independent. In particular, they contain no endianess or datatype
    information. Nevertheless it can be useful for reading in simply formatted
    or binary data quickly.

    """)

add_newdoc('numpy.core.multiarray','frombuffer',
    """frombuffer(buffer=, dtype=float, count=-1, offset=0)

    Returns a 1-d array of data type dtype from buffer.

    Parameters
    ----------
    buffer
        An object that exposes the buffer interface
    dtype : data-type
        Data type of the returned array.
    count : int
        Number of items to read. -1 means all data in the buffer.
    offset : int
        Number of bytes to jump from the start of the buffer before reading

    Notes
    -----
    If the buffer has data that is not in machine byte-order, then
    use a proper data type descriptor. The data will not be
    byteswapped, but the array will manage it in future operations.

    """)

add_newdoc('numpy.core.multiarray','concatenate',
    """concatenate((a1, a2, ...), axis=0)

    Join arrays together.

    The tuple of sequences (a1, a2, ...) are joined along the given axis
    (default is the first one) into a single numpy array.

    Examples
    --------
    >>> concatenate( ([0,1,2], [5,6,7]) )
    array([0, 1, 2, 5, 6, 7])

    """)

add_newdoc('numpy.core.multiarray','inner',
    """inner(a,b)

    Returns the dot product of two arrays, which has shape a.shape[:-1] +
    b.shape[:-1] with elements computed by the product of the elements
    from the last dimensions of a and b.

    """)

add_newdoc('numpy.core','fastCopyAndTranspose',
    """_fastCopyAndTranspose(a)""")

add_newdoc('numpy.core.multiarray','correlate',
    """cross_correlate(a,v, mode=0)""")

add_newdoc('numpy.core.multiarray','arange',
    """arange([start,] stop[, step,], dtype=None)

    For integer arguments, just like range() except it returns an array
    whose type can be specified by the keyword argument dtype.  If dtype
    is not specified, the type of the result is deduced from the type of
    the arguments.

    For floating point arguments, the length of the result is ceil((stop -
    start)/step).  This rule may result in the last element of the result
    being greater than stop.

    """)

add_newdoc('numpy.core.multiarray','_get_ndarray_c_version',
    """_get_ndarray_c_version()

    Return the compile time NDARRAY_VERSION number.

    """)

add_newdoc('numpy.core.multiarray','_reconstruct',
    """_reconstruct(subtype, shape, dtype)

    Construct an empty array. Used by Pickles.

    """)


add_newdoc('numpy.core.multiarray','set_string_function',
    """set_string_function(f, repr=1)

    Set the python function f to be the function used to obtain a pretty
    printable string version of an array whenever an array is printed.
    f(M) should expect an array argument M, and should return a string
    consisting of the desired representation of M for printing.

    """)

add_newdoc('numpy.core.multiarray','set_numeric_ops',
    """set_numeric_ops(op=func, ...)

    Set some or all of the number methods for all array objects.  Do not
    forget **dict can be used as the argument list.  Return the functions
    that were replaced, which can be stored and set later.

    """)

add_newdoc('numpy.core.multiarray','where',
    """where(condition, x, y) or where(condition)

    Return elements from `x` or `y`, depending on `condition`.

    Parameters
    ----------
    condition : array of bool
        When True, yield x, otherwise yield y.
    x,y : 1-dimensional arrays
        Values from which to choose.

    Notes
    -----
    This is equivalent to

        [xv if c else yv for (c,xv,yv) in zip(condition,x,y)]

    The result is shaped like `condition` and has elements of `x`
    or `y` where `condition` is respectively True or False.

    In the special case, where only `condition` is given, the
    tuple condition.nonzero() is returned, instead.

    Examples
    --------
    >>> where([True,False,True],[1,2,3],[4,5,6])
    array([1, 5, 3])

    """)


add_newdoc('numpy.core.multiarray','lexsort',
    """lexsort(keys=, axis=-1) -> array of indices. Argsort with list of keys.

    Perform an indirect sort using a list of keys. The first key is sorted,
    then the second, and so on through the list of keys. At each step the
    previous order is preserved when equal keys are encountered. The result is
    a sort on multiple keys.  If the keys represented columns of a spreadsheet,
    for example, this would sort using multiple columns (the last key being
    used for the primary sort order, the second-to-last key for the secondary
    sort order, and so on).

    Parameters
    ----------
    keys : (k,N) array or tuple of (N,) sequences
        Array containing values that the returned indices should sort, or
        a sequence of things that can be converted to arrays of the same shape.

    axis : integer
        Axis to be indirectly sorted.  Default is -1 (i.e. last axis).

    Returns
    -------
    indices : (N,) integer array
        Array of indices that sort the keys along the specified axis.

    See Also
    --------
    argsort : indirect sort
    sort : inplace sort

    Examples
    --------
    >>> a = [1,5,1,4,3,6,7]
    >>> b = [9,4,0,4,0,4,3]
    >>> ind = lexsort((b,a))
    >>> print ind
    [2 0 4 3 1 5 6]
    >>> print take(a,ind)
    [1 1 3 4 5 6 7]
    >>> print take(b,ind)
    [0 9 0 4 4 4 3]

    """)

add_newdoc('numpy.core.multiarray','can_cast',
    """can_cast(from=d1, to=d2)

    Returns True if data type d1 can be cast to data type d2 without
    losing precision.

    """)

add_newdoc('numpy.core.multiarray','newbuffer',
    """newbuffer(size)

    Return a new uninitialized buffer object of size bytes

    """)

add_newdoc('numpy.core.multiarray','getbuffer',
    """getbuffer(obj [,offset[, size]])

    Create a buffer object from the given object referencing a slice of
    length size starting at offset.  Default is the entire buffer. A
    read-write buffer is attempted followed by a read-only buffer.

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
    """An array object represents a multidimensional, homogeneous array
    of fixed-size items.  An associated data-type-descriptor object
    details the data-type in an array (including byteorder and any
    fields).  An array can be constructed using the numpy.array
    command. Arrays are sequence, mapping and numeric objects.
    More information is available in the numpy module and by looking
    at the methods and attributes of an array.

    ndarray.__new__(subtype, shape=, dtype=float, buffer=None,
                    offset=0, strides=None, order=None)

     There are two modes of creating an array using __new__:
     1) If buffer is None, then only shape, dtype, and order
        are used
     2) If buffer is an object exporting the buffer interface, then
        all keywords are interpreted.
     The dtype parameter can be any object that can be interpreted
        as a numpy.dtype object.

     No __init__ method is needed because the array is fully
     initialized after the __new__ method.

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
    """Base object if memory is from some other object.

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
    """Imaginary part of the array.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('itemsize',
    """Length of one element in bytes.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('flags',
    """Special object providing array flags.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('flat',
    """A 1-d flat iterator.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('nbytes',
    """Number of bytes in the array.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('ndim',
    """Number of array dimensions.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('real',
    """Real part of the array.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('shape',
    """Tuple of array dimensions.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('size',
    """Number of elements in the array.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('strides',
    """Tuple of bytes to step in each dimension.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('T',
    """Same as self.transpose() except self is returned for self.ndim < 2.

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


add_newdoc('numpy.core.multiarray', 'ndarray', ('__array_wrap__',
    """a.__array_wrap__(obj) -> Object of same type as a from ndarray obj.

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
    """a.all(axis=None, out=None)

    Check if all of the elements of `a` are true.

    Performs a logical_and over the given axis and returns the result

    Parameters
    ----------
    axis : {None, integer}
        Axis to perform the operation over.
        If None, perform over flattened array.
    out : {None, array}, optional
        Array into which the result can be placed. Its type is preserved
        and it must be of the right shape to hold the output.

    See Also
    --------
    all : equivalent function

     """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('any',
    """a.any(axis=None, out=None)

    Check if any of the elements of `a` are true.

    Performs a logical_or over the given axis and returns the result

    Parameters
    ----------
    axis : {None, integer}
        Axis to perform the operation over.
        If None, perform over flattened array and return a scalar.
    out : {None, array}, optional
        Array into which the result can be placed. Its type is preserved
        and it must be of the right shape to hold the output.

    See Also
    --------
    any : equivalent function

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('argmax',
    """a.argmax(axis=None, out=None)

    Returns array of indices of the maximum values along the given axis.

    Parameters
    ----------
    axis : {None, integer}
        If None, the index is into the flattened array, otherwise along
        the specified axis
    out : {None, array}, optional
        Array into which the result can be placed. Its type is preserved
        and it must be of the right shape to hold the output.

    Returns
    -------
    index_array : {integer_array}

    Examples
    --------
    >>> a = arange(6).reshape(2,3)
    >>> a.argmax()
    5
    >>> a.argmax(0)
    array([1, 1, 1])
    >>> a.argmax(1)
    array([2, 2])

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('argmin',
    """a.argmin(axis=None, out=None)

    Return array of indices to the minimum values along the given axis.

    Parameters
    ----------
    axis : {None, integer}
        If None, the index is into the flattened array, otherwise along
        the specified axis
    out : {None, array}, optional
        Array into which the result can be placed. Its type is preserved
        and it must be of the right shape to hold the output.

    Returns
    -------
    index_array : {integer_array}

    Examples
    --------
    >>> a = arange(6).reshape(2,3)
    >>> a.argmin()
    0
    >>> a.argmin(0)
    array([0, 0, 0])
    >>> a.argmin(1)
    array([0, 0])

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('argsort',
    """a.argsort(axis=-1, kind='quicksort', order=None) -> indices

    Perform an indirect sort along the given axis using the algorithm specified
    by the kind keyword. It returns an array of indices of the same shape as
    'a' that index data along the given axis in sorted order.

    Parameters
    ----------
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

    Returns
    -------
    indices : integer array
        Array of indices that sort 'a' along the specified axis.

    See Also
    --------
    lexsort : indirect stable sort with multiple keys
    sort : inplace sort

    Notes
    -----
    The various sorts are characterized by average speed, worst case
    performance, need for work space, and whether they are stable. A stable
    sort keeps items with the same key in the same relative order. The three
    available algorithms have the following properties:

    ============ ======= ============= ============ ========
        kind      speed    worst case   work space   stable
    ============ ======= ============= ============ ========
     'quicksort'    1     O(n^2)            0         no
     'mergesort'    2     O(n*log(n))      ~n/2       yes
     'heapsort'     3     O(n*log(n))       0         no
    ============ ======= ============= ============ ========

    All the sort algorithms make temporary copies of the data when the
    sort is not along the last axis. Consequently, sorts along the
    last axis are faster and use less space than sorts along other
    axis.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('astype',
    """a.astype(t) -> Copy of array cast to type t.

    Cast array m to type t.  t can be either a string representing a typecode,
    or a python type object of type int, float, or complex.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('byteswap',
    """a.byteswap(False) -> View or copy. Swap the bytes in the array.

    Swap the bytes in the array.  Return the byteswapped array.  If the first
    argument is True, byteswap in-place and return a reference to self.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('choose',
    """
    a.choose(choices, out=None, mode='raise')

    Use an index array to construct a new array from a set of choices.

    Given an array of integers and a set of n choice arrays, this method
    will create a new array that merges each of the choice arrays.  Where a
    value in `a` is i, the new array will have the value that choices[i]
    contains in the same place.

    Parameters
    ----------
    choices : sequence of arrays
        Choice arrays. The index array and all of the choices should be
        broadcastable to the same shape.
    out : array, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype
    mode : {'raise', 'wrap', 'clip'}, optional
        Specifies how out-of-bounds indices will behave.
        'raise' : raise an error
        'wrap' : wrap around
        'clip' : clip to the range

    Returns
    -------
    merged_array : array

    See Also
    --------
    choose : equivalent function

    Examples
    --------
    >>> choices = [[0, 1, 2, 3], [10, 11, 12, 13],
    ...   [20, 21, 22, 23], [30, 31, 32, 33]]
    >>> a = array([2, 3, 1, 0], dtype=int)
    >>> a.choose(choices)
    array([20, 31, 12,  3])
    >>> a = array([2, 4, 1, 0], dtype=int)
    >>> a.choose(choices, mode='clip')
    array([20, 31, 12,  3])
    >>> a.choose(choices, mode='wrap')
    array([20,  1, 12,  3])

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('clip',
    """a.clip(a_min, a_max, out=None)

    Return an array whose values are limited to [a_min, a_max].

    Parameters
    ----------
    a_min
        Minimum value
    a_max
        Maximum value
    out : {None, array}, optional
        Array into which the clipped values can be placed.  Its type
        is preserved and it must be of the right shape to hold the
        output.

    Returns
    -------
    clipped_array : array
        A new array whose elements are same as for a, but values
        < a_min are replaced with a_min, and > a_max with a_max.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('compress',
    """a.compress(condition, axis=None, out=None)

    Return selected slices of an array along given axis.

    Parameters
    ----------
    condition : {array}
        Boolean 1-d array selecting which entries to return. If len(condition)
        is less than the size of a along the axis, then output is truncated
        to length of condition array.
    axis : {None, integer}
        Axis along which to take slices. If None, work on the flattened array.
    out : array, optional
        Output array.  Its type is preserved and it must be of the right
        shape to hold the output.

    Returns
    -------
    compressed_array : array
        A copy of a, without the slices along axis for which condition is false.

    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> a.compress([0, 1], axis=0)
    array([[3, 4]])
    >>> a.compress([1], axis=1)
    array([[1],
           [3]])
    >>> a.compress([0,1,1])
    array([2, 3])

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
    """a.copy([order])

    Return a copy of the array.

    Parameters
    ----------
    order : {'C', 'F', 'A'}, optional
        If order is 'C' (False) then the result is contiguous (default).
        If order is 'Fortran' (True) then the result has fortran order.
        If order is 'Any' (None) then the result has fortran order
        only if the array already is in fortran order.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('cumprod',
    """a.cumprod(axis=None, dtype=None, out=None)

    Return the cumulative product of the elements along the given axis.

    The cumulative product is taken over the flattened array by
    default, otherwise over the specified axis.

    Parameters
    ----------
    axis : {None, -1, int}, optional
        Axis along which the product is computed. The default
        (``axis``= None) is to compute over the flattened array.
    dtype : {None, dtype}, optional
        Determines the type of the returned array and of the accumulator
        where the elements are multiplied. If dtype has the value None and
        the type of a is an integer type of precision less than the default
        platform integer, then the default platform integer precision is
        used.  Otherwise, the dtype is the same as that of a.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type will be cast if necessary.

    Returns
    -------
    cumprod : ndarray.
        A new array holding the result is returned unless out is
        specified, in which case a reference to out is returned.

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('cumsum',
    """a.cumsum(axis=None, dtype=None, out=None)

    Return the cumulative sum of the elements along the given axis.

    The cumulative sum is calculated over the flattened array by
    default, otherwise over the specified axis.

    Parameters
    ----------
    axis : {None, -1, int}, optional
        Axis along which the sum is computed. The default
        (``axis``= None) is to compute over the flattened array.
    dtype : {None, dtype}, optional
        Determines the type of the returned array and of the accumulator
        where the elements are summed. If dtype has the value None and
        the type of a is an integer type of precision less than the default
        platform integer, then the default platform integer precision is
        used.  Otherwise, the dtype is the same as that of a.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type will be cast if necessary.

    Returns
    -------
    cumsum : ndarray.
        A new array holding the result is returned unless ``out`` is
        specified, in which case a reference to ``out`` is returned.

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('diagonal',
    """a.diagonal(offset=0, axis1=0, axis2=1) -> diagonals

    If a is 2-d, return the diagonal of self with the given offset, i.e., the
    collection of elements of the form a[i,i+offset]. If a is n-d with n > 2,
    then the axes specified by axis1 and axis2 are used to determine the 2-d
    subarray whose diagonal is returned. The shape of the resulting array can
    be determined by removing axis1 and axis2 and appending an index to the
    right equal to the size of the resulting diagonals.

    Parameters
    ----------
    offset : integer
        Offset of the diagonal from the main diagonal. Can be both positive
        and negative. Defaults to main diagonal.
    axis1 : integer
        Axis to be used as the first axis of the 2-d subarrays from which
        the diagonals should be taken. Defaults to first index.
    axis2 : integer
        Axis to be used as the second axis of the 2-d subarrays from which
        the diagonals should be taken. Defaults to second index.

    Returns
    -------
    array_of_diagonals : same type as original array
        If a is 2-d, then a 1-d array containing the diagonal is returned.
        If a is n-d, n > 2, then an array of diagonals is returned.

    See Also
    --------
    diag : matlab workalike for 1-d and 2-d arrays.
    diagflat : creates diagonal arrays
    trace : sum along diagonals

    Examples
    --------
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
    """a.fill(value)

    Fill the array with a scalar value.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('flatten',
    """a.flatten([order])

    Return a 1-d array (always copy)

    Parameters
    ----------
    order : {'C', 'F'}
        Whether to flatten in C or Fortran order.

    Notes
    -----
    a.flatten('F') == a.T.flatten('C')

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
    """a.max(axis=None, out=None)

    Return the maximum along a given axis.

    Parameters
    ----------
    axis : {None, int}, optional
        Axis along which to operate.  By default, ``axis`` is None and the
        flattened input is used.
    out : array_like, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.

    Returns
    -------
    amax : array_like
        New array holding the result.
        If ``out`` was specified, ``out`` is returned.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('mean',
    """a.mean(axis=None, dtype=None, out=None) -> mean

    Returns the average of the array elements.  The average is taken over the
    flattened array by default, otherwise over the specified axis.

    Parameters
    ----------
    axis : integer
        Axis along which the means are computed. The default is
        to compute the mean of the flattened array.
    dtype : type
        Type to use in computing the means. For arrays of
        integer type the default is float32, for arrays of float types it
        is the same as the array type.
    out : ndarray
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type will be cast if
        necessary.

    Returns
    -------
    mean : The return type varies, see above.
        A new array holding the result is returned unless out is specified,
        in which case a reference to out is returned.

    See Also
    --------
    var : variance
    std : standard deviation

    Notes
    -----
    The mean is the sum of the elements along the axis divided by the
    number of elements.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('min',
    """a.min(axis=None, out=None)

    Return the minimum along a given axis.

    Parameters
    ----------
    axis : {None, int}, optional
        Axis along which to operate.  By default, ``axis`` is None and the
        flattened input is used.
    out : array_like, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.

    Returns
    -------
    amin : array_like
        New array holding the result.
        If ``out`` was specified, ``out`` is returned.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('newbyteorder',
    """a.newbyteorder(byteorder)

    Equivalent to a.view(a.dtype.newbytorder(byteorder))

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('nonzero',
    """a.nonzero()

    Returns a tuple of arrays, one for each dimension of a, containing
    the indices of the non-zero elements in that dimension. The
    corresponding non-zero values can be obtained with::

        a[a.nonzero()].

    To group the indices by element, rather than dimension, use::

        transpose(a.nonzero())

    instead. The result of this is always a 2d array, with a row for
    each non-zero element.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('prod',
    """a.prod(axis=None, dtype=None, out=None)

    Return the product of the array elements over the given axis

    Parameters
    ----------
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

    Returns
    -------
    product_along_axis : {array, scalar}, see dtype parameter above.
        Returns an array whose shape is the same as a with the specified
        axis removed. Returns a 0d array when a is 1d or axis=None.
        Returns a reference to the specified output array if specified.

    See Also
    --------
    prod : equivalent function

    Examples
    --------
    >>> prod([1.,2.])
    2.0
    >>> prod([1.,2.], dtype=int32)
    2
    >>> prod([[1.,2.],[3.,4.]])
    24.0
    >>> prod([[1.,2.],[3.,4.]], axis=1)
    array([  2.,  12.])

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('ptp',
    """a.ptp(axis=None, out=None)

    Return (maximum - minimum) along the the given dimension
    (i.e. peak-to-peak value).

    Parameters
    ----------
    axis : {None, int}, optional
        Axis along which to find the peaks.  If None (default) the
        flattened array is used.
    out : array_like
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type will be cast if necessary.

    Returns
    -------
    ptp : ndarray.
        A new array holding the result, unless ``out`` was
        specified, in which case a reference to ``out`` is returned.

    Examples
    --------
    >>> x = np.arange(4).reshape((2,2))
    >>> x
    array([[0, 1],
           [2, 3]])
    >>> x.ptp(0)
    array([2, 2])
    >>> x.ptp(1)
    array([1, 1])

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('put',
    """a.put(indices, values, mode='raise')

    Set a.flat[n] = values[n] for all n in indices.
    If values is shorter than indices, it will repeat.

    Parameters
    ----------
    indices : array_like
        Target indices, interpreted as integers.
    values : array_like
        Values to place in `a` at target indices.
    mode : {'raise', 'wrap', 'clip'}
        Specifies how out-of-bounds indices will behave.
        'raise' -- raise an error
        'wrap' -- wrap around
        'clip' -- clip to the range

    Notes
    -----
    If v is shorter than mask it will be repeated as necessary.  In particular v
    can be a scalar or length 1 array.  The routine put is the equivalent of the
    following (although the loop is in C for speed):

        ind = array(indices, copy=False)
        v = array(values, copy=False).astype(a.dtype)
        for i in ind: a.flat[i] = v[i]

    Examples
    --------
    >>> x = np.arange(5)
    >>> x.put([0,2,4],[-1,-2,-3])
    >>> print x
    [-1  1 -2  3 -3]

    """))


add_newdoc('numpy.core.multiarray', 'putmask',
    """putmask(a, mask, values)

    Sets a.flat[n] = values[n] for each n where mask.flat[n] is true.

    If values is not the same size as `a` and `mask` then it will repeat.
    This gives behavior different from a[mask] = values.

    Parameters
    ----------
    a : {array_like}
        Array to put data into
    mask : {array_like}
        Boolean mask array
    values : {array_like}
        Values to put

    """)


add_newdoc('numpy.core.multiarray', 'ndarray', ('ravel',
    """a.ravel([order])

    Return a 1d array containing the elements of a (copy only if needed).

    The elements in the new array are taken in the order specified by
    the order keyword. The new array is a view of a if possible,
    otherwise it is a copy.

    Parameters
    ----------
    order : {'C','F'}, optional
        If order is 'C' the elements are taken in row major order. If order
        is 'F' they are taken in column major order.

    Returns
    -------
    1d_array : {array}

    See Also
    --------
    ndarray.flat : 1d iterator over the array.
    ndarray.flatten : 1d array copy of the elements of a in C order.

    Examples
    --------
    >>> x = array([[1,2,3],[4,5,6]])
    >>> x
    array([[1, 2, 3],
          [4, 5, 6]])
    >>> x.ravel()
    array([1, 2, 3, 4, 5, 6])

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('repeat',
    """a.repeat(repeats, axis=None)

    Repeat elements of an array.

    Parameters
    ----------
    a : {array_like}
        Input array.
    repeats : {integer, integer_array}
        The number of repetitions for each element. If a plain integer, then
        it is applied to all elements. If an array, it needs to be of the
        same length as the chosen axis.
    axis : {None, integer}, optional
        The axis along which to repeat values. If None, then this method
        will operated on the flattened array `a` and return a similarly flat
        result.

    Returns
    -------
    repeated_array : array

    See also
    --------
    tile : tile an array

    Examples
    --------
    >>> x = array([[1,2],[3,4]])
    >>> x.repeat(2)
    array([1, 1, 2, 2, 3, 3, 4, 4])
    >>> x.repeat(3, axis=1)
    array([[1, 1, 1, 2, 2, 2],
           [3, 3, 3, 4, 4, 4]])
    >>> x.repeat([1, 2], axis=0)
    array([[1, 2],
           [3, 4],
           [3, 4]])

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('reshape',
    """a.reshape(shape, order='C')
    a.reshape(*shape, order='C')

    Returns an array containing the data of a, but with a new shape.

    The result is a view to the original array; if this is not possible,
    a ValueError is raised.

    Parameters
    ----------
    shape : shape tuple or int
       The new shape should be compatible with the original shape. If an
       integer, then the result will be a 1D array of that length.
    order : {'C', 'F'}, optional
        Determines whether the array data should be viewed as in C
        (row-major) order or FORTRAN (column-major) order.

    Returns
    -------
    reshaped_array : array
        A new view to the array.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('resize',
    """a.resize(new_shape, refcheck=True, order=False)

    Change size and shape of self inplace.  Array must own its own memory and
    not be referenced by other arrays. Returns None.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('round',
    """a.round(decimals=0, out=None)

    Return an array rounded a to the given number of decimals.

    The real and imaginary parts of complex numbers are rounded separately. The
    result of rounding a float is a float so the type must be cast if integers
    are desired.  Nothing is done if the input is an integer array and the
    decimals parameter has a value >= 0.

    Parameters
    ----------
    decimals : {0, integer}, optional
        Number of decimal places to round to. When decimals is negative it
        specifies the number of positions to the left of the decimal point.
    out : {None, array}, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type will be cast if
        necessary.

    Returns
    -------
    rounded_array : {array}
        If out=None, returns a new array of the same type as a containing
        the rounded values, otherwise a reference to the output array is
        returned.

    See Also
    --------
    around : equivalent function

    Notes
    -----
    Numpy rounds to even. Thus 1.5 and 2.5 round to 2.0, -0.5 and 0.5 round
    to 0.0, etc. Results may also be surprising due to the inexact
    representation of decimal fractions in IEEE floating point and the
    errors introduced when scaling by powers of ten.

    Examples
    --------
    >>> x = array([.5, 1.5, 2.5, 3.5, 4.5])
    >>> x.round()
    array([ 0.,  2.,  2.,  4.,  4.])
    >>> x = array([1,2,3,11])
    >>> x.round(decimals=1)
    array([ 1,  2,  3, 11])
    >>> x.round(decimals=-1)
    array([ 0,  0,  0, 10])

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('searchsorted',
    """a.searchsorted(v, side='left')

    Find the indices into a sorted array such that if the corresponding keys in
    v were inserted before the indices the order of a would be preserved.  If
    side='left', then the first such index is returned. If side='right', then
    the last such index is returned. If there is no such index because the key
    is out of bounds, then the length of a is returned, i.e., the key would
    need to be appended. The returned index array has the same shape as v.

    Parameters
    ----------
    v : array or list type
        Array of keys to be searched for in a.
    side : string
        Possible values are : 'left', 'right'. Default is 'left'. Return
        the first or last index where the key could be inserted.

    Returns
    -------
    indices : integer array
        The returned array has the same shape as v.

    See also
    --------
    sort
    histogram

    Notes
    -----
    The array a must be 1-d and is assumed to be sorted in ascending order.
    Searchsorted uses binary search to find the required insertion points.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('setfield',
    """m.setfield(value, dtype, offset) -> None.
    places val into field of the given array defined by the data type and offset.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('setflags',
    """a.setflags(write=None, align=None, uic=None)

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('sort',
    """a.sort(axis=-1, kind='quicksort', order=None) -> None.

    Perform an inplace sort along the given axis using the algorithm specified
    by the kind keyword.

    Parameters
    ----------
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

    See Also
    --------
    argsort : indirect sort
    lexsort : indirect stable sort on multiple keys
    searchsorted : find keys in sorted array

    Notes
    -----

    The various sorts are characterized by average speed, worst case
    performance, need for work space, and whether they are stable. A stable
    sort keeps items with the same key in the same relative order. The three
    available algorithms have the following properties:

    =========== ======= ============= ============ =======
       kind      speed   worst case    work space  stable
    =========== ======= ============= ============ =======
    'quicksort'    1     O(n^2)            0          no
    'mergesort'    2     O(n*log(n))      ~n/2        yes
    'heapsort'     3     O(n*log(n))       0          no
    =========== ======= ============= ============ =======

    All the sort algorithms make temporary copies of the data when the sort is
    not along the last axis. Consequently, sorts along the last axis are faster
    and use less space than sorts along other axis.
    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('squeeze',
    """m.squeeze()

    Remove single-dimensional entries from the shape of a.

    Examples
    --------
    >>> x = array([[[1,1,1],[2,2,2],[3,3,3]]])
    >>> x.shape
    (1, 3, 3)
    >>> x.squeeze().shape
    (3, 3)

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('std',
    """a.std(axis=None, dtype=None, out=None, ddof=0)

    Returns the standard deviation of the array elements, a measure of the
    spread of a distribution. The standard deviation is computed for the
    flattened array by default, otherwise over the specified axis.

    Parameters
    ----------
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
    ddof : {0, integer}
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is N-ddof.

    Returns
    -------
    standard deviation : The return type varies, see above.
        A new array holding the result is returned unless out is specified,
        in which case a reference to out is returned.

    See Also
    --------
    var : variance
    mean : average

    Notes
    -----
    The standard deviation is the square root of the average of the squared
    deviations from the mean, i.e. var = sqrt(mean(abs(x - x.mean())**2)).  The
    computed standard deviation is computed by dividing by the number of
    elements, N-ddof. The option ddof defaults to zero, that is, a biased
    estimate. Note that for complex numbers std takes the absolute value before
    squaring, so that the result is always real and nonnegative.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('sum',
    """a.sum(axis=None, dtype=None, out=None)

    Return the sum of the array elements over the given axis

    Parameters
    ----------
    axis : {None, integer}
        Axis over which the sum is taken. If None is used, then the sum is
        over all the array elements.
    dtype : {None, dtype}, optional
        Determines the type of the returned array and of the accumulator where
        the elements are summed. If dtype has the value None and the type of a
        is an integer type of precision less than the default platform integer,
        then the default platform integer precision is used.  Otherwise, the
        dtype is the same as that of a.
    out : {None, array}, optional
        Array into which the sum can be placed. Its type is preserved and it
        must be of the right shape to hold the output.

    Returns
    -------
    sum_along_axis : {array, scalar}, see dtype parameter above.
        Returns an array whose shape is the same as a with the specified axis
        removed. Returns a 0d array when a is 1d or axis=None.  Returns a
        reference to the specified output array if specified.

    See Also
    --------
    sum : equivalent function

    Examples
    --------
    >>> array([0.5, 1.5]).sum()
    2.0
    >>> array([0.5, 1.5]).sum(dtype=int32)
    1
    >>> array([[0, 1], [0, 5]]).sum(axis=0)
    array([0, 6])
    >>> array([[0, 1], [0, 5]]).sum(axis=1)
    array([1, 5])
    >>> ones(128, dtype=int8).sum(dtype=int8) # overflow!
    -128

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('swapaxes',
    """a.swapaxes(axis1, axis2)

    Return a view of the array with axis1 and axis2 interchanged.

    Parameters
    ----------
    axis1 : int
        First axis.
    axis2 : int
        Second axis.

    Examples
    --------
    >>> x = np.array([[1,2,3]])
    >>> x.swapaxes(0,1)
    array([[1],
           [2],
           [3]])

    >>> x = np.array([[[0,1],[2,3]],[[4,5],[6,7]]])
    >>> x
    array([[[0, 1],
            [2, 3]],

           [[4, 5],
            [6, 7]]])
    >>> x.swapaxes(0,2)
    array([[[0, 4],
            [2, 6]],

           [[1, 5],
            [3, 7]]])

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('take',
    """a.take(indices, axis=None, out=None, mode='raise')

    Return an array formed from the elements of a at the given indices.

    This method does the same thing as "fancy" indexing; however, it can
    be easier to use if you need to specify a given axis.

    Parameters
    ----------
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

    Returns
    -------
    subarray : array
        The returned array has the same type as a.

    See Also
    --------
    take : equivalent function

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('tofile',
    """a.tofile(fid, sep="", format="%s")

    Write the data to a file.

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
    """a.tolist()

    Return the array as nested lists.

    Copy the data portion of the array to a hierarchical Python list and return
    that list. Data items are converted to the nearest compatible Python type.

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
    """a.trace(offset=0, axis1=0, axis2=1, dtype=None, out=None)

    Return the sum along diagonals of the array.

    If a is 2-d, returns the sum along the diagonal of self with the given
    offset, i.e., the collection of elements of the form a[i,i+offset]. If a
    has more than two dimensions, then the axes specified by axis1 and axis2
    are used to determine the 2-d subarray whose trace is returned. The shape
    of the resulting array can be determined by removing axis1 and axis2 and
    appending an index to the right equal to the size of the resulting
    diagonals.

    Parameters
    ----------
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
        Array into which the sum can be placed. Its type is preserved and
        it must be of the right shape to hold the output.

    Returns
    -------
    sum_along_diagonals : array
        If a is 2-d, a 0-d array containing the diagonal is
        returned.  If a has larger dimensions, then an array of
        diagonals is returned.

    Examples
    --------
    >>> eye(3).trace()
    3.0
    >>> a = arange(8).reshape((2,2,2))
    >>> a.trace()
    array([6, 8])

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('transpose',
    """a.transpose(*axes)

    Returns a view of 'a' with axes transposed. If no axes are given,
    or None is passed, switches the order of the axes. For a 2-d
    array, this is the usual matrix transpose. If axes are given,
    they describe how the axes are permuted.

    Examples
    --------
    >>> a = array([[1,2],[3,4]])
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
    """a.var(axis=None, dtype=None, out=None, ddof=0) -> variance

    Returns the variance of the array elements, a measure of the spread of a
    distribution.  The variance is computed for the flattened array by default,
    otherwise over the specified axis.

    Parameters
    ----------
    axis : integer
        Axis along which the variance is computed. The default is to
        compute the variance of the flattened array.
    dtype : data-type
        Type to use in computing the variance. For arrays of integer type
        the default is float32, for arrays of float types it is the same as
        the array type.
    out : ndarray
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type will be cast if
        necessary.
    ddof : {0, integer},
        Means Delta Degrees of Freedom.  The divisor used in calculation is
        N - ddof.

    Returns
    -------
    variance : The return type varies, see above.
        A new array holding the result is returned unless out is specified,
        in which case a reference to out is returned.

    See Also
    --------
    std : standard deviation
    mean: average

    Notes
    -----
    The variance is the average of the squared deviations from the mean,
    i.e.  var = mean(abs(x - x.mean())**2).  The mean is computed by
    dividing by N-ddof, where N is the number of elements. The argument
    ddof defaults to zero; for an unbiased estimate supply ddof=1. Note
    that for complex numbers the absolute value is taken before squaring,
    so that the result is always real and nonnegative.

    """))


add_newdoc('numpy.core.multiarray', 'ndarray', ('view',
    """a.view(dtype=None, type=None)

    New view of array with the same data.

    Parameters
    ----------
    dtype : data-type
        Data-type descriptor of the returned view, e.g. float32 or int16.
    type : python type
        Type of the returned view, e.g. ndarray or matrix.

    Examples
    --------
    >>> x = np.array([(1,2)],dtype=[('a',np.int8),('b',np.int8)])
    >>> y = x.view(dtype=np.int16, type=np.matrix)

    >>> print y.dtype
    int16

    >>> print type(y)
    <class 'numpy.core.defmatrix.matrix'>

    """))

add_newdoc('numpy.core.umath','geterrobj',
    """geterrobj()

    Used internally by `geterr`.

    Returns
    -------
    errobj : list
        Internal numpy buffer size, error mask, error callback function.

    """)

add_newdoc('numpy.core.umath','seterrobj',
    """seterrobj()

    Used internally by `seterr`.

    Parameters
    ----------
    errobj : list
        [buffer_size, error_mask, callback_func]

    See Also
    --------
    seterrcall

    """)

add_newdoc("numpy.core","ufunc",
    """Functions that operate element by element on whole arrays.

    Unary ufuncs:
    =============

    op(X, out=None)
    Apply op to X elementwise

    Parameters
    ----------
    X : array-like
    out : array-like
        An array to store the output. Must be the same shape as X.

    Returns
    -------
    r : array-like
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
    X : array-like
    Y : array-like
    out : array-like
        An array to store the output. Must be the same shape as the
        output would have.

    Returns
    -------
    r : array-like
        The return value; if out is provided, r will be equal to out.

    """)


add_newdoc("numpy.core","ufunc",("reduce",
    """reduce(array,axis=0,dtype=None,out=None)

    Reduce applies the operator to all elements of the array producing
    a single result.

    For a one-dimensional array, reduce produces results equivalent to:
    r = op.identity
    for i in xrange(len(A)):
        r = op(r,A[i])
    return r

    For example, add.reduce() is equivalent to sum().

    Parameters:
    -----------

    array : array-like
        The array to act on.
    axis : integer
        The axis along which to apply the reduction.
    dtype : data type or None
        The type used to represent the intermediate results. Defaults
        to the data type of the output array if this is provided, or
        the data type of the input array if no output array is provided.
    out : array-like or None
        A location into which the result is stored. If not provided a
        freshly-allocated array is returned.

    Returns:
    --------

    r : array
        The reduced values. If out was supplied, r is equal to out.

    Example:
    --------
    >>> np.multiply.reduce([2,3,5])
    30

    """))

add_newdoc("numpy.core","ufunc",("accumulate",
    """accumulate(array,axis=None,dtype=None,out=None)

    Accumulate applies the operator to all elements of the array producing
    cumulative results.

    For a one-dimensional array, accumulate produces results equivalent to:
    r = np.empty(len(A))
    t = op.identity
    for i in xrange(len(A)):
        t = op(t,A[i])
        r[i] = t
    return r

    For example, add.accumulate() is equivalent to cumsum().

    Parameters:
    -----------

    array : array-like
        The array to act on.
    axis : integer
        The axis along which to apply the accumulation.
    dtype : data type or None
        The type used to represent the intermediate results. Defaults
        to the data type of the output array if this is provided, or
        the data type of the input array if no output array is provided.
    out : array-like or None
        A location into which the result is stored. If not provided a
        freshly-allocated array is returned.

    Returns:
    --------

    r : array
        The accumulated values. If out was supplied, r is equal to out.

    Example:
    --------
    >>> np.multiply.accumulate([2,3,5])
    array([2,6,30])

    """))

add_newdoc("numpy.core","ufunc",("reduceat",
    """reduceat(self,array,indices,axis=None,dtype=None,out=None)

    Reduceat performs a reduce over an axis using the indices as a guide

    op.reduceat(array,indices)  computes
    op.reduce(array[indices[i]:indices[i+1]])
    for i=0..end with an implicit indices[i+1]=len(array)
    assumed when i=end-1

    if indices[i+1] <= indices[i]+1
    then the result is array[indices[i]] for that value

    op.accumulate(array) is the same as
    op.reduceat(array,indices)[::2]
    where indices is range(len(array)-1) with a zero placed
    in every other sample:
    indices = zeros(len(array)*2-1)
    indices[1::2] = range(1,len(array))

    output shape is based on the size of indices

    Parameters:
    -----------

    array : array-like
        The array to act on.
    indices : array-like
        Indices specifying ranges to reduce.
    axis : integer
        The axis along which to apply the reduceat.
    dtype : data type or None
        The type used to represent the intermediate results. Defaults
        to the data type of the output array if this is provided, or
        the data type of the input array if no output array is provided.
    out : array-like or None
        A location into which the result is stored. If not provided a
        freshly-allocated array is returned.

    Returns:
    --------

    r : array
        The reduced values. If out was supplied, r is equal to out.

    Example:
    --------
    To take the running sum of four successive values:
    >>> np.multiply.reduceat(np.arange(8),[0,4, 1,5, 2,6, 3,7])[::2]
    array([ 6, 10, 14, 18])

    """))

add_newdoc("numpy.core","ufunc",("outer",
    """outer(A,B)

    Compute the result of applying op to all pairs (a,b)

    op.outer(A,B) is equivalent to
    op(A[:,:,...,:,newaxis,...,newaxis]*B[newaxis,...,newaxis,:,...,:])
    where A has B.ndim new axes appended and B has A.ndim new axes prepended.

    For A and B one-dimensional, this is equivalent to
    r = empty(len(A),len(B))
    for i in xrange(len(A)):
        for j in xrange(len(B)):
            r[i,j] = A[i]*B[j]
    If A and B are higher-dimensional, the result has dimension A.ndim+B.ndim

    Parameters:
    -----------

    A : array-like
    B : array-like

    Returns:
    --------

    r : array
    Example:
    --------
    >>> np.multiply.outer([1,2,3],[4,5,6])
    array([[ 4,  5,  6],
           [ 8, 10, 12],
           [12, 15, 18]])

    """))
