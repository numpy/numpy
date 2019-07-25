Array API
=========

.. sectionauthor:: Travis E. Oliphant

|    The test of a first-rate intelligence is the ability to hold two
|    opposed ideas in the mind at the same time, and still retain the
|    ability to function.
|    --- *F. Scott Fitzgerald*

|    For a successful technology, reality must take precedence over public
|    relations, for Nature cannot be fooled.
|    --- *Richard P. Feynman*

.. index::
   pair: ndarray; C-API
   pair: C-API; array


Array structure and data access
-------------------------------

These macros access the :c:type:`PyArrayObject` structure members and are
defined in ``ndarraytypes.h``. The input argument, *arr*, can be any
:c:type:`PyObject *<PyObject>` that is directly interpretable as a
:c:type:`PyArrayObject *` (any instance of the :c:data:`PyArray_Type`
and itssub-types).

.. c:function:: int PyArray_NDIM(PyArrayObject *arr)

    The number of dimensions in the array.

.. c:function:: int PyArray_FLAGS(PyArrayObject* arr)

    Returns an integer representing the :ref:`array-flags<array-flags>`.

.. c:function:: int PyArray_TYPE(PyArrayObject* arr)

    Return the (builtin) typenumber for the elements of this array.

.. c:function:: int PyArray_SETITEM( \
        PyArrayObject* arr, void* itemptr, PyObject* obj)

    Convert obj and place it in the ndarray, *arr*, at the place
    pointed to by itemptr. Return -1 if an error occurs or 0 on
    success.

.. c:function:: void PyArray_ENABLEFLAGS(PyArrayObject* arr, int flags)

    .. versionadded:: 1.7

    Enables the specified array flags. This function does no validation,
    and assumes that you know what you're doing.

.. c:function:: void PyArray_CLEARFLAGS(PyArrayObject* arr, int flags)

    .. versionadded:: 1.7

    Clears the specified array flags. This function does no validation,
    and assumes that you know what you're doing.

.. c:function:: void *PyArray_DATA(PyArrayObject *arr)

.. c:function:: char *PyArray_BYTES(PyArrayObject *arr)

    These two macros are similar and obtain the pointer to the
    data-buffer for the array. The first macro can (and should be)
    assigned to a particular pointer where the second is for generic
    processing. If you have not guaranteed a contiguous and/or aligned
    array then be sure you understand how to access the data in the
    array to avoid memory and/or alignment problems.

.. c:function:: npy_intp *PyArray_DIMS(PyArrayObject *arr)

    Returns a pointer to the dimensions/shape of the array. The
    number of elements matches the number of dimensions
    of the array. Can return ``NULL`` for 0-dimensional arrays.

.. c:function:: npy_intp *PyArray_SHAPE(PyArrayObject *arr)

    .. versionadded:: 1.7

    A synonym for :c:func:`PyArray_DIMS`, named to be consistent with the
    `shape <numpy.ndarray.shape>` usage within Python.

.. c:function:: npy_intp *PyArray_STRIDES(PyArrayObject* arr)

    Returns a pointer to the strides of the array. The
    number of elements matches the number of dimensions
    of the array.

.. c:function:: npy_intp PyArray_DIM(PyArrayObject* arr, int n)

    Return the shape in the *n* :math:`^{\textrm{th}}` dimension.

.. c:function:: npy_intp PyArray_STRIDE(PyArrayObject* arr, int n)

    Return the stride in the *n* :math:`^{\textrm{th}}` dimension.

.. c:function:: npy_intp PyArray_ITEMSIZE(PyArrayObject* arr)

    Return the itemsize for the elements of this array.

    Note that, in the old API that was deprecated in version 1.7, this function
    had the return type ``int``.

.. c:function:: npy_intp PyArray_SIZE(PyArrayObject* arr)

    Returns the total size (in number of elements) of the array.

.. c:function:: npy_intp PyArray_Size(PyArrayObject* obj)

    Returns 0 if *obj* is not a sub-class of ndarray. Otherwise,
    returns the total number of elements in the array. Safer version
    of :c:func:`PyArray_SIZE` (*obj*).

.. c:function:: npy_intp PyArray_NBYTES(PyArrayObject* arr)

    Returns the total number of bytes consumed by the array.

.. c:function:: PyObject *PyArray_BASE(PyArrayObject* arr)

    This returns the base object of the array. In most cases, this
    means the object which owns the memory the array is pointing at.

    If you are constructing an array using the C API, and specifying
    your own memory, you should use the function :c:func:`PyArray_SetBaseObject`
    to set the base to an object which owns the memory.

    If the (deprecated) :c:data:`NPY_ARRAY_UPDATEIFCOPY` or the
    :c:data:`NPY_ARRAY_WRITEBACKIFCOPY` flags are set, it has a different
    meaning, namely base is the array into which the current array will
    be copied upon copy resolution. This overloading of the base property
    for two functions is likely to change in a future version of NumPy.

.. c:function:: PyArray_Descr *PyArray_DESCR(PyArrayObject* arr)

    Returns a borrowed reference to the dtype property of the array.

.. c:function:: PyArray_Descr *PyArray_DTYPE(PyArrayObject* arr)

    .. versionadded:: 1.7

    A synonym for PyArray_DESCR, named to be consistent with the
    'dtype' usage within Python.

.. c:function:: PyObject *PyArray_GETITEM(PyArrayObject* arr, void* itemptr)

    Get a Python object of a builtin type from the ndarray, *arr*,
    at the location pointed to by itemptr. Return ``NULL`` on failure.

    `numpy.ndarray.item` is identical to PyArray_GETITEM.


Data access
^^^^^^^^^^^

These functions and macros provide easy access to elements of the
ndarray from C. These work for all arrays. You may need to take care
when accessing the data in the array, however, if it is not in machine
byte-order, misaligned, or not writeable. In other words, be sure to
respect the state of the flags unless you know what you are doing, or
have previously guaranteed an array that is writeable, aligned, and in
machine byte-order using :c:func:`PyArray_FromAny`. If you wish to handle all
types of arrays, the copyswap function for each type is useful for
handling misbehaved arrays. Some platforms (e.g. Solaris) do not like
misaligned data and will crash if you de-reference a misaligned
pointer. Other platforms (e.g. x86 Linux) will just work more slowly
with misaligned data.

.. c:function:: void* PyArray_GetPtr(PyArrayObject* aobj, npy_intp* ind)

    Return a pointer to the data of the ndarray, *aobj*, at the
    N-dimensional index given by the c-array, *ind*, (which must be
    at least *aobj* ->nd in size). You may want to typecast the
    returned pointer to the data type of the ndarray.

.. c:function:: void* PyArray_GETPTR1(PyArrayObject* obj, npy_intp i)

.. c:function:: void* PyArray_GETPTR2( \
        PyArrayObject* obj, npy_intp i, npy_intp j)

.. c:function:: void* PyArray_GETPTR3( \
        PyArrayObject* obj, npy_intp i, npy_intp j, npy_intp k)

.. c:function:: void* PyArray_GETPTR4( \
        PyArrayObject* obj, npy_intp i, npy_intp j, npy_intp k, npy_intp l)

    Quick, inline access to the element at the given coordinates in
    the ndarray, *obj*, which must have respectively 1, 2, 3, or 4
    dimensions (this is not checked). The corresponding *i*, *j*,
    *k*, and *l* coordinates can be any integer but will be
    interpreted as ``npy_intp``. You may want to typecast the
    returned pointer to the data type of the ndarray.

.. _array-flags:

Array flags
-----------

The ``flags`` attribute of the ``PyArrayObject`` structure contains
important information about the memory used by the array (pointed to
by the data member) This flag information must be kept accurate or
strange results and even segfaults may result.

There are 6 (binary) flags that describe the memory area used by the
data buffer.  These constants are defined in ``arrayobject.h`` and
determine the bit-position of the flag.  Python exposes a nice
attribute- based interface as well as a dictionary-like interface for
getting (and, if appropriate, setting) these flags.

Memory areas of all kinds can be pointed to by an ndarray, necessitating
these flags.  If you get an arbitrary ``PyArrayObject`` in C-code, you
need to be aware of the flags that are set.  If you need to guarantee
a certain kind of array (like :c:data:`NPY_ARRAY_C_CONTIGUOUS` and
:c:data:`NPY_ARRAY_BEHAVED`), then pass these requirements into the
PyArray_FromAny function.


Basic Array Flags
^^^^^^^^^^^^^^^^^

An ndarray can have a data segment that is not a simple contiguous
chunk of well-behaved memory you can manipulate. It may not be aligned
with word boundaries (very important on some platforms). It might have
its data in a different byte-order than the machine recognizes. It
might not be writeable. It might be in Fortan-contiguous order. The
array flags are used to indicate what can be said about data
associated with an array.

In versions 1.6 and earlier of NumPy, the following flags
did not have the _ARRAY_ macro namespace in them. That form
of the constant names is deprecated in 1.7.

.. c:var:: NPY_ARRAY_C_CONTIGUOUS

    The data area is in C-style contiguous order (last index varies the
    fastest).

.. c:var:: NPY_ARRAY_F_CONTIGUOUS

    The data area is in Fortran-style contiguous order (first index varies
    the fastest).

.. note::

    Arrays can be both C-style and Fortran-style contiguous simultaneously.
    This is clear for 1-dimensional arrays, but can also be true for higher
    dimensional arrays.

    Even for contiguous arrays a stride for a given dimension
    ``arr.strides[dim]`` may be *arbitrary* if ``arr.shape[dim] == 1``
    or the array has no elements.
    It does *not* generally hold that ``self.strides[-1] == self.itemsize``
    for C-style contiguous arrays or ``self.strides[0] == self.itemsize`` for
    Fortran-style contiguous arrays is true. The correct way to access the
    ``itemsize`` of an array from the C API is ``PyArray_ITEMSIZE(arr)``.

    .. seealso:: :ref:`Internal memory layout of an ndarray <arrays.ndarray>`

.. c:var:: NPY_ARRAY_OWNDATA

    The data area is owned by this array.

.. c:var:: NPY_ARRAY_ALIGNED

    The data area and all array elements are aligned appropriately.

.. c:var:: NPY_ARRAY_WRITEABLE

    The data area can be written to.

    Notice that the above 3 flags are defined so that a new, well-
    behaved array has these flags defined as true.

.. c:var:: NPY_ARRAY_WRITEBACKIFCOPY

    The data area represents a (well-behaved) copy whose information
    should be transferred back to the original when
    :c:func:`PyArray_ResolveWritebackIfCopy` is called.

    This is a special flag that is set if this array represents a copy
    made because a user required certain flags in
    :c:func:`PyArray_FromAny` and a copy had to be made of some other
    array (and the user asked for this flag to be set in such a
    situation). The base attribute then points to the "misbehaved"
    array (which is set read_only). :c:func`PyArray_ResolveWritebackIfCopy`
    will copy its contents back to the "misbehaved"
    array (casting if necessary) and will reset the "misbehaved" array
    to :c:data:`NPY_ARRAY_WRITEABLE`. If the "misbehaved" array was not
    :c:data:`NPY_ARRAY_WRITEABLE` to begin with then :c:func:`PyArray_FromAny`
    would have returned an error because :c:data:`NPY_ARRAY_WRITEBACKIFCOPY`
    would not have been possible.

.. c:var:: NPY_ARRAY_UPDATEIFCOPY

    A deprecated version of :c:data:`NPY_ARRAY_WRITEBACKIFCOPY` which
    depends upon ``dealloc`` to trigger the writeback. For backwards
    compatibility, :c:func:`PyArray_ResolveWritebackIfCopy` is called at
    ``dealloc`` but relying
    on that behavior is deprecated and not supported in PyPy.

:c:func:`PyArray_UpdateFlags` (obj, flags) will update the ``obj->flags``
for ``flags`` which can be any of :c:data:`NPY_ARRAY_C_CONTIGUOUS`,
:c:data:`NPY_ARRAY_F_CONTIGUOUS`, :c:data:`NPY_ARRAY_ALIGNED`, or
:c:data:`NPY_ARRAY_WRITEABLE`.


Combinations of array flags
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. c:var:: NPY_ARRAY_BEHAVED

    :c:data:`NPY_ARRAY_ALIGNED` \| :c:data:`NPY_ARRAY_WRITEABLE`

.. c:var:: NPY_ARRAY_CARRAY

    :c:data:`NPY_ARRAY_C_CONTIGUOUS` \| :c:data:`NPY_ARRAY_BEHAVED`

.. c:var:: NPY_ARRAY_CARRAY_RO

    :c:data:`NPY_ARRAY_C_CONTIGUOUS` \| :c:data:`NPY_ARRAY_ALIGNED`

.. c:var:: NPY_ARRAY_FARRAY

    :c:data:`NPY_ARRAY_F_CONTIGUOUS` \| :c:data:`NPY_ARRAY_BEHAVED`

.. c:var:: NPY_ARRAY_FARRAY_RO

    :c:data:`NPY_ARRAY_F_CONTIGUOUS` \| :c:data:`NPY_ARRAY_ALIGNED`

.. c:var:: NPY_ARRAY_DEFAULT

    :c:data:`NPY_ARRAY_CARRAY`

.. c:var:: NPY_ARRAY_UPDATE_ALL

    :c:data:`NPY_ARRAY_C_CONTIGUOUS` \| :c:data:`NPY_ARRAY_F_CONTIGUOUS` \| :c:data:`NPY_ARRAY_ALIGNED`


Flag-like constants
^^^^^^^^^^^^^^^^^^^

These constants are used in :c:func:`PyArray_FromAny` (and its macro forms) to
specify desired properties of the new array.

.. c:var:: NPY_ARRAY_FORCECAST

    Cast to the desired type, even if it can't be done without losing
    information.

.. c:var:: NPY_ARRAY_ENSURECOPY

    Make sure the resulting array is a copy of the original.

.. c:var:: NPY_ARRAY_ENSUREARRAY

    Make sure the resulting object is an actual ndarray, and not a sub-class.

.. c:var:: NPY_ARRAY_NOTSWAPPED

    Only used in :c:func:`PyArray_CheckFromAny` to over-ride the byteorder
    of the data-type object passed in.

.. c:var:: NPY_ARRAY_BEHAVED_NS

    :c:data:`NPY_ARRAY_ALIGNED` \| :c:data:`NPY_ARRAY_WRITEABLE` \| :c:data:`NPY_ARRAY_NOTSWAPPED`


Flag checking
^^^^^^^^^^^^^

For all of these macros *arr* must be an instance of a (subclass of)
:c:data:`PyArray_Type`.

.. c:function:: PyArray_CHKFLAGS(arr, flags)

    The first parameter, arr, must be an ndarray or subclass. The
    parameter, *flags*, should be an integer consisting of bitwise
    combinations of the possible flags an array can have:
    :c:data:`NPY_ARRAY_C_CONTIGUOUS`, :c:data:`NPY_ARRAY_F_CONTIGUOUS`,
    :c:data:`NPY_ARRAY_OWNDATA`, :c:data:`NPY_ARRAY_ALIGNED`,
    :c:data:`NPY_ARRAY_WRITEABLE`, :c:data:`NPY_ARRAY_WRITEBACKIFCOPY`,
    :c:data:`NPY_ARRAY_UPDATEIFCOPY`.

.. c:function:: PyArray_IS_C_CONTIGUOUS(arr)

    Evaluates true if *arr* is C-style contiguous.

.. c:function:: PyArray_IS_F_CONTIGUOUS(arr)

    Evaluates true if *arr* is Fortran-style contiguous.

.. c:function:: PyArray_ISFORTRAN(arr)

    Evaluates true if *arr* is Fortran-style contiguous and *not*
    C-style contiguous. :c:func:`PyArray_IS_F_CONTIGUOUS`
    is the correct way to test for Fortran-style contiguity.

.. c:function:: PyArray_ISWRITEABLE(arr)

    Evaluates true if the data area of *arr* can be written to

.. c:function:: PyArray_ISALIGNED(arr)

    Evaluates true if the data area of *arr* is properly aligned on
    the machine.

.. c:function:: PyArray_ISBEHAVED(arr)

    Evaluates true if the data area of *arr* is aligned and writeable
    and in machine byte-order according to its descriptor.

.. c:function:: PyArray_ISBEHAVED_RO(arr)

    Evaluates true if the data area of *arr* is aligned and in machine
    byte-order.

.. c:function:: PyArray_ISCARRAY(arr)

    Evaluates true if the data area of *arr* is C-style contiguous,
    and :c:func:`PyArray_ISBEHAVED` (*arr*) is true.

.. c:function:: PyArray_ISFARRAY(arr)

    Evaluates true if the data area of *arr* is Fortran-style
    contiguous and :c:func:`PyArray_ISBEHAVED` (*arr*) is true.

.. c:function:: PyArray_ISCARRAY_RO(arr)

    Evaluates true if the data area of *arr* is C-style contiguous,
    aligned, and in machine byte-order.

.. c:function:: PyArray_ISFARRAY_RO(arr)

    Evaluates true if the data area of *arr* is Fortran-style
    contiguous, aligned, and in machine byte-order **.**

.. c:function:: PyArray_ISONESEGMENT(arr)

    Evaluates true if the data area of *arr* consists of a single
    (C-style or Fortran-style) contiguous segment.

.. c:function:: void PyArray_UpdateFlags(PyArrayObject* arr, int flagmask)

    The :c:data:`NPY_ARRAY_C_CONTIGUOUS`, :c:data:`NPY_ARRAY_ALIGNED`, and
    :c:data:`NPY_ARRAY_F_CONTIGUOUS` array flags can be "calculated" from the
    array object itself. This routine updates one or more of these
    flags of *arr* as specified in *flagmask* by performing the
    required calculation.


.. warning::

    It is important to keep the flags updated (using
    :c:func:`PyArray_UpdateFlags` can help) whenever a manipulation with an
    array is performed that might cause them to change. Later
    calculations in NumPy that rely on the state of these flags do not
    repeat the calculation to update them.

Creating arrays
---------------


From scratch
^^^^^^^^^^^^

.. c:function:: PyObject* PyArray_NewFromDescr( \
        PyTypeObject* subtype, PyArray_Descr* descr, int nd, npy_intp const* dims, \
        npy_intp const* strides, void* data, int flags, PyObject* obj)

    This function steals a reference to *descr*. The easiest way to get one
    is using :c:func:`PyArray_DescrFromType`.

    This is the main array creation function. Most new arrays are
    created with this flexible function.

    The returned object is an object of Python-type *subtype*, which
    must be a subtype of :c:data:`PyArray_Type`.  The array has *nd*
    dimensions, described by *dims*. The data-type descriptor of the
    new array is *descr*.

    If *subtype* is of an array subclass instead of the base
    :c:data:`&PyArray_Type<PyArray_Type>`, then *obj* is the object to pass to
    the :obj:`~numpy.class.__array_finalize__` method of the subclass.

    If *data* is ``NULL``, then new unitinialized memory will be allocated and
    *flags* can be non-zero to indicate a Fortran-style contiguous array. Use
    :c:func:`PyArray_FILLWBYTE` to initialize the memory.

    If *data* is not ``NULL``, then it is assumed to point to the memory
    to be used for the array and the *flags* argument is used as the
    new flags for the array (except the state of :c:data:`NPY_OWNDATA`,
    :c:data:`NPY_ARRAY_WRITEBACKIFCOPY` and :c:data:`NPY_ARRAY_UPDATEIFCOPY`
    flags of the new array will be reset).

    In addition, if *data* is non-NULL, then *strides* can
    also be provided. If *strides* is ``NULL``, then the array strides
    are computed as C-style contiguous (default) or Fortran-style
    contiguous (*flags* is nonzero for *data* = ``NULL`` or *flags* &
    :c:data:`NPY_ARRAY_F_CONTIGUOUS` is nonzero non-NULL *data*). Any
    provided *dims* and *strides* are copied into newly allocated
    dimension and strides arrays for the new array object.

    :c:func:`PyArray_CheckStrides` can help verify non- ``NULL`` stride
    information.

    If ``data`` is provided, it must stay alive for the life of the array. One
    way to manage this is through :c:func:`PyArray_SetBaseObject`

.. c:function:: PyObject* PyArray_NewLikeArray( \
        PyArrayObject* prototype, NPY_ORDER order, PyArray_Descr* descr, \
        int subok)

    .. versionadded:: 1.6

    This function steals a reference to *descr* if it is not NULL.

    This array creation routine allows for the convenient creation of
    a new array matching an existing array's shapes and memory layout,
    possibly changing the layout and/or data type.

    When *order* is :c:data:`NPY_ANYORDER`, the result order is
    :c:data:`NPY_FORTRANORDER` if *prototype* is a fortran array,
    :c:data:`NPY_CORDER` otherwise.  When *order* is
    :c:data:`NPY_KEEPORDER`, the result order matches that of *prototype*, even
    when the axes of *prototype* aren't in C or Fortran order.

    If *descr* is NULL, the data type of *prototype* is used.

    If *subok* is 1, the newly created array will use the sub-type of
    *prototype* to create the new array, otherwise it will create a
    base-class array.

.. c:function:: PyObject* PyArray_New( \
        PyTypeObject* subtype, int nd, npy_intp const* dims, int type_num, \
        npy_intp const* strides, void* data, int itemsize, int flags, \
        PyObject* obj)

    This is similar to :c:func:`PyArray_NewFromDescr` (...) except you
    specify the data-type descriptor with *type_num* and *itemsize*,
    where *type_num* corresponds to a builtin (or user-defined)
    type. If the type always has the same number of bytes, then
    itemsize is ignored. Otherwise, itemsize specifies the particular
    size of this array.



.. warning::

    If data is passed to :c:func:`PyArray_NewFromDescr` or :c:func:`PyArray_New`,
    this memory must not be deallocated until the new array is
    deleted.  If this data came from another Python object, this can
    be accomplished using :c:func:`Py_INCREF` on that object and setting the
    base member of the new array to point to that object. If strides
    are passed in they must be consistent with the dimensions, the
    itemsize, and the data of the array.

.. c:function:: PyObject* PyArray_SimpleNew(int nd, npy_intp const* dims, int typenum)

    Create a new uninitialized array of type, *typenum*, whose size in
    each of *nd* dimensions is given by the integer array, *dims*.The memory
    for the array is uninitialized (unless typenum is :c:data:`NPY_OBJECT`
    in which case each element in the array is set to NULL). The
    *typenum* argument allows specification of any of the builtin
    data-types such as :c:data:`NPY_FLOAT` or :c:data:`NPY_LONG`. The
    memory for the array can be set to zero if desired using
    :c:func:`PyArray_FILLWBYTE` (return_object, 0).This function cannot be
    used to create a flexible-type array (no itemsize given).

.. c:function:: PyObject* PyArray_SimpleNewFromData( \
        int nd, npy_intp const* dims, int typenum, void* data)

    Create an array wrapper around *data* pointed to by the given
    pointer. The array flags will have a default that the data area is
    well-behaved and C-style contiguous. The shape of the array is
    given by the *dims* c-array of length *nd*. The data-type of the
    array is indicated by *typenum*. If data comes from another
    reference-counted Python object, the reference count on this object
    should be increased after the pointer is passed in, and the base member
    of the returned ndarray should point to the Python object that owns
    the data. This will ensure that the provided memory is not
    freed while the returned array is in existence. To free memory as soon
    as the ndarray is deallocated, set the OWNDATA flag on the returned ndarray.

.. c:function:: PyObject* PyArray_SimpleNewFromDescr( \
        int nd, npy_int const* dims, PyArray_Descr* descr)

    This function steals a reference to *descr*.

    Create a new array with the provided data-type descriptor, *descr*,
    of the shape determined by *nd* and *dims*.

.. c:function:: PyArray_FILLWBYTE(PyObject* obj, int val)

    Fill the array pointed to by *obj* ---which must be a (subclass
    of) ndarray---with the contents of *val* (evaluated as a byte).
    This macro calls memset, so obj must be contiguous.

.. c:function:: PyObject* PyArray_Zeros( \
        int nd, npy_intp const* dims, PyArray_Descr* dtype, int fortran)

    Construct a new *nd* -dimensional array with shape given by *dims*
    and data type given by *dtype*. If *fortran* is non-zero, then a
    Fortran-order array is created, otherwise a C-order array is
    created. Fill the memory with zeros (or the 0 object if *dtype*
    corresponds to :c:type:`NPY_OBJECT` ).

.. c:function:: PyObject* PyArray_ZEROS( \
        int nd, npy_intp const* dims, int type_num, int fortran)

    Macro form of :c:func:`PyArray_Zeros` which takes a type-number instead
    of a data-type object.

.. c:function:: PyObject* PyArray_Empty( \
        int nd, npy_intp const* dims, PyArray_Descr* dtype, int fortran)

    Construct a new *nd* -dimensional array with shape given by *dims*
    and data type given by *dtype*. If *fortran* is non-zero, then a
    Fortran-order array is created, otherwise a C-order array is
    created. The array is uninitialized unless the data type
    corresponds to :c:type:`NPY_OBJECT` in which case the array is
    filled with :c:data:`Py_None`.

.. c:function:: PyObject* PyArray_EMPTY( \
        int nd, npy_intp const* dims, int typenum, int fortran)

    Macro form of :c:func:`PyArray_Empty` which takes a type-number,
    *typenum*, instead of a data-type object.

.. c:function:: PyObject* PyArray_Arange( \
        double start, double stop, double step, int typenum)

    Construct a new 1-dimensional array of data-type, *typenum*, that
    ranges from *start* to *stop* (exclusive) in increments of *step*
    . Equivalent to **arange** (*start*, *stop*, *step*, dtype).

.. c:function:: PyObject* PyArray_ArangeObj( \
        PyObject* start, PyObject* stop, PyObject* step, PyArray_Descr* descr)

    Construct a new 1-dimensional array of data-type determined by
    ``descr``, that ranges from ``start`` to ``stop`` (exclusive) in
    increments of ``step``. Equivalent to arange( ``start``,
    ``stop``, ``step``, ``typenum`` ).

.. c:function:: int PyArray_SetBaseObject(PyArrayObject* arr, PyObject* obj)

    .. versionadded:: 1.7

    This function **steals a reference** to ``obj`` and sets it as the
    base property of ``arr``.

    If you construct an array by passing in your own memory buffer as
    a parameter, you need to set the array's `base` property to ensure
    the lifetime of the memory buffer is appropriate.

    The return value is 0 on success, -1 on failure.

    If the object provided is an array, this function traverses the
    chain of `base` pointers so that each array points to the owner
    of the memory directly. Once the base is set, it may not be changed
    to another value.

From other objects
^^^^^^^^^^^^^^^^^^

.. c:function:: PyObject* PyArray_FromAny( \
        PyObject* op, PyArray_Descr* dtype, int min_depth, int max_depth, \
        int requirements, PyObject* context)

    This is the main function used to obtain an array from any nested
    sequence, or object that exposes the array interface, *op*. The
    parameters allow specification of the required *dtype*, the
    minimum (*min_depth*) and maximum (*max_depth*) number of
    dimensions acceptable, and other *requirements* for the array. This
    function **steals a reference** to the dtype argument, which needs
    to be a :c:type:`PyArray_Descr` structure
    indicating the desired data-type (including required
    byteorder). The *dtype* argument may be ``NULL``, indicating that any
    data-type (and byteorder) is acceptable. Unless
    :c:data:`NPY_ARRAY_FORCECAST` is present in ``flags``,
    this call will generate an error if the data
    type cannot be safely obtained from the object. If you want to use
    ``NULL`` for the *dtype* and ensure the array is notswapped then
    use :c:func:`PyArray_CheckFromAny`. A value of 0 for either of the
    depth parameters causes the parameter to be ignored. Any of the
    following array flags can be added (*e.g.* using \|) to get the
    *requirements* argument. If your code can handle general (*e.g.*
    strided, byte-swapped, or unaligned arrays) then *requirements*
    may be 0. Also, if *op* is not already an array (or does not
    expose the array interface), then a new array will be created (and
    filled from *op* using the sequence protocol). The new array will
    have :c:data:`NPY_ARRAY_DEFAULT` as its flags member. The *context* argument
    is passed to the :obj:`~numpy.class.__array__` method of *op* and is only used if
    the array is constructed that way. Almost always this
    parameter is ``NULL``.

    .. c:var:: NPY_ARRAY_C_CONTIGUOUS

        Make sure the returned array is C-style contiguous

    .. c:var:: NPY_ARRAY_F_CONTIGUOUS

        Make sure the returned array is Fortran-style contiguous.

    .. c:var:: NPY_ARRAY_ALIGNED

        Make sure the returned array is aligned on proper boundaries for its
        data type. An aligned array has the data pointer and every strides
        factor as a multiple of the alignment factor for the data-type-
        descriptor.

    .. c:var:: NPY_ARRAY_WRITEABLE

        Make sure the returned array can be written to.

    .. c:var:: NPY_ARRAY_ENSURECOPY

        Make sure a copy is made of *op*. If this flag is not
        present, data is not copied if it can be avoided.

    .. c:var:: NPY_ARRAY_ENSUREARRAY

        Make sure the result is a base-class ndarray. By
        default, if *op* is an instance of a subclass of
        ndarray, an instance of that same subclass is returned. If
        this flag is set, an ndarray object will be returned instead.

    .. c:var:: NPY_ARRAY_FORCECAST

        Force a cast to the output type even if it cannot be done
        safely.  Without this flag, a data cast will occur only if it
        can be done safely, otherwise an error is raised.

    .. c:var:: NPY_ARRAY_WRITEBACKIFCOPY

        If *op* is already an array, but does not satisfy the
        requirements, then a copy is made (which will satisfy the
        requirements). If this flag is present and a copy (of an object
        that is already an array) must be made, then the corresponding
        :c:data:`NPY_ARRAY_WRITEBACKIFCOPY` flag is set in the returned
        copy and *op* is made to be read-only. You must be sure to call
        :c:func:`PyArray_ResolveWritebackIfCopy` to copy the contents
        back into *op* and the *op* array
        will be made writeable again. If *op* is not writeable to begin
        with, or if it is not already an array, then an error is raised.

    .. c:var:: NPY_ARRAY_UPDATEIFCOPY

        Deprecated. Use :c:data:`NPY_ARRAY_WRITEBACKIFCOPY`, which is similar.
        This flag "automatically" copies the data back when the returned
        array is deallocated, which is not supported in all python
        implementations.

    .. c:var:: NPY_ARRAY_BEHAVED

        :c:data:`NPY_ARRAY_ALIGNED` \| :c:data:`NPY_ARRAY_WRITEABLE`

    .. c:var:: NPY_ARRAY_CARRAY

        :c:data:`NPY_ARRAY_C_CONTIGUOUS` \| :c:data:`NPY_ARRAY_BEHAVED`

    .. c:var:: NPY_ARRAY_CARRAY_RO

        :c:data:`NPY_ARRAY_C_CONTIGUOUS` \| :c:data:`NPY_ARRAY_ALIGNED`

    .. c:var:: NPY_ARRAY_FARRAY

        :c:data:`NPY_ARRAY_F_CONTIGUOUS` \| :c:data:`NPY_ARRAY_BEHAVED`

    .. c:var:: NPY_ARRAY_FARRAY_RO

        :c:data:`NPY_ARRAY_F_CONTIGUOUS` \| :c:data:`NPY_ARRAY_ALIGNED`

    .. c:var:: NPY_ARRAY_DEFAULT

        :c:data:`NPY_ARRAY_CARRAY`

    .. c:var:: NPY_ARRAY_IN_ARRAY

        :c:data:`NPY_ARRAY_C_CONTIGUOUS` \| :c:data:`NPY_ARRAY_ALIGNED`

    .. c:var:: NPY_ARRAY_IN_FARRAY

        :c:data:`NPY_ARRAY_F_CONTIGUOUS` \| :c:data:`NPY_ARRAY_ALIGNED`

    .. c:var:: NPY_OUT_ARRAY

        :c:data:`NPY_ARRAY_C_CONTIGUOUS` \| :c:data:`NPY_ARRAY_WRITEABLE` \|
        :c:data:`NPY_ARRAY_ALIGNED`

    .. c:var:: NPY_ARRAY_OUT_ARRAY

        :c:data:`NPY_ARRAY_C_CONTIGUOUS` \| :c:data:`NPY_ARRAY_ALIGNED` \|
        :c:data:`NPY_ARRAY_WRITEABLE`

    .. c:var:: NPY_ARRAY_OUT_FARRAY

        :c:data:`NPY_ARRAY_F_CONTIGUOUS` \| :c:data:`NPY_ARRAY_WRITEABLE` \|
        :c:data:`NPY_ARRAY_ALIGNED`

    .. c:var:: NPY_ARRAY_INOUT_ARRAY

        :c:data:`NPY_ARRAY_C_CONTIGUOUS` \| :c:data:`NPY_ARRAY_WRITEABLE` \|
        :c:data:`NPY_ARRAY_ALIGNED` \| :c:data:`NPY_ARRAY_WRITEBACKIFCOPY` \|
        :c:data:`NPY_ARRAY_UPDATEIFCOPY`

    .. c:var:: NPY_ARRAY_INOUT_FARRAY

        :c:data:`NPY_ARRAY_F_CONTIGUOUS` \| :c:data:`NPY_ARRAY_WRITEABLE` \|
        :c:data:`NPY_ARRAY_ALIGNED` \| :c:data:`NPY_ARRAY_WRITEBACKIFCOPY` \|
        :c:data:`NPY_ARRAY_UPDATEIFCOPY`

.. c:function:: int PyArray_GetArrayParamsFromObject( \
        PyObject* op, PyArray_Descr* requested_dtype, npy_bool writeable, \
        PyArray_Descr** out_dtype, int* out_ndim, npy_intp* out_dims, \
        PyArrayObject** out_arr, PyObject* context)

    .. versionadded:: 1.6

    Retrieves the array parameters for viewing/converting an arbitrary
    PyObject* to a NumPy array. This allows the "innate type and shape"
    of Python list-of-lists to be discovered without
    actually converting to an array. PyArray_FromAny calls this function
    to analyze its input.

    In some cases, such as structured arrays and the :obj:`~numpy.class.__array__` interface,
    a data type needs to be used to make sense of the object.  When
    this is needed, provide a Descr for 'requested_dtype', otherwise
    provide NULL. This reference is not stolen. Also, if the requested
    dtype doesn't modify the interpretation of the input, out_dtype will
    still get the "innate" dtype of the object, not the dtype passed
    in 'requested_dtype'.

    If writing to the value in 'op' is desired, set the boolean
    'writeable' to 1.  This raises an error when 'op' is a scalar, list
    of lists, or other non-writeable 'op'. This differs from passing
    :c:data:`NPY_ARRAY_WRITEABLE` to PyArray_FromAny, where the writeable array may
    be a copy of the input.

    When success (0 return value) is returned, either out_arr
    is filled with a non-NULL PyArrayObject and
    the rest of the parameters are untouched, or out_arr is
    filled with NULL, and the rest of the parameters are filled.

    Typical usage:

    .. code-block:: c

        PyArrayObject *arr = NULL;
        PyArray_Descr *dtype = NULL;
        int ndim = 0;
        npy_intp dims[NPY_MAXDIMS];

        if (PyArray_GetArrayParamsFromObject(op, NULL, 1, &dtype,
                                            &ndim, &dims, &arr, NULL) < 0) {
            return NULL;
        }
        if (arr == NULL) {
            /*
            ... validate/change dtype, validate flags, ndim, etc ...
             Could make custom strides here too */
            arr = PyArray_NewFromDescr(&PyArray_Type, dtype, ndim,
                                        dims, NULL,
                                        fortran ? NPY_ARRAY_F_CONTIGUOUS : 0,
                                        NULL);
            if (arr == NULL) {
                return NULL;
            }
            if (PyArray_CopyObject(arr, op) < 0) {
                Py_DECREF(arr);
                return NULL;
            }
        }
        else {
            /*
            ... in this case the other parameters weren't filled, just
                validate and possibly copy arr itself ...
            */
        }
        /*
        ... use arr ...
        */

.. c:function:: PyObject* PyArray_CheckFromAny( \
        PyObject* op, PyArray_Descr* dtype, int min_depth, int max_depth, \
        int requirements, PyObject* context)

    Nearly identical to :c:func:`PyArray_FromAny` (...) except
    *requirements* can contain :c:data:`NPY_ARRAY_NOTSWAPPED` (over-riding the
    specification in *dtype*) and :c:data:`NPY_ARRAY_ELEMENTSTRIDES` which
    indicates that the array should be aligned in the sense that the
    strides are multiples of the element size.

    In versions 1.6 and earlier of NumPy, the following flags
    did not have the _ARRAY_ macro namespace in them. That form
    of the constant names is deprecated in 1.7.

.. c:var:: NPY_ARRAY_NOTSWAPPED

    Make sure the returned array has a data-type descriptor that is in
    machine byte-order, over-riding any specification in the *dtype*
    argument. Normally, the byte-order requirement is determined by
    the *dtype* argument. If this flag is set and the dtype argument
    does not indicate a machine byte-order descriptor (or is NULL and
    the object is already an array with a data-type descriptor that is
    not in machine byte- order), then a new data-type descriptor is
    created and used with its byte-order field set to native.

.. c:var:: NPY_ARRAY_BEHAVED_NS

    :c:data:`NPY_ARRAY_ALIGNED` \| :c:data:`NPY_ARRAY_WRITEABLE` \| :c:data:`NPY_ARRAY_NOTSWAPPED`

.. c:var:: NPY_ARRAY_ELEMENTSTRIDES

    Make sure the returned array has strides that are multiples of the
    element size.

.. c:function:: PyObject* PyArray_FromArray( \
        PyArrayObject* op, PyArray_Descr* newtype, int requirements)

    Special case of :c:func:`PyArray_FromAny` for when *op* is already an
    array but it needs to be of a specific *newtype* (including
    byte-order) or has certain *requirements*.

.. c:function:: PyObject* PyArray_FromStructInterface(PyObject* op)

    Returns an ndarray object from a Python object that exposes the
    :obj:`__array_struct__` attribute and follows the array interface
    protocol. If the object does not contain this attribute then a
    borrowed reference to :c:data:`Py_NotImplemented` is returned.

.. c:function:: PyObject* PyArray_FromInterface(PyObject* op)

    Returns an ndarray object from a Python object that exposes the
    :obj:`__array_interface__` attribute following the array interface
    protocol. If the object does not contain this attribute then a
    borrowed reference to :c:data:`Py_NotImplemented` is returned.

.. c:function:: PyObject* PyArray_FromArrayAttr( \
        PyObject* op, PyArray_Descr* dtype, PyObject* context)

    Return an ndarray object from a Python object that exposes the
    :obj:`~numpy.class.__array__` method. The :obj:`~numpy.class.__array__` method can take 0, 1, or 2
    arguments ([dtype, context]) where *context* is used to pass
    information about where the :obj:`~numpy.class.__array__` method is being called
    from (currently only used in ufuncs).

.. c:function:: PyObject* PyArray_ContiguousFromAny( \
        PyObject* op, int typenum, int min_depth, int max_depth)

    This function returns a (C-style) contiguous and behaved function
    array from any nested sequence or array interface exporting
    object, *op*, of (non-flexible) type given by the enumerated
    *typenum*, of minimum depth *min_depth*, and of maximum depth
    *max_depth*. Equivalent to a call to :c:func:`PyArray_FromAny` with
    requirements set to :c:data:`NPY_ARRAY_DEFAULT` and the type_num member of the
    type argument set to *typenum*.

.. c:function:: PyObject *PyArray_FromObject( \
        PyObject *op, int typenum, int min_depth, int max_depth)

    Return an aligned and in native-byteorder array from any nested
    sequence or array-interface exporting object, op, of a type given by
    the enumerated typenum. The minimum number of dimensions the array can
    have is given by min_depth while the maximum is max_depth. This is
    equivalent to a call to :c:func:`PyArray_FromAny` with requirements set to
    BEHAVED.

.. c:function:: PyObject* PyArray_EnsureArray(PyObject* op)

    This function **steals a reference** to ``op`` and makes sure that
    ``op`` is a base-class ndarray. It special cases array scalars,
    but otherwise calls :c:func:`PyArray_FromAny` ( ``op``, NULL, 0, 0,
    :c:data:`NPY_ARRAY_ENSUREARRAY`, NULL).

.. c:function:: PyObject* PyArray_FromString( \
        char* string, npy_intp slen, PyArray_Descr* dtype, npy_intp num, \
        char* sep)

    Construct a one-dimensional ndarray of a single type from a binary
    or (ASCII) text ``string`` of length ``slen``. The data-type of
    the array to-be-created is given by ``dtype``. If num is -1, then
    **copy** the entire string and return an appropriately sized
    array, otherwise, ``num`` is the number of items to **copy** from
    the string. If ``sep`` is NULL (or ""), then interpret the string
    as bytes of binary data, otherwise convert the sub-strings
    separated by ``sep`` to items of data-type ``dtype``. Some
    data-types may not be readable in text mode and an error will be
    raised if that occurs. All errors return NULL.

.. c:function:: PyObject* PyArray_FromFile( \
        FILE* fp, PyArray_Descr* dtype, npy_intp num, char* sep)

    Construct a one-dimensional ndarray of a single type from a binary
    or text file. The open file pointer is ``fp``, the data-type of
    the array to be created is given by ``dtype``. This must match
    the data in the file. If ``num`` is -1, then read until the end of
    the file and return an appropriately sized array, otherwise,
    ``num`` is the number of items to read. If ``sep`` is NULL (or
    ""), then read from the file in binary mode, otherwise read from
    the file in text mode with ``sep`` providing the item
    separator. Some array types cannot be read in text mode in which
    case an error is raised.

.. c:function:: PyObject* PyArray_FromBuffer( \
        PyObject* buf, PyArray_Descr* dtype, npy_intp count, npy_intp offset)

    Construct a one-dimensional ndarray of a single type from an
    object, ``buf``, that exports the (single-segment) buffer protocol
    (or has an attribute __buffer\__ that returns an object that
    exports the buffer protocol). A writeable buffer will be tried
    first followed by a read- only buffer. The :c:data:`NPY_ARRAY_WRITEABLE`
    flag of the returned array will reflect which one was
    successful. The data is assumed to start at ``offset`` bytes from
    the start of the memory location for the object. The type of the
    data in the buffer will be interpreted depending on the data- type
    descriptor, ``dtype.`` If ``count`` is negative then it will be
    determined from the size of the buffer and the requested itemsize,
    otherwise, ``count`` represents how many elements should be
    converted from the buffer.

.. c:function:: int PyArray_CopyInto(PyArrayObject* dest, PyArrayObject* src)

    Copy from the source array, ``src``, into the destination array,
    ``dest``, performing a data-type conversion if necessary. If an
    error occurs return -1 (otherwise 0). The shape of ``src`` must be
    broadcastable to the shape of ``dest``. The data areas of dest
    and src must not overlap.

.. c:function:: int PyArray_MoveInto(PyArrayObject* dest, PyArrayObject* src)

    Move data from the source array, ``src``, into the destination
    array, ``dest``, performing a data-type conversion if
    necessary. If an error occurs return -1 (otherwise 0). The shape
    of ``src`` must be broadcastable to the shape of ``dest``. The
    data areas of dest and src may overlap.

.. c:function:: PyArrayObject* PyArray_GETCONTIGUOUS(PyObject* op)

    If ``op`` is already (C-style) contiguous and well-behaved then
    just return a reference, otherwise return a (contiguous and
    well-behaved) copy of the array. The parameter op must be a
    (sub-class of an) ndarray and no checking for that is done.

.. c:function:: PyObject* PyArray_FROM_O(PyObject* obj)

    Convert ``obj`` to an ndarray. The argument can be any nested
    sequence or object that exports the array interface. This is a
    macro form of :c:func:`PyArray_FromAny` using ``NULL``, 0, 0, 0 for the
    other arguments. Your code must be able to handle any data-type
    descriptor and any combination of data-flags to use this macro.

.. c:function:: PyObject* PyArray_FROM_OF(PyObject* obj, int requirements)

    Similar to :c:func:`PyArray_FROM_O` except it can take an argument
    of *requirements* indicating properties the resulting array must
    have. Available requirements that can be enforced are
    :c:data:`NPY_ARRAY_C_CONTIGUOUS`, :c:data:`NPY_ARRAY_F_CONTIGUOUS`,
    :c:data:`NPY_ARRAY_ALIGNED`, :c:data:`NPY_ARRAY_WRITEABLE`,
    :c:data:`NPY_ARRAY_NOTSWAPPED`, :c:data:`NPY_ARRAY_ENSURECOPY`,
    :c:data:`NPY_ARRAY_WRITEBACKIFCOPY`, :c:data:`NPY_ARRAY_UPDATEIFCOPY`,
    :c:data:`NPY_ARRAY_FORCECAST`, and
    :c:data:`NPY_ARRAY_ENSUREARRAY`. Standard combinations of flags can also
    be used:

.. c:function:: PyObject* PyArray_FROM_OT(PyObject* obj, int typenum)

    Similar to :c:func:`PyArray_FROM_O` except it can take an argument of
    *typenum* specifying the type-number the returned array.

.. c:function:: PyObject* PyArray_FROM_OTF( \
        PyObject* obj, int typenum, int requirements)

    Combination of :c:func:`PyArray_FROM_OF` and :c:func:`PyArray_FROM_OT`
    allowing both a *typenum* and a *flags* argument to be provided.

.. c:function:: PyObject* PyArray_FROMANY( \
        PyObject* obj, int typenum, int min, int max, int requirements)

    Similar to :c:func:`PyArray_FromAny` except the data-type is
    specified using a typenumber. :c:func:`PyArray_DescrFromType`
    (*typenum*) is passed directly to :c:func:`PyArray_FromAny`. This
    macro also adds :c:data:`NPY_ARRAY_DEFAULT` to requirements if
    :c:data:`NPY_ARRAY_ENSURECOPY` is passed in as requirements.

.. c:function:: PyObject *PyArray_CheckAxis( \
        PyObject* obj, int* axis, int requirements)

    Encapsulate the functionality of functions and methods that take
    the axis= keyword and work properly with None as the axis
    argument. The input array is ``obj``, while ``*axis`` is a
    converted integer (so that >=MAXDIMS is the None value), and
    ``requirements`` gives the needed properties of ``obj``. The
    output is a converted version of the input so that requirements
    are met and if needed a flattening has occurred. On output
    negative values of ``*axis`` are converted and the new value is
    checked to ensure consistency with the shape of ``obj``.    


.. index::
   pair: ndarray; C-API
