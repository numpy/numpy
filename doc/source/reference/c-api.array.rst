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

These macros all access the :c:type:`PyArrayObject` structure members. The input
argument, arr, can be any :c:type:`PyObject *<PyObject>` that is directly interpretable
as a :c:type:`PyArrayObject *` (any instance of the :c:data:`PyArray_Type` and its
sub-types).

.. c:function:: int PyArray_NDIM(PyArrayObject *arr)

    The number of dimensions in the array.

.. c:function:: npy_intp *PyArray_DIMS(PyArrayObject *arr)

    Returns a pointer to the dimensions/shape of the array. The
    number of elements matches the number of dimensions
    of the array.

.. c:function:: npy_intp *PyArray_SHAPE(PyArrayObject *arr)

    .. versionadded:: 1.7

    A synonym for PyArray_DIMS, named to be consistent with the
    'shape' usage within Python.

.. c:function:: void *PyArray_DATA(PyArrayObject *arr)

.. c:function:: char *PyArray_BYTES(PyArrayObject *arr)

    These two macros are similar and obtain the pointer to the
    data-buffer for the array. The first macro can (and should be)
    assigned to a particular pointer where the second is for generic
    processing. If you have not guaranteed a contiguous and/or aligned
    array then be sure you understand how to access the data in the
    array to avoid memory and/or alignment problems.

.. c:function:: npy_intp *PyArray_STRIDES(PyArrayObject* arr)

    Returns a pointer to the strides of the array. The
    number of elements matches the number of dimensions
    of the array.

.. c:function:: npy_intp PyArray_DIM(PyArrayObject* arr, int n)

    Return the shape in the *n* :math:`^{\textrm{th}}` dimension.

.. c:function:: npy_intp PyArray_STRIDE(PyArrayObject* arr, int n)

    Return the stride in the *n* :math:`^{\textrm{th}}` dimension.

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

.. c:function:: void PyArray_ENABLEFLAGS(PyArrayObject* arr, int flags)

    .. versionadded:: 1.7

    Enables the specified array flags. This function does no validation,
    and assumes that you know what you're doing.

.. c:function:: void PyArray_CLEARFLAGS(PyArrayObject* arr, int flags)

    .. versionadded:: 1.7

    Clears the specified array flags. This function does no validation,
    and assumes that you know what you're doing.

.. c:function:: int PyArray_FLAGS(PyArrayObject* arr)

.. c:function:: npy_intp PyArray_ITEMSIZE(PyArrayObject* arr)

    Return the itemsize for the elements of this array.

    Note that, in the old API that was deprecated in version 1.7, this function
    had the return type ``int``.

.. c:function:: int PyArray_TYPE(PyArrayObject* arr)

    Return the (builtin) typenumber for the elements of this array.

.. c:function:: PyObject *PyArray_GETITEM(PyArrayObject* arr, void* itemptr)

    Get a Python object of a builtin type from the ndarray, *arr*, 
    at the location pointed to by itemptr. Return ``NULL`` on failure.
    
    `numpy.ndarray.item` is identical to PyArray_GETITEM.
    
.. c:function:: int PyArray_SETITEM( \
        PyArrayObject* arr, void* itemptr, PyObject* obj)

    Convert obj and place it in the ndarray, *arr*, at the place
    pointed to by itemptr. Return -1 if an error occurs or 0 on
    success.

.. c:function:: npy_intp PyArray_SIZE(PyArrayObject* arr)

    Returns the total size (in number of elements) of the array.

.. c:function:: npy_intp PyArray_Size(PyArrayObject* obj)

    Returns 0 if *obj* is not a sub-class of ndarray. Otherwise,
    returns the total number of elements in the array. Safer version
    of :c:func:`PyArray_SIZE` (*obj*).

.. c:function:: npy_intp PyArray_NBYTES(PyArrayObject* arr)

    Returns the total number of bytes consumed by the array.


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


Creating arrays
---------------


From scratch
^^^^^^^^^^^^

.. c:function:: PyObject* PyArray_NewFromDescr( \
        PyTypeObject* subtype, PyArray_Descr* descr, int nd, npy_intp* dims, \
        npy_intp* strides, void* data, int flags, PyObject* obj)

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
    :c:func:`PyArray_FILLWBYTE` to initialze the memory.

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
        PyTypeObject* subtype, int nd, npy_intp* dims, int type_num, \
        npy_intp* strides, void* data, int itemsize, int flags, PyObject* obj)

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

.. c:function:: PyObject* PyArray_SimpleNew(int nd, npy_intp* dims, int typenum)

    Create a new uninitialized array of type, *typenum*, whose size in
    each of *nd* dimensions is given by the integer array, *dims*.
    This function cannot be used to create a flexible-type array (no
    itemsize given).

.. c:function:: PyObject* PyArray_SimpleNewFromData( \
        int nd, npy_intp* dims, int typenum, void* data)

    Create an array wrapper around *data* pointed to by the given
    pointer. The array flags will have a default that the data area is
    well-behaved and C-style contiguous. The shape of the array is
    given by the *dims* c-array of length *nd*. The data-type of the
    array is indicated by *typenum*.

.. c:function:: PyObject* PyArray_SimpleNewFromDescr( \
        int nd, npy_intp* dims, PyArray_Descr* descr)

    This function steals a reference to *descr*.

    Create a new array with the provided data-type descriptor, *descr*,
    of the shape determined by *nd* and *dims*.

.. c:function:: PyArray_FILLWBYTE(PyObject* obj, int val)

    Fill the array pointed to by *obj* ---which must be a (subclass
    of) ndarray---with the contents of *val* (evaluated as a byte).
    This macro calls memset, so obj must be contiguous.

.. c:function:: PyObject* PyArray_Zeros( \
        int nd, npy_intp* dims, PyArray_Descr* dtype, int fortran)

    Construct a new *nd* -dimensional array with shape given by *dims*
    and data type given by *dtype*. If *fortran* is non-zero, then a
    Fortran-order array is created, otherwise a C-order array is
    created. Fill the memory with zeros (or the 0 object if *dtype*
    corresponds to :c:type:`NPY_OBJECT` ).

.. c:function:: PyObject* PyArray_ZEROS( \
        int nd, npy_intp* dims, int type_num, int fortran)

    Macro form of :c:func:`PyArray_Zeros` which takes a type-number instead
    of a data-type object.

.. c:function:: PyObject* PyArray_Empty( \
        int nd, npy_intp* dims, PyArray_Descr* dtype, int fortran)

    Construct a new *nd* -dimensional array with shape given by *dims*
    and data type given by *dtype*. If *fortran* is non-zero, then a
    Fortran-order array is created, otherwise a C-order array is
    created. The array is uninitialized unless the data type
    corresponds to :c:type:`NPY_OBJECT` in which case the array is
    filled with :c:data:`Py_None`.

.. c:function:: PyObject* PyArray_EMPTY( \
        int nd, npy_intp* dims, int typenum, int fortran)

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
    allowing both a *typenum* and a *flags* argument to be provided..

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


Dealing with types
------------------


General check of Python Type
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. c:function:: PyArray_Check(PyObject *op)

    Evaluates true if *op* is a Python object whose type is a sub-type
    of :c:data:`PyArray_Type`.

.. c:function:: PyArray_CheckExact(PyObject *op)

    Evaluates true if *op* is a Python object with type
    :c:data:`PyArray_Type`.

.. c:function:: PyArray_HasArrayInterface(PyObject *op, PyObject *out)

    If ``op`` implements any part of the array interface, then ``out``
    will contain a new reference to the newly created ndarray using
    the interface or ``out`` will contain ``NULL`` if an error during
    conversion occurs. Otherwise, out will contain a borrowed
    reference to :c:data:`Py_NotImplemented` and no error condition is set.

.. c:function:: PyArray_HasArrayInterfaceType(op, type, context, out)

    If ``op`` implements any part of the array interface, then ``out``
    will contain a new reference to the newly created ndarray using
    the interface or ``out`` will contain ``NULL`` if an error during
    conversion occurs. Otherwise, out will contain a borrowed
    reference to Py_NotImplemented and no error condition is set.
    This version allows setting of the type and context in the part of
    the array interface that looks for the :obj:`~numpy.class.__array__` attribute.

.. c:function:: PyArray_IsZeroDim(op)

    Evaluates true if *op* is an instance of (a subclass of)
    :c:data:`PyArray_Type` and has 0 dimensions.

.. c:function:: PyArray_IsScalar(op, cls)

    Evaluates true if *op* is an instance of :c:data:`Py{cls}ArrType_Type`.

.. c:function:: PyArray_CheckScalar(op)

    Evaluates true if *op* is either an array scalar (an instance of a
    sub-type of :c:data:`PyGenericArr_Type` ), or an instance of (a
    sub-class of) :c:data:`PyArray_Type` whose dimensionality is 0.

.. c:function:: PyArray_IsPythonNumber(op)

    Evaluates true if *op* is an instance of a builtin numeric type (int,
    float, complex, long, bool)

.. c:function:: PyArray_IsPythonScalar(op)

    Evaluates true if *op* is a builtin Python scalar object (int,
    float, complex, str, unicode, long, bool).

.. c:function:: PyArray_IsAnyScalar(op)

    Evaluates true if *op* is either a Python scalar object (see
    :c:func:`PyArray_IsPythonScalar`) or an array scalar (an instance of a sub-
    type of :c:data:`PyGenericArr_Type` ).

.. c:function:: PyArray_CheckAnyScalar(op)

    Evaluates true if *op* is a Python scalar object (see
    :c:func:`PyArray_IsPythonScalar`), an array scalar (an instance of a
    sub-type of :c:data:`PyGenericArr_Type`) or an instance of a sub-type of
    :c:data:`PyArray_Type` whose dimensionality is 0.


Data-type checking
^^^^^^^^^^^^^^^^^^

For the typenum macros, the argument is an integer representing an
enumerated array data type. For the array type checking macros the
argument must be a :c:type:`PyObject *<PyObject>` that can be directly interpreted as a
:c:type:`PyArrayObject *`.

.. c:function:: PyTypeNum_ISUNSIGNED(num)

.. c:function:: PyDataType_ISUNSIGNED(descr)

.. c:function:: PyArray_ISUNSIGNED(obj)

    Type represents an unsigned integer.

.. c:function:: PyTypeNum_ISSIGNED(num)

.. c:function:: PyDataType_ISSIGNED(descr)

.. c:function:: PyArray_ISSIGNED(obj)

    Type represents a signed integer.

.. c:function:: PyTypeNum_ISINTEGER(num)

.. c:function:: PyDataType_ISINTEGER(descr)

.. c:function:: PyArray_ISINTEGER(obj)

    Type represents any integer.

.. c:function:: PyTypeNum_ISFLOAT(num)

.. c:function:: PyDataType_ISFLOAT(descr)

.. c:function:: PyArray_ISFLOAT(obj)

    Type represents any floating point number.

.. c:function:: PyTypeNum_ISCOMPLEX(num)

.. c:function:: PyDataType_ISCOMPLEX(descr)

.. c:function:: PyArray_ISCOMPLEX(obj)

    Type represents any complex floating point number.

.. c:function:: PyTypeNum_ISNUMBER(num)

.. c:function:: PyDataType_ISNUMBER(descr)

.. c:function:: PyArray_ISNUMBER(obj)

    Type represents any integer, floating point, or complex floating point
    number.

.. c:function:: PyTypeNum_ISSTRING(num)

.. c:function:: PyDataType_ISSTRING(descr)

.. c:function:: PyArray_ISSTRING(obj)

    Type represents a string data type.

.. c:function:: PyTypeNum_ISPYTHON(num)

.. c:function:: PyDataType_ISPYTHON(descr)

.. c:function:: PyArray_ISPYTHON(obj)

    Type represents an enumerated type corresponding to one of the
    standard Python scalar (bool, int, float, or complex).

.. c:function:: PyTypeNum_ISFLEXIBLE(num)

.. c:function:: PyDataType_ISFLEXIBLE(descr)

.. c:function:: PyArray_ISFLEXIBLE(obj)

    Type represents one of the flexible array types ( :c:data:`NPY_STRING`,
    :c:data:`NPY_UNICODE`, or :c:data:`NPY_VOID` ).

.. c:function:: PyDataType_ISUNSIZED(descr):

    Type has no size information attached, and can be resized. Should only be
    called on flexible dtypes. Types that are attached to an array will always
    be sized, hence the array form of this macro not existing.

.. c:function:: PyTypeNum_ISUSERDEF(num)

.. c:function:: PyDataType_ISUSERDEF(descr)

.. c:function:: PyArray_ISUSERDEF(obj)

    Type represents a user-defined type.

.. c:function:: PyTypeNum_ISEXTENDED(num)

.. c:function:: PyDataType_ISEXTENDED(descr)

.. c:function:: PyArray_ISEXTENDED(obj)

    Type is either flexible or user-defined.

.. c:function:: PyTypeNum_ISOBJECT(num)

.. c:function:: PyDataType_ISOBJECT(descr)

.. c:function:: PyArray_ISOBJECT(obj)

    Type represents object data type.

.. c:function:: PyTypeNum_ISBOOL(num)

.. c:function:: PyDataType_ISBOOL(descr)

.. c:function:: PyArray_ISBOOL(obj)

    Type represents Boolean data type.

.. c:function:: PyDataType_HASFIELDS(descr)

.. c:function:: PyArray_HASFIELDS(obj)

    Type has fields associated with it.

.. c:function:: PyArray_ISNOTSWAPPED(m)

    Evaluates true if the data area of the ndarray *m* is in machine
    byte-order according to the array's data-type descriptor.

.. c:function:: PyArray_ISBYTESWAPPED(m)

    Evaluates true if the data area of the ndarray *m* is **not** in
    machine byte-order according to the array's data-type descriptor.

.. c:function:: Bool PyArray_EquivTypes( \
        PyArray_Descr* type1, PyArray_Descr* type2)

    Return :c:data:`NPY_TRUE` if *type1* and *type2* actually represent
    equivalent types for this platform (the fortran member of each
    type is ignored). For example, on 32-bit platforms,
    :c:data:`NPY_LONG` and :c:data:`NPY_INT` are equivalent. Otherwise
    return :c:data:`NPY_FALSE`.

.. c:function:: Bool PyArray_EquivArrTypes( \
        PyArrayObject* a1, PyArrayObject * a2)

    Return :c:data:`NPY_TRUE` if *a1* and *a2* are arrays with equivalent
    types for this platform.

.. c:function:: Bool PyArray_EquivTypenums(int typenum1, int typenum2)

    Special case of :c:func:`PyArray_EquivTypes` (...) that does not accept
    flexible data types but may be easier to call.

.. c:function:: int PyArray_EquivByteorders({byteorder} b1, {byteorder} b2)

    True if byteorder characters ( :c:data:`NPY_LITTLE`,
    :c:data:`NPY_BIG`, :c:data:`NPY_NATIVE`, :c:data:`NPY_IGNORE` ) are
    either equal or equivalent as to their specification of a native
    byte order. Thus, on a little-endian machine :c:data:`NPY_LITTLE`
    and :c:data:`NPY_NATIVE` are equivalent where they are not
    equivalent on a big-endian machine.


Converting data types
^^^^^^^^^^^^^^^^^^^^^

.. c:function:: PyObject* PyArray_Cast(PyArrayObject* arr, int typenum)

    Mainly for backwards compatibility to the Numeric C-API and for
    simple casts to non-flexible types. Return a new array object with
    the elements of *arr* cast to the data-type *typenum* which must
    be one of the enumerated types and not a flexible type.

.. c:function:: PyObject* PyArray_CastToType( \
        PyArrayObject* arr, PyArray_Descr* type, int fortran)

    Return a new array of the *type* specified, casting the elements
    of *arr* as appropriate. The fortran argument specifies the
    ordering of the output array.

.. c:function:: int PyArray_CastTo(PyArrayObject* out, PyArrayObject* in)

    As of 1.6, this function simply calls :c:func:`PyArray_CopyInto`,
    which handles the casting.

    Cast the elements of the array *in* into the array *out*. The
    output array should be writeable, have an integer-multiple of the
    number of elements in the input array (more than one copy can be
    placed in out), and have a data type that is one of the builtin
    types.  Returns 0 on success and -1 if an error occurs.

.. c:function:: PyArray_VectorUnaryFunc* PyArray_GetCastFunc( \
        PyArray_Descr* from, int totype)

    Return the low-level casting function to cast from the given
    descriptor to the builtin type number. If no casting function
    exists return ``NULL`` and set an error. Using this function
    instead of direct access to *from* ->f->cast will allow support of
    any user-defined casting functions added to a descriptors casting
    dictionary.

.. c:function:: int PyArray_CanCastSafely(int fromtype, int totype)

    Returns non-zero if an array of data type *fromtype* can be cast
    to an array of data type *totype* without losing information. An
    exception is that 64-bit integers are allowed to be cast to 64-bit
    floating point values even though this can lose precision on large
    integers so as not to proliferate the use of long doubles without
    explicit requests. Flexible array types are not checked according
    to their lengths with this function.

.. c:function:: int PyArray_CanCastTo( \
        PyArray_Descr* fromtype, PyArray_Descr* totype)

    :c:func:`PyArray_CanCastTypeTo` supersedes this function in
    NumPy 1.6 and later.

    Equivalent to PyArray_CanCastTypeTo(fromtype, totype, NPY_SAFE_CASTING).

.. c:function:: int PyArray_CanCastTypeTo( \
        PyArray_Descr* fromtype, PyArray_Descr* totype, NPY_CASTING casting)

    .. versionadded:: 1.6

    Returns non-zero if an array of data type *fromtype* (which can
    include flexible types) can be cast safely to an array of data
    type *totype* (which can include flexible types) according to
    the casting rule *casting*. For simple types with :c:data:`NPY_SAFE_CASTING`,
    this is basically a wrapper around :c:func:`PyArray_CanCastSafely`, but
    for flexible types such as strings or unicode, it produces results
    taking into account their sizes. Integer and float types can only be cast
    to a string or unicode type using :c:data:`NPY_SAFE_CASTING` if the string
    or unicode type is big enough to hold the max value of the integer/float
    type being cast from.

.. c:function:: int PyArray_CanCastArrayTo( \
        PyArrayObject* arr, PyArray_Descr* totype, NPY_CASTING casting)

    .. versionadded:: 1.6

    Returns non-zero if *arr* can be cast to *totype* according
    to the casting rule given in *casting*.  If *arr* is an array
    scalar, its value is taken into account, and non-zero is also
    returned when the value will not overflow or be truncated to
    an integer when converting to a smaller type.

    This is almost the same as the result of
    PyArray_CanCastTypeTo(PyArray_MinScalarType(arr), totype, casting),
    but it also handles a special case arising because the set
    of uint values is not a subset of the int values for types with the
    same number of bits.

.. c:function:: PyArray_Descr* PyArray_MinScalarType(PyArrayObject* arr)

    .. versionadded:: 1.6

    If *arr* is an array, returns its data type descriptor, but if
    *arr* is an array scalar (has 0 dimensions), it finds the data type
    of smallest size to which the value may be converted
    without overflow or truncation to an integer.

    This function will not demote complex to float or anything to
    boolean, but will demote a signed integer to an unsigned integer
    when the scalar value is positive.

.. c:function:: PyArray_Descr* PyArray_PromoteTypes( \
        PyArray_Descr* type1, PyArray_Descr* type2)

    .. versionadded:: 1.6

    Finds the data type of smallest size and kind to which *type1* and
    *type2* may be safely converted. This function is symmetric and
    associative. A string or unicode result will be the proper size for
    storing the max value of the input types converted to a string or unicode.

.. c:function:: PyArray_Descr* PyArray_ResultType( \
        npy_intp narrs, PyArrayObject**arrs, npy_intp ndtypes, \
        PyArray_Descr**dtypes)

    .. versionadded:: 1.6

    This applies type promotion to all the inputs,
    using the NumPy rules for combining scalars and arrays, to
    determine the output type of a set of operands.  This is the
    same result type that ufuncs produce. The specific algorithm
    used is as follows.

    Categories are determined by first checking which of boolean,
    integer (int/uint), or floating point (float/complex) the maximum
    kind of all the arrays and the scalars are.

    If there are only scalars or the maximum category of the scalars
    is higher than the maximum category of the arrays,
    the data types are combined with :c:func:`PyArray_PromoteTypes`
    to produce the return value.

    Otherwise, PyArray_MinScalarType is called on each array, and
    the resulting data types are all combined with
    :c:func:`PyArray_PromoteTypes` to produce the return value.

    The set of int values is not a subset of the uint values for types
    with the same number of bits, something not reflected in
    :c:func:`PyArray_MinScalarType`, but handled as a special case in
    PyArray_ResultType.

.. c:function:: int PyArray_ObjectType(PyObject* op, int mintype)

    This function is superceded by :c:func:`PyArray_MinScalarType` and/or
    :c:func:`PyArray_ResultType`.

    This function is useful for determining a common type that two or
    more arrays can be converted to. It only works for non-flexible
    array types as no itemsize information is passed. The *mintype*
    argument represents the minimum type acceptable, and *op*
    represents the object that will be converted to an array. The
    return value is the enumerated typenumber that represents the
    data-type that *op* should have.

.. c:function:: void PyArray_ArrayType( \
        PyObject* op, PyArray_Descr* mintype, PyArray_Descr* outtype)

    This function is superceded by :c:func:`PyArray_ResultType`.

    This function works similarly to :c:func:`PyArray_ObjectType` (...)
    except it handles flexible arrays. The *mintype* argument can have
    an itemsize member and the *outtype* argument will have an
    itemsize member at least as big but perhaps bigger depending on
    the object *op*.

.. c:function:: PyArrayObject** PyArray_ConvertToCommonType( \
        PyObject* op, int* n)

    The functionality this provides is largely superceded by iterator
    :c:type:`NpyIter` introduced in 1.6, with flag
    :c:data:`NPY_ITER_COMMON_DTYPE` or with the same dtype parameter for
    all operands.

    Convert a sequence of Python objects contained in *op* to an array
    of ndarrays each having the same data type. The type is selected
    based on the typenumber (larger type number is chosen over a
    smaller one) ignoring objects that are only scalars. The length of
    the sequence is returned in *n*, and an *n* -length array of
    :c:type:`PyArrayObject` pointers is the return value (or ``NULL`` if an
    error occurs). The returned array must be freed by the caller of
    this routine (using :c:func:`PyDataMem_FREE` ) and all the array objects
    in it ``DECREF`` 'd or a memory-leak will occur. The example
    template-code below shows a typically usage:

    .. code-block:: c

        mps = PyArray_ConvertToCommonType(obj, &n);
        if (mps==NULL) return NULL;
        {code}
        <before return>
        for (i=0; i<n; i++) Py_DECREF(mps[i]);
        PyDataMem_FREE(mps);
        {return}

.. c:function:: char* PyArray_Zero(PyArrayObject* arr)

    A pointer to newly created memory of size *arr* ->itemsize that
    holds the representation of 0 for that type. The returned pointer,
    *ret*, **must be freed** using :c:func:`PyDataMem_FREE` (ret) when it is
    not needed anymore.

.. c:function:: char* PyArray_One(PyArrayObject* arr)

    A pointer to newly created memory of size *arr* ->itemsize that
    holds the representation of 1 for that type. The returned pointer,
    *ret*, **must be freed** using :c:func:`PyDataMem_FREE` (ret) when it
    is not needed anymore.

.. c:function:: int PyArray_ValidType(int typenum)

    Returns :c:data:`NPY_TRUE` if *typenum* represents a valid type-number
    (builtin or user-defined or character code). Otherwise, this
    function returns :c:data:`NPY_FALSE`.


New data types
^^^^^^^^^^^^^^

.. c:function:: void PyArray_InitArrFuncs(PyArray_ArrFuncs* f)

    Initialize all function pointers and members to ``NULL``.

.. c:function:: int PyArray_RegisterDataType(PyArray_Descr* dtype)

    Register a data-type as a new user-defined data type for
    arrays. The type must have most of its entries filled in. This is
    not always checked and errors can produce segfaults. In
    particular, the typeobj member of the ``dtype`` structure must be
    filled with a Python type that has a fixed-size element-size that
    corresponds to the elsize member of *dtype*. Also the ``f``
    member must have the required functions: nonzero, copyswap,
    copyswapn, getitem, setitem, and cast (some of the cast functions
    may be ``NULL`` if no support is desired). To avoid confusion, you
    should choose a unique character typecode but this is not enforced
    and not relied on internally.

    A user-defined type number is returned that uniquely identifies
    the type. A pointer to the new structure can then be obtained from
    :c:func:`PyArray_DescrFromType` using the returned type number. A -1 is
    returned if an error occurs.  If this *dtype* has already been
    registered (checked only by the address of the pointer), then
    return the previously-assigned type-number.

.. c:function:: int PyArray_RegisterCastFunc( \
        PyArray_Descr* descr, int totype, PyArray_VectorUnaryFunc* castfunc)

    Register a low-level casting function, *castfunc*, to convert
    from the data-type, *descr*, to the given data-type number,
    *totype*. Any old casting function is over-written. A ``0`` is
    returned on success or a ``-1`` on failure.

.. c:function:: int PyArray_RegisterCanCast( \
        PyArray_Descr* descr, int totype, NPY_SCALARKIND scalar)

    Register the data-type number, *totype*, as castable from
    data-type object, *descr*, of the given *scalar* kind. Use
    *scalar* = :c:data:`NPY_NOSCALAR` to register that an array of data-type
    *descr* can be cast safely to a data-type whose type_number is
    *totype*.


Special functions for NPY_OBJECT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. c:function:: int PyArray_INCREF(PyArrayObject* op)

    Used for an array, *op*, that contains any Python objects. It
    increments the reference count of every object in the array
    according to the data-type of *op*. A -1 is returned if an error
    occurs, otherwise 0 is returned.

.. c:function:: void PyArray_Item_INCREF(char* ptr, PyArray_Descr* dtype)

    A function to INCREF all the objects at the location *ptr*
    according to the data-type *dtype*. If *ptr* is the start of a
    structured type with an object at any offset, then this will (recursively)
    increment the reference count of all object-like items in the
    structured type.

.. c:function:: int PyArray_XDECREF(PyArrayObject* op)

    Used for an array, *op*, that contains any Python objects. It
    decrements the reference count of every object in the array
    according to the data-type of *op*. Normal return value is 0. A
    -1 is returned if an error occurs.

.. c:function:: void PyArray_Item_XDECREF(char* ptr, PyArray_Descr* dtype)

    A function to XDECREF all the object-like items at the location
    *ptr* as recorded in the data-type, *dtype*. This works
    recursively so that if ``dtype`` itself has fields with data-types
    that contain object-like items, all the object-like fields will be
    XDECREF ``'d``.

.. c:function:: void PyArray_FillObjectArray(PyArrayObject* arr, PyObject* obj)

    Fill a newly created array with a single value obj at all
    locations in the structure with object data-types. No checking is
    performed but *arr* must be of data-type :c:type:`NPY_OBJECT` and be
    single-segment and uninitialized (no previous objects in
    position). Use :c:func:`PyArray_DECREF` (*arr*) if you need to
    decrement all the items in the object array prior to calling this
    function.

.. c:function:: int PyArray_SetUpdateIfCopyBase(PyArrayObject* arr, PyArrayObject* base)

    Precondition: ``arr`` is a copy of ``base`` (though possibly with different
    strides, ordering, etc.) Set the UPDATEIFCOPY flag and ``arr->base`` so
    that when ``arr`` is destructed, it will copy any changes back to ``base``.
    DEPRECATED, use :c:func:`PyArray_SetWritebackIfCopyBase``.

    Returns 0 for success, -1 for failure.

.. c:function:: int PyArray_SetWritebackIfCopyBase(PyArrayObject* arr, PyArrayObject* base)

    Precondition: ``arr`` is a copy of ``base`` (though possibly with different
    strides, ordering, etc.) Sets the :c:data:`NPY_ARRAY_WRITEBACKIFCOPY` flag
    and ``arr->base``, and set ``base`` to READONLY. Call
    :c:func:`PyArray_ResolveWritebackIfCopy` before calling
    `Py_DECREF`` in order copy any changes back to ``base`` and
    reset the READONLY flag.

    Returns 0 for success, -1 for failure.


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
:c:data:`PyArray_Type`, but no checking is done.

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


Array method alternative API
----------------------------


Conversion
^^^^^^^^^^

.. c:function:: PyObject* PyArray_GetField( \
        PyArrayObject* self, PyArray_Descr* dtype, int offset)

    Equivalent to :meth:`ndarray.getfield<numpy.ndarray.getfield>` (*self*, *dtype*, *offset*). Return
    a new array of the given *dtype* using the data in the current
    array at a specified *offset* in bytes. The *offset* plus the
    itemsize of the new array type must be less than *self*
    ->descr->elsize or an error is raised. The same shape and strides
    as the original array are used. Therefore, this function has the
    effect of returning a field from a structured array. But, it can also
    be used to select specific bytes or groups of bytes from any array
    type.

.. c:function:: int PyArray_SetField( \
        PyArrayObject* self, PyArray_Descr* dtype, int offset, PyObject* val)

    Equivalent to :meth:`ndarray.setfield<numpy.ndarray.setfield>` (*self*, *val*, *dtype*, *offset*
    ). Set the field starting at *offset* in bytes and of the given
    *dtype* to *val*. The *offset* plus *dtype* ->elsize must be less
    than *self* ->descr->elsize or an error is raised. Otherwise, the
    *val* argument is converted to an array and copied into the field
    pointed to. If necessary, the elements of *val* are repeated to
    fill the destination array, But, the number of elements in the
    destination must be an integer multiple of the number of elements
    in *val*.

.. c:function:: PyObject* PyArray_Byteswap(PyArrayObject* self, Bool inplace)

    Equivalent to :meth:`ndarray.byteswap<numpy.ndarray.byteswap>` (*self*, *inplace*). Return an array
    whose data area is byteswapped. If *inplace* is non-zero, then do
    the byteswap inplace and return a reference to self. Otherwise,
    create a byteswapped copy and leave self unchanged.

.. c:function:: PyObject* PyArray_NewCopy(PyArrayObject* old, NPY_ORDER order)

    Equivalent to :meth:`ndarray.copy<numpy.ndarray.copy>` (*self*, *fortran*). Make a copy of the
    *old* array. The returned array is always aligned and writeable
    with data interpreted the same as the old array. If *order* is
    :c:data:`NPY_CORDER`, then a C-style contiguous array is returned. If
    *order* is :c:data:`NPY_FORTRANORDER`, then a Fortran-style contiguous
    array is returned. If *order is* :c:data:`NPY_ANYORDER`, then the array
    returned is Fortran-style contiguous only if the old one is;
    otherwise, it is C-style contiguous.

.. c:function:: PyObject* PyArray_ToList(PyArrayObject* self)

    Equivalent to :meth:`ndarray.tolist<numpy.ndarray.tolist>` (*self*). Return a nested Python list
    from *self*.

.. c:function:: PyObject* PyArray_ToString(PyArrayObject* self, NPY_ORDER order)

    Equivalent to :meth:`ndarray.tobytes<numpy.ndarray.tobytes>` (*self*, *order*). Return the bytes
    of this array in a Python string.

.. c:function:: PyObject* PyArray_ToFile( \
        PyArrayObject* self, FILE* fp, char* sep, char* format)

    Write the contents of *self* to the file pointer *fp* in C-style
    contiguous fashion. Write the data as binary bytes if *sep* is the
    string ""or ``NULL``. Otherwise, write the contents of *self* as
    text using the *sep* string as the item separator. Each item will
    be printed to the file.  If the *format* string is not ``NULL`` or
    "", then it is a Python print statement format string showing how
    the items are to be written.

.. c:function:: int PyArray_Dump(PyObject* self, PyObject* file, int protocol)

    Pickle the object in *self* to the given *file* (either a string
    or a Python file object). If *file* is a Python string it is
    considered to be the name of a file which is then opened in binary
    mode. The given *protocol* is used (if *protocol* is negative, or
    the highest available is used). This is a simple wrapper around
    cPickle.dump(*self*, *file*, *protocol*).

.. c:function:: PyObject* PyArray_Dumps(PyObject* self, int protocol)

    Pickle the object in *self* to a Python string and return it. Use
    the Pickle *protocol* provided (or the highest available if
    *protocol* is negative).

.. c:function:: int PyArray_FillWithScalar(PyArrayObject* arr, PyObject* obj)

    Fill the array, *arr*, with the given scalar object, *obj*. The
    object is first converted to the data type of *arr*, and then
    copied into every location. A -1 is returned if an error occurs,
    otherwise 0 is returned.

.. c:function:: PyObject* PyArray_View( \
        PyArrayObject* self, PyArray_Descr* dtype, PyTypeObject *ptype)

    Equivalent to :meth:`ndarray.view<numpy.ndarray.view>` (*self*, *dtype*). Return a new
    view of the array *self* as possibly a different data-type, *dtype*,
    and different array subclass *ptype*.

    If *dtype* is ``NULL``, then the returned array will have the same
    data type as *self*. The new data-type must be consistent with the
    size of *self*. Either the itemsizes must be identical, or *self* must
    be single-segment and the total number of bytes must be the same.
    In the latter case the dimensions of the returned array will be
    altered in the last (or first for Fortran-style contiguous arrays)
    dimension. The data area of the returned array and self is exactly
    the same.


Shape Manipulation
^^^^^^^^^^^^^^^^^^

.. c:function:: PyObject* PyArray_Newshape( \
        PyArrayObject* self, PyArray_Dims* newshape, NPY_ORDER order)

    Result will be a new array (pointing to the same memory location
    as *self* if possible), but having a shape given by *newshape*.
    If the new shape is not compatible with the strides of *self*,
    then a copy of the array with the new specified shape will be
    returned.

.. c:function:: PyObject* PyArray_Reshape(PyArrayObject* self, PyObject* shape)

    Equivalent to :meth:`ndarray.reshape<numpy.ndarray.reshape>` (*self*, *shape*) where *shape* is a
    sequence. Converts *shape* to a :c:type:`PyArray_Dims` structure and
    calls :c:func:`PyArray_Newshape` internally.
    For back-ward compatibility -- Not recommended

.. c:function:: PyObject* PyArray_Squeeze(PyArrayObject* self)

    Equivalent to :meth:`ndarray.squeeze<numpy.ndarray.squeeze>` (*self*). Return a new view of *self*
    with all of the dimensions of length 1 removed from the shape.

.. warning::

    matrix objects are always 2-dimensional. Therefore,
    :c:func:`PyArray_Squeeze` has no effect on arrays of matrix sub-class.

.. c:function:: PyObject* PyArray_SwapAxes(PyArrayObject* self, int a1, int a2)

    Equivalent to :meth:`ndarray.swapaxes<numpy.ndarray.swapaxes>` (*self*, *a1*, *a2*). The returned
    array is a new view of the data in *self* with the given axes,
    *a1* and *a2*, swapped.

.. c:function:: PyObject* PyArray_Resize( \
        PyArrayObject* self, PyArray_Dims* newshape, int refcheck, \
        NPY_ORDER fortran)

    Equivalent to :meth:`ndarray.resize<numpy.ndarray.resize>` (*self*, *newshape*, refcheck
    ``=`` *refcheck*, order= fortran ). This function only works on
    single-segment arrays. It changes the shape of *self* inplace and
    will reallocate the memory for *self* if *newshape* has a
    different total number of elements then the old shape. If
    reallocation is necessary, then *self* must own its data, have
    *self* - ``>base==NULL``, have *self* - ``>weakrefs==NULL``, and
    (unless refcheck is 0) not be referenced by any other array.
    The fortran argument can be :c:data:`NPY_ANYORDER`, :c:data:`NPY_CORDER`,
    or :c:data:`NPY_FORTRANORDER`. It currently has no effect. Eventually
    it could be used to determine how the resize operation should view
    the data when constructing a differently-dimensioned array.
    Returns None on success and NULL on error.

.. c:function:: PyObject* PyArray_Transpose( \
        PyArrayObject* self, PyArray_Dims* permute)

    Equivalent to :meth:`ndarray.transpose<numpy.ndarray.transpose>` (*self*, *permute*). Permute the
    axes of the ndarray object *self* according to the data structure
    *permute* and return the result. If *permute* is ``NULL``, then
    the resulting array has its axes reversed. For example if *self*
    has shape :math:`10\times20\times30`, and *permute* ``.ptr`` is
    (0,2,1) the shape of the result is :math:`10\times30\times20.` If
    *permute* is ``NULL``, the shape of the result is
    :math:`30\times20\times10.`

.. c:function:: PyObject* PyArray_Flatten(PyArrayObject* self, NPY_ORDER order)

    Equivalent to :meth:`ndarray.flatten<numpy.ndarray.flatten>` (*self*, *order*). Return a 1-d copy
    of the array. If *order* is :c:data:`NPY_FORTRANORDER` the elements are
    scanned out in Fortran order (first-dimension varies the
    fastest). If *order* is :c:data:`NPY_CORDER`, the elements of ``self``
    are scanned in C-order (last dimension varies the fastest). If
    *order* :c:data:`NPY_ANYORDER`, then the result of
    :c:func:`PyArray_ISFORTRAN` (*self*) is used to determine which order
    to flatten.

.. c:function:: PyObject* PyArray_Ravel(PyArrayObject* self, NPY_ORDER order)

    Equivalent to *self*.ravel(*order*). Same basic functionality
    as :c:func:`PyArray_Flatten` (*self*, *order*) except if *order* is 0
    and *self* is C-style contiguous, the shape is altered but no copy
    is performed.


Item selection and manipulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. c:function:: PyObject* PyArray_TakeFrom( \
        PyArrayObject* self, PyObject* indices, int axis, PyArrayObject* ret, \
        NPY_CLIPMODE clipmode)

    Equivalent to :meth:`ndarray.take<numpy.ndarray.take>` (*self*, *indices*, *axis*, *ret*,
    *clipmode*) except *axis* =None in Python is obtained by setting
    *axis* = :c:data:`NPY_MAXDIMS` in C. Extract the items from self
    indicated by the integer-valued *indices* along the given *axis.*
    The clipmode argument can be :c:data:`NPY_RAISE`, :c:data:`NPY_WRAP`, or
    :c:data:`NPY_CLIP` to indicate what to do with out-of-bound indices. The
    *ret* argument can specify an output array rather than having one
    created internally.

.. c:function:: PyObject* PyArray_PutTo( \
        PyArrayObject* self, PyObject* values, PyObject* indices, \
        NPY_CLIPMODE clipmode)

    Equivalent to *self*.put(*values*, *indices*, *clipmode*
    ). Put *values* into *self* at the corresponding (flattened)
    *indices*. If *values* is too small it will be repeated as
    necessary.

.. c:function:: PyObject* PyArray_PutMask( \
        PyArrayObject* self, PyObject* values, PyObject* mask)

    Place the *values* in *self* wherever corresponding positions
    (using a flattened context) in *mask* are true. The *mask* and
    *self* arrays must have the same total number of elements. If
    *values* is too small, it will be repeated as necessary.

.. c:function:: PyObject* PyArray_Repeat( \
        PyArrayObject* self, PyObject* op, int axis)

    Equivalent to :meth:`ndarray.repeat<numpy.ndarray.repeat>` (*self*, *op*, *axis*). Copy the
    elements of *self*, *op* times along the given *axis*. Either
    *op* is a scalar integer or a sequence of length *self*
    ->dimensions[ *axis* ] indicating how many times to repeat each
    item along the axis.

.. c:function:: PyObject* PyArray_Choose( \
        PyArrayObject* self, PyObject* op, PyArrayObject* ret, \
        NPY_CLIPMODE clipmode)

    Equivalent to :meth:`ndarray.choose<numpy.ndarray.choose>` (*self*, *op*, *ret*, *clipmode*).
    Create a new array by selecting elements from the sequence of
    arrays in *op* based on the integer values in *self*. The arrays
    must all be broadcastable to the same shape and the entries in
    *self* should be between 0 and len(*op*). The output is placed
    in *ret* unless it is ``NULL`` in which case a new output is
    created. The *clipmode* argument determines behavior for when
    entries in *self* are not between 0 and len(*op*).

    .. c:var:: NPY_RAISE

        raise a ValueError;

    .. c:var:: NPY_WRAP

        wrap values < 0 by adding len(*op*) and values >=len(*op*)
        by subtracting len(*op*) until they are in range;

    .. c:var:: NPY_CLIP

        all values are clipped to the region [0, len(*op*) ).


.. c:function:: PyObject* PyArray_Sort(PyArrayObject* self, int axis, NPY_SORTKIND kind)

    Equivalent to :meth:`ndarray.sort<numpy.ndarray.sort>` (*self*, *axis*, *kind*).
    Return an array with the items of *self* sorted along *axis*. The array
    is sorted using the algorithm denoted by *kind* , which is an integer/enum pointing
    to the type of sorting algorithms used.

.. c:function:: PyObject* PyArray_ArgSort(PyArrayObject* self, int axis)

    Equivalent to :meth:`ndarray.argsort<numpy.ndarray.argsort>` (*self*, *axis*).
    Return an array of indices such that selection of these indices
    along the given ``axis`` would return a sorted version of *self*. If *self* ->descr
    is a data-type with fields defined, then self->descr->names is used
    to determine the sort order. A comparison where the first field is equal
    will use the second field and so on. To alter the sort order of a
    structured array, create a new data-type with a different order of names
    and construct a view of the array with that new data-type.

.. c:function:: PyObject* PyArray_LexSort(PyObject* sort_keys, int axis)

    Given a sequence of arrays (*sort_keys*) of the same shape,
    return an array of indices (similar to :c:func:`PyArray_ArgSort` (...))
    that would sort the arrays lexicographically. A lexicographic sort
    specifies that when two keys are found to be equal, the order is
    based on comparison of subsequent keys. A merge sort (which leaves
    equal entries unmoved) is required to be defined for the
    types. The sort is accomplished by sorting the indices first using
    the first *sort_key* and then using the second *sort_key* and so
    forth. This is equivalent to the lexsort(*sort_keys*, *axis*)
    Python command. Because of the way the merge-sort works, be sure
    to understand the order the *sort_keys* must be in (reversed from
    the order you would use when comparing two elements).

    If these arrays are all collected in a structured array, then
    :c:func:`PyArray_Sort` (...) can also be used to sort the array
    directly.

.. c:function:: PyObject* PyArray_SearchSorted( \
        PyArrayObject* self, PyObject* values, NPY_SEARCHSIDE side, \
        PyObject* perm)

    Equivalent to :meth:`ndarray.searchsorted<numpy.ndarray.searchsorted>` (*self*, *values*, *side*,
    *perm*). Assuming *self* is a 1-d array in ascending order, then the
    output is an array of indices the same shape as *values* such that, if
    the elements in *values* were inserted before the indices, the order of
    *self* would be preserved. No checking is done on whether or not self is
    in ascending order.

    The *side* argument indicates whether the index returned should be that of
    the first suitable location (if :c:data:`NPY_SEARCHLEFT`) or of the last
    (if :c:data:`NPY_SEARCHRIGHT`).

    The *sorter* argument, if not ``NULL``, must be a 1D array of integer
    indices the same length as *self*, that sorts it into ascending order.
    This is typically the result of a call to :c:func:`PyArray_ArgSort` (...)
    Binary search is used to find the required insertion points.

.. c:function:: int PyArray_Partition( \
        PyArrayObject *self, PyArrayObject * ktharray, int axis, \
        NPY_SELECTKIND which)

    Equivalent to :meth:`ndarray.partition<numpy.ndarray.partition>` (*self*, *ktharray*, *axis*,
    *kind*). Partitions the array so that the values of the element indexed by
    *ktharray* are in the positions they would be if the array is fully sorted
    and places all elements smaller than the kth before and all elements equal
    or greater after the kth element. The ordering of all elements within the
    partitions is undefined.
    If *self*->descr is a data-type with fields defined, then
    self->descr->names is used to determine the sort order. A comparison where
    the first field is equal will use the second field and so on. To alter the
    sort order of a structured array, create a new data-type with a different
    order of names and construct a view of the array with that new data-type.
    Returns zero on success and -1 on failure.

.. c:function:: PyObject* PyArray_ArgPartition( \
        PyArrayObject *op, PyArrayObject * ktharray, int axis, \
        NPY_SELECTKIND which)

    Equivalent to :meth:`ndarray.argpartition<numpy.ndarray.argpartition>` (*self*, *ktharray*, *axis*,
    *kind*). Return an array of indices such that selection of these indices
    along the given ``axis`` would return a partitioned version of *self*.

.. c:function:: PyObject* PyArray_Diagonal( \
        PyArrayObject* self, int offset, int axis1, int axis2)

    Equivalent to :meth:`ndarray.diagonal<numpy.ndarray.diagonal>` (*self*, *offset*, *axis1*, *axis2*
    ). Return the *offset* diagonals of the 2-d arrays defined by
    *axis1* and *axis2*.

.. c:function:: npy_intp PyArray_CountNonzero(PyArrayObject* self)

    .. versionadded:: 1.6

    Counts the number of non-zero elements in the array object *self*.

.. c:function:: PyObject* PyArray_Nonzero(PyArrayObject* self)

    Equivalent to :meth:`ndarray.nonzero<numpy.ndarray.nonzero>` (*self*). Returns a tuple of index
    arrays that select elements of *self* that are nonzero. If (nd=
    :c:func:`PyArray_NDIM` ( ``self`` ))==1, then a single index array is
    returned. The index arrays have data type :c:data:`NPY_INTP`. If a
    tuple is returned (nd :math:`\neq` 1), then its length is nd.

.. c:function:: PyObject* PyArray_Compress( \
        PyArrayObject* self, PyObject* condition, int axis, PyArrayObject* out)

    Equivalent to :meth:`ndarray.compress<numpy.ndarray.compress>` (*self*, *condition*, *axis*
    ). Return the elements along *axis* corresponding to elements of
    *condition* that are true.


Calculation
^^^^^^^^^^^

.. tip::

    Pass in :c:data:`NPY_MAXDIMS` for axis in order to achieve the same
    effect that is obtained by passing in *axis* = :const:`None` in Python
    (treating the array as a 1-d array).

.. c:function:: PyObject* PyArray_ArgMax( \
        PyArrayObject* self, int axis, PyArrayObject* out)

    Equivalent to :meth:`ndarray.argmax<numpy.ndarray.argmax>` (*self*, *axis*). Return the index of
    the largest element of *self* along *axis*.

.. c:function:: PyObject* PyArray_ArgMin( \
        PyArrayObject* self, int axis, PyArrayObject* out)

    Equivalent to :meth:`ndarray.argmin<numpy.ndarray.argmin>` (*self*, *axis*). Return the index of
    the smallest element of *self* along *axis*.




.. note::

    The out argument specifies where to place the result. If out is
    NULL, then the output array is created, otherwise the output is
    placed in out which must be the correct size and type. A new
    reference to the output array is always returned even when out
    is not NULL. The caller of the routine has the responsibility
    to ``DECREF`` out if not NULL or a memory-leak will occur.

.. c:function:: PyObject* PyArray_Max( \
        PyArrayObject* self, int axis, PyArrayObject* out)

    Equivalent to :meth:`ndarray.max<numpy.ndarray.max>` (*self*, *axis*). Returns the largest
    element of *self* along the given *axis*. When the result is a single
    element, returns a numpy scalar instead of an ndarray.

.. c:function:: PyObject* PyArray_Min( \
        PyArrayObject* self, int axis, PyArrayObject* out)

    Equivalent to :meth:`ndarray.min<numpy.ndarray.min>` (*self*, *axis*). Return the smallest
    element of *self* along the given *axis*. When the result is a single
    element, returns a numpy scalar instead of an ndarray.


.. c:function:: PyObject* PyArray_Ptp( \
        PyArrayObject* self, int axis, PyArrayObject* out)

    Equivalent to :meth:`ndarray.ptp<numpy.ndarray.ptp>` (*self*, *axis*). Return the difference
    between the largest element of *self* along *axis* and the
    smallest element of *self* along *axis*. When the result is a single
    element, returns a numpy scalar instead of an ndarray.




.. note::

    The rtype argument specifies the data-type the reduction should
    take place over. This is important if the data-type of the array
    is not "large" enough to handle the output. By default, all
    integer data-types are made at least as large as :c:data:`NPY_LONG`
    for the "add" and "multiply" ufuncs (which form the basis for
    mean, sum, cumsum, prod, and cumprod functions).

.. c:function:: PyObject* PyArray_Mean( \
        PyArrayObject* self, int axis, int rtype, PyArrayObject* out)

    Equivalent to :meth:`ndarray.mean<numpy.ndarray.mean>` (*self*, *axis*, *rtype*). Returns the
    mean of the elements along the given *axis*, using the enumerated
    type *rtype* as the data type to sum in. Default sum behavior is
    obtained using :c:data:`NPY_NOTYPE` for *rtype*.

.. c:function:: PyObject* PyArray_Trace( \
        PyArrayObject* self, int offset, int axis1, int axis2, int rtype, \
        PyArrayObject* out)

    Equivalent to :meth:`ndarray.trace<numpy.ndarray.trace>` (*self*, *offset*, *axis1*, *axis2*,
    *rtype*). Return the sum (using *rtype* as the data type of
    summation) over the *offset* diagonal elements of the 2-d arrays
    defined by *axis1* and *axis2* variables. A positive offset
    chooses diagonals above the main diagonal. A negative offset
    selects diagonals below the main diagonal.

.. c:function:: PyObject* PyArray_Clip( \
        PyArrayObject* self, PyObject* min, PyObject* max)

    Equivalent to :meth:`ndarray.clip<numpy.ndarray.clip>` (*self*, *min*, *max*). Clip an array,
    *self*, so that values larger than *max* are fixed to *max* and
    values less than *min* are fixed to *min*.

.. c:function:: PyObject* PyArray_Conjugate(PyArrayObject* self)

    Equivalent to :meth:`ndarray.conjugate<numpy.ndarray.conjugate>` (*self*).
    Return the complex conjugate of *self*. If *self* is not of
    complex data type, then return *self* with a reference.

.. c:function:: PyObject* PyArray_Round( \
        PyArrayObject* self, int decimals, PyArrayObject* out)

    Equivalent to :meth:`ndarray.round<numpy.ndarray.round>` (*self*, *decimals*, *out*). Returns
    the array with elements rounded to the nearest decimal place. The
    decimal place is defined as the :math:`10^{-\textrm{decimals}}`
    digit so that negative *decimals* cause rounding to the nearest 10's, 100's, etc. If out is ``NULL``, then the output array is created, otherwise the output is placed in *out* which must be the correct size and type.

.. c:function:: PyObject* PyArray_Std( \
        PyArrayObject* self, int axis, int rtype, PyArrayObject* out)

    Equivalent to :meth:`ndarray.std<numpy.ndarray.std>` (*self*, *axis*, *rtype*). Return the
    standard deviation using data along *axis* converted to data type
    *rtype*.

.. c:function:: PyObject* PyArray_Sum( \
        PyArrayObject* self, int axis, int rtype, PyArrayObject* out)

    Equivalent to :meth:`ndarray.sum<numpy.ndarray.sum>` (*self*, *axis*, *rtype*). Return 1-d
    vector sums of elements in *self* along *axis*. Perform the sum
    after converting data to data type *rtype*.

.. c:function:: PyObject* PyArray_CumSum( \
        PyArrayObject* self, int axis, int rtype, PyArrayObject* out)

    Equivalent to :meth:`ndarray.cumsum<numpy.ndarray.cumsum>` (*self*, *axis*, *rtype*). Return
    cumulative 1-d sums of elements in *self* along *axis*. Perform
    the sum after converting data to data type *rtype*.

.. c:function:: PyObject* PyArray_Prod( \
        PyArrayObject* self, int axis, int rtype, PyArrayObject* out)

    Equivalent to :meth:`ndarray.prod<numpy.ndarray.prod>` (*self*, *axis*, *rtype*). Return 1-d
    products of elements in *self* along *axis*. Perform the product
    after converting data to data type *rtype*.

.. c:function:: PyObject* PyArray_CumProd( \
        PyArrayObject* self, int axis, int rtype, PyArrayObject* out)

    Equivalent to :meth:`ndarray.cumprod<numpy.ndarray.cumprod>` (*self*, *axis*, *rtype*). Return
    1-d cumulative products of elements in ``self`` along ``axis``.
    Perform the product after converting data to data type ``rtype``.

.. c:function:: PyObject* PyArray_All( \
        PyArrayObject* self, int axis, PyArrayObject* out)

    Equivalent to :meth:`ndarray.all<numpy.ndarray.all>` (*self*, *axis*). Return an array with
    True elements for every 1-d sub-array of ``self`` defined by
    ``axis`` in which all the elements are True.

.. c:function:: PyObject* PyArray_Any( \
        PyArrayObject* self, int axis, PyArrayObject* out)

    Equivalent to :meth:`ndarray.any<numpy.ndarray.any>` (*self*, *axis*). Return an array with
    True elements for every 1-d sub-array of *self* defined by *axis*
    in which any of the elements are True.

Functions
---------


Array Functions
^^^^^^^^^^^^^^^

.. c:function:: int PyArray_AsCArray( \
        PyObject** op, void* ptr, npy_intp* dims, int nd, int typenum, \
        int itemsize)

    Sometimes it is useful to access a multidimensional array as a
    C-style multi-dimensional array so that algorithms can be
    implemented using C's a[i][j][k] syntax. This routine returns a
    pointer, *ptr*, that simulates this kind of C-style array, for
    1-, 2-, and 3-d ndarrays.

    :param op:

        The address to any Python object. This Python object will be replaced
        with an equivalent well-behaved, C-style contiguous, ndarray of the
        given data type specified by the last two arguments. Be sure that
        stealing a reference in this way to the input object is justified.

    :param ptr:

        The address to a (ctype* for 1-d, ctype** for 2-d or ctype*** for 3-d)
        variable where ctype is the equivalent C-type for the data type. On
        return, *ptr* will be addressable as a 1-d, 2-d, or 3-d array.

    :param dims:

        An output array that contains the shape of the array object. This
        array gives boundaries on any looping that will take place.

    :param nd:

        The dimensionality of the array (1, 2, or 3).

    :param typenum:

        The expected data type of the array.

    :param itemsize:

        This argument is only needed when *typenum* represents a
        flexible array. Otherwise it should be 0.

.. note::

    The simulation of a C-style array is not complete for 2-d and 3-d
    arrays. For example, the simulated arrays of pointers cannot be passed
    to subroutines expecting specific, statically-defined 2-d and 3-d
    arrays. To pass to functions requiring those kind of inputs, you must
    statically define the required array and copy data.

.. c:function:: int PyArray_Free(PyObject* op, void* ptr)

    Must be called with the same objects and memory locations returned
    from :c:func:`PyArray_AsCArray` (...). This function cleans up memory
    that otherwise would get leaked.

.. c:function:: PyObject* PyArray_Concatenate(PyObject* obj, int axis)

    Join the sequence of objects in *obj* together along *axis* into a
    single array. If the dimensions or types are not compatible an
    error is raised.

.. c:function:: PyObject* PyArray_InnerProduct(PyObject* obj1, PyObject* obj2)

    Compute a product-sum over the last dimensions of *obj1* and
    *obj2*. Neither array is conjugated.

.. c:function:: PyObject* PyArray_MatrixProduct(PyObject* obj1, PyObject* obj)

    Compute a product-sum over the last dimension of *obj1* and the
    second-to-last dimension of *obj2*. For 2-d arrays this is a
    matrix-product. Neither array is conjugated.

.. c:function:: PyObject* PyArray_MatrixProduct2( \
        PyObject* obj1, PyObject* obj, PyArrayObject* out)

    .. versionadded:: 1.6

    Same as PyArray_MatrixProduct, but store the result in *out*.  The
    output array must have the correct shape, type, and be
    C-contiguous, or an exception is raised.

.. c:function:: PyObject* PyArray_EinsteinSum( \
        char* subscripts, npy_intp nop, PyArrayObject** op_in, \
        PyArray_Descr* dtype, NPY_ORDER order, NPY_CASTING casting, \
        PyArrayObject* out)

    .. versionadded:: 1.6

    Applies the Einstein summation convention to the array operands
    provided, returning a new array or placing the result in *out*.
    The string in *subscripts* is a comma separated list of index
    letters. The number of operands is in *nop*, and *op_in* is an
    array containing those operands. The data type of the output can
    be forced with *dtype*, the output order can be forced with *order*
    (:c:data:`NPY_KEEPORDER` is recommended), and when *dtype* is specified,
    *casting* indicates how permissive the data conversion should be.

    See the :func:`~numpy.einsum` function for more details.

.. c:function:: PyObject* PyArray_CopyAndTranspose(PyObject \* op)

    A specialized copy and transpose function that works only for 2-d
    arrays. The returned array is a transposed copy of *op*.

.. c:function:: PyObject* PyArray_Correlate( \
        PyObject* op1, PyObject* op2, int mode)

    Compute the 1-d correlation of the 1-d arrays *op1* and *op2*
    . The correlation is computed at each output point by multiplying
    *op1* by a shifted version of *op2* and summing the result. As a
    result of the shift, needed values outside of the defined range of
    *op1* and *op2* are interpreted as zero. The mode determines how
    many shifts to return: 0 - return only shifts that did not need to
    assume zero- values; 1 - return an object that is the same size as
    *op1*, 2 - return all possible shifts (any overlap at all is
    accepted).

    .. rubric:: Notes

    This does not compute the usual correlation: if op2 is larger than op1, the
    arguments are swapped, and the conjugate is never taken for complex arrays.
    See PyArray_Correlate2 for the usual signal processing correlation.

.. c:function:: PyObject* PyArray_Correlate2( \
        PyObject* op1, PyObject* op2, int mode)

    Updated version of PyArray_Correlate, which uses the usual definition of
    correlation for 1d arrays. The correlation is computed at each output point
    by multiplying *op1* by a shifted version of *op2* and summing the result.
    As a result of the shift, needed values outside of the defined range of
    *op1* and *op2* are interpreted as zero. The mode determines how many
    shifts to return: 0 - return only shifts that did not need to assume zero-
    values; 1 - return an object that is the same size as *op1*, 2 - return all
    possible shifts (any overlap at all is accepted).

    .. rubric:: Notes

    Compute z as follows::

      z[k] = sum_n op1[n] * conj(op2[n+k])

.. c:function:: PyObject* PyArray_Where( \
        PyObject* condition, PyObject* x, PyObject* y)

    If both ``x`` and ``y`` are ``NULL``, then return
    :c:func:`PyArray_Nonzero` (*condition*). Otherwise, both *x* and *y*
    must be given and the object returned is shaped like *condition*
    and has elements of *x* and *y* where *condition* is respectively
    True or False.


Other functions
^^^^^^^^^^^^^^^

.. c:function:: Bool PyArray_CheckStrides( \
        int elsize, int nd, npy_intp numbytes, npy_intp* dims, \
        npy_intp* newstrides)

    Determine if *newstrides* is a strides array consistent with the
    memory of an *nd* -dimensional array with shape ``dims`` and
    element-size, *elsize*. The *newstrides* array is checked to see
    if jumping by the provided number of bytes in each direction will
    ever mean jumping more than *numbytes* which is the assumed size
    of the available memory segment. If *numbytes* is 0, then an
    equivalent *numbytes* is computed assuming *nd*, *dims*, and
    *elsize* refer to a single-segment array. Return :c:data:`NPY_TRUE` if
    *newstrides* is acceptable, otherwise return :c:data:`NPY_FALSE`.

.. c:function:: npy_intp PyArray_MultiplyList(npy_intp* seq, int n)

.. c:function:: int PyArray_MultiplyIntList(int* seq, int n)

    Both of these routines multiply an *n* -length array, *seq*, of
    integers and return the result. No overflow checking is performed.

.. c:function:: int PyArray_CompareLists(npy_intp* l1, npy_intp* l2, int n)

    Given two *n* -length arrays of integers, *l1*, and *l2*, return
    1 if the lists are identical; otherwise, return 0.


Auxiliary Data With Object Semantics
------------------------------------

.. versionadded:: 1.7.0

.. c:type:: NpyAuxData

When working with more complex dtypes which are composed of other dtypes,
such as the struct dtype, creating inner loops that manipulate the dtypes
requires carrying along additional data. NumPy supports this idea
through a struct :c:type:`NpyAuxData`, mandating a few conventions so that
it is possible to do this.

Defining an :c:type:`NpyAuxData` is similar to defining a class in C++,
but the object semantics have to be tracked manually since the API is in C.
Here's an example for a function which doubles up an element using
an element copier function as a primitive.::

    typedef struct {
        NpyAuxData base;
        ElementCopier_Func *func;
        NpyAuxData *funcdata;
    } eldoubler_aux_data;

    void free_element_doubler_aux_data(NpyAuxData *data)
    {
        eldoubler_aux_data *d = (eldoubler_aux_data *)data;
        /* Free the memory owned by this auxdata */
        NPY_AUXDATA_FREE(d->funcdata);
        PyArray_free(d);
    }

    NpyAuxData *clone_element_doubler_aux_data(NpyAuxData *data)
    {
        eldoubler_aux_data *ret = PyArray_malloc(sizeof(eldoubler_aux_data));
        if (ret == NULL) {
            return NULL;
        }

        /* Raw copy of all data */
        memcpy(ret, data, sizeof(eldoubler_aux_data));

        /* Fix up the owned auxdata so we have our own copy */
        ret->funcdata = NPY_AUXDATA_CLONE(ret->funcdata);
        if (ret->funcdata == NULL) {
            PyArray_free(ret);
            return NULL;
        }

        return (NpyAuxData *)ret;
    }

    NpyAuxData *create_element_doubler_aux_data(
                                ElementCopier_Func *func,
                                NpyAuxData *funcdata)
    {
        eldoubler_aux_data *ret = PyArray_malloc(sizeof(eldoubler_aux_data));
        if (ret == NULL) {
            PyErr_NoMemory();
            return NULL;
        }
        memset(&ret, 0, sizeof(eldoubler_aux_data));
        ret->base->free = &free_element_doubler_aux_data;
        ret->base->clone = &clone_element_doubler_aux_data;
        ret->func = func;
        ret->funcdata = funcdata;

        return (NpyAuxData *)ret;
    }

.. c:type:: NpyAuxData_FreeFunc

    The function pointer type for NpyAuxData free functions.

.. c:type:: NpyAuxData_CloneFunc

    The function pointer type for NpyAuxData clone functions. These
    functions should never set the Python exception on error, because
    they may be called from a multi-threaded context.

.. c:function:: NPY_AUXDATA_FREE(auxdata)

    A macro which calls the auxdata's free function appropriately,
    does nothing if auxdata is NULL.

.. c:function:: NPY_AUXDATA_CLONE(auxdata)

    A macro which calls the auxdata's clone function appropriately,
    returning a deep copy of the auxiliary data.

Array Iterators
---------------

As of NumPy 1.6.0, these array iterators are superceded by
the new array iterator, :c:type:`NpyIter`.

An array iterator is a simple way to access the elements of an
N-dimensional array quickly and efficiently. Section `2
<#sec-array-iterator>`__ provides more description and examples of
this useful approach to looping over an array.

.. c:function:: PyObject* PyArray_IterNew(PyObject* arr)

    Return an array iterator object from the array, *arr*. This is
    equivalent to *arr*. **flat**. The array iterator object makes
    it easy to loop over an N-dimensional non-contiguous array in
    C-style contiguous fashion.

.. c:function:: PyObject* PyArray_IterAllButAxis(PyObject* arr, int \*axis)

    Return an array iterator that will iterate over all axes but the
    one provided in *\*axis*. The returned iterator cannot be used
    with :c:func:`PyArray_ITER_GOTO1D`. This iterator could be used to
    write something similar to what ufuncs do wherein the loop over
    the largest axis is done by a separate sub-routine. If *\*axis* is
    negative then *\*axis* will be set to the axis having the smallest
    stride and that axis will be used.

.. c:function:: PyObject *PyArray_BroadcastToShape( \
        PyObject* arr, npy_intp *dimensions, int nd)

    Return an array iterator that is broadcast to iterate as an array
    of the shape provided by *dimensions* and *nd*.

.. c:function:: int PyArrayIter_Check(PyObject* op)

    Evaluates true if *op* is an array iterator (or instance of a
    subclass of the array iterator type).

.. c:function:: void PyArray_ITER_RESET(PyObject* iterator)

    Reset an *iterator* to the beginning of the array.

.. c:function:: void PyArray_ITER_NEXT(PyObject* iterator)

    Incremement the index and the dataptr members of the *iterator* to
    point to the next element of the array. If the array is not
    (C-style) contiguous, also increment the N-dimensional coordinates
    array.

.. c:function:: void *PyArray_ITER_DATA(PyObject* iterator)

    A pointer to the current element of the array.

.. c:function:: void PyArray_ITER_GOTO( \
        PyObject* iterator, npy_intp* destination)

    Set the *iterator* index, dataptr, and coordinates members to the
    location in the array indicated by the N-dimensional c-array,
    *destination*, which must have size at least *iterator*
    ->nd_m1+1.

.. c:function:: PyArray_ITER_GOTO1D(PyObject* iterator, npy_intp index)

    Set the *iterator* index and dataptr to the location in the array
    indicated by the integer *index* which points to an element in the
    C-styled flattened array.

.. c:function:: int PyArray_ITER_NOTDONE(PyObject* iterator)

    Evaluates TRUE as long as the iterator has not looped through all of
    the elements, otherwise it evaluates FALSE.


Broadcasting (multi-iterators)
------------------------------

.. c:function:: PyObject* PyArray_MultiIterNew(int num, ...)

    A simplified interface to broadcasting. This function takes the
    number of arrays to broadcast and then *num* extra ( :c:type:`PyObject *<PyObject>`
    ) arguments. These arguments are converted to arrays and iterators
    are created. :c:func:`PyArray_Broadcast` is then called on the resulting
    multi-iterator object. The resulting, broadcasted mult-iterator
    object is then returned. A broadcasted operation can then be
    performed using a single loop and using :c:func:`PyArray_MultiIter_NEXT`
    (..)

.. c:function:: void PyArray_MultiIter_RESET(PyObject* multi)

    Reset all the iterators to the beginning in a multi-iterator
    object, *multi*.

.. c:function:: void PyArray_MultiIter_NEXT(PyObject* multi)

    Advance each iterator in a multi-iterator object, *multi*, to its
    next (broadcasted) element.

.. c:function:: void *PyArray_MultiIter_DATA(PyObject* multi, int i)

    Return the data-pointer of the *i* :math:`^{\textrm{th}}` iterator
    in a multi-iterator object.

.. c:function:: void PyArray_MultiIter_NEXTi(PyObject* multi, int i)

    Advance the pointer of only the *i* :math:`^{\textrm{th}}` iterator.

.. c:function:: void PyArray_MultiIter_GOTO( \
        PyObject* multi, npy_intp* destination)

    Advance each iterator in a multi-iterator object, *multi*, to the
    given :math:`N` -dimensional *destination* where :math:`N` is the
    number of dimensions in the broadcasted array.

.. c:function:: void PyArray_MultiIter_GOTO1D(PyObject* multi, npy_intp index)

    Advance each iterator in a multi-iterator object, *multi*, to the
    corresponding location of the *index* into the flattened
    broadcasted array.

.. c:function:: int PyArray_MultiIter_NOTDONE(PyObject* multi)

    Evaluates TRUE as long as the multi-iterator has not looped
    through all of the elements (of the broadcasted result), otherwise
    it evaluates FALSE.

.. c:function:: int PyArray_Broadcast(PyArrayMultiIterObject* mit)

    This function encapsulates the broadcasting rules. The *mit*
    container should already contain iterators for all the arrays that
    need to be broadcast. On return, these iterators will be adjusted
    so that iteration over each simultaneously will accomplish the
    broadcasting. A negative number is returned if an error occurs.

.. c:function:: int PyArray_RemoveSmallest(PyArrayMultiIterObject* mit)

    This function takes a multi-iterator object that has been
    previously "broadcasted," finds the dimension with the smallest
    "sum of strides" in the broadcasted result and adapts all the
    iterators so as not to iterate over that dimension (by effectively
    making them of length-1 in that dimension). The corresponding
    dimension is returned unless *mit* ->nd is 0, then -1 is
    returned. This function is useful for constructing ufunc-like
    routines that broadcast their inputs correctly and then call a
    strided 1-d version of the routine as the inner-loop.  This 1-d
    version is usually optimized for speed and for this reason the
    loop should be performed over the axis that won't require large
    stride jumps.

Neighborhood iterator
---------------------

.. versionadded:: 1.4.0

Neighborhood iterators are subclasses of the iterator object, and can be used
to iter over a neighborhood of a point. For example, you may want to iterate
over every voxel of a 3d image, and for every such voxel, iterate over an
hypercube. Neighborhood iterator automatically handle boundaries, thus making
this kind of code much easier to write than manual boundaries handling, at the
cost of a slight overhead.

.. c:function:: PyObject* PyArray_NeighborhoodIterNew( \
        PyArrayIterObject* iter, npy_intp bounds, int mode, \
        PyArrayObject* fill_value)

    This function creates a new neighborhood iterator from an existing
    iterator.  The neighborhood will be computed relatively to the position
    currently pointed by *iter*, the bounds define the shape of the
    neighborhood iterator, and the mode argument the boundaries handling mode.

    The *bounds* argument is expected to be a (2 * iter->ao->nd) arrays, such
    as the range bound[2*i]->bounds[2*i+1] defines the range where to walk for
    dimension i (both bounds are included in the walked coordinates). The
    bounds should be ordered for each dimension (bounds[2*i] <= bounds[2*i+1]).

    The mode should be one of:

    * NPY_NEIGHBORHOOD_ITER_ZERO_PADDING: zero padding. Outside bounds values
      will be 0.
    * NPY_NEIGHBORHOOD_ITER_ONE_PADDING: one padding, Outside bounds values
      will be 1.
    * NPY_NEIGHBORHOOD_ITER_CONSTANT_PADDING: constant padding. Outside bounds
      values will be the same as the first item in fill_value.
    * NPY_NEIGHBORHOOD_ITER_MIRROR_PADDING: mirror padding. Outside bounds
      values will be as if the array items were mirrored. For example, for the
      array [1, 2, 3, 4], x[-2] will be 2, x[-2] will be 1, x[4] will be 4,
      x[5] will be 1, etc...
    * NPY_NEIGHBORHOOD_ITER_CIRCULAR_PADDING: circular padding. Outside bounds
      values will be as if the array was repeated. For example, for the
      array [1, 2, 3, 4], x[-2] will be 3, x[-2] will be 4, x[4] will be 1,
      x[5] will be 2, etc...

    If the mode is constant filling (NPY_NEIGHBORHOOD_ITER_CONSTANT_PADDING),
    fill_value should point to an array object which holds the filling value
    (the first item will be the filling value if the array contains more than
    one item). For other cases, fill_value may be NULL.

    - The iterator holds a reference to iter
    - Return NULL on failure (in which case the reference count of iter is not
      changed)
    - iter itself can be a Neighborhood iterator: this can be useful for .e.g
      automatic boundaries handling
    - the object returned by this function should be safe to use as a normal
      iterator
    - If the position of iter is changed, any subsequent call to
      PyArrayNeighborhoodIter_Next is undefined behavior, and
      PyArrayNeighborhoodIter_Reset must be called.

    .. code-block:: c

       PyArrayIterObject *iter;
       PyArrayNeighborhoodIterObject *neigh_iter;
       iter = PyArray_IterNew(x);

       /*For a 3x3 kernel */
       bounds = {-1, 1, -1, 1};
       neigh_iter = (PyArrayNeighborhoodIterObject*)PyArrayNeighborhoodIter_New(
            iter, bounds, NPY_NEIGHBORHOOD_ITER_ZERO_PADDING, NULL);

       for(i = 0; i < iter->size; ++i) {
            for (j = 0; j < neigh_iter->size; ++j) {
                    /* Walk around the item currently pointed by iter->dataptr */
                    PyArrayNeighborhoodIter_Next(neigh_iter);
            }

            /* Move to the next point of iter */
            PyArrayIter_Next(iter);
            PyArrayNeighborhoodIter_Reset(neigh_iter);
       }

.. c:function:: int PyArrayNeighborhoodIter_Reset( \
        PyArrayNeighborhoodIterObject* iter)

    Reset the iterator position to the first point of the neighborhood. This
    should be called whenever the iter argument given at
    PyArray_NeighborhoodIterObject is changed (see example)

.. c:function:: int PyArrayNeighborhoodIter_Next( \
        PyArrayNeighborhoodIterObject* iter)

    After this call, iter->dataptr points to the next point of the
    neighborhood. Calling this function after every point of the
    neighborhood has been visited is undefined.

Array Scalars
-------------

.. c:function:: PyObject* PyArray_Return(PyArrayObject* arr)

    This function steals a reference to *arr*.

    This function checks to see if *arr* is a 0-dimensional array and,
    if so, returns the appropriate array scalar. It should be used
    whenever 0-dimensional arrays could be returned to Python.

.. c:function:: PyObject* PyArray_Scalar( \
        void* data, PyArray_Descr* dtype, PyObject* itemsize)

    Return an array scalar object of the given enumerated *typenum*
    and *itemsize* by **copying** from memory pointed to by *data*
    . If *swap* is nonzero then this function will byteswap the data
    if appropriate to the data-type because array scalars are always
    in correct machine-byte order.

.. c:function:: PyObject* PyArray_ToScalar(void* data, PyArrayObject* arr)

    Return an array scalar object of the type and itemsize indicated
    by the array object *arr* copied from the memory pointed to by
    *data* and swapping if the data in *arr* is not in machine
    byte-order.

.. c:function:: PyObject* PyArray_FromScalar( \
        PyObject* scalar, PyArray_Descr* outcode)

    Return a 0-dimensional array of type determined by *outcode* from
    *scalar* which should be an array-scalar object. If *outcode* is
    NULL, then the type is determined from *scalar*.

.. c:function:: void PyArray_ScalarAsCtype(PyObject* scalar, void* ctypeptr)

    Return in *ctypeptr* a pointer to the actual value in an array
    scalar. There is no error checking so *scalar* must be an
    array-scalar object, and ctypeptr must have enough space to hold
    the correct type. For flexible-sized types, a pointer to the data
    is copied into the memory of *ctypeptr*, for all other types, the
    actual data is copied into the address pointed to by *ctypeptr*.

.. c:function:: void PyArray_CastScalarToCtype( \
        PyObject* scalar, void* ctypeptr, PyArray_Descr* outcode)

    Return the data (cast to the data type indicated by *outcode*)
    from the array-scalar, *scalar*, into the memory pointed to by
    *ctypeptr* (which must be large enough to handle the incoming
    memory).

.. c:function:: PyObject* PyArray_TypeObjectFromType(int type)

    Returns a scalar type-object from a type-number, *type*
    . Equivalent to :c:func:`PyArray_DescrFromType` (*type*)->typeobj
    except for reference counting and error-checking. Returns a new
    reference to the typeobject on success or ``NULL`` on failure.

.. c:function:: NPY_SCALARKIND PyArray_ScalarKind( \
        int typenum, PyArrayObject** arr)

    See the function :c:func:`PyArray_MinScalarType` for an alternative
    mechanism introduced in NumPy 1.6.0.

    Return the kind of scalar represented by *typenum* and the array
    in *\*arr* (if *arr* is not ``NULL`` ). The array is assumed to be
    rank-0 and only used if *typenum* represents a signed integer. If
    *arr* is not ``NULL`` and the first element is negative then
    :c:data:`NPY_INTNEG_SCALAR` is returned, otherwise
    :c:data:`NPY_INTPOS_SCALAR` is returned. The possible return values
    are :c:data:`NPY_{kind}_SCALAR` where ``{kind}`` can be **INTPOS**,
    **INTNEG**, **FLOAT**, **COMPLEX**, **BOOL**, or **OBJECT**.
    :c:data:`NPY_NOSCALAR` is also an enumerated value
    :c:type:`NPY_SCALARKIND` variables can take on.

.. c:function:: int PyArray_CanCoerceScalar( \
        char thistype, char neededtype, NPY_SCALARKIND scalar)

    See the function :c:func:`PyArray_ResultType` for details of
    NumPy type promotion, updated in NumPy 1.6.0.

    Implements the rules for scalar coercion. Scalars are only
    silently coerced from thistype to neededtype if this function
    returns nonzero.  If scalar is :c:data:`NPY_NOSCALAR`, then this
    function is equivalent to :c:func:`PyArray_CanCastSafely`. The rule is
    that scalars of the same KIND can be coerced into arrays of the
    same KIND. This rule means that high-precision scalars will never
    cause low-precision arrays of the same KIND to be upcast.


Data-type descriptors
---------------------



.. warning::

    Data-type objects must be reference counted so be aware of the
    action on the data-type reference of different C-API calls. The
    standard rule is that when a data-type object is returned it is a
    new reference.  Functions that take :c:type:`PyArray_Descr *` objects and
    return arrays steal references to the data-type their inputs
    unless otherwise noted. Therefore, you must own a reference to any
    data-type object used as input to such a function.

.. c:function:: int PyArray_DescrCheck(PyObject* obj)

    Evaluates as true if *obj* is a data-type object ( :c:type:`PyArray_Descr *` ).

.. c:function:: PyArray_Descr* PyArray_DescrNew(PyArray_Descr* obj)

    Return a new data-type object copied from *obj* (the fields
    reference is just updated so that the new object points to the
    same fields dictionary if any).

.. c:function:: PyArray_Descr* PyArray_DescrNewFromType(int typenum)

    Create a new data-type object from the built-in (or
    user-registered) data-type indicated by *typenum*. All builtin
    types should not have any of their fields changed. This creates a
    new copy of the :c:type:`PyArray_Descr` structure so that you can fill
    it in as appropriate. This function is especially needed for
    flexible data-types which need to have a new elsize member in
    order to be meaningful in array construction.

.. c:function:: PyArray_Descr* PyArray_DescrNewByteorder( \
        PyArray_Descr* obj, char newendian)

    Create a new data-type object with the byteorder set according to
    *newendian*. All referenced data-type objects (in subdescr and
    fields members of the data-type object) are also changed
    (recursively). If a byteorder of :c:data:`NPY_IGNORE` is encountered it
    is left alone. If newendian is :c:data:`NPY_SWAP`, then all byte-orders
    are swapped. Other valid newendian values are :c:data:`NPY_NATIVE`,
    :c:data:`NPY_LITTLE`, and :c:data:`NPY_BIG` which all cause the returned
    data-typed descriptor (and all it's
    referenced data-type descriptors) to have the corresponding byte-
    order.

.. c:function:: PyArray_Descr* PyArray_DescrFromObject( \
        PyObject* op, PyArray_Descr* mintype)

    Determine an appropriate data-type object from the object *op*
    (which should be a "nested" sequence object) and the minimum
    data-type descriptor mintype (which can be ``NULL`` ). Similar in
    behavior to array(*op*).dtype. Don't confuse this function with
    :c:func:`PyArray_DescrConverter`. This function essentially looks at
    all the objects in the (nested) sequence and determines the
    data-type from the elements it finds.

.. c:function:: PyArray_Descr* PyArray_DescrFromScalar(PyObject* scalar)

    Return a data-type object from an array-scalar object. No checking
    is done to be sure that *scalar* is an array scalar. If no
    suitable data-type can be determined, then a data-type of
    :c:data:`NPY_OBJECT` is returned by default.

.. c:function:: PyArray_Descr* PyArray_DescrFromType(int typenum)

    Returns a data-type object corresponding to *typenum*. The
    *typenum* can be one of the enumerated types, a character code for
    one of the enumerated types, or a user-defined type. If you want to use a
    flexible size array, then you need to ``flexible typenum`` and set the
    results ``elsize`` parameter to the desired size. The typenum is one of the
    :c:data:`NPY_TYPES`.

.. c:function:: int PyArray_DescrConverter(PyObject* obj, PyArray_Descr** dtype)

    Convert any compatible Python object, *obj*, to a data-type object
    in *dtype*. A large number of Python objects can be converted to
    data-type objects. See :ref:`arrays.dtypes` for a complete
    description. This version of the converter converts None objects
    to a :c:data:`NPY_DEFAULT_TYPE` data-type object. This function can
    be used with the "O&" character code in :c:func:`PyArg_ParseTuple`
    processing.

.. c:function:: int PyArray_DescrConverter2( \
        PyObject* obj, PyArray_Descr** dtype)

    Convert any compatible Python object, *obj*, to a data-type
    object in *dtype*. This version of the converter converts None
    objects so that the returned data-type is ``NULL``. This function
    can also be used with the "O&" character in PyArg_ParseTuple
    processing.

.. c:function:: int Pyarray_DescrAlignConverter( \
        PyObject* obj, PyArray_Descr** dtype)

    Like :c:func:`PyArray_DescrConverter` except it aligns C-struct-like
    objects on word-boundaries as the compiler would.

.. c:function:: int Pyarray_DescrAlignConverter2( \
        PyObject* obj, PyArray_Descr** dtype)

    Like :c:func:`PyArray_DescrConverter2` except it aligns C-struct-like
    objects on word-boundaries as the compiler would.

.. c:function:: PyObject *PyArray_FieldNames(PyObject* dict)

    Take the fields dictionary, *dict*, such as the one attached to a
    data-type object and construct an ordered-list of field names such
    as is stored in the names field of the :c:type:`PyArray_Descr` object.


Conversion Utilities
--------------------


For use with :c:func:`PyArg_ParseTuple`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All of these functions can be used in :c:func:`PyArg_ParseTuple` (...) with
the "O&" format specifier to automatically convert any Python object
to the required C-object. All of these functions return
:c:data:`NPY_SUCCEED` if successful and :c:data:`NPY_FAIL` if not. The first
argument to all of these function is a Python object. The second
argument is the **address** of the C-type to convert the Python object
to.


.. warning::

    Be sure to understand what steps you should take to manage the
    memory when using these conversion functions. These functions can
    require freeing memory, and/or altering the reference counts of
    specific objects based on your use.

.. c:function:: int PyArray_Converter(PyObject* obj, PyObject** address)

    Convert any Python object to a :c:type:`PyArrayObject`. If
    :c:func:`PyArray_Check` (*obj*) is TRUE then its reference count is
    incremented and a reference placed in *address*. If *obj* is not
    an array, then convert it to an array using :c:func:`PyArray_FromAny`
    . No matter what is returned, you must DECREF the object returned
    by this routine in *address* when you are done with it.

.. c:function:: int PyArray_OutputConverter( \
        PyObject* obj, PyArrayObject** address)

    This is a default converter for output arrays given to
    functions. If *obj* is :c:data:`Py_None` or ``NULL``, then *\*address*
    will be ``NULL`` but the call will succeed. If :c:func:`PyArray_Check` (
    *obj*) is TRUE then it is returned in *\*address* without
    incrementing its reference count.

.. c:function:: int PyArray_IntpConverter(PyObject* obj, PyArray_Dims* seq)

    Convert any Python sequence, *obj*, smaller than :c:data:`NPY_MAXDIMS`
    to a C-array of :c:type:`npy_intp`. The Python object could also be a
    single number. The *seq* variable is a pointer to a structure with
    members ptr and len. On successful return, *seq* ->ptr contains a
    pointer to memory that must be freed, by calling :c:func:`PyDimMem_FREE`,
    to avoid a memory leak. The restriction on memory size allows this
    converter to be conveniently used for sequences intended to be
    interpreted as array shapes.

.. c:function:: int PyArray_BufferConverter(PyObject* obj, PyArray_Chunk* buf)

    Convert any Python object, *obj*, with a (single-segment) buffer
    interface to a variable with members that detail the object's use
    of its chunk of memory. The *buf* variable is a pointer to a
    structure with base, ptr, len, and flags members. The
    :c:type:`PyArray_Chunk` structure is binary compatible with the
    Python's buffer object (through its len member on 32-bit platforms
    and its ptr member on 64-bit platforms or in Python 2.5). On
    return, the base member is set to *obj* (or its base if *obj* is
    already a buffer object pointing to another object). If you need
    to hold on to the memory be sure to INCREF the base member. The
    chunk of memory is pointed to by *buf* ->ptr member and has length
    *buf* ->len. The flags member of *buf* is :c:data:`NPY_BEHAVED_RO` with
    the :c:data:`NPY_ARRAY_WRITEABLE` flag set if *obj* has a writeable buffer
    interface.

.. c:function:: int PyArray_AxisConverter(PyObject \* obj, int* axis)

    Convert a Python object, *obj*, representing an axis argument to
    the proper value for passing to the functions that take an integer
    axis. Specifically, if *obj* is None, *axis* is set to
    :c:data:`NPY_MAXDIMS` which is interpreted correctly by the C-API
    functions that take axis arguments.

.. c:function:: int PyArray_BoolConverter(PyObject* obj, Bool* value)

    Convert any Python object, *obj*, to :c:data:`NPY_TRUE` or
    :c:data:`NPY_FALSE`, and place the result in *value*.

.. c:function:: int PyArray_ByteorderConverter(PyObject* obj, char* endian)

    Convert Python strings into the corresponding byte-order
    character:
    '>', '<', 's', '=', or '\|'.

.. c:function:: int PyArray_SortkindConverter(PyObject* obj, NPY_SORTKIND* sort)

    Convert Python strings into one of :c:data:`NPY_QUICKSORT` (starts
    with 'q' or 'Q'), :c:data:`NPY_HEAPSORT` (starts with 'h' or 'H'),
    :c:data:`NPY_MERGESORT` (starts with 'm' or 'M') or :c:data:`NPY_STABLESORT`
    (starts with 't' or 'T'). :c:data:`NPY_MERGESORT` and :c:data:`NPY_STABLESORT`
    are aliased to each other for backwards compatibility and may refer to one
    of several stable sorting algorithms depending on the data type.

.. c:function:: int PyArray_SearchsideConverter( \
        PyObject* obj, NPY_SEARCHSIDE* side)

    Convert Python strings into one of :c:data:`NPY_SEARCHLEFT` (starts with 'l'
    or 'L'), or :c:data:`NPY_SEARCHRIGHT` (starts with 'r' or 'R').

.. c:function:: int PyArray_OrderConverter(PyObject* obj, NPY_ORDER* order)

   Convert the Python strings 'C', 'F', 'A', and 'K' into the :c:type:`NPY_ORDER`
   enumeration :c:data:`NPY_CORDER`, :c:data:`NPY_FORTRANORDER`,
   :c:data:`NPY_ANYORDER`, and :c:data:`NPY_KEEPORDER`.

.. c:function:: int PyArray_CastingConverter( \
        PyObject* obj, NPY_CASTING* casting)

   Convert the Python strings 'no', 'equiv', 'safe', 'same_kind', and
   'unsafe' into the :c:type:`NPY_CASTING` enumeration :c:data:`NPY_NO_CASTING`,
   :c:data:`NPY_EQUIV_CASTING`, :c:data:`NPY_SAFE_CASTING`,
   :c:data:`NPY_SAME_KIND_CASTING`, and :c:data:`NPY_UNSAFE_CASTING`.

.. c:function:: int PyArray_ClipmodeConverter( \
        PyObject* object, NPY_CLIPMODE* val)

    Convert the Python strings 'clip', 'wrap', and 'raise' into the
    :c:type:`NPY_CLIPMODE` enumeration :c:data:`NPY_CLIP`, :c:data:`NPY_WRAP`,
    and :c:data:`NPY_RAISE`.

.. c:function:: int PyArray_ConvertClipmodeSequence( \
        PyObject* object, NPY_CLIPMODE* modes, int n)

   Converts either a sequence of clipmodes or a single clipmode into
   a C array of :c:type:`NPY_CLIPMODE` values. The number of clipmodes *n*
   must be known before calling this function. This function is provided
   to help functions allow a different clipmode for each dimension.

Other conversions
^^^^^^^^^^^^^^^^^

.. c:function:: int PyArray_PyIntAsInt(PyObject* op)

    Convert all kinds of Python objects (including arrays and array
    scalars) to a standard integer. On error, -1 is returned and an
    exception set. You may find useful the macro:

    .. code-block:: c

        #define error_converting(x) (((x) == -1) && PyErr_Occurred()

.. c:function:: npy_intp PyArray_PyIntAsIntp(PyObject* op)

    Convert all kinds of Python objects (including arrays and array
    scalars) to a (platform-pointer-sized) integer. On error, -1 is
    returned and an exception set.

.. c:function:: int PyArray_IntpFromSequence( \
        PyObject* seq, npy_intp* vals, int maxvals)

    Convert any Python sequence (or single Python number) passed in as
    *seq* to (up to) *maxvals* pointer-sized integers and place them
    in the *vals* array. The sequence can be smaller then *maxvals* as
    the number of converted objects is returned.

.. c:function:: int PyArray_TypestrConvert(int itemsize, int gentype)

    Convert typestring characters (with *itemsize*) to basic
    enumerated data types. The typestring character corresponding to
    signed and unsigned integers, floating point numbers, and
    complex-floating point numbers are recognized and converted. Other
    values of gentype are returned. This function can be used to
    convert, for example, the string 'f4' to :c:data:`NPY_FLOAT32`.


Miscellaneous
-------------


Importing the API
^^^^^^^^^^^^^^^^^

In order to make use of the C-API from another extension module, the
:c:func:`import_array` function must be called. If the extension module is
self-contained in a single .c file, then that is all that needs to be
done. If, however, the extension module involves multiple files where
the C-API is needed then some additional steps must be taken.

.. c:function:: void import_array(void)

    This function must be called in the initialization section of a
    module that will make use of the C-API. It imports the module
    where the function-pointer table is stored and points the correct
    variable to it.

.. c:macro:: PY_ARRAY_UNIQUE_SYMBOL

.. c:macro:: NO_IMPORT_ARRAY

    Using these #defines you can use the C-API in multiple files for a
    single extension module. In each file you must define
    :c:macro:`PY_ARRAY_UNIQUE_SYMBOL` to some name that will hold the
    C-API (*e.g.* myextension_ARRAY_API). This must be done **before**
    including the numpy/arrayobject.h file. In the module
    initialization routine you call :c:func:`import_array`. In addition,
    in the files that do not have the module initialization
    sub_routine define :c:macro:`NO_IMPORT_ARRAY` prior to including
    numpy/arrayobject.h.

    Suppose I have two files coolmodule.c and coolhelper.c which need
    to be compiled and linked into a single extension module. Suppose
    coolmodule.c contains the required initcool module initialization
    function (with the import_array() function called). Then,
    coolmodule.c would have at the top:

    .. code-block:: c

        #define PY_ARRAY_UNIQUE_SYMBOL cool_ARRAY_API
        #include numpy/arrayobject.h

    On the other hand, coolhelper.c would contain at the top:

    .. code-block:: c

        #define NO_IMPORT_ARRAY
        #define PY_ARRAY_UNIQUE_SYMBOL cool_ARRAY_API
        #include numpy/arrayobject.h

    You can also put the common two last lines into an extension-local
    header file as long as you make sure that NO_IMPORT_ARRAY is
    #defined before #including that file.

    Internally, these #defines work as follows:

        * If neither is defined, the C-API is declared to be
          :c:type:`static void**`, so it is only visible within the
          compilation unit that #includes numpy/arrayobject.h.
        * If :c:macro:`PY_ARRAY_UNIQUE_SYMBOL` is #defined, but
          :c:macro:`NO_IMPORT_ARRAY` is not, the C-API is declared to
          be :c:type:`void**`, so that it will also be visible to other
          compilation units.
        * If :c:macro:`NO_IMPORT_ARRAY` is #defined, regardless of
          whether :c:macro:`PY_ARRAY_UNIQUE_SYMBOL` is, the C-API is
          declared to be :c:type:`extern void**`, so it is expected to
          be defined in another compilation unit.
        * Whenever :c:macro:`PY_ARRAY_UNIQUE_SYMBOL` is #defined, it
          also changes the name of the variable holding the C-API, which
          defaults to :c:data:`PyArray_API`, to whatever the macro is
          #defined to.

Checking the API Version
^^^^^^^^^^^^^^^^^^^^^^^^

Because python extensions are not used in the same way as usual libraries on
most platforms, some errors cannot be automatically detected at build time or
even runtime. For example, if you build an extension using a function available
only for numpy >= 1.3.0, and you import the extension later with numpy 1.2, you
will not get an import error (but almost certainly a segmentation fault when
calling the function). That's why several functions are provided to check for
numpy versions. The macros :c:data:`NPY_VERSION`  and
:c:data:`NPY_FEATURE_VERSION` corresponds to the numpy version used to build the
extension, whereas the versions returned by the functions
PyArray_GetNDArrayCVersion and PyArray_GetNDArrayCFeatureVersion corresponds to
the runtime numpy's version.

The rules for ABI and API compatibilities can be summarized as follows:

    * Whenever :c:data:`NPY_VERSION` != PyArray_GetNDArrayCVersion, the
      extension has to be recompiled (ABI incompatibility).
    * :c:data:`NPY_VERSION` == PyArray_GetNDArrayCVersion and
      :c:data:`NPY_FEATURE_VERSION` <= PyArray_GetNDArrayCFeatureVersion means
      backward compatible changes.

ABI incompatibility is automatically detected in every numpy's version. API
incompatibility detection was added in numpy 1.4.0. If you want to supported
many different numpy versions with one extension binary, you have to build your
extension with the lowest NPY_FEATURE_VERSION as possible.

.. c:function:: unsigned int PyArray_GetNDArrayCVersion(void)

    This just returns the value :c:data:`NPY_VERSION`. :c:data:`NPY_VERSION`
    changes whenever a backward incompatible change at the ABI level. Because
    it is in the C-API, however, comparing the output of this function from the
    value defined in the current header gives a way to test if the C-API has
    changed thus requiring a re-compilation of extension modules that use the
    C-API. This is automatically checked in the function :c:func:`import_array`.

.. c:function:: unsigned int PyArray_GetNDArrayCFeatureVersion(void)

    .. versionadded:: 1.4.0

    This just returns the value :c:data:`NPY_FEATURE_VERSION`.
    :c:data:`NPY_FEATURE_VERSION` changes whenever the API changes (e.g. a
    function is added). A changed value does not always require a recompile.

Internal Flexibility
^^^^^^^^^^^^^^^^^^^^

.. c:function:: int PyArray_SetNumericOps(PyObject* dict)

    NumPy stores an internal table of Python callable objects that are
    used to implement arithmetic operations for arrays as well as
    certain array calculation methods. This function allows the user
    to replace any or all of these Python objects with their own
    versions. The keys of the dictionary, *dict*, are the named
    functions to replace and the paired value is the Python callable
    object to use. Care should be taken that the function used to
    replace an internal array operation does not itself call back to
    that internal array operation (unless you have designed the
    function to handle that), or an unchecked infinite recursion can
    result (possibly causing program crash). The key names that
    represent operations that can be replaced are:

        **add**, **subtract**, **multiply**, **divide**,
        **remainder**, **power**, **square**, **reciprocal**,
        **ones_like**, **sqrt**, **negative**, **positive**,
        **absolute**, **invert**, **left_shift**, **right_shift**,
        **bitwise_and**, **bitwise_xor**, **bitwise_or**,
        **less**, **less_equal**, **equal**, **not_equal**,
        **greater**, **greater_equal**, **floor_divide**,
        **true_divide**, **logical_or**, **logical_and**,
        **floor**, **ceil**, **maximum**, **minimum**, **rint**.


    These functions are included here because they are used at least once
    in the array object's methods. The function returns -1 (without
    setting a Python Error) if one of the objects being assigned is not
    callable.

    .. deprecated:: 1.16

.. c:function:: PyObject* PyArray_GetNumericOps(void)

    Return a Python dictionary containing the callable Python objects
    stored in the internal arithmetic operation table. The keys of
    this dictionary are given in the explanation for :c:func:`PyArray_SetNumericOps`.

    .. deprecated:: 1.16

.. c:function:: void PyArray_SetStringFunction(PyObject* op, int repr)

    This function allows you to alter the tp_str and tp_repr methods
    of the array object to any Python function. Thus you can alter
    what happens for all arrays when str(arr) or repr(arr) is called
    from Python. The function to be called is passed in as *op*. If
    *repr* is non-zero, then this function will be called in response
    to repr(arr), otherwise the function will be called in response to
    str(arr). No check on whether or not *op* is callable is
    performed. The callable passed in to *op* should expect an array
    argument and should return a string to be printed.


Memory management
^^^^^^^^^^^^^^^^^

.. c:function:: char* PyDataMem_NEW(size_t nbytes)

.. c:function:: PyDataMem_FREE(char* ptr)

.. c:function:: char* PyDataMem_RENEW(void * ptr, size_t newbytes)

    Macros to allocate, free, and reallocate memory. These macros are used
    internally to create arrays.

.. c:function:: npy_intp*  PyDimMem_NEW(int nd)

.. c:function:: PyDimMem_FREE(char* ptr)

.. c:function:: npy_intp* PyDimMem_RENEW(void* ptr, size_t newnd)

    Macros to allocate, free, and reallocate dimension and strides memory.

.. c:function:: void* PyArray_malloc(size_t nbytes)

.. c:function:: PyArray_free(void* ptr)

.. c:function:: void* PyArray_realloc(npy_intp* ptr, size_t nbytes)

    These macros use different memory allocators, depending on the
    constant :c:data:`NPY_USE_PYMEM`. The system malloc is used when
    :c:data:`NPY_USE_PYMEM` is 0, if :c:data:`NPY_USE_PYMEM` is 1, then
    the Python memory allocator is used.

.. c:function:: int PyArray_ResolveWritebackIfCopy(PyArrayObject* obj)

    If ``obj.flags`` has :c:data:`NPY_ARRAY_WRITEBACKIFCOPY` or (deprecated)
    :c:data:`NPY_ARRAY_UPDATEIFCOPY`, this function clears the flags, `DECREF` s
    `obj->base` and makes it writeable, and sets ``obj->base`` to NULL. It then
    copies ``obj->data`` to `obj->base->data`, and returns the error state of
    the copy operation. This is the opposite of
    :c:func:`PyArray_SetWritebackIfCopyBase`. Usually this is called once
    you are finished with ``obj``, just before ``Py_DECREF(obj)``. It may be called
    multiple times, or with ``NULL`` input. See also
    :c:func:`PyArray_DiscardWritebackIfCopy`.

    Returns 0 if nothing was done, -1 on error, and 1 if action was taken.

Threading support
^^^^^^^^^^^^^^^^^

These macros are only meaningful if :c:data:`NPY_ALLOW_THREADS`
evaluates True during compilation of the extension module. Otherwise,
these macros are equivalent to whitespace. Python uses a single Global
Interpreter Lock (GIL) for each Python process so that only a single
thread may execute at a time (even on multi-cpu machines). When
calling out to a compiled function that may take time to compute (and
does not have side-effects for other threads like updated global
variables), the GIL should be released so that other Python threads
can run while the time-consuming calculations are performed. This can
be accomplished using two groups of macros. Typically, if one macro in
a group is used in a code block, all of them must be used in the same
code block. Currently, :c:data:`NPY_ALLOW_THREADS` is defined to the
python-defined :c:data:`WITH_THREADS` constant unless the environment
variable :c:data:`NPY_NOSMP` is set in which case
:c:data:`NPY_ALLOW_THREADS` is defined to be 0.

Group 1
"""""""

    This group is used to call code that may take some time but does not
    use any Python C-API calls. Thus, the GIL should be released during
    its calculation.

    .. c:macro:: NPY_BEGIN_ALLOW_THREADS

        Equivalent to :c:macro:`Py_BEGIN_ALLOW_THREADS` except it uses
        :c:data:`NPY_ALLOW_THREADS` to determine if the macro if
        replaced with white-space or not.

    .. c:macro:: NPY_END_ALLOW_THREADS

        Equivalent to :c:macro:`Py_END_ALLOW_THREADS` except it uses
        :c:data:`NPY_ALLOW_THREADS` to determine if the macro if
        replaced with white-space or not.

    .. c:macro:: NPY_BEGIN_THREADS_DEF

        Place in the variable declaration area. This macro sets up the
        variable needed for storing the Python state.

    .. c:macro:: NPY_BEGIN_THREADS

        Place right before code that does not need the Python
        interpreter (no Python C-API calls). This macro saves the
        Python state and releases the GIL.

    .. c:macro:: NPY_END_THREADS

        Place right after code that does not need the Python
        interpreter. This macro acquires the GIL and restores the
        Python state from the saved variable.

    .. c:function:: NPY_BEGIN_THREADS_DESCR(PyArray_Descr *dtype)

        Useful to release the GIL only if *dtype* does not contain
        arbitrary Python objects which may need the Python interpreter
        during execution of the loop. Equivalent to

    .. c:function:: NPY_END_THREADS_DESCR(PyArray_Descr *dtype)

        Useful to regain the GIL in situations where it was released
        using the BEGIN form of this macro.

    .. c:function:: NPY_BEGIN_THREADS_THRESHOLDED(int loop_size)

        Useful to release the GIL only if *loop_size* exceeds a
        minimum threshold, currently set to 500. Should be matched
        with a :c:macro:`NPY_END_THREADS` to regain the GIL.

Group 2
"""""""

    This group is used to re-acquire the Python GIL after it has been
    released. For example, suppose the GIL has been released (using the
    previous calls), and then some path in the code (perhaps in a
    different subroutine) requires use of the Python C-API, then these
    macros are useful to acquire the GIL. These macros accomplish
    essentially a reverse of the previous three (acquire the LOCK saving
    what state it had) and then re-release it with the saved state.

    .. c:macro:: NPY_ALLOW_C_API_DEF

        Place in the variable declaration area to set up the necessary
        variable.

    .. c:macro:: NPY_ALLOW_C_API

        Place before code that needs to call the Python C-API (when it is
        known that the GIL has already been released).

    .. c:macro:: NPY_DISABLE_C_API

        Place after code that needs to call the Python C-API (to re-release
        the GIL).

.. tip::

    Never use semicolons after the threading support macros.


Priority
^^^^^^^^

.. c:var:: NPY_PRIORITY

    Default priority for arrays.

.. c:var:: NPY_SUBTYPE_PRIORITY

    Default subtype priority.

.. c:var:: NPY_SCALAR_PRIORITY

    Default scalar priority (very small)

.. c:function:: double PyArray_GetPriority(PyObject* obj, double def)

    Return the :obj:`~numpy.class.__array_priority__` attribute (converted to a
    double) of *obj* or *def* if no attribute of that name
    exists. Fast returns that avoid the attribute lookup are provided
    for objects of type :c:data:`PyArray_Type`.


Default buffers
^^^^^^^^^^^^^^^

.. c:var:: NPY_BUFSIZE

    Default size of the user-settable internal buffers.

.. c:var:: NPY_MIN_BUFSIZE

    Smallest size of user-settable internal buffers.

.. c:var:: NPY_MAX_BUFSIZE

    Largest size allowed for the user-settable buffers.


Other constants
^^^^^^^^^^^^^^^

.. c:var:: NPY_NUM_FLOATTYPE

    The number of floating-point types

.. c:var:: NPY_MAXDIMS

    The maximum number of dimensions allowed in arrays.

.. c:var:: NPY_VERSION

    The current version of the ndarray object (check to see if this
    variable is defined to guarantee the numpy/arrayobject.h header is
    being used).

.. c:var:: NPY_FALSE

    Defined as 0 for use with Bool.

.. c:var:: NPY_TRUE

    Defined as 1 for use with Bool.

.. c:var:: NPY_FAIL

    The return value of failed converter functions which are called using
    the "O&" syntax in :c:func:`PyArg_ParseTuple`-like functions.

.. c:var:: NPY_SUCCEED

    The return value of successful converter functions which are called
    using the "O&" syntax in :c:func:`PyArg_ParseTuple`-like functions.


Miscellaneous Macros
^^^^^^^^^^^^^^^^^^^^

.. c:function:: PyArray_SAMESHAPE(PyArrayObject *a1, PyArrayObject *a2)

    Evaluates as True if arrays *a1* and *a2* have the same shape.

.. c:macro:: PyArray_MAX(a,b)

    Returns the maximum of *a* and *b*. If (*a*) or (*b*) are
    expressions they are evaluated twice.

.. c:macro:: PyArray_MIN(a,b)

    Returns the minimum of *a* and *b*. If (*a*) or (*b*) are
    expressions they are evaluated twice.

.. c:macro:: PyArray_CLT(a,b)

.. c:macro:: PyArray_CGT(a,b)

.. c:macro:: PyArray_CLE(a,b)

.. c:macro:: PyArray_CGE(a,b)

.. c:macro:: PyArray_CEQ(a,b)

.. c:macro:: PyArray_CNE(a,b)

    Implements the complex comparisons between two complex numbers
    (structures with a real and imag member) using NumPy's definition
    of the ordering which is lexicographic: comparing the real parts
    first and then the complex parts if the real parts are equal.

.. c:function:: PyArray_REFCOUNT(PyObject* op)

    Returns the reference count of any Python object.

.. c:function:: PyArray_DiscardWritebackIfCopy(PyObject* obj)

    If ``obj.flags`` has :c:data:`NPY_ARRAY_WRITEBACKIFCOPY` or (deprecated)
    :c:data:`NPY_ARRAY_UPDATEIFCOPY`, this function clears the flags, `DECREF` s
    `obj->base` and makes it writeable, and sets ``obj->base`` to NULL. In
    contrast to :c:func:`PyArray_DiscardWritebackIfCopy` it makes no attempt
    to copy the data from `obj->base` This undoes
    :c:func:`PyArray_SetWritebackIfCopyBase`. Usually this is called after an
    error when you are finished with ``obj``, just before ``Py_DECREF(obj)``.
    It may be called multiple times, or with ``NULL`` input.

.. c:function:: PyArray_XDECREF_ERR(PyObject* obj)

    Deprecated in 1.14, use :c:func:`PyArray_DiscardWritebackIfCopy`
    followed by ``Py_XDECREF``

    DECREF's an array object which may have the (deprecated)
    :c:data:`NPY_ARRAY_UPDATEIFCOPY` or :c:data:`NPY_ARRAY_WRITEBACKIFCOPY`
    flag set without causing the contents to be copied back into the
    original array. Resets the :c:data:`NPY_ARRAY_WRITEABLE` flag on the base
    object. This is useful for recovering from an error condition when
    writeback semantics are used, but will lead to wrong results.


Enumerated Types
^^^^^^^^^^^^^^^^

.. c:type:: NPY_SORTKIND

    A special variable-type which can take on the values :c:data:`NPY_{KIND}`
    where ``{KIND}`` is

        **QUICKSORT**, **HEAPSORT**, **MERGESORT**, **STABLESORT**

    .. c:var:: NPY_NSORTS

       Defined to be the number of sorts. It is fixed at three by the need for
       backwards compatibility, and consequently :c:data:`NPY_MERGESORT` and
       :c:data:`NPY_STABLESORT` are aliased to each other and may refer to one
       of several stable sorting algorithms depending on the data type.


.. c:type:: NPY_SCALARKIND

    A special variable type indicating the number of "kinds" of
    scalars distinguished in determining scalar-coercion rules. This
    variable can take on the values :c:data:`NPY_{KIND}` where ``{KIND}`` can be

        **NOSCALAR**, **BOOL_SCALAR**, **INTPOS_SCALAR**,
        **INTNEG_SCALAR**, **FLOAT_SCALAR**, **COMPLEX_SCALAR**,
        **OBJECT_SCALAR**

    .. c:var:: NPY_NSCALARKINDS

       Defined to be the number of scalar kinds
       (not including :c:data:`NPY_NOSCALAR`).

.. c:type:: NPY_ORDER

    An enumeration type indicating the element order that an array should be
    interpreted in. When a brand new array is created, generally
    only **NPY_CORDER** and **NPY_FORTRANORDER** are used, whereas
    when one or more inputs are provided, the order can be based on them.

    .. c:var:: NPY_ANYORDER

        Fortran order if all the inputs are Fortran, C otherwise.

    .. c:var:: NPY_CORDER

        C order.

    .. c:var:: NPY_FORTRANORDER

        Fortran order.

    .. c:var:: NPY_KEEPORDER

        An order as close to the order of the inputs as possible, even
        if the input is in neither C nor Fortran order.

.. c:type:: NPY_CLIPMODE

    A variable type indicating the kind of clipping that should be
    applied in certain functions.

    .. c:var:: NPY_RAISE

        The default for most operations, raises an exception if an index
        is out of bounds.

    .. c:var:: NPY_CLIP

        Clips an index to the valid range if it is out of bounds.

    .. c:var:: NPY_WRAP

        Wraps an index to the valid range if it is out of bounds.

.. c:type:: NPY_CASTING

    .. versionadded:: 1.6

    An enumeration type indicating how permissive data conversions should
    be. This is used by the iterator added in NumPy 1.6, and is intended
    to be used more broadly in a future version.

    .. c:var:: NPY_NO_CASTING

        Only allow identical types.

    .. c:var:: NPY_EQUIV_CASTING

       Allow identical and casts involving byte swapping.

    .. c:var:: NPY_SAFE_CASTING

       Only allow casts which will not cause values to be rounded,
       truncated, or otherwise changed.

    .. c:var:: NPY_SAME_KIND_CASTING

       Allow any safe casts, and casts between types of the same kind.
       For example, float64 -> float32 is permitted with this rule.

    .. c:var:: NPY_UNSAFE_CASTING

       Allow any cast, no matter what kind of data loss may occur.

.. index::
   pair: ndarray; C-API
