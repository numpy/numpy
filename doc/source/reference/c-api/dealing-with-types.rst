Dealing with types
==================


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
