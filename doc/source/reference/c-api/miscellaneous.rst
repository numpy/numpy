Miscellaneous
=============

General check of Python Type
----------------------------

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



Importing the API
-----------------

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
------------------------

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
--------------------

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
-----------------

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
-----------------

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
^^^^^^^

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
        during execution of the loop.

    .. c:function:: NPY_END_THREADS_DESCR(PyArray_Descr *dtype)

        Useful to regain the GIL in situations where it was released
        using the BEGIN form of this macro.

    .. c:function:: NPY_BEGIN_THREADS_THRESHOLDED(int loop_size)

        Useful to release the GIL only if *loop_size* exceeds a
        minimum threshold, currently set to 500. Should be matched
        with a :c:macro:`NPY_END_THREADS` to regain the GIL.

Group 2
^^^^^^^

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
--------

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
---------------

.. c:var:: NPY_BUFSIZE

    Default size of the user-settable internal buffers.

.. c:var:: NPY_MIN_BUFSIZE

    Smallest size of user-settable internal buffers.

.. c:var:: NPY_MAX_BUFSIZE

    Largest size allowed for the user-settable buffers.


Other constants
---------------

.. c:var:: NPY_NUM_FLOATTYPE

    The number of floating-point types

.. c:var:: NPY_MAXDIMS

    The maximum number of dimensions allowed in arrays.

.. c:var:: NPY_MAXARGS

    The maximum number of array arguments that can be used in functions.

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
--------------------

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
----------------

.. c:type:: NPY_SORTKIND

    A special variable-type which can take on different values to indicate
    the sorting algorithm being used.

    .. c:var:: NPY_QUICKSORT

    .. c:var:: NPY_HEAPSORT

    .. c:var:: NPY_MERGESORT

    .. c:var:: NPY_STABLESORT

        Used as an alias of :c:data:`NPY_MERGESORT` and vica versa.

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
