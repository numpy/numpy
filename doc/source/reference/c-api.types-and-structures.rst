*****************************
Python Types and C-Structures
*****************************

.. sectionauthor:: Travis E. Oliphant

Several new types are defined in the C-code. Most of these are
accessible from Python, but a few are not exposed due to their limited
use. Every new Python type has an associated :ctype:`PyObject *` with an
internal structure that includes a pointer to a "method table" that
defines how the new object behaves in Python. When you receive a
Python object into C code, you always get a pointer to a
:ctype:`PyObject` structure. Because a :ctype:`PyObject` structure is
very generic and defines only :cmacro:`PyObject_HEAD`, by itself it
is not very interesting. However, different objects contain more
details after the :cmacro:`PyObject_HEAD` (but you have to cast to the
correct type to access them --- or use accessor functions or macros).


New Python Types Defined
========================

Python types are the functional equivalent in C of classes in Python.
By constructing a new Python type you make available a new object for
Python. The ndarray object is an example of a new type defined in C.
New types are defined in C by two basic steps:

1. creating a C-structure (usually named :ctype:`Py{Name}Object`) that is
   binary- compatible with the :ctype:`PyObject` structure itself but holds
   the additional information needed for that particular object;

2. populating the :ctype:`PyTypeObject` table (pointed to by the ob_type
   member of the :ctype:`PyObject` structure) with pointers to functions
   that implement the desired behavior for the type.

Instead of special method names which define behavior for Python
classes, there are "function tables" which point to functions that
implement the desired results. Since Python 2.2, the PyTypeObject
itself has become dynamic which allows C types that can be "sub-typed
"from other C-types in C, and sub-classed in Python. The children
types inherit the attributes and methods from their parent(s).

There are two major new types: the ndarray ( :cdata:`PyArray_Type` )
and the ufunc ( :cdata:`PyUFunc_Type` ). Additional types play a
supportive role: the :cdata:`PyArrayIter_Type`, the
:cdata:`PyArrayMultiIter_Type`, and the :cdata:`PyArrayDescr_Type`
. The :cdata:`PyArrayIter_Type` is the type for a flat iterator for an
ndarray (the object that is returned when getting the flat
attribute). The :cdata:`PyArrayMultiIter_Type` is the type of the
object returned when calling ``broadcast`` (). It handles iteration
and broadcasting over a collection of nested sequences. Also, the
:cdata:`PyArrayDescr_Type` is the data-type-descriptor type whose
instances describe the data.  Finally, there are 21 new scalar-array
types which are new Python scalars corresponding to each of the
fundamental data types available for arrays. An additional 10 other
types are place holders that allow the array scalars to fit into a
hierarchy of actual Python types.


PyArray_Type
------------

.. cvar:: PyArray_Type

   The Python type of the ndarray is :cdata:`PyArray_Type`. In C, every
   ndarray is a pointer to a :ctype:`PyArrayObject` structure. The ob_type
   member of this structure contains a pointer to the :cdata:`PyArray_Type`
   typeobject.

.. ctype:: PyArrayObject

   The :ctype:`PyArrayObject` C-structure contains all of the required
   information for an array. All instances of an ndarray (and its
   subclasses) will have this structure.  For future compatibility,
   these structure members should normally be accessed using the
   provided macros. If you need a shorter name, then you can make use
   of :ctype:`NPY_AO` which is defined to be equivalent to
   :ctype:`PyArrayObject`.

   .. code-block:: c

      typedef struct PyArrayObject {
          PyObject_HEAD
          char *data;
          int nd;
          npy_intp *dimensions;
          npy_intp *strides;
          PyObject *base;
          PyArray_Descr *descr;
          int flags;
          PyObject *weakreflist;
      } PyArrayObject;

.. cmacro:: PyArrayObject.PyObject_HEAD

    This is needed by all Python objects. It consists of (at least)
    a reference count member ( ``ob_refcnt`` ) and a pointer to the
    typeobject ( ``ob_type`` ). (Other elements may also be present
    if Python was compiled with special options see
    Include/object.h in the Python source tree for more
    information). The ob_type member points to a Python type
    object.

.. cmember:: char *PyArrayObject.data

    A pointer to the first element of the array. This pointer can
    (and normally should) be recast to the data type of the array.

.. cmember:: int PyArrayObject.nd

    An integer providing the number of dimensions for this
    array. When nd is 0, the array is sometimes called a rank-0
    array. Such arrays have undefined dimensions and strides and
    cannot be accessed. :cdata:`NPY_MAXDIMS` is the largest number of
    dimensions for any array.

.. cmember:: npy_intp PyArrayObject.dimensions

    An array of integers providing the shape in each dimension as
    long as nd :math:`\geq` 1. The integer is always large enough
    to hold a pointer on the platform, so the dimension size is
    only limited by memory.

.. cmember:: npy_intp *PyArrayObject.strides

    An array of integers providing for each dimension the number of
    bytes that must be skipped to get to the next element in that
    dimension.

.. cmember:: PyObject *PyArrayObject.base

    This member is used to hold a pointer to another Python object that
    is related to this array. There are two use cases: 1) If this array
    does not own its own memory, then base points to the Python object
    that owns it (perhaps another array object), 2) If this array has
    the :cdata:`NPY_ARRAY_UPDATEIFCOPY` flag set, then this array is
    a working copy of a "misbehaved" array. As soon as this array is
    deleted, the array pointed to by base will be updated with the
    contents of this array.

.. cmember:: PyArray_Descr *PyArrayObject.descr

    A pointer to a data-type descriptor object (see below). The
    data-type descriptor object is an instance of a new built-in
    type which allows a generic description of memory. There is a
    descriptor structure for each data type supported. This
    descriptor structure contains useful information about the type
    as well as a pointer to a table of function pointers to
    implement specific functionality.

.. cmember:: int PyArrayObject.flags

    Flags indicating how the memory pointed to by data is to be
    interpreted. Possible flags are :cdata:`NPY_ARRAY_C_CONTIGUOUS`,
    :cdata:`NPY_ARRAY_F_CONTIGUOUS`, :cdata:`NPY_ARRAY_OWNDATA`,
    :cdata:`NPY_ARRAY_ALIGNED`, :cdata:`NPY_ARRAY_WRITEABLE`, and
    :cdata:`NPY_ARRAY_UPDATEIFCOPY`.

.. cmember:: PyObject *PyArrayObject.weakreflist

    This member allows array objects to have weak references (using the
    weakref module).


PyArrayDescr_Type
-----------------

.. cvar:: PyArrayDescr_Type

   The :cdata:`PyArrayDescr_Type` is the built-in type of the
   data-type-descriptor objects used to describe how the bytes comprising
   the array are to be interpreted.  There are 21 statically-defined
   :ctype:`PyArray_Descr` objects for the built-in data-types. While these
   participate in reference counting, their reference count should never
   reach zero.  There is also a dynamic table of user-defined
   :ctype:`PyArray_Descr` objects that is also maintained. Once a
   data-type-descriptor object is "registered" it should never be
   deallocated either. The function :cfunc:`PyArray_DescrFromType` (...) can
   be used to retrieve a :ctype:`PyArray_Descr` object from an enumerated
   type-number (either built-in or user- defined).

.. ctype:: PyArray_Descr

   The format of the :ctype:`PyArray_Descr` structure that lies at the
   heart of the :cdata:`PyArrayDescr_Type` is

   .. code-block:: c

      typedef struct {
          PyObject_HEAD
          PyTypeObject *typeobj;
          char kind;
          char type;
          char byteorder;
          char unused;
          int flags;
          int type_num;
          int elsize;
          int alignment;
          PyArray_ArrayDescr *subarray;
          PyObject *fields;
          PyArray_ArrFuncs *f;
      } PyArray_Descr;

.. cmember:: PyTypeObject *PyArray_Descr.typeobj

    Pointer to a typeobject that is the corresponding Python type for
    the elements of this array. For the builtin types, this points to
    the corresponding array scalar. For user-defined types, this
    should point to a user-defined typeobject. This typeobject can
    either inherit from array scalars or not. If it does not inherit
    from array scalars, then the :cdata:`NPY_USE_GETITEM` and
    :cdata:`NPY_USE_SETITEM` flags should be set in the ``flags`` member.

.. cmember:: char PyArray_Descr.kind

    A character code indicating the kind of array (using the array
    interface typestring notation). A 'b' represents Boolean, a 'i'
    represents signed integer, a 'u' represents unsigned integer, 'f'
    represents floating point, 'c' represents complex floating point, 'S'
    represents 8-bit character string, 'U' represents 32-bit/character
    unicode string, and 'V' repesents arbitrary.

.. cmember:: char PyArray_Descr.type

    A traditional character code indicating the data type.

.. cmember:: char PyArray_Descr.byteorder

    A character indicating the byte-order: '>' (big-endian), '<' (little-
    endian), '=' (native), '\|' (irrelevant, ignore). All builtin data-
    types have byteorder '='.

.. cmember:: int PyArray_Descr.flags

    A data-type bit-flag that determines if the data-type exhibits object-
    array like behavior. Each bit in this member is a flag which are named
    as:

    .. cvar:: NPY_ITEM_REFCOUNT

    .. cvar:: NPY_ITEM_HASOBJECT

        Indicates that items of this data-type must be reference
        counted (using :cfunc:`Py_INCREF` and :cfunc:`Py_DECREF` ).

    .. cvar:: NPY_ITEM_LISTPICKLE

        Indicates arrays of this data-type must be converted to a list
        before pickling.

    .. cvar:: NPY_ITEM_IS_POINTER

        Indicates the item is a pointer to some other data-type

    .. cvar:: NPY_NEEDS_INIT

        Indicates memory for this data-type must be initialized (set
        to 0) on creation.

    .. cvar:: NPY_NEEDS_PYAPI

        Indicates this data-type requires the Python C-API during
        access (so don't give up the GIL if array access is going to
        be needed).

    .. cvar:: NPY_USE_GETITEM

        On array access use the ``f->getitem`` function pointer
        instead of the standard conversion to an array scalar. Must
        use if you don't define an array scalar to go along with
        the data-type.

    .. cvar:: NPY_USE_SETITEM

        When creating a 0-d array from an array scalar use
        ``f->setitem`` instead of the standard copy from an array
        scalar. Must use if you don't define an array scalar to go
        along with the data-type.

    .. cvar:: NPY_FROM_FIELDS

        The bits that are inherited for the parent data-type if these
        bits are set in any field of the data-type. Currently (
        :cdata:`NPY_NEEDS_INIT` \| :cdata:`NPY_LIST_PICKLE` \|
        :cdata:`NPY_ITEM_REFCOUNT` \| :cdata:`NPY_NEEDS_PYAPI` ).

    .. cvar:: NPY_OBJECT_DTYPE_FLAGS

        Bits set for the object data-type: ( :cdata:`NPY_LIST_PICKLE`
        \| :cdata:`NPY_USE_GETITEM` \| :cdata:`NPY_ITEM_IS_POINTER` \|
        :cdata:`NPY_REFCOUNT` \| :cdata:`NPY_NEEDS_INIT` \|
        :cdata:`NPY_NEEDS_PYAPI`).

    .. cfunction:: PyDataType_FLAGCHK(PyArray_Descr *dtype, int flags)

        Return true if all the given flags are set for the data-type
        object.

    .. cfunction:: PyDataType_REFCHK(PyArray_Descr *dtype)

        Equivalent to :cfunc:`PyDataType_FLAGCHK` (*dtype*,
 	:cdata:`NPY_ITEM_REFCOUNT`).

.. cmember:: int PyArray_Descr.type_num

    A number that uniquely identifies the data type. For new data-types,
    this number is assigned when the data-type is registered.

.. cmember:: int PyArray_Descr.elsize

    For data types that are always the same size (such as long), this
    holds the size of the data type. For flexible data types where
    different arrays can have a different elementsize, this should be
    0.

.. cmember:: int PyArray_Descr.alignment

    A number providing alignment information for this data type.
    Specifically, it shows how far from the start of a 2-element
    structure (whose first element is a ``char`` ), the compiler
    places an item of this type: ``offsetof(struct {char c; type v;},
    v)``

.. cmember:: PyArray_ArrayDescr *PyArray_Descr.subarray

    If this is non- ``NULL``, then this data-type descriptor is a
    C-style contiguous array of another data-type descriptor. In
    other-words, each element that this descriptor describes is
    actually an array of some other base descriptor. This is most
    useful as the data-type descriptor for a field in another
    data-type descriptor. The fields member should be ``NULL`` if this
    is non- ``NULL`` (the fields member of the base descriptor can be
    non- ``NULL`` however). The :ctype:`PyArray_ArrayDescr` structure is
    defined using

    .. code-block:: c

       typedef struct {
           PyArray_Descr *base;
           PyObject *shape;
       } PyArray_ArrayDescr;

    The elements of this structure are:

    .. cmember:: PyArray_Descr *PyArray_ArrayDescr.base

        The data-type-descriptor object of the base-type.

    .. cmember:: PyObject *PyArray_ArrayDescr.shape

        The shape (always C-style contiguous) of the sub-array as a Python
        tuple.


.. cmember:: PyObject *PyArray_Descr.fields

    If this is non-NULL, then this data-type-descriptor has fields
    described by a Python dictionary whose keys are names (and also
    titles if given) and whose values are tuples that describe the
    fields. Recall that a data-type-descriptor always describes a
    fixed-length set of bytes. A field is a named sub-region of that
    total, fixed-length collection. A field is described by a tuple
    composed of another data- type-descriptor and a byte
    offset. Optionally, the tuple may contain a title which is
    normally a Python string. These tuples are placed in this
    dictionary keyed by name (and also title if given).

.. cmember:: PyArray_ArrFuncs *PyArray_Descr.f

    A pointer to a structure containing functions that the type needs
    to implement internal features. These functions are not the same
    thing as the universal functions (ufuncs) described later. Their
    signatures can vary arbitrarily.

.. ctype:: PyArray_ArrFuncs

    Functions implementing internal features. Not all of these
    function pointers must be defined for a given type. The required
    members are ``nonzero``, ``copyswap``, ``copyswapn``, ``setitem``,
    ``getitem``, and ``cast``. These are assumed to be non- ``NULL``
    and ``NULL`` entries will cause a program crash. The other
    functions may be ``NULL`` which will just mean reduced
    functionality for that data-type. (Also, the nonzero function will
    be filled in with a default function if it is ``NULL`` when you
    register a user-defined data-type).

    .. code-block:: c

       typedef struct {
           PyArray_VectorUnaryFunc *cast[NPY_NTYPES];
           PyArray_GetItemFunc *getitem;
           PyArray_SetItemFunc *setitem;
           PyArray_CopySwapNFunc *copyswapn;
           PyArray_CopySwapFunc *copyswap;
           PyArray_CompareFunc *compare;
           PyArray_ArgFunc *argmax;
           PyArray_DotFunc *dotfunc;
           PyArray_ScanFunc *scanfunc;
           PyArray_FromStrFunc *fromstr;
           PyArray_NonzeroFunc *nonzero;
           PyArray_FillFunc *fill;
           PyArray_FillWithScalarFunc *fillwithscalar;
           PyArray_SortFunc *sort[NPY_NSORTS];
           PyArray_ArgSortFunc *argsort[NPY_NSORTS];
           PyObject *castdict;
           PyArray_ScalarKindFunc *scalarkind;
           int **cancastscalarkindto;
           int *cancastto;
           int listpickle
       } PyArray_ArrFuncs;

    The concept of a behaved segment is used in the description of the
    function pointers. A behaved segment is one that is aligned and in
    native machine byte-order for the data-type. The ``nonzero``,
    ``copyswap``, ``copyswapn``, ``getitem``, and ``setitem``
    functions can (and must) deal with mis-behaved arrays. The other
    functions require behaved memory segments.

    .. cmember:: void cast(void *from, void *to, npy_intp n, void *fromarr,
       void *toarr)

        An array of function pointers to cast from the current type to
        all of the other builtin types. Each function casts a
        contiguous, aligned, and notswapped buffer pointed at by
        *from* to a contiguous, aligned, and notswapped buffer pointed
        at by *to* The number of items to cast is given by *n*, and
        the arguments *fromarr* and *toarr* are interpreted as
        PyArrayObjects for flexible arrays to get itemsize
        information.

    .. cmember:: PyObject *getitem(void *data, void *arr)

        A pointer to a function that returns a standard Python object
        from a single element of the array object *arr* pointed to by
        *data*. This function must be able to deal with "misbehaved
        "(misaligned and/or swapped) arrays correctly.

    .. cmember:: int setitem(PyObject *item, void *data, void *arr)

        A pointer to a function that sets the Python object *item*
        into the array, *arr*, at the position pointed to by *data*
        . This function deals with "misbehaved" arrays. If successful,
        a zero is returned, otherwise, a negative one is returned (and
        a Python error set).

    .. cmember:: void copyswapn(void *dest, npy_intp dstride, void *src,
       npy_intp sstride, npy_intp n, int swap, void *arr)

    .. cmember:: void copyswap(void *dest, void *src, int swap, void *arr)

        These members are both pointers to functions to copy data from
        *src* to *dest* and *swap* if indicated. The value of arr is
        only used for flexible ( :cdata:`NPY_STRING`, :cdata:`NPY_UNICODE`,
        and :cdata:`NPY_VOID` ) arrays (and is obtained from
        ``arr->descr->elsize`` ). The second function copies a single
        value, while the first loops over n values with the provided
        strides. These functions can deal with misbehaved *src*
        data. If *src* is NULL then no copy is performed. If *swap* is
        0, then no byteswapping occurs. It is assumed that *dest* and
        *src* do not overlap. If they overlap, then use ``memmove``
        (...) first followed by ``copyswap(n)`` with NULL valued
        ``src``.

    .. cmember:: int compare(const void* d1, const void* d2, void* arr)

        A pointer to a function that compares two elements of the
        array, ``arr``, pointed to by ``d1`` and ``d2``. This
        function requires behaved arrays. The return value is 1 if *
        ``d1`` > * ``d2``, 0 if * ``d1`` == * ``d2``, and -1 if *
        ``d1`` < * ``d2``. The array object arr is used to retrieve
        itemsize and field information for flexible arrays.

    .. cmember:: int argmax(void* data, npy_intp n, npy_intp* max_ind,
       void* arr)

        A pointer to a function that retrieves the index of the
        largest of ``n`` elements in ``arr`` beginning at the element
        pointed to by ``data``. This function requires that the
        memory segment be contiguous and behaved. The return value is
        always 0. The index of the largest element is returned in
        ``max_ind``.

    .. cmember:: void dotfunc(void* ip1, npy_intp is1, void* ip2, npy_intp is2,
       void* op, npy_intp n, void* arr)

        A pointer to a function that multiplies two ``n`` -length
        sequences together, adds them, and places the result in
        element pointed to by ``op`` of ``arr``. The start of the two
        sequences are pointed to by ``ip1`` and ``ip2``. To get to
        the next element in each sequence requires a jump of ``is1``
        and ``is2`` *bytes*, respectively. This function requires
        behaved (though not necessarily contiguous) memory.

    .. cmember:: int scanfunc(FILE* fd, void* ip , void* sep , void* arr)

        A pointer to a function that scans (scanf style) one element
        of the corresponding type from the file descriptor ``fd`` into
        the array memory pointed to by ``ip``. The array is assumed
        to be behaved. If ``sep`` is not NULL, then a separator string
        is also scanned from the file before returning. The last
        argument ``arr`` is the array to be scanned into. A 0 is
        returned if the scan is successful. A negative number
        indicates something went wrong: -1 means the end of file was
        reached before the separator string could be scanned, -4 means
        that the end of file was reached before the element could be
        scanned, and -3 means that the element could not be
        interpreted from the format string. Requires a behaved array.

    .. cmember:: int fromstr(char* str, void* ip, char** endptr, void* arr)

        A pointer to a function that converts the string pointed to by
        ``str`` to one element of the corresponding type and places it
        in the memory location pointed to by ``ip``. After the
        conversion is completed, ``*endptr`` points to the rest of the
        string. The last argument ``arr`` is the array into which ip
        points (needed for variable-size data- types). Returns 0 on
        success or -1 on failure. Requires a behaved array.

    .. cmember:: Bool nonzero(void* data, void* arr)

        A pointer to a function that returns TRUE if the item of
        ``arr`` pointed to by ``data`` is nonzero. This function can
        deal with misbehaved arrays.

    .. cmember:: void fill(void* data, npy_intp length, void* arr)

        A pointer to a function that fills a contiguous array of given
        length with data. The first two elements of the array must
        already be filled- in. From these two values, a delta will be
        computed and the values from item 3 to the end will be
        computed by repeatedly adding this computed delta. The data
        buffer must be well-behaved.

    .. cmember:: void fillwithscalar(void* buffer, npy_intp length,
       void* value, void* arr)

        A pointer to a function that fills a contiguous ``buffer`` of
        the given ``length`` with a single scalar ``value`` whose
        address is given. The final argument is the array which is
        needed to get the itemsize for variable-length arrays.

    .. cmember:: int sort(void* start, npy_intp length, void* arr)

        An array of function pointers to a particular sorting
        algorithms. A particular sorting algorithm is obtained using a
        key (so far :cdata:`NPY_QUICKSORT`, :data`NPY_HEAPSORT`, and
        :cdata:`NPY_MERGESORT` are defined). These sorts are done
        in-place assuming contiguous and aligned data.

    .. cmember:: int argsort(void* start, npy_intp* result, npy_intp length,
       void \*arr)

        An array of function pointers to sorting algorithms for this
        data type. The same sorting algorithms as for sort are
        available. The indices producing the sort are returned in
        result (which must be initialized with indices 0 to length-1
        inclusive).

    .. cmember:: PyObject *castdict

        Either ``NULL`` or a dictionary containing low-level casting
        functions for user- defined data-types. Each function is
        wrapped in a :ctype:`PyCObject *` and keyed by the data-type number.

    .. cmember:: NPY_SCALARKIND scalarkind(PyArrayObject* arr)

        A function to determine how scalars of this type should be
        interpreted. The argument is ``NULL`` or a 0-dimensional array
        containing the data (if that is needed to determine the kind
        of scalar). The return value must be of type
        :ctype:`NPY_SCALARKIND`.

    .. cmember:: int **cancastscalarkindto

        Either ``NULL`` or an array of :ctype:`NPY_NSCALARKINDS`
        pointers. These pointers should each be either ``NULL`` or a
        pointer to an array of integers (terminated by
        :cdata:`NPY_NOTYPE`) indicating data-types that a scalar of
        this data-type of the specified kind can be cast to safely
        (this usually means without losing precision).

    .. cmember:: int *cancastto

        Either ``NULL`` or an array of integers (terminated by
        :cdata:`NPY_NOTYPE` ) indicated data-types that this data-type
        can be cast to safely (this usually means without losing
        precision).

    .. cmember:: int listpickle

        Unused.

The :cdata:`PyArray_Type` typeobject implements many of the features of
Python objects including the tp_as_number, tp_as_sequence,
tp_as_mapping, and tp_as_buffer interfaces. The rich comparison
(tp_richcompare) is also used along with new-style attribute lookup
for methods (tp_methods) and properties (tp_getset). The
:cdata:`PyArray_Type` can also be sub-typed.

.. tip::

    The tp_as_number methods use a generic approach to call whatever
    function has been registered for handling the operation. The
    function PyNumeric_SetOps(..) can be used to register functions to
    handle particular mathematical operations (for all arrays). When
    the umath module is imported, it sets the numeric operations for
    all arrays to the corresponding ufuncs.  The tp_str and tp_repr
    methods can also be altered using PyString_SetStringFunction(...).


PyUFunc_Type
------------

.. cvar:: PyUFunc_Type

   The ufunc object is implemented by creation of the
   :cdata:`PyUFunc_Type`. It is a very simple type that implements only
   basic getattribute behavior, printing behavior, and has call
   behavior which allows these objects to act like functions. The
   basic idea behind the ufunc is to hold a reference to fast
   1-dimensional (vector) loops for each data type that supports the
   operation. These one-dimensional loops all have the same signature
   and are the key to creating a new ufunc. They are called by the
   generic looping code as appropriate to implement the N-dimensional
   function. There are also some generic 1-d loops defined for
   floating and complexfloating arrays that allow you to define a
   ufunc using a single scalar function (*e.g.* atanh).


.. ctype:: PyUFuncObject

   The core of the ufunc is the :ctype:`PyUFuncObject` which contains all
   the information needed to call the underlying C-code loops that
   perform the actual work. It has the following structure:

   .. code-block:: c

      typedef struct {
          PyObject_HEAD
          int nin;
          int nout;
          int nargs;
          int identity;
          PyUFuncGenericFunction *functions;
          void **data;
          int ntypes;
          int check_return;
          char *name;
          char *types;
          char *doc;
          void *ptr;
          PyObject *obj;
          PyObject *userloops;
          npy_uint32 *op_flags;
          npy_uint32 *iter_flags;
      } PyUFuncObject;

   .. cmacro:: PyUFuncObject.PyObject_HEAD

       required for all Python objects.

   .. cmember:: int PyUFuncObject.nin

       The number of input arguments.

   .. cmember:: int PyUFuncObject.nout

       The number of output arguments.

   .. cmember:: int PyUFuncObject.nargs

       The total number of arguments (*nin* + *nout*). This must be
       less than :cdata:`NPY_MAXARGS`.

   .. cmember:: int PyUFuncObject.identity

       Either :cdata:`PyUFunc_One`, :cdata:`PyUFunc_Zero`, or
       :cdata:`PyUFunc_None` to indicate the identity for this operation.
       It is only used for a reduce-like call on an empty array.

   .. cmember:: void PyUFuncObject.functions(char** args, npy_intp* dims,
      npy_intp* steps, void* extradata)

       An array of function pointers --- one for each data type
       supported by the ufunc. This is the vector loop that is called
       to implement the underlying function *dims* [0] times. The
       first argument, *args*, is an array of *nargs* pointers to
       behaved memory. Pointers to the data for the input arguments
       are first, followed by the pointers to the data for the output
       arguments. How many bytes must be skipped to get to the next
       element in the sequence is specified by the corresponding entry
       in the *steps* array. The last argument allows the loop to
       receive extra information.  This is commonly used so that a
       single, generic vector loop can be used for multiple
       functions. In this case, the actual scalar function to call is
       passed in as *extradata*. The size of this function pointer
       array is ntypes.

   .. cmember:: void **PyUFuncObject.data

       Extra data to be passed to the 1-d vector loops or ``NULL`` if
       no extra-data is needed. This C-array must be the same size (
       *i.e.* ntypes) as the functions array. ``NULL`` is used if
       extra_data is not needed. Several C-API calls for UFuncs are
       just 1-d vector loops that make use of this extra data to
       receive a pointer to the actual function to call.

   .. cmember:: int PyUFuncObject.ntypes

       The number of supported data types for the ufunc. This number
       specifies how many different 1-d loops (of the builtin data types) are
       available.

   .. cmember:: int PyUFuncObject.check_return

       Obsolete and unused. However, it is set by the corresponding entry in
       the main ufunc creation routine: :cfunc:`PyUFunc_FromFuncAndData` (...).

   .. cmember:: char *PyUFuncObject.name

       A string name for the ufunc. This is used dynamically to build
       the __doc\__ attribute of ufuncs.

   .. cmember:: char *PyUFuncObject.types

       An array of *nargs* :math:`\times` *ntypes* 8-bit type_numbers
       which contains the type signature for the function for each of
       the supported (builtin) data types. For each of the *ntypes*
       functions, the corresponding set of type numbers in this array
       shows how the *args* argument should be interpreted in the 1-d
       vector loop. These type numbers do not have to be the same type
       and mixed-type ufuncs are supported.

   .. cmember:: char *PyUFuncObject.doc

       Documentation for the ufunc. Should not contain the function
       signature as this is generated dynamically when __doc\__ is
       retrieved.

   .. cmember:: void *PyUFuncObject.ptr

       Any dynamically allocated memory. Currently, this is used for dynamic
       ufuncs created from a python function to store room for the types,
       data, and name members.

   .. cmember:: PyObject *PyUFuncObject.obj

       For ufuncs dynamically created from python functions, this member
       holds a reference to the underlying Python function.

   .. cmember:: PyObject *PyUFuncObject.userloops

       A dictionary of user-defined 1-d vector loops (stored as CObject ptrs)
       for user-defined types. A loop may be registered by the user for any
       user-defined type. It is retrieved by type number. User defined type
       numbers are always larger than :cdata:`NPY_USERDEF`.


   .. cmember:: npy_uint32 PyUFuncObject.op_flags

       Override the default operand flags for each ufunc operand.

   .. cmember:: npy_uint32 PyUFuncObject.iter_flags

       Override the default nditer flags for the ufunc.

PyArrayIter_Type
----------------

.. cvar:: PyArrayIter_Type

   This is an iterator object that makes it easy to loop over an N-dimensional
   array. It is the object returned from the flat attribute of an
   ndarray. It is also used extensively throughout the implementation
   internals to loop over an N-dimensional array. The tp_as_mapping
   interface is implemented so that the iterator object can be indexed
   (using 1-d indexing), and a few methods are implemented through the
   tp_methods table. This object implements the next method and can be
   used anywhere an iterator can be used in Python.

.. ctype:: PyArrayIterObject

   The C-structure corresponding to an object of :cdata:`PyArrayIter_Type` is
   the :ctype:`PyArrayIterObject`. The :ctype:`PyArrayIterObject` is used to
   keep track of a pointer into an N-dimensional array. It contains associated
   information used to quickly march through the array. The pointer can
   be adjusted in three basic ways: 1) advance to the "next" position in
   the array in a C-style contiguous fashion, 2) advance to an arbitrary
   N-dimensional coordinate in the array, and 3) advance to an arbitrary
   one-dimensional index into the array. The members of the
   :ctype:`PyArrayIterObject` structure are used in these
   calculations. Iterator objects keep their own dimension and strides
   information about an array. This can be adjusted as needed for
   "broadcasting," or to loop over only specific dimensions.

   .. code-block:: c

      typedef struct {
          PyObject_HEAD
          int   nd_m1;
          npy_intp  index;
          npy_intp  size;
          npy_intp  coordinates[NPY_MAXDIMS];
          npy_intp  dims_m1[NPY_MAXDIMS];
          npy_intp  strides[NPY_MAXDIMS];
          npy_intp  backstrides[NPY_MAXDIMS];
          npy_intp  factors[NPY_MAXDIMS];
          PyArrayObject *ao;
          char  *dataptr;
          Bool  contiguous;
      } PyArrayIterObject;

   .. cmember:: int PyArrayIterObject.nd_m1

       :math:`N-1` where :math:`N` is the number of dimensions in the
       underlying array.

   .. cmember:: npy_intp PyArrayIterObject.index

       The current 1-d index into the array.

   .. cmember:: npy_intp PyArrayIterObject.size

       The total size of the underlying array.

   .. cmember:: npy_intp *PyArrayIterObject.coordinates

       An :math:`N` -dimensional index into the array.

   .. cmember:: npy_intp *PyArrayIterObject.dims_m1

       The size of the array minus 1 in each dimension.

   .. cmember:: npy_intp *PyArrayIterObject.strides

       The strides of the array. How many bytes needed to jump to the next
       element in each dimension.

   .. cmember:: npy_intp *PyArrayIterObject.backstrides

       How many bytes needed to jump from the end of a dimension back
       to its beginning. Note that *backstrides* [k]= *strides* [k]*d
       *ims_m1* [k], but it is stored here as an optimization.

   .. cmember:: npy_intp *PyArrayIterObject.factors

       This array is used in computing an N-d index from a 1-d index. It
       contains needed products of the dimensions.

   .. cmember:: PyArrayObject *PyArrayIterObject.ao

       A pointer to the underlying ndarray this iterator was created to
       represent.

   .. cmember:: char *PyArrayIterObject.dataptr

       This member points to an element in the ndarray indicated by the
       index.

   .. cmember:: Bool PyArrayIterObject.contiguous

       This flag is true if the underlying array is
       :cdata:`NPY_ARRAY_C_CONTIGUOUS`. It is used to simplify
       calculations when possible.


How to use an array iterator on a C-level is explained more fully in
later sections. Typically, you do not need to concern yourself with
the internal structure of the iterator object, and merely interact
with it through the use of the macros :cfunc:`PyArray_ITER_NEXT` (it),
:cfunc:`PyArray_ITER_GOTO` (it, dest), or :cfunc:`PyArray_ITER_GOTO1D` (it,
index). All of these macros require the argument *it* to be a
:ctype:`PyArrayIterObject *`.


PyArrayMultiIter_Type
---------------------

.. cvar:: PyArrayMultiIter_Type

   This type provides an iterator that encapsulates the concept of
   broadcasting. It allows :math:`N` arrays to be broadcast together
   so that the loop progresses in C-style contiguous fashion over the
   broadcasted array. The corresponding C-structure is the
   :ctype:`PyArrayMultiIterObject` whose memory layout must begin any
   object, *obj*, passed in to the :cfunc:`PyArray_Broadcast` (obj)
   function. Broadcasting is performed by adjusting array iterators so
   that each iterator represents the broadcasted shape and size, but
   has its strides adjusted so that the correct element from the array
   is used at each iteration.


.. ctype:: PyArrayMultiIterObject

   .. code-block:: c

      typedef struct {
          PyObject_HEAD
          int numiter;
          npy_intp size;
          npy_intp index;
          int nd;
          npy_intp dimensions[NPY_MAXDIMS];
          PyArrayIterObject *iters[NPY_MAXDIMS];
      } PyArrayMultiIterObject;

   .. cmacro:: PyArrayMultiIterObject.PyObject_HEAD

       Needed at the start of every Python object (holds reference count and
       type identification).

   .. cmember:: int PyArrayMultiIterObject.numiter

       The number of arrays that need to be broadcast to the same shape.

   .. cmember:: npy_intp PyArrayMultiIterObject.size

       The total broadcasted size.

   .. cmember:: npy_intp PyArrayMultiIterObject.index

       The current (1-d) index into the broadcasted result.

   .. cmember:: int PyArrayMultiIterObject.nd

       The number of dimensions in the broadcasted result.

   .. cmember:: npy_intp *PyArrayMultiIterObject.dimensions

       The shape of the broadcasted result (only ``nd`` slots are used).

   .. cmember:: PyArrayIterObject **PyArrayMultiIterObject.iters

       An array of iterator objects that holds the iterators for the arrays
       to be broadcast together. On return, the iterators are adjusted for
       broadcasting.

PyArrayNeighborhoodIter_Type
----------------------------

.. cvar:: PyArrayNeighborhoodIter_Type

   This is an iterator object that makes it easy to loop over an N-dimensional
   neighborhood.

.. ctype:: PyArrayNeighborhoodIterObject

   The C-structure corresponding to an object of
   :cdata:`PyArrayNeighborhoodIter_Type` is the
   :ctype:`PyArrayNeighborhoodIterObject`.

PyArrayFlags_Type
-----------------

.. cvar:: PyArrayFlags_Type

   When the flags attribute is retrieved from Python, a special
   builtin object of this type is constructed. This special type makes
   it easier to work with the different flags by accessing them as
   attributes or by accessing them as if the object were a dictionary
   with the flag names as entries.


ScalarArrayTypes
----------------

There is a Python type for each of the different built-in data types
that can be present in the array Most of these are simple wrappers
around the corresponding data type in C. The C-names for these types
are :cdata:`Py{TYPE}ArrType_Type` where ``{TYPE}`` can be

    **Bool**, **Byte**, **Short**, **Int**, **Long**, **LongLong**,
    **UByte**, **UShort**, **UInt**, **ULong**, **ULongLong**,
    **Half**, **Float**, **Double**, **LongDouble**, **CFloat**, **CDouble**,
    **CLongDouble**, **String**, **Unicode**, **Void**, and
    **Object**.

These type names are part of the C-API and can therefore be created in
extension C-code. There is also a :cdata:`PyIntpArrType_Type` and a
:cdata:`PyUIntpArrType_Type` that are simple substitutes for one of the
integer types that can hold a pointer on the platform. The structure
of these scalar objects is not exposed to C-code. The function
:cfunc:`PyArray_ScalarAsCtype` (..) can be used to extract the C-type value
from the array scalar and the function :cfunc:`PyArray_Scalar` (...) can be
used to construct an array scalar from a C-value.


Other C-Structures
==================

A few new C-structures were found to be useful in the development of
NumPy. These C-structures are used in at least one C-API call and are
therefore documented here. The main reason these structures were
defined is to make it easy to use the Python ParseTuple C-API to
convert from Python objects to a useful C-Object.


PyArray_Dims
------------

.. ctype:: PyArray_Dims

   This structure is very useful when shape and/or strides information is
   supposed to be interpreted. The structure is:

   .. code-block:: c

      typedef struct {
          npy_intp *ptr;
          int len;
      } PyArray_Dims;

   The members of this structure are

   .. cmember:: npy_intp *PyArray_Dims.ptr

       A pointer to a list of (:ctype:`npy_intp`) integers which usually
       represent array shape or array strides.

   .. cmember:: int PyArray_Dims.len

       The length of the list of integers. It is assumed safe to
       access *ptr* [0] to *ptr* [len-1].


PyArray_Chunk
-------------

.. ctype:: PyArray_Chunk

   This is equivalent to the buffer object structure in Python up to
   the ptr member. On 32-bit platforms (*i.e.* if :cdata:`NPY_SIZEOF_INT`
   == :cdata:`NPY_SIZEOF_INTP` ) or in Python 2.5, the len member also
   matches an equivalent member of the buffer object. It is useful to
   represent a generic single- segment chunk of memory.

   .. code-block:: c

      typedef struct {
          PyObject_HEAD
          PyObject *base;
          void *ptr;
          npy_intp len;
          int flags;
      } PyArray_Chunk;

   The members are

   .. cmacro:: PyArray_Chunk.PyObject_HEAD

       Necessary for all Python objects. Included here so that the
       :ctype:`PyArray_Chunk` structure matches that of the buffer object
       (at least to the len member).

   .. cmember:: PyObject *PyArray_Chunk.base

       The Python object this chunk of memory comes from. Needed so that
       memory can be accounted for properly.

   .. cmember:: void *PyArray_Chunk.ptr

       A pointer to the start of the single-segment chunk of memory.

   .. cmember:: npy_intp PyArray_Chunk.len

       The length of the segment in bytes.

   .. cmember:: int PyArray_Chunk.flags

       Any data flags (*e.g.* :cdata:`NPY_ARRAY_WRITEABLE` ) that should
       be used to interpret the memory.


PyArrayInterface
----------------

.. seealso:: :ref:`arrays.interface`

.. ctype:: PyArrayInterface

   The :ctype:`PyArrayInterface` structure is defined so that NumPy and
   other extension modules can use the rapid array interface
   protocol. The :obj:`__array_struct__` method of an object that
   supports the rapid array interface protocol should return a
   :ctype:`PyCObject` that contains a pointer to a :ctype:`PyArrayInterface`
   structure with the relevant details of the array. After the new
   array is created, the attribute should be ``DECREF``'d which will
   free the :ctype:`PyArrayInterface` structure. Remember to ``INCREF`` the
   object (whose :obj:`__array_struct__` attribute was retrieved) and
   point the base member of the new :ctype:`PyArrayObject` to this same
   object. In this way the memory for the array will be managed
   correctly.

   .. code-block:: c

      typedef struct {
          int two;
          int nd;
          char typekind;
          int itemsize;
          int flags;
          npy_intp *shape;
          npy_intp *strides;
          void *data;
          PyObject *descr;
      } PyArrayInterface;

   .. cmember:: int PyArrayInterface.two

       the integer 2 as a sanity check.

   .. cmember:: int PyArrayInterface.nd

       the number of dimensions in the array.

   .. cmember:: char PyArrayInterface.typekind

       A character indicating what kind of array is present according to the
       typestring convention with 't' -> bitfield, 'b' -> Boolean, 'i' ->
       signed integer, 'u' -> unsigned integer, 'f' -> floating point, 'c' ->
       complex floating point, 'O' -> object, 'S' -> string, 'U' -> unicode,
       'V' -> void.

   .. cmember:: int PyArrayInterface.itemsize

       The number of bytes each item in the array requires.

   .. cmember:: int PyArrayInterface.flags

       Any of the bits :cdata:`NPY_ARRAY_C_CONTIGUOUS` (1),
       :cdata:`NPY_ARRAY_F_CONTIGUOUS` (2), :cdata:`NPY_ARRAY_ALIGNED` (0x100),
       :cdata:`NPY_ARRAY_NOTSWAPPED` (0x200), or :cdata:`NPY_ARRAY_WRITEABLE`
       (0x400) to indicate something about the data. The
       :cdata:`NPY_ARRAY_ALIGNED`, :cdata:`NPY_ARRAY_C_CONTIGUOUS`, and
       :cdata:`NPY_ARRAY_F_CONTIGUOUS` flags can actually be determined from
       the other parameters. The flag :cdata:`NPY_ARR_HAS_DESCR`
       (0x800) can also be set to indicate to objects consuming the
       version 3 array interface that the descr member of the
       structure is present (it will be ignored by objects consuming
       version 2 of the array interface).

   .. cmember:: npy_intp *PyArrayInterface.shape

       An array containing the size of the array in each dimension.

   .. cmember:: npy_intp *PyArrayInterface.strides

       An array containing the number of bytes to jump to get to the next
       element in each dimension.

   .. cmember:: void *PyArrayInterface.data

       A pointer *to* the first element of the array.

   .. cmember:: PyObject *PyArrayInterface.descr

       A Python object describing the data-type in more detail (same
       as the *descr* key in :obj:`__array_interface__`). This can be
       ``NULL`` if *typekind* and *itemsize* provide enough
       information. This field is also ignored unless
       :cdata:`ARR_HAS_DESCR` flag is on in *flags*.


Internally used structures
--------------------------

Internally, the code uses some additional Python objects primarily for
memory management. These types are not accessible directly from
Python, and are not exposed to the C-API. They are included here only
for completeness and assistance in understanding the code.


.. ctype:: PyUFuncLoopObject

   A loose wrapper for a C-structure that contains the information
   needed for looping. This is useful if you are trying to understand
   the ufunc looping code. The :ctype:`PyUFuncLoopObject` is the associated
   C-structure. It is defined in the ``ufuncobject.h`` header.

.. ctype:: PyUFuncReduceObject

   A loose wrapper for the C-structure that contains the information
   needed for reduce-like methods of ufuncs. This is useful if you are
   trying to understand the reduce, accumulate, and reduce-at
   code. The :ctype:`PyUFuncReduceObject` is the associated C-structure. It
   is defined in the ``ufuncobject.h`` header.

.. ctype:: PyUFunc_Loop1d

   A simple linked-list of C-structures containing the information needed
   to define a 1-d loop for a ufunc for every defined signature of a
   user-defined data-type.

.. cvar:: PyArrayMapIter_Type

   Advanced indexing is handled with this Python type. It is simply a
   loose wrapper around the C-structure containing the variables
   needed for advanced array indexing. The associated C-structure,
   :ctype:`PyArrayMapIterObject`, is useful if you are trying to
   understand the advanced-index mapping code. It is defined in the
   ``arrayobject.h`` header. This type is not exposed to Python and
   could be replaced with a C-structure. As a Python type it takes
   advantage of reference- counted memory management.
