*******************
How to extend NumPy
*******************

|    That which is static and repetitive is boring. That which is dynamic
|    and random is confusing. In between lies art.
|    --- *John A. Locke*

|    Science is a differential equation. Religion is a boundary condition.
|    --- *Alan Turing*


.. _writing-an-extension:

Writing an extension module
===========================

While the ndarray object is designed to allow rapid computation in
Python, it is also designed to be general-purpose and satisfy a wide-
variety of computational needs. As a result, if absolute speed is
essential, there is no replacement for a well-crafted, compiled loop
specific to your application and hardware. This is one of the reasons
that numpy includes f2py so that an easy-to-use mechanisms for linking
(simple) C/C++ and (arbitrary) Fortran code directly into Python are
available. You are encouraged to use and improve this mechanism. The
purpose of this section is not to document this tool but to document
the more basic steps to writing an extension module that this tool
depends on.

.. index::
   single: extension module

When an extension module is written, compiled, and installed to
somewhere in the Python path (sys.path), the code can then be imported
into Python as if it were a standard python file. It will contain
objects and methods that have been defined and compiled in C code. The
basic steps for doing this in Python are well-documented and you can
find more information in the documentation for Python itself available
online at `www.python.org <https://www.python.org>`_ .

In addition to the Python C-API, there is a full and rich C-API for
NumPy allowing sophisticated manipulations on a C-level. However, for
most applications, only a few API calls will typically be used. If all
you need to do is extract a pointer to memory along with some shape
information to pass to another calculation routine, then you will use
very different calls, then if you are trying to create a new array-
like type or add a new data type for ndarrays. This chapter documents
the API calls and macros that are most commonly used.


Required subroutine
===================

There is exactly one function that must be defined in your C-code in
order for Python to use it as an extension module. The function must
be called init{name} where {name} is the name of the module from
Python. This function must be declared so that it is visible to code
outside of the routine. Besides adding the methods and constants you
desire, this subroutine must also contain calls like ``import_array()``
and/or ``import_ufunc()`` depending on which C-API is needed. Forgetting
to place these commands will show itself as an ugly segmentation fault
(crash) as soon as any C-API subroutine is actually called. It is
actually possible to have multiple init{name} functions in a single
file in which case multiple modules will be defined by that file.
However, there are some tricks to get that to work correctly and it is
not covered here.

A minimal ``init{name}`` method looks like:

.. code-block:: c

    PyMODINIT_FUNC
    init{name}(void)
    {
       (void)Py_InitModule({name}, mymethods);
       import_array();
    }

The mymethods must be an array (usually statically declared) of
PyMethodDef structures which contain method names, actual C-functions,
a variable indicating whether the method uses keyword arguments or
not, and docstrings. These are explained in the next section. If you
want to add constants to the module, then you store the returned value
from Py_InitModule which is a module object. The most general way to
add items to the module is to get the module dictionary using
PyModule_GetDict(module). With the module dictionary, you can add
whatever you like to the module manually. An easier way to add objects
to the module is to use one of three additional Python C-API calls
that do not require a separate extraction of the module dictionary.
These are documented in the Python documentation, but repeated here
for convenience:

.. c:function:: int PyModule_AddObject( \
        PyObject* module, char* name, PyObject* value)

.. c:function:: int PyModule_AddIntConstant( \
        PyObject* module, char* name, long value)

.. c:function:: int PyModule_AddStringConstant( \
        PyObject* module, char* name, char* value)

    All three of these functions require the *module* object (the
    return value of Py_InitModule). The *name* is a string that
    labels the value in the module. Depending on which function is
    called, the *value* argument is either a general object
    (:c:func:`PyModule_AddObject` steals a reference to it), an integer
    constant, or a string constant.


Defining functions
==================

The second argument passed in to the Py_InitModule function is a
structure that makes it easy to to define functions in the module. In
the example given above, the mymethods structure would have been
defined earlier in the file (usually right before the init{name}
subroutine) to:

.. code-block:: c

    static PyMethodDef mymethods[] = {
        { nokeywordfunc,nokeyword_cfunc,
          METH_VARARGS,
          Doc string},
        { keywordfunc, keyword_cfunc,
          METH_VARARGS|METH_KEYWORDS,
          Doc string},
        {NULL, NULL, 0, NULL} /* Sentinel */
    }

Each entry in the mymethods array is a :c:type:`PyMethodDef` structure
containing 1) the Python name, 2) the C-function that implements the
function, 3) flags indicating whether or not keywords are accepted for
this function, and 4) The docstring for the function. Any number of
functions may be defined for a single module by adding more entries to
this table. The last entry must be all NULL as shown to act as a
sentinel. Python looks for this entry to know that all of the
functions for the module have been defined.

The last thing that must be done to finish the extension module is to
actually write the code that performs the desired functions. There are
two kinds of functions: those that don't accept keyword arguments, and
those that do.


Functions without keyword arguments
-----------------------------------

Functions that don't accept keyword arguments should be written as:

.. code-block:: c

    static PyObject*
    nokeyword_cfunc (PyObject *dummy, PyObject *args)
    {
        /* convert Python arguments */
        /* do function */
        /* return something */
    }

The dummy argument is not used in this context and can be safely
ignored. The *args* argument contains all of the arguments passed in
to the function as a tuple. You can do anything you want at this
point, but usually the easiest way to manage the input arguments is to
call :c:func:`PyArg_ParseTuple` (args, format_string,
addresses_to_C_variables...) or :c:func:`PyArg_UnpackTuple` (tuple, "name" ,
min, max, ...). A good description of how to use the first function is
contained in the Python C-API reference manual under section 5.5
(Parsing arguments and building values). You should pay particular
attention to the "O&" format which uses converter functions to go
between the Python object and the C object. All of the other format
functions can be (mostly) thought of as special cases of this general
rule. There are several converter functions defined in the NumPy C-API
that may be of use. In particular, the :c:func:`PyArray_DescrConverter`
function is very useful to support arbitrary data-type specification.
This function transforms any valid data-type Python object into a
:c:type:`PyArray_Descr *` object. Remember to pass in the address of the
C-variables that should be filled in.

There are lots of examples of how to use :c:func:`PyArg_ParseTuple`
throughout the NumPy source code. The standard usage is like this:

.. code-block:: c

    PyObject *input;
    PyArray_Descr *dtype;
    if (!PyArg_ParseTuple(args, "OO&", &input,
                          PyArray_DescrConverter,
                          &dtype)) return NULL;

It is important to keep in mind that you get a *borrowed* reference to
the object when using the "O" format string. However, the converter
functions usually require some form of memory handling. In this
example, if the conversion is successful, *dtype* will hold a new
reference to a :c:type:`PyArray_Descr *` object, while *input* will hold a
borrowed reference. Therefore, if this conversion were mixed with
another conversion (say to an integer) and the data-type conversion
was successful but the integer conversion failed, then you would need
to release the reference count to the data-type object before
returning. A typical way to do this is to set *dtype* to ``NULL``
before calling :c:func:`PyArg_ParseTuple` and then use :c:func:`Py_XDECREF`
on *dtype* before returning.

After the input arguments are processed, the code that actually does
the work is written (likely calling other functions as needed). The
final step of the C-function is to return something. If an error is
encountered then ``NULL`` should be returned (making sure an error has
actually been set). If nothing should be returned then increment
:c:data:`Py_None` and return it. If a single object should be returned then
it is returned (ensuring that you own a reference to it first). If
multiple objects should be returned then you need to return a tuple.
The :c:func:`Py_BuildValue` (format_string, c_variables...) function makes
it easy to build tuples of Python objects from C variables. Pay
special attention to the difference between 'N' and 'O' in the format
string or you can easily create memory leaks. The 'O' format string
increments the reference count of the :c:type:`PyObject *<PyObject>` C-variable it
corresponds to, while the 'N' format string steals a reference to the
corresponding :c:type:`PyObject *<PyObject>` C-variable. You should use 'N' if you have
already created a reference for the object and just want to give that
reference to the tuple. You should use 'O' if you only have a borrowed
reference to an object and need to create one to provide for the
tuple.


Functions with keyword arguments
--------------------------------

These functions are very similar to functions without keyword
arguments. The only difference is that the function signature is:

.. code-block:: c

    static PyObject*
    keyword_cfunc (PyObject *dummy, PyObject *args, PyObject *kwds)
    {
    ...
    }

The kwds argument holds a Python dictionary whose keys are the names
of the keyword arguments and whose values are the corresponding
keyword-argument values. This dictionary can be processed however you
see fit. The easiest way to handle it, however, is to replace the
:c:func:`PyArg_ParseTuple` (args, format_string, addresses...) function with
a call to :c:func:`PyArg_ParseTupleAndKeywords` (args, kwds, format_string,
char \*kwlist[], addresses...). The kwlist parameter to this function
is a ``NULL`` -terminated array of strings providing the expected
keyword arguments.  There should be one string for each entry in the
format_string. Using this function will raise a TypeError if invalid
keyword arguments are passed in.

For more help on this function please see section 1.8 (Keyword
Parameters for Extension Functions) of the Extending and Embedding
tutorial in the Python documentation.


Reference counting
------------------

The biggest difficulty when writing extension modules is reference
counting. It is an important reason for the popularity of f2py, weave,
Cython, ctypes, etc.... If you mis-handle reference counts you can get
problems from memory-leaks to segmentation faults. The only strategy I
know of to handle reference counts correctly is blood, sweat, and
tears. First, you force it into your head that every Python variable
has a reference count. Then, you understand exactly what each function
does to the reference count of your objects, so that you can properly
use DECREF and INCREF when you need them. Reference counting can
really test the amount of patience and diligence you have towards your
programming craft. Despite the grim depiction, most cases of reference
counting are quite straightforward with the most common difficulty
being not using DECREF on objects before exiting early from a routine
due to some error. In second place, is the common error of not owning
the reference on an object that is passed to a function or macro that
is going to steal the reference ( *e.g.* :c:func:`PyTuple_SET_ITEM`, and
most functions that take :c:type:`PyArray_Descr` objects).

.. index::
   single: reference counting

Typically you get a new reference to a variable when it is created or
is the return value of some function (there are some prominent
exceptions, however --- such as getting an item out of a tuple or a
dictionary). When you own the reference, you are responsible to make
sure that :c:func:`Py_DECREF` (var) is called when the variable is no
longer necessary (and no other function has "stolen" its
reference). Also, if you are passing a Python object to a function
that will "steal" the reference, then you need to make sure you own it
(or use :c:func:`Py_INCREF` to get your own reference). You will also
encounter the notion of borrowing a reference. A function that borrows
a reference does not alter the reference count of the object and does
not expect to "hold on "to the reference. It's just going to use the
object temporarily.  When you use :c:func:`PyArg_ParseTuple` or
:c:func:`PyArg_UnpackTuple` you receive a borrowed reference to the
objects in the tuple and should not alter their reference count inside
your function. With practice, you can learn to get reference counting
right, but it can be frustrating at first.

One common source of reference-count errors is the :c:func:`Py_BuildValue`
function. Pay careful attention to the difference between the 'N'
format character and the 'O' format character. If you create a new
object in your subroutine (such as an output array), and you are
passing it back in a tuple of return values, then you should most-
likely use the 'N' format character in :c:func:`Py_BuildValue`. The 'O'
character will increase the reference count by one. This will leave
the caller with two reference counts for a brand-new array.  When the
variable is deleted and the reference count decremented by one, there
will still be that extra reference count, and the array will never be
deallocated. You will have a reference-counting induced memory leak.
Using the 'N' character will avoid this situation as it will return to
the caller an object (inside the tuple) with a single reference count.

.. index::
   single: reference counting




Dealing with array objects
==========================

Most extension modules for NumPy will need to access the memory for an
ndarray object (or one of it's sub-classes). The easiest way to do
this doesn't require you to know much about the internals of NumPy.
The method is to

1. Ensure you are dealing with a well-behaved array (aligned, in machine
   byte-order and single-segment) of the correct type and number of
   dimensions.

    1. By converting it from some Python object using
       :c:func:`PyArray_FromAny` or a macro built on it.

    2. By constructing a new ndarray of your desired shape and type
       using :c:func:`PyArray_NewFromDescr` or a simpler macro or function
       based on it.


2. Get the shape of the array and a pointer to its actual data.

3. Pass the data and shape information on to a subroutine or other
   section of code that actually performs the computation.

4. If you are writing the algorithm, then I recommend that you use the
   stride information contained in the array to access the elements of
   the array (the :c:func:`PyArray_GETPTR` macros make this painless). Then,
   you can relax your requirements so as not to force a single-segment
   array and the data-copying that might result.

Each of these sub-topics is covered in the following sub-sections.


Converting an arbitrary sequence object
---------------------------------------

The main routine for obtaining an array from any Python object that
can be converted to an array is :c:func:`PyArray_FromAny`. This
function is very flexible with many input arguments. Several macros
make it easier to use the basic function. :c:func:`PyArray_FROM_OTF` is
arguably the most useful of these macros for the most common uses.  It
allows you to convert an arbitrary Python object to an array of a
specific builtin data-type ( *e.g.* float), while specifying a
particular set of requirements ( *e.g.* contiguous, aligned, and
writeable). The syntax is

:c:func:`PyArray_FROM_OTF`

    Return an ndarray from any Python object, *obj*, that can be
    converted to an array. The number of dimensions in the returned
    array is determined by the object. The desired data-type of the
    returned array is provided in *typenum* which should be one of the
    enumerated types. The *requirements* for the returned array can be
    any combination of standard array flags.  Each of these arguments
    is explained in more detail below. You receive a new reference to
    the array on success. On failure, ``NULL`` is returned and an
    exception is set.

    *obj*

        The object can be any Python object convertible to an ndarray.
        If the object is already (a subclass of) the ndarray that
        satisfies the requirements then a new reference is returned.
        Otherwise, a new array is constructed. The contents of *obj*
        are copied to the new array unless the array interface is used
        so that data does not have to be copied. Objects that can be
        converted to an array include: 1) any nested sequence object,
        2) any object exposing the array interface, 3) any object with
        an :obj:`~numpy.class.__array__` method (which should return an ndarray),
        and 4) any scalar object (becomes a zero-dimensional
        array). Sub-classes of the ndarray that otherwise fit the
        requirements will be passed through. If you want to ensure
        a base-class ndarray, then use :c:data:`NPY_ARRAY_ENSUREARRAY` in the
        requirements flag. A copy is made only if necessary. If you
        want to guarantee a copy, then pass in :c:data:`NPY_ARRAY_ENSURECOPY`
        to the requirements flag.

    *typenum*

        One of the enumerated types or :c:data:`NPY_NOTYPE` if the data-type
        should be determined from the object itself. The C-based names
        can be used:

            :c:data:`NPY_BOOL`, :c:data:`NPY_BYTE`, :c:data:`NPY_UBYTE`,
            :c:data:`NPY_SHORT`, :c:data:`NPY_USHORT`, :c:data:`NPY_INT`,
            :c:data:`NPY_UINT`, :c:data:`NPY_LONG`, :c:data:`NPY_ULONG`,
            :c:data:`NPY_LONGLONG`, :c:data:`NPY_ULONGLONG`, :c:data:`NPY_DOUBLE`,
            :c:data:`NPY_LONGDOUBLE`, :c:data:`NPY_CFLOAT`, :c:data:`NPY_CDOUBLE`,
            :c:data:`NPY_CLONGDOUBLE`, :c:data:`NPY_OBJECT`.

        Alternatively, the bit-width names can be used as supported on the
        platform. For example:

            :c:data:`NPY_INT8`, :c:data:`NPY_INT16`, :c:data:`NPY_INT32`,
            :c:data:`NPY_INT64`, :c:data:`NPY_UINT8`,
            :c:data:`NPY_UINT16`, :c:data:`NPY_UINT32`,
            :c:data:`NPY_UINT64`, :c:data:`NPY_FLOAT32`,
            :c:data:`NPY_FLOAT64`, :c:data:`NPY_COMPLEX64`,
            :c:data:`NPY_COMPLEX128`.

        The object will be converted to the desired type only if it
        can be done without losing precision. Otherwise ``NULL`` will
        be returned and an error raised. Use :c:data:`NPY_ARRAY_FORCECAST` in the
        requirements flag to override this behavior.

    *requirements*

        The memory model for an ndarray admits arbitrary strides in
        each dimension to advance to the next element of the array.
        Often, however, you need to interface with code that expects a
        C-contiguous or a Fortran-contiguous memory layout. In
        addition, an ndarray can be misaligned (the address of an
        element is not at an integral multiple of the size of the
        element) which can cause your program to crash (or at least
        work more slowly) if you try and dereference a pointer into
        the array data. Both of these problems can be solved by
        converting the Python object into an array that is more
        "well-behaved" for your specific usage.

        The requirements flag allows specification of what kind of
        array is acceptable. If the object passed in does not satisfy
        this requirements then a copy is made so that thre returned
        object will satisfy the requirements. these ndarray can use a
        very generic pointer to memory.  This flag allows specification
        of the desired properties of the returned array object. All
        of the flags are explained in the detailed API chapter. The
        flags most commonly needed are :c:data:`NPY_ARRAY_IN_ARRAY`,
        :c:data:`NPY_OUT_ARRAY`, and :c:data:`NPY_ARRAY_INOUT_ARRAY`:

        :c:data:`NPY_ARRAY_IN_ARRAY`

            Equivalent to :c:data:`NPY_ARRAY_C_CONTIGUOUS` \|
            :c:data:`NPY_ARRAY_ALIGNED`. This combination of flags is useful
            for arrays that must be in C-contiguous order and aligned.
            These kinds of arrays are usually input arrays for some
            algorithm.

        .. c:var:: NPY_ARRAY_OUT_ARRAY

            Equivalent to :c:data:`NPY_ARRAY_C_CONTIGUOUS` \|
            :c:data:`NPY_ARRAY_ALIGNED` \| :c:data:`NPY_ARRAY_WRITEABLE`. This
            combination of flags is useful to specify an array that is
            in C-contiguous order, is aligned, and can be written to
            as well. Such an array is usually returned as output
            (although normally such output arrays are created from
            scratch).

        :c:data:`NPY_ARRAY_INOUT_ARRAY`

            Equivalent to :c:data:`NPY_ARRAY_C_CONTIGUOUS` \|
            :c:data:`NPY_ARRAY_ALIGNED` \| :c:data:`NPY_ARRAY_WRITEABLE` \|
            :c:data:`NPY_ARRAY_WRITEBACKIFCOPY` \|
            :c:data:`NPY_ARRAY_UPDATEIFCOPY`. This combination of flags is
            useful to specify an array that will be used for both
            input and output. :c:func:`PyArray_ResolveWritebackIfCopy`
            must be called before :func:`Py_DECREF` at
            the end of the interface routine to write back the temporary data
            into the original array passed in. Use
            of the :c:data:`NPY_ARRAY_WRITEBACKIFCOPY` or
            :c:data:`NPY_ARRAY_UPDATEIFCOPY` flags requires that the input
            object is already an array (because other objects cannot
            be automatically updated in this fashion). If an error
            occurs use :c:func:`PyArray_DiscardWritebackIfCopy` (obj) on an
            array with these flags set. This will set the underlying base array
            writable without causing the contents to be copied
            back into the original array.


        Other useful flags that can be OR'd as additional requirements are:

        :c:data:`NPY_ARRAY_FORCECAST`

            Cast to the desired type, even if it can't be done without losing
            information.

        :c:data:`NPY_ARRAY_ENSURECOPY`

            Make sure the resulting array is a copy of the original.

        :c:data:`NPY_ARRAY_ENSUREARRAY`

            Make sure the resulting object is an actual ndarray and not a sub-
            class.

.. note::

    Whether or not an array is byte-swapped is determined by the
    data-type of the array. Native byte-order arrays are always
    requested by :c:func:`PyArray_FROM_OTF` and so there is no need for
    a :c:data:`NPY_ARRAY_NOTSWAPPED` flag in the requirements argument. There
    is also no way to get a byte-swapped array from this routine.


Creating a brand-new ndarray
----------------------------

Quite often new arrays must be created from within extension-module
code. Perhaps an output array is needed and you don't want the caller
to have to supply it. Perhaps only a temporary array is needed to hold
an intermediate calculation. Whatever the need there are simple ways
to get an ndarray object of whatever data-type is needed. The most
general function for doing this is :c:func:`PyArray_NewFromDescr`. All array
creation functions go through this heavily re-used code. Because of
its flexibility, it can be somewhat confusing to use. As a result,
simpler forms exist that are easier to use.

:c:func:`PyArray_SimpleNew`

    This function allocates new memory and places it in an ndarray
    with *nd* dimensions whose shape is determined by the array of
    at least *nd* items pointed to by *dims*. The memory for the
    array is uninitialized (unless typenum is :c:data:`NPY_OBJECT` in
    which case each element in the array is set to NULL). The
    *typenum* argument allows specification of any of the builtin
    data-types such as :c:data:`NPY_FLOAT` or :c:data:`NPY_LONG`. The
    memory for the array can be set to zero if desired using
    :c:func:`PyArray_FILLWBYTE` (return_object, 0).

:c:func:`PyArray_SimpleNewFromData`

    Sometimes, you want to wrap memory allocated elsewhere into an
    ndarray object for downstream use. This routine makes it
    straightforward to do that. The first three arguments are the same
    as in :c:func:`PyArray_SimpleNew`, the final argument is a pointer to a
    block of contiguous memory that the ndarray should use as it's
    data-buffer which will be interpreted in C-style contiguous
    fashion. A new reference to an ndarray is returned, but the
    ndarray will not own its data. When this ndarray is deallocated,
    the pointer will not be freed.

    You should ensure that the provided memory is not freed while the
    returned array is in existence. The easiest way to handle this is
    if data comes from another reference-counted Python object. The
    reference count on this object should be increased after the
    pointer is passed in, and the base member of the returned ndarray
    should point to the Python object that owns the data. Then, when
    the ndarray is deallocated, the base-member will be DECREF'd
    appropriately. If you want the memory to be freed as soon as the
    ndarray is deallocated then simply set the OWNDATA flag on the
    returned ndarray.


Getting at ndarray memory and accessing elements of the ndarray
---------------------------------------------------------------

If obj is an ndarray (:c:type:`PyArrayObject *`), then the data-area of the
ndarray is pointed to by the void* pointer :c:func:`PyArray_DATA` (obj) or
the char* pointer :c:func:`PyArray_BYTES` (obj). Remember that (in general)
this data-area may not be aligned according to the data-type, it may
represent byte-swapped data, and/or it may not be writeable. If the
data area is aligned and in native byte-order, then how to get at a
specific element of the array is determined only by the array of
npy_intp variables, :c:func:`PyArray_STRIDES` (obj). In particular, this
c-array of integers shows how many **bytes** must be added to the
current element pointer to get to the next element in each dimension.
For arrays less than 4-dimensions there are :c:func:`PyArray_GETPTR{k}`
(obj, ...) macros where {k} is the integer 1, 2, 3, or 4 that make
using the array strides easier. The arguments .... represent {k} non-
negative integer indices into the array. For example, suppose ``E`` is
a 3-dimensional ndarray. A (void*) pointer to the element ``E[i,j,k]``
is obtained as :c:func:`PyArray_GETPTR3` (E, i, j, k).

As explained previously, C-style contiguous arrays and Fortran-style
contiguous arrays have particular striding patterns. Two array flags
(:c:data:`NPY_ARRAY_C_CONTIGUOUS` and :c:data:`NPY_ARRAY_F_CONTIGUOUS`) indicate
whether or not the striding pattern of a particular array matches the
C-style contiguous or Fortran-style contiguous or neither. Whether or
not the striding pattern matches a standard C or Fortran one can be
tested Using :c:func:`PyArray_ISCONTIGUOUS` (obj) and
:c:func:`PyArray_ISFORTRAN` (obj) respectively. Most third-party
libraries expect contiguous arrays.  But, often it is not difficult to
support general-purpose striding. I encourage you to use the striding
information in your own code whenever possible, and reserve
single-segment requirements for wrapping third-party code. Using the
striding information provided with the ndarray rather than requiring a
contiguous striding reduces copying that otherwise must be made.


Example
=======

.. index::
   single: extension module

The following example shows how you might write a wrapper that accepts
two input arguments (that will be converted to an array) and an output
argument (that must be an array). The function returns None and
updates the output array. Note the updated use of WRITEBACKIFCOPY semantics
for NumPy v1.14 and above

.. code-block:: c

    static PyObject *
    example_wrapper(PyObject *dummy, PyObject *args)
    {
        PyObject *arg1=NULL, *arg2=NULL, *out=NULL;
        PyObject *arr1=NULL, *arr2=NULL, *oarr=NULL;

        if (!PyArg_ParseTuple(args, "OOO!", &arg1, &arg2,
            &PyArray_Type, &out)) return NULL;

        arr1 = PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        if (arr1 == NULL) return NULL;
        arr2 = PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        if (arr2 == NULL) goto fail;
    #if NPY_API_VERSION >= 0x0000000c
        oarr = PyArray_FROM_OTF(out, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    #else
        oarr = PyArray_FROM_OTF(out, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
    #endif
        if (oarr == NULL) goto fail;

        /* code that makes use of arguments */
        /* You will probably need at least
           nd = PyArray_NDIM(<..>)    -- number of dimensions
           dims = PyArray_DIMS(<..>)  -- npy_intp array of length nd
                                         showing length in each dim.
           dptr = (double *)PyArray_DATA(<..>) -- pointer to data.

           If an error occurs goto fail.
         */

        Py_DECREF(arr1);
        Py_DECREF(arr2);
    #if NPY_API_VERSION >= 0x0000000c
        PyArray_ResolveWritebackIfCopy(oarr);
    #endif
        Py_DECREF(oarr);
        Py_INCREF(Py_None);
        return Py_None;

     fail:
        Py_XDECREF(arr1);
        Py_XDECREF(arr2);
    #if NPY_API_VERSION >= 0x0000000c
        PyArray_DiscardWritebackIfCopy(oarr);
    #endif
        Py_XDECREF(oarr);
        return NULL;
    }
