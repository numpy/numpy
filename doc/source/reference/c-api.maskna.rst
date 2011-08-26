Array NA Mask API
==================

.. sectionauthor:: Mark Wiebe

.. index::
   pair: maskna; C-API
   pair: C-API; maskna

.. versionadded:: 1.7

NA Masks in Arrays
------------------

NumPy supports the idea of NA (Not Available) missing values in its
arrays.  In the design document leading up to the implementation, two
mechanisms for this were proposed, NA masks and NA bitpatterns. NA masks
have been implemented as the first representation of these values. This
mechanism supports working with NA values similar to what the R language
provides, and when combined with views, allows one to temporarily mark
elements as NA without affecting the original data.

The C API has been updated with mechanisms to allow NumPy extensions
to work with these masks, and this document provides some examples and
reference for the NA mask-related functions.

The NA Object
-------------

The main *numpy* namespace in Python has a new object called *NA*.
This is an instance of :ctype:`NpyNA`, which is a Python object
representing an NA value. This object is analogous to the NumPy
scalars, and is returned by :cfunc:`PyArray_Return` instead of
a scalar where appropriate.

The global *numpy.NA* object is accessible from C as :cdata:`Npy_NA`.
This is an NA value with no data type or multi-NA payload. Use it
just as you would Py_None, except use :cfunc:`NpyNA_Check` to
see if an object is an :ctype:`NpyNA`, because :cdata:`Npy_NA` isn't
the only instance of NA possible.

If you want to see whether a general PyObject* is NA, you should
use the API function :cfunc:`NpyNA_FromObject` with *suppress_error*
set to true. If this returns NULL, the object is not an NA, and if
it returns an NpyNA instance, the object is NA and you can then
access its *dtype* and *payload* fields as needed.

To make new :ctype:`NpyNA` objects, use
:cfunc:`NpyNA_FromDTypeAndPayload`. The functions
:cfunc:`NpyNA_GetDType`, :cfunc:`NpyNA_IsMultiNA`, and
:cfunc:`NpyNA_GetPayload` provide access to the data members.

Working With NA-Masked Arrays
-----------------------------

The starting point for many C-API functions which manipulate NumPy
arrays is the function :cfunc:`PyArray_FromAny`. This function converts
a general PyObject* object into a NumPy ndarray, based on options
specified in the flags. To avoid surprises, this function does
not allow NA-masked arrays to pass through by default.

To allow third-party code to work with NA-masked arrays which contain
no NAs, :cfunc:`PyArray_FromAny` will make a copy of the array into
a new array without an NA-mask, and return that. This allows for
proper interoperability in cases where it's possible until functions
are updated to provide optimal code paths for NA-masked arrays.

To update a function with NA-mask support, add the flag
:cdata:`NPY_ARRAY_ALLOWNA` when calling :cfunc:`PyArray_FromAny`.
This allows NA-masked arrays to pass through untouched, and will
convert PyObject lists containing NA values into NA-masked arrays
instead of the alternative of switching to object arrays.

To check whether an array has an NA-mask, use the function
:cfunc:`PyArray_HASMASKNA`, which checks the appropriate flag.
There are a number of things that one will typically want to do
when encountering an NA-masked array. We'll go through a few
of these cases.

Forbidding Any NA Values
~~~~~~~~~~~~~~~~~~~~~~~~

The simplest case is to forbid any NA values. Note that it is better
to still be aware of the NA mask and explicitly test for NA values
than to leave out the :cdata:`NPY_ARRAY_ALLOWNA`, because it is possible
to avoid the extra copy that :cfunc:`PyArray_FromAny` will make. The
check for NAs will go something like this::

    PyArrayObject *arr = ...;
    int containsna;

    /* ContainsNA checks HASMASKNA() for you */
    containsna = PyArray_ContainsNA(arr, NULL, NULL);
    /* Error case */
    if (containsna < 0) {
        return NULL;
    }
    /* If it found an NA */
    else if (containsna) {
        PyErr_SetString(PyExc_ValueError,
                "this operation does not support arrays with NA values");
        return NULL;
    }

After this check, you can be certain that the array doesn't contain any
NA values, and can proceed accordingly. For example, if you iterate
over the elements of the array, you may pass the flag
:cdata:`NPY_ITER_IGNORE_MASKNA` to iterate over the data without
touching the NA-mask at all.

Manipulating NA Values
~~~~~~~~~~~~~~~~~~~~~~

The semantics of the NA-mask demand that whenever an array element
is hidden by the NA-mask, no computations are permitted to modify
the data backing that element. The :ctype:`NpyIter` provides
a number of flags to assist with visiting both the array data
and the mask data simultaneously, and preserving the masking semantics
even when buffering is required.

The main flag for iterating over NA-masked arrays is
:cdata:`NPY_ITER_USE_MASKNA`. For each iterator operand which has this
flag specified, a new operand is added to the end of the iterator operand
list, and is set to iterate over the original operand's NA-mask. Operands
which do not have an NA mask are permitted as well when they are flagged
as read-only. The new operand in this case points to a single exposed
mask value and all its strides are zero. The latter feature is useful
when combining multiple read-only inputs, where some of them have masks.

Accumulating NA Values
~~~~~~~~~~~~~~~~~~~~~~

More complex operations, like the NumPy ufunc reduce functions, need
to take extra care to follow the masking semantics. If we accumulate
the NA mask and the data values together, we could discover half way
through that the output is NA, and that we have violated the contract
to never change the underlying output value when it is being assigned
NA.

The solution to this problem is to first accumulate the NA-mask as necessary
to produce the output's NA-mask, then accumulate the data values without
touching NA-masked values in the output. The parameter *preservena* in
functions like :cfunc:`PyArray_AssignArray` can assist when initializing
values in such an algorithm.

Example NA-Masked Operation in C
--------------------------------

As an example, let's implement a simple binary NA-masked operation
for the double dtype. We'll make a divide operation which turns
divide by zero into NA instead of Inf or NaN.

To start, we define the function prototype and some basic
:ctype:`NpyIter` boilerplate setup. We'll make a function which
supports an optional *out* parameter, which may be NULL.::

    static PyArrayObject*
    SpecialDivide(PyArrayObject* a, PyArrayObject* b, PyArrayObject *out)
    {
        NpyIter *iter = NULL;
        PyArrayObject *op[3];
        PyArray_Descr *dtypes[3];
        npy_uint32 flags, op_flags[3];

        /* Iterator construction parameters */
        op[0] = a;
        op[1] = b;
        op[2] = out;

        dtypes[0] = PyArray_DescrFromType(NPY_DOUBLE);
        if (dtypes[0] == NULL) {
            return NULL;
        }
        dtypes[1] = dtypes[0];
        dtypes[2] = dtypes[0];

        flags = NPY_ITER_BUFFERED |
                NPY_ITER_EXTERNAL_LOOP |
                NPY_ITER_GROWINNER |
                NPY_ITER_REFS_OK |
                NPY_ITER_ZEROSIZE_OK;

        /* Every operand gets the flag NPY_ITER_USE_MASKNA */
        op_flags[0] = NPY_ITER_READONLY |
                      NPY_ITER_ALIGNED |
                      NPY_ITER_USE_MASKNA;
        op_flags[1] = op_flags[0];
        op_flags[2] = NPY_ITER_WRITEONLY |
                      NPY_ITER_ALIGNED |
                      NPY_ITER_USE_MASKNA |
                      NPY_ITER_NO_BROADCAST |
                      NPY_ITER_ALLOCATE;

        iter = NpyIter_MultiNew(3, op, flags, NPY_KEEPORDER,
                                NPY_SAME_KIND_CASTING, op_flags, dtypes);
        /* Don't need the dtype reference anymore */
        Py_DECREF(dtypes[0]);
        if (iter == NULL) {
            return NULL;
        }

At this point, the input operands have been validated according to
the casting rule, the shapes of the arrays have been broadcast together,
and any buffering necessary has been prepared. This means we can
dive into the inner loop of this function.::

    ...
        if (NpyIter_GetIterSize(iter) > 0) {
            NpyIter_IterNextFunc *iternext;
            char **dataptr;
            npy_intp *stridesptr, *countptr;

            /* Variables needed for looping */
            iternext = NpyIter_GetIterNext(iter, NULL);
            if (iternext == NULL) {
                NpyIter_Deallocate(iter);
                return NULL;
            }
            dataptr = NpyIter_GetDataPtrArray(iter);
            stridesptr = NpyIter_GetInnerStrideArray(iter);
            countptr = NpyIter_GetInnerLoopSizePtr(iter);

The loop gets a bit messy when dealing with NA-masks, because it
doubles the number of operands being processed in the iterator. Here
we are naming things clearly so that the content of the innermost loop
can be easy to work with.::

    ...
            do {
                /* Data pointers and strides needed for innermost loop */
                char *data_a = dataptr[0], *data_b = dataptr[1];
                char *data_out = dataptr[2];
                char *maskna_a = dataptr[3], *maskna_b = dataptr[4];
                char *maskna_out = dataptr[5];
                npy_intp stride_a = stridesptr[0], stride_b = stridesptr[1];
                npy_intp stride_out = strides[2];
                npy_intp maskna_stride_a = stridesptr[3];
                npy_intp maskna_stride_b = stridesptr[4];
                npy_intp maskna_stride_out = stridesptr[5];
                npy_intp i, count = *countptr;

                for (i = 0; i < count; ++i) {

Here is the code for performing one special division. We use
the functions :cfunc:`NpyMaskValue_IsExposed` and
:cfunc:`NpyMaskValue_Create` to work with the masks, in order to be
as general as possible. These are inline functions, and the compiler
optimizer should be able to produce the same result as if you performed
these operations directly inline here.::

    ...
                    /* If neither of the inputs are NA */
                    if (NpyMaskValue_IsExposed((npy_mask)*maskna_a) &&
                                NpyMaskValue_IsExposed((npy_mask)*maskna_b)) {
                        double a_val = *(double *)data_a;
                        double b_val = *(double *)data_b;
                        /* Do the divide if 'b' isn't zero */
                        if (b_val != 0.0) {
                            *(double *)data_out = a_val / b_val;
                            /* Need to also set this element to exposed */
                            *maskna_out = NpyMaskValue_Create(1, 0);
                        }
                        /* Otherwise output an NA without touching its data */
                        else {
                            *maskna_out = NpyMaskValue_Create(0, 0);
                        }
                    }
                    /* Turn the output into NA without touching its data */
                    else {
                        *maskna_out = NpyMaskValue_Create(0, 0);
                    }

                    data_a += stride_a;
                    data_b += stride_b;
                    data_out += stride_out;
                    maskna_a += maskna_stride_a;
                    maskna_b += maskna_stride_b;
                    maskna_out += maskna_stride_out;
                }
            } while (iternext(iter));
        }

A little bit more boilerplate for returning the result from the iterator,
and the function is done.::

    ...
        if (out == NULL) {
            out = NpyIter_GetOperandArray(iter)[2];
        }
        Py_INCREF(out);
        NpyIter_Deallocate(iter);

        return out;
    }

To run this example, you can create a simple module with a C-file spdiv_mod.c
consisting of::

    #include <Python.h>
    #include <numpy/arrayobject.h>

    /* INSERT SpecialDivide source code here */

    static PyObject *
    spdiv(PyObject *self, PyObject *args, PyObject *kwds)
    {
        PyArrayObject *a, *b, *out = NULL;
        static char *kwlist[] = {"a", "b", "out", NULL};

        if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&|O&", kwlist, 
                                &PyArray_AllowNAConverter, &a,
                                &PyArray_AllowNAConverter, &b,
                                &PyArray_OutputAllowNAConverter, &out)) {
            return NULL;
        }

        /*
         * The usual NumPy way is to only use PyArray_Return when
         * the 'out' parameter is not provided.
         */
        if (out == NULL) {
            return PyArray_Return(SpecialDivide(a, b, out));
        }
        else {
            return (PyObject *)SpecialDivide(a, b, out);
        }
    }

    static PyMethodDef SpDivMethods[] = {
        {"spdiv", (PyCFunction)spdiv, METH_VARARGS | METH_KEYWORDS, NULL},
        {NULL, NULL, 0, NULL}
    };


    PyMODINIT_FUNC initspdiv_mod(void)
    {
        PyObject *m;

        m = Py_InitModule("spdiv_mod", SpDivMethods);
        if (m == NULL) {
            return;
        }

        /* Make sure NumPy is initialized */
        import_array();
    }

Create a setup.py file like::

    #!/usr/bin/env python
    def configuration(parent_package='',top_path=None):
        from numpy.distutils.misc_util import Configuration
        config = Configuration('.',parent_package,top_path)
        config.add_extension('spdiv_mod',['spdiv_mod.c'])
        return config

    if __name__ == "__main__":
        from numpy.distutils.core import setup
        setup(configuration=configuration)

With these two files in a directory by itself, run::

    $ python setup.py build_ext --inplace

and the file spdiv_mod.so (or .dll) will be placed in the same directory.
Now you can try out this sample, to see how it behaves.::

    >>> import numpy as np
    >>> from spdiv_mod import spdiv

Because we used :cfunc:`PyArray_Return` when wrapping SpecialDivide,
it returns scalars like any typical NumPy function does::

    >>> spdiv(1, 2)
    0.5
    >>> spdiv(2, 0)
    NA(dtype='float64')
    >>> spdiv(np.NA, 1.5)
    NA(dtype='float64')

Here we can see how NAs propagate, and how 0 in the output turns into NA
as desired.::

    >>> a = np.arange(6)
    >>> b = np.array([0,np.NA,0,2,1,0])
    >>> spdiv(a, b)
    array([  NA,   NA,   NA,  1.5,  4. ,   NA])

Finally, we can see the masking behavior by creating a masked
view of an array. The ones in *c_orig* are preserved whereever
NA got assigned.::

    >>> c_orig = np.ones(6)
    >>> c = c_orig.view(maskna=True)
    >>> spdiv(a, b, out=c)
    array([  NA,   NA,   NA,  1.5,  4. ,   NA])
    >>> c_orig
    array([ 1. ,  1. ,  1. ,  1.5,  4. ,  1. ])

NA Object Data Type
-------------------

.. ctype:: NpyNA

    This is the C object corresponding to objects of type
    numpy.NAType. The fields themselves are hidden from consumers of the
    API, you must use the functions provided to create new NA objects
    and get their properties.

    This object contains two fields, a :ctype:`PyArray_Descr *` dtype
    which is either NULL or indicates the data type the NA represents,
    and a payload which is there for the future addition of multi-NA support.

.. cvar:: Npy_NA

    This is a global singleton, similar to Py_None, which is the
    *numpy.NA* object. Note that unlike Py_None, multiple NAs may be
    created, for instance with different multi-NA payloads or with
    different dtypes. If you want to return an NA with no payload
    or dtype, return a new reference to Npy_NA.

NA Object Functions
-------------------

.. cfunction:: NpyNA_Check(obj)

    Evaluates to true if *obj* is an instance of :ctype:`NpyNA`.

.. cfunction:: PyArray_Descr* NpyNA_GetDType(NpyNA* na)

    Returns the *dtype* field of the NA object, which is NULL when
    the NA has no dtype.  Does not raise an error.

.. cfunction:: npy_bool NpyNA_IsMultiNA(NpyNA* na)

    Returns true if the NA has a multi-NA payload, false otherwise.

.. cfunction:: int NpyNA_GetPayload(NpyNA* na)

    Gets the multi-NA payload of the NA, or 0 if *na* doesn't have
    a multi-NA payload.

.. cfunction:: NpyNA* NpyNA_FromObject(PyObject* obj, int suppress_error)

    If *obj* represents an object which is NA, for example if it
    is an :ctype:`NpyNA`, or a zero-dimensional NA-masked array with
    its value hidden by the mask, returns a new reference to an
    :ctype:`NpyNA` object representing *obj*. Otherwise returns
    NULL.

    If *suppress_error* is true, this function doesn't raise an exception
    when the input isn't NA and it returns NULL, otherwise it does.

.. cfunction:: NpyNA* NpyNA_FromDTypeAndPayload(PyArray_Descr *dtype, int multina, int payload)


    Constructs a new :ctype:`NpyNA` instance with the specified *dtype*
    and *payload*. For an NA with no dtype, provide NULL in *dtype*.
    
    Until multi-NA is implemented, just pass 0 for both *multina*
    and *payload*.

NA Mask Functions
-----------------

A mask dtype can be one of three different possibilities. It can
be :cdata:`NPY_BOOL`, :cdata:`NPY_MASK`, or a struct dtype whose
fields are all mask dtypes.

A mask of :cdata:`NPY_BOOL` can just indicate True, with underlying
value 1, for an element that is exposed, and False, with underlying
value 0, for an element that is hidden.

A mask of :cdata:`NPY_MASK` can additionally carry a payload which
is a value from 0 to 127. This allows for missing data implementations
based on such masks to support multiple reasons for data being missing.

A mask of a struct dtype can only pair up with another struct dtype
with the same field names. In this way, each field of the mask controls
the masking for the corresponding field in the associated data array.

Inline functions to work with masks are as follows.

.. cfunction:: npy_bool NpyMaskValue_IsExposed(npy_mask mask)

    Returns true if the data element corresponding to the mask element
    can be modified, false if not.

.. cfunction:: npy_uint8 NpyMaskValue_GetPayload(npy_mask mask)

    Returns the payload contained in the mask. The return value
    is between 0 and 127.

.. cfunction:: npy_mask NpyMaskValue_Create(npy_bool exposed, npy_int8 payload)

    Creates a mask from a flag indicating whether the element is exposed
    or not and a payload value.

NA Mask Array Functions
-----------------------

.. cfunction:: int PyArray_AllocateMaskNA(PyArrayObject *arr, npy_bool ownmaskna, npy_bool multina, npy_mask defaultmask)

    Allocates an NA mask for the array *arr* if necessary. If *ownmaskna*
    if false, it only allocates an NA mask if none exists, but if
    *ownmaskna* is true, it also allocates one if the NA mask is a view
    into another array's NA mask. Here are the two most common usage
    patterns::

        /* Use this to make sure 'arr' has an NA mask */
        if (PyArray_AllocateMaskNA(arr, 0, 0, 1) < 0) {
            return NULL;
        }

        /* Use this to make sure 'arr' owns an NA mask */
        if (PyArray_AllocateMaskNA(arr, 1, 0, 1) < 0) {
            return NULL;
        }

    The parameter *multina* is provided for future expansion, when
    mult-NA support is added to NumPy. This will affect the dtype of
    the NA mask, which currently must be always NPY_BOOL, but will be
    NPY_MASK for arrays multi-NA when this is implemented.

    When a new NA mask is allocated, and the mask needs to be filled,
    it uses the value *defaultmask*. In nearly all cases, this should be set
    to 1, indicating that the elements are exposed. If a mask is allocated
    just because of *ownmaskna*, the existing mask values are copied
    into the newly allocated mask.

    This function returns 0 for success, -1 for failure.

.. cfunction:: npy_bool PyArray_HasNASupport(PyArrayObject *arr)

    Returns true if *arr* is an array which supports NA. This function
    exists because the design for adding NA proposed two mechanisms
    for NAs in NumPy, NA masks and NA bitpatterns. Currently, just
    NA masks have been implemented, but when NA bitpatterns are implemented
    this would return true for arrays with an NA bitpattern dtype as well.

.. cfunction:: int PyArray_ContainsNA(PyArrayObject *arr, PyArrayObject *wheremask, npy_bool *whichna)

    Checks whether the array *arr* contains any NA values.

    If *wheremask* is non-NULL, it must be an NPY_BOOL mask which can
    broadcast onto *arr*. Whereever the where mask is True, *arr*
    is checked for NA, and whereever it is False, the *arr* value is
    ignored.

    The parameter *whichna* is provided for future expansion to multi-NA
    support. When implemented, this parameter will be a 128 element
    array of npy_bool, with the value True for the NA values that are
    being looked for.

    This function returns 1 when the array contains NA values, 0 when
    it does not, and -1 when a error has occurred.

.. cfunction:: int PyArray_AssignNA(PyArrayObject *arr, NpyNA *na, PyArrayObject *wheremask, npy_bool preservena, npy_bool *preservewhichna)

    Assigns the given *na* value to elements of *arr*.
    
    If *wheremask* is non-NULL, it must be an NPY_BOOL array broadcastable
    onto *arr*, and only elements of *arr* with a corresponding value
    of True in *wheremask* will have *na* assigned.

    The parameters *preservena* and *preservewhichna* are provided for
    future expansion to multi-NA support. With a single NA value, one
    NA cannot be distinguished from another, so preserving NA values
    does not make sense. With multiple NA values, preserving NA values
    becomes an important concept because that implies not overwriting the
    multi-NA payloads. The parameter *preservewhichna* will be a 128 element
    array of npy_bool, indicating which NA payloads to preserve.

    This function returns 0 for success, -1 for failure.

.. cfunction:: int PyArray_AssignMaskNA(PyArrayObject *arr, npy_mask maskvalue, PyArrayObject *wheremask, npy_bool preservena, npy_bool *preservewhichna)

    Assigns the given NA mask *maskvalue* to elements of *arr*.

    If *wheremask* is non-NULL, it must be an NPY_BOOL array broadcastable
    onto *arr*, and only elements of *arr* with a corresponding value
    of True in *wheremask* will have the NA *maskvalue* assigned.

    The parameters *preservena* and *preservewhichna* are provided for
    future expansion to multi-NA support. With a single NA value, one
    NA cannot be distinguished from another, so preserving NA values
    does not make sense. With multiple NA values, preserving NA values
    becomes an important concept because that implies not overwriting the
    multi-NA payloads. The parameter *preservewhichna* will be a 128 element
    array of npy_bool, indicating which NA payloads to preserve.

    This function returns 0 for success, -1 for failure.

