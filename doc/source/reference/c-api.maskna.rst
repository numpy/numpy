Array NA Mask API
==================

.. sectionauthor:: Mark Wiebe

.. index::
   pair: maskna; C-API
   pair: C-API; maskna

.. versionadded:: 1.7

NA Masks in Arrays
------------------

NumPy supports the idea of NA (Not Available) missing values in its arrays.
In the design document leading up to the implementation, two mechanisms
for this were proposed, NA masks and NA bitpatterns. NA masks have been
implemented as the first representation of these values. This mechanism
supports working with NA values similar to the approach taking in the R
project, while when combined with views, allows one to temporarily mark
elements as NA since the mask is independent of the raw array data.

The C API has been updated with mechanisms to allow NumPy extensions
to work with these masks, and this document provides some examples and
reference for the NA mask-related functions.

The NA Object
-------------

The main *numpy* namespace in Python has a new object called *NA*.
This is an instance of :ctype:`NpyNA *`, which is a Python object
representing an NA value. This object is analogous to the NumPy
scalars, and is returned by :cfunc:`PyArray_Return` instead of
a scalar where appropriate.

The global *numpy.NA* object is accessible from C as :cdata:`Npy_NA`.
This is an NA value with no data type or multi-NA payload. Use it
just as you would Py_None, except use :cfunc:`NpyNA_Check` to
see if an object is an :ctype:`NpyNA *`, because :cdata:`Npy_NA` isn't
the only instance of NA possible.

If you want to see whether a general PyObject* is NA, you should
use the API function :cfunc:`NpyNA_FromObject` with *suppress_error*
set to true. If this returns NULL, the object is not an NA, and if
it returns an NpyNA instance, the object is NA and you can then
access its *dtype* and *payload* fields as needed.

To make new :ctype:`NpyNA *` objects, use
:cfunc:`NpyNA_FromDTypeAndPayload`, and the functions
:cfunc:`NpyNA_GetDType`, :cfunc:`NpyNA_IsMultiNA`, and
:cfunc:`NpyNA_GetPayload` provide access to the data members.

Example NA-Masked Operation in C
--------------------------------

NA Object Data Type
-------------------

.. ctype:: NpyNA *

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

    Evaluates to true if *obj* is an instance of :ctype:`NpyNA *`.

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
    is an :ctype:`NpyNA *`, or a zero-dimensional NA-masked array with
    its value hidden by the mask, returns a new reference to an
    :ctype:`NpyNA *` object representing *obj*. Otherwise returns
    NULL.

    If *suppress_error* is true, this function doesn't raise an exception
    when the input isn't NA and it returns NULL, otherwise it does.

.. cfunction:: NpyNA* NpyNA_FromDTypeAndPayload(PyArray_Descr *dtype, int multina, int payload)


    Constructs a new :ctype:`NpyNA *` instance with the specified *dtype*
    and *payload*. For an NA with no dtype, provide NULL in *dtype*.

    Until multi-NA is implemented, just pass 0 for both *multina*
    and *payload*.

NA Mask Functions
-----------------

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
