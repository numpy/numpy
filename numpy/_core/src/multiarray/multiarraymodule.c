/*
  Python Multiarray Module -- A useful collection of functions for creating and
  using ndarrays

  Original file
  Copyright (c) 1995, 1996, 1997 Jim Hugunin, hugunin@mit.edu

  Modified for numpy in 2005

  Travis E. Oliphant
  oliphant@ee.byu.edu
  Brigham Young University
*/
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _UMATHMODULE
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include <numpy/npy_common.h>
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "multiarraymodule.h"
#include "numpy/npy_math.h"
#include "npy_argparse.h"
#include "npy_config.h"
#include "npy_pycompat.h"
#include "npy_import.h"
#include "npy_static_data.h"
#include "convert_datatype.h"
#include "legacy_dtype_implementation.h"

NPY_NO_EXPORT int NPY_NUMUSERTYPES = 0;

/* Internal APIs */
#include "alloc.h"
#include "abstractdtypes.h"
#include "array_coercion.h"
#include "arrayfunction_override.h"
#include "arraytypes.h"
#include "arrayobject.h"
#include "array_converter.h"
#include "hashdescr.h"
#include "descriptor.h"
#include "dragon4.h"
#include "flagsobject.h"
#include "calculation.h"
#include "number.h"
#include "scalartypes.h"
#include "convert_datatype.h"
#include "conversion_utils.h"
#include "nditer_pywrap.h"
#define NPY_ITERATOR_IMPLEMENTATION_CODE
#include "nditer_impl.h"
#include "methods.h"
#include "_datetime.h"
#include "datetime_strings.h"
#include "datetime_busday.h"
#include "datetime_busdaycal.h"
#include "item_selection.h"
#include "shape.h"
#include "ctors.h"
#include "array_assign.h"
#include "common.h"
#include "cblasfuncs.h"
#include "vdot.h"
#include "templ_common.h" /* for npy_mul_sizes_with_overflow */
#include "compiled_base.h"
#include "mem_overlap.h"
#include "convert.h" /* for PyArray_AssignZero */
#include "lowlevel_strided_loops.h"
#include "dtype_transfer.h"
#include "stringdtype/dtype.h"

#include "get_attr_string.h"
#include "public_dtype_api.h"  /* _fill_dtype_api */
#include "textreading/readtext.h"  /* _readtext_from_file_object */

#include "npy_dlpack.h"

#include "umathmodule.h"

#include "unique.h"

/*
 *****************************************************************************
 **                    INCLUDE GENERATED CODE                               **
 *****************************************************************************
 */
/* __ufunc_api.c define is the PyUFunc_API table: */
#include "__ufunc_api.c"

NPY_NO_EXPORT int initscalarmath(PyObject *);
NPY_NO_EXPORT int set_matmul_flags(PyObject *d); /* in ufunc_object.c */
/* From umath/string_ufuncs.cpp/h */
NPY_NO_EXPORT PyObject *
_umath_strings_richcompare(
        PyArrayObject *self, PyArrayObject *other, int cmp_op, int rstrip);


NPY_NO_EXPORT int
get_legacy_print_mode(void) {
    /* Get the C value of the legacy printing mode.
     *
     * It is stored as a Python context variable so we access it via the C
     * API. For simplicity the mode is encoded as an integer where INT_MAX
     * means no legacy mode, and '113'/'121'/'125' means 1.13/1.21/1.25 legacy
     * mode; and 0 maps to INT_MAX. We can upgrade this if we have more
     * complex requirements in the future.
     */
    PyObject *format_options = NULL;
    PyContextVar_Get(npy_static_pydata.format_options, NULL, &format_options);
    if (format_options == NULL) {
        PyErr_SetString(PyExc_SystemError,
                        "NumPy internal error: unable to get format_options "
                        "context variable");
        return -1;
    }
    PyObject *legacy_print_mode = NULL;
    if (PyDict_GetItemRef(format_options, npy_interned_str.legacy,
                          &legacy_print_mode) == -1) {
        return -1;
    }
    Py_DECREF(format_options);
    if (legacy_print_mode == NULL) {
        PyErr_SetString(PyExc_SystemError,
                        "NumPy internal error: unable to get legacy print "
                        "mode");
        return -1;
    }
    Py_ssize_t ret = PyLong_AsSsize_t(legacy_print_mode);
    Py_DECREF(legacy_print_mode);
    if (error_converting(ret)) {
        return -1;
    }
    if (ret > INT_MAX) {
        return INT_MAX;
    }
    return (int)ret;
}


/*NUMPY_API
 * Get Priority from object
 */
NPY_NO_EXPORT double
PyArray_GetPriority(PyObject *obj, double default_)
{
    PyObject *ret;
    double priority = NPY_PRIORITY;

    if (PyArray_CheckExact(obj)) {
        return priority;
    }
    else if (PyArray_CheckAnyScalarExact(obj)) {
        return NPY_SCALAR_PRIORITY;
    }

    if (PyArray_LookupSpecial_OnInstance(
            obj, npy_interned_str.array_priority, &ret) < 0) {
        /* TODO[gh-14801]: propagate crashes during attribute access? */
        PyErr_Clear();
        return default_;
    }
    else if (ret == NULL) {
        return default_;
    }

    priority = PyFloat_AsDouble(ret);
    Py_DECREF(ret);
    if (error_converting(priority)) {
        /* TODO[gh-14801]: propagate crashes for bad priority? */
        PyErr_Clear();
        return default_;
    }
    return priority;
}

/*NUMPY_API
 * Multiply a List of ints
 */
NPY_NO_EXPORT int
PyArray_MultiplyIntList(int const *l1, int n)
{
    int s = 1;

    while (n--) {
        s *= (*l1++);
    }
    return s;
}

/*NUMPY_API
 * Multiply a List
 */
NPY_NO_EXPORT npy_intp
PyArray_MultiplyList(npy_intp const *l1, int n)
{
    npy_intp s = 1;

    while (n--) {
        s *= (*l1++);
    }
    return s;
}

/*NUMPY_API
 * Multiply a List of Non-negative numbers with over-flow detection.
 */
NPY_NO_EXPORT npy_intp
PyArray_OverflowMultiplyList(npy_intp const *l1, int n)
{
    npy_intp prod = 1;
    int i;

    for (i = 0; i < n; i++) {
        npy_intp dim = l1[i];

        if (dim == 0) {
            return 0;
        }
        if (npy_mul_sizes_with_overflow(&prod, prod, dim)) {
            return -1;
        }
    }
    return prod;
}

/*NUMPY_API
 * Produce a pointer into array
 */
NPY_NO_EXPORT void *
PyArray_GetPtr(PyArrayObject *obj, npy_intp const* ind)
{
    int n = PyArray_NDIM(obj);
    npy_intp *strides = PyArray_STRIDES(obj);
    char *dptr = PyArray_DATA(obj);

    while (n--) {
        dptr += (*strides++) * (*ind++);
    }
    return (void *)dptr;
}

/*NUMPY_API
 * Compare Lists
 */
NPY_NO_EXPORT int
PyArray_CompareLists(npy_intp const *l1, npy_intp const *l2, int n)
{
    int i;

    for (i = 0; i < n; i++) {
        if (l1[i] != l2[i]) {
            return 0;
        }
    }
    return 1;
}

/*
 * simulates a C-style 1-3 dimensional array which can be accessed using
 * ptr[i]  or ptr[i][j] or ptr[i][j][k] -- requires pointer allocation
 * for 2-d and 3-d.
 *
 * For 2-d and up, ptr is NOT equivalent to a statically defined
 * 2-d or 3-d array.  In particular, it cannot be passed into a
 * function that requires a true pointer to a fixed-size array.
 */

/*NUMPY_API
 * Simulate a C-array
 * steals a reference to typedescr -- can be NULL
 */
NPY_NO_EXPORT int
PyArray_AsCArray(PyObject **op, void *ptr, npy_intp *dims, int nd,
                 PyArray_Descr* typedescr)
{
    PyArrayObject *ap;
    npy_intp n, m, i, j;
    char **ptr2;
    char ***ptr3;

    if ((nd < 1) || (nd > 3)) {
        PyErr_SetString(PyExc_ValueError,
                        "C arrays of only 1-3 dimensions available");
        Py_XDECREF(typedescr);
        return -1;
    }
    if ((ap = (PyArrayObject*)PyArray_FromAny(*op, typedescr, nd, nd,
                                      NPY_ARRAY_CARRAY, NULL)) == NULL) {
        return -1;
    }
    switch(nd) {
    case 1:
        *((char **)ptr) = PyArray_DATA(ap);
        break;
    case 2:
        n = PyArray_DIMS(ap)[0];
        ptr2 = (char **)PyArray_malloc(n * sizeof(char *));
        if (!ptr2) {
            PyErr_NoMemory();
            return -1;
        }
        for (i = 0; i < n; i++) {
            ptr2[i] = PyArray_BYTES(ap) + i*PyArray_STRIDES(ap)[0];
        }
        *((char ***)ptr) = ptr2;
        break;
    case 3:
        n = PyArray_DIMS(ap)[0];
        m = PyArray_DIMS(ap)[1];
        ptr3 = (char ***)PyArray_malloc(n*(m+1) * sizeof(char *));
        if (!ptr3) {
            PyErr_NoMemory();
            return -1;
        }
        for (i = 0; i < n; i++) {
            ptr3[i] = (char **) &ptr3[n + m * i];
            for (j = 0; j < m; j++) {
                ptr3[i][j] = PyArray_BYTES(ap) + i*PyArray_STRIDES(ap)[0] + j*PyArray_STRIDES(ap)[1];
            }
        }
        *((char ****)ptr) = ptr3;
    }
    if (nd) {
        memcpy(dims, PyArray_DIMS(ap), nd*sizeof(npy_intp));
    }
    *op = (PyObject *)ap;
    return 0;
}


/*NUMPY_API
 * Free pointers created if As2D is called
 */
NPY_NO_EXPORT int
PyArray_Free(PyObject *op, void *ptr)
{
    PyArrayObject *ap = (PyArrayObject *)op;

    if ((PyArray_NDIM(ap) < 1) || (PyArray_NDIM(ap) > 3)) {
        return -1;
    }
    if (PyArray_NDIM(ap) >= 2) {
        PyArray_free(ptr);
    }
    Py_DECREF(ap);
    return 0;
}

/*
 * Get the ndarray subclass with the highest priority
 */
NPY_NO_EXPORT PyTypeObject *
PyArray_GetSubType(int narrays, PyArrayObject **arrays) {
    PyTypeObject *subtype = &PyArray_Type;
    double priority = NPY_PRIORITY;
    int i;

    /* Get the priority subtype for the array */
    for (i = 0; i < narrays; ++i) {
        if (Py_TYPE(arrays[i]) != subtype) {
            double pr = PyArray_GetPriority((PyObject *)(arrays[i]), 0.0);
            if (pr > priority) {
                priority = pr;
                subtype = Py_TYPE(arrays[i]);
            }
        }
    }

    return subtype;
}


/*
 * Concatenates a list of ndarrays.
 */
NPY_NO_EXPORT PyArrayObject *
PyArray_ConcatenateArrays(int narrays, PyArrayObject **arrays, int axis,
                          PyArrayObject* ret, PyArray_Descr *dtype,
                          NPY_CASTING casting)
{
    int iarrays, idim, ndim;
    npy_intp shape[NPY_MAXDIMS];
    PyArrayObject_fields *sliding_view = NULL;

    if (narrays <= 0) {
        PyErr_SetString(PyExc_ValueError,
                        "need at least one array to concatenate");
        return NULL;
    }

    /* All the arrays must have the same 'ndim' */
    ndim = PyArray_NDIM(arrays[0]);

    if (ndim == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "zero-dimensional arrays cannot be concatenated");
        return NULL;
    }

    /* Handle standard Python negative indexing */
    if (check_and_adjust_axis(&axis, ndim) < 0) {
        return NULL;
    }

    /*
     * Figure out the final concatenated shape starting from the first
     * array's shape.
     */
    memcpy(shape, PyArray_SHAPE(arrays[0]), ndim * sizeof(shape[0]));
    for (iarrays = 1; iarrays < narrays; ++iarrays) {
        npy_intp *arr_shape;

        if (PyArray_NDIM(arrays[iarrays]) != ndim) {
            PyErr_Format(PyExc_ValueError,
                         "all the input arrays must have same number of "
                         "dimensions, but the array at index %d has %d "
                         "dimension(s) and the array at index %d has %d "
                         "dimension(s)",
                         0, ndim, iarrays, PyArray_NDIM(arrays[iarrays]));
            return NULL;
        }
        arr_shape = PyArray_SHAPE(arrays[iarrays]);

        for (idim = 0; idim < ndim; ++idim) {
            /* Build up the size of the concatenation axis */
            if (idim == axis) {
                shape[idim] += arr_shape[idim];
            }
            /* Validate that the rest of the dimensions match */
            else if (shape[idim] != arr_shape[idim]) {
                PyErr_Format(PyExc_ValueError,
                             "all the input array dimensions except for the "
                             "concatenation axis must match exactly, but "
                             "along dimension %d, the array at index %d has "
                             "size %d and the array at index %d has size %d",
                             idim, 0, shape[idim], iarrays, arr_shape[idim]);
                return NULL;
            }
        }
    }

    if (ret != NULL) {
        assert(dtype == NULL);
        if (PyArray_NDIM(ret) != ndim) {
            PyErr_SetString(PyExc_ValueError,
                            "Output array has wrong dimensionality");
            return NULL;
        }
        if (!PyArray_CompareLists(shape, PyArray_SHAPE(ret), ndim)) {
            PyErr_SetString(PyExc_ValueError,
                            "Output array is the wrong shape");
            return NULL;
        }
        Py_INCREF(ret);
    }
    else {
        npy_intp s, strides[NPY_MAXDIMS];
        int strideperm[NPY_MAXDIMS];

        /* Get the priority subtype for the array */
        PyTypeObject *subtype = PyArray_GetSubType(narrays, arrays);
        PyArray_Descr *descr = PyArray_FindConcatenationDescriptor(
                narrays, arrays, dtype);
        if (descr == NULL) {
            return NULL;
        }

        /*
         * Figure out the permutation to apply to the strides to match
         * the memory layout of the input arrays, using ambiguity
         * resolution rules matching that of the NpyIter.
         */
        PyArray_CreateMultiSortedStridePerm(narrays, arrays, ndim, strideperm);
        s = descr->elsize;
        for (idim = ndim-1; idim >= 0; --idim) {
            int iperm = strideperm[idim];
            strides[iperm] = s;
            s *= shape[iperm];
        }

        /* Allocate the array for the result. This steals the 'dtype' reference. */
        ret = (PyArrayObject *)PyArray_NewFromDescr_int(
                subtype, descr, ndim, shape, strides, NULL, 0, NULL,
                NULL, _NPY_ARRAY_ALLOW_EMPTY_STRING);
        if (ret == NULL) {
            return NULL;
        }
    }

    /*
     * Create a view which slides through ret for assigning the
     * successive input arrays.
     */
    sliding_view = (PyArrayObject_fields *)PyArray_View(ret,
                                                        NULL, &PyArray_Type);
    if (sliding_view == NULL) {
        Py_DECREF(ret);
        return NULL;
    }
    for (iarrays = 0; iarrays < narrays; ++iarrays) {
        /* Set the dimension to match the input array's */
        sliding_view->dimensions[axis] = PyArray_SHAPE(arrays[iarrays])[axis];

        /* Copy the data for this array */
        if (PyArray_AssignArray((PyArrayObject *)sliding_view, arrays[iarrays],
                            NULL, casting) < 0) {
            Py_DECREF(sliding_view);
            Py_DECREF(ret);
            return NULL;
        }

        /* Slide to the start of the next window */
        sliding_view->data += sliding_view->dimensions[axis] *
                                 sliding_view->strides[axis];
    }

    Py_DECREF(sliding_view);
    return ret;
}

/*
 * Concatenates a list of ndarrays, flattening each in the specified order.
 */
NPY_NO_EXPORT PyArrayObject *
PyArray_ConcatenateFlattenedArrays(int narrays, PyArrayObject **arrays,
                                   NPY_ORDER order, PyArrayObject *ret,
                                   PyArray_Descr *dtype, NPY_CASTING casting,
                                   npy_bool casting_not_passed)
{
    int iarrays;
    npy_intp shape = 0;
    PyArrayObject_fields *sliding_view = NULL;

    if (narrays <= 0) {
        PyErr_SetString(PyExc_ValueError,
                        "need at least one array to concatenate");
        return NULL;
    }

    /*
     * Figure out the final concatenated shape starting from the first
     * array's shape.
     */
    for (iarrays = 0; iarrays < narrays; ++iarrays) {
        shape += PyArray_SIZE(arrays[iarrays]);
        /* Check for overflow */
        if (shape < 0) {
            PyErr_SetString(PyExc_ValueError,
                            "total number of elements "
                            "too large to concatenate");
            return NULL;
        }
    }

    if (ret != NULL) {
        assert(dtype == NULL);
        if (PyArray_NDIM(ret) != 1) {
            PyErr_SetString(PyExc_ValueError,
                            "Output array must be 1D");
            return NULL;
        }
        if (shape != PyArray_SIZE(ret)) {
            PyErr_SetString(PyExc_ValueError,
                            "Output array is the wrong size");
            return NULL;
        }
        Py_INCREF(ret);
    }
    else {
        npy_intp stride;

        /* Get the priority subtype for the array */
        PyTypeObject *subtype = PyArray_GetSubType(narrays, arrays);

        PyArray_Descr *descr = PyArray_FindConcatenationDescriptor(
                narrays, arrays, dtype);
        if (descr == NULL) {
            return NULL;
        }

        stride = descr->elsize;

        /* Allocate the array for the result. This steals the 'dtype' reference. */
        ret = (PyArrayObject *)PyArray_NewFromDescr_int(
                subtype, descr,  1, &shape, &stride, NULL, 0, NULL,
                NULL, _NPY_ARRAY_ALLOW_EMPTY_STRING);
        if (ret == NULL) {
            return NULL;
        }
        assert(PyArray_DESCR(ret) == descr);
    }

    /*
     * Create a view which slides through ret for assigning the
     * successive input arrays.
     */
    sliding_view = (PyArrayObject_fields *)PyArray_View(ret,
                                                        NULL, &PyArray_Type);
    if (sliding_view == NULL) {
        Py_DECREF(ret);
        return NULL;
    }

    for (iarrays = 0; iarrays < narrays; ++iarrays) {
        /* Adjust the window dimensions for this array */
        sliding_view->dimensions[0] = PyArray_SIZE(arrays[iarrays]);

        if (!PyArray_CanCastArrayTo(
                arrays[iarrays], PyArray_DESCR(ret), casting)) {
            npy_set_invalid_cast_error(
                    PyArray_DESCR(arrays[iarrays]), PyArray_DESCR(ret),
                    casting, PyArray_NDIM(arrays[iarrays]) == 0);
            Py_DECREF(sliding_view);
            Py_DECREF(ret);
            return NULL;
        }

        /* Copy the data for this array */
        if (PyArray_CopyAsFlat((PyArrayObject *)sliding_view, arrays[iarrays],
                            order) < 0) {
            Py_DECREF(sliding_view);
            Py_DECREF(ret);
            return NULL;
        }

        /* Slide to the start of the next window */
        sliding_view->data +=
            sliding_view->strides[0] * PyArray_SIZE(arrays[iarrays]);
    }

    Py_DECREF(sliding_view);
    return ret;
}


/**
 * Implementation for np.concatenate
 *
 * @param op Sequence of arrays to concatenate
 * @param axis Axis to concatenate along
 * @param ret output array to fill
 * @param dtype Forced output array dtype (cannot be combined with ret)
 * @param casting Casting mode used
 * @param casting_not_passed Deprecation helper
 */
NPY_NO_EXPORT PyObject *
PyArray_ConcatenateInto(PyObject *op,
        int axis, PyArrayObject *ret, PyArray_Descr *dtype,
        NPY_CASTING casting, npy_bool casting_not_passed)
{
    int iarrays, narrays;
    PyArrayObject **arrays;

    if (!PySequence_Check(op)) {
        PyErr_SetString(PyExc_TypeError,
                        "The first input argument needs to be a sequence");
        return NULL;
    }
    if (ret != NULL && dtype != NULL) {
        PyErr_SetString(PyExc_TypeError,
                "concatenate() only takes `out` or `dtype` as an "
                "argument, but both were provided.");
        return NULL;
    }

    /* Convert the input list into arrays */
    narrays = PySequence_Size(op);
    if (narrays < 0) {
        return NULL;
    }
    arrays = PyArray_malloc(narrays * sizeof(arrays[0]));
    if (arrays == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    for (iarrays = 0; iarrays < narrays; ++iarrays) {
        PyObject *item = PySequence_GetItem(op, iarrays);
        if (item == NULL) {
            narrays = iarrays;
            goto fail;
        }
        arrays[iarrays] = (PyArrayObject *)PyArray_FROM_O(item);
        if (arrays[iarrays] == NULL) {
            Py_DECREF(item);
            narrays = iarrays;
            goto fail;
        }
        npy_mark_tmp_array_if_pyscalar(item, arrays[iarrays], NULL);
        Py_DECREF(item);
    }

    if (axis == NPY_RAVEL_AXIS) {
        ret = PyArray_ConcatenateFlattenedArrays(
                narrays, arrays, NPY_CORDER, ret, dtype,
                casting, casting_not_passed);
    }
    else {
        ret = PyArray_ConcatenateArrays(
                narrays, arrays, axis, ret, dtype, casting);
    }

    for (iarrays = 0; iarrays < narrays; ++iarrays) {
        Py_DECREF(arrays[iarrays]);
    }
    PyArray_free(arrays);

    return (PyObject *)ret;

fail:
    /* 'narrays' was set to how far we got in the conversion */
    for (iarrays = 0; iarrays < narrays; ++iarrays) {
        Py_DECREF(arrays[iarrays]);
    }
    PyArray_free(arrays);

    return NULL;
}

/*NUMPY_API
 * Concatenate
 *
 * Concatenate an arbitrary Python sequence into an array.
 * op is a python object supporting the sequence interface.
 * Its elements will be concatenated together to form a single
 * multidimensional array. If axis is NPY_MAXDIMS or bigger, then
 * each sequence object will be flattened before concatenation
*/
NPY_NO_EXPORT PyObject *
PyArray_Concatenate(PyObject *op, int axis)
{
    /* retain legacy behaviour for casting */
    NPY_CASTING casting;
    if (axis >= NPY_MAXDIMS) {
        casting = NPY_UNSAFE_CASTING;
    }
    else {
        casting = NPY_SAME_KIND_CASTING;
    }
    return PyArray_ConcatenateInto(
            op, axis, NULL, NULL, casting, 0);
}

static int
_signbit_set(PyArrayObject *arr)
{
    static char bitmask = (char) 0x80;
    char *ptr;  /* points to the npy_byte to test */
    char byteorder;
    int elsize;

    elsize = PyArray_ITEMSIZE(arr);
    byteorder = PyArray_DESCR(arr)->byteorder;
    ptr = PyArray_DATA(arr);
    if (elsize > 1 &&
        (byteorder == NPY_LITTLE ||
         (byteorder == NPY_NATIVE &&
          PyArray_ISNBO(NPY_LITTLE)))) {
        ptr += elsize - 1;
    }
    return ((*ptr & bitmask) != 0);
}


/*NUMPY_API
 * ScalarKind
 *
 * Returns the scalar kind of a type number, with an
 * optional tweak based on the scalar value itself.
 * If no scalar is provided, it returns INTPOS_SCALAR
 * for both signed and unsigned integers, otherwise
 * it checks the sign of any signed integer to choose
 * INTNEG_SCALAR when appropriate.
 */
NPY_NO_EXPORT NPY_SCALARKIND
PyArray_ScalarKind(int typenum, PyArrayObject **arr)
{
    NPY_SCALARKIND ret = NPY_NOSCALAR;

    if ((unsigned int)typenum < NPY_NTYPES_LEGACY) {
        ret = _npy_scalar_kinds_table[typenum];
        /* Signed integer types are INTNEG in the table */
        if (ret == NPY_INTNEG_SCALAR) {
            if (!arr || !_signbit_set(*arr)) {
                ret = NPY_INTPOS_SCALAR;
            }
        }
    } else if (PyTypeNum_ISUSERDEF(typenum)) {
        PyArray_Descr* descr = PyArray_DescrFromType(typenum);

        if (PyDataType_GetArrFuncs(descr)->scalarkind) {
            ret = PyDataType_GetArrFuncs(descr)->scalarkind((arr ? *arr : NULL));
        }
        Py_DECREF(descr);
    }

    return ret;
}

/*NUMPY_API
 *
 * Determines whether the data type 'thistype', with
 * scalar kind 'scalar', can be coerced into 'neededtype'.
 */
NPY_NO_EXPORT int
PyArray_CanCoerceScalar(int thistype, int neededtype,
                        NPY_SCALARKIND scalar)
{
    PyArray_Descr* from;
    int *castlist;

    /* If 'thistype' is not a scalar, it must be safely castable */
    if (scalar == NPY_NOSCALAR) {
        return PyArray_CanCastSafely(thistype, neededtype);
    }
    if ((unsigned int)neededtype < NPY_NTYPES_LEGACY) {
        NPY_SCALARKIND neededscalar;

        if (scalar == NPY_OBJECT_SCALAR) {
            return PyArray_CanCastSafely(thistype, neededtype);
        }

        /*
         * The lookup table gives us exactly what we need for
         * this comparison, which PyArray_ScalarKind would not.
         *
         * The rule is that positive scalars can be coerced
         * to a signed ints, but negative scalars cannot be coerced
         * to unsigned ints.
         *   _npy_scalar_kinds_table[int]==NEGINT > POSINT,
         *      so 1 is returned, but
         *   _npy_scalar_kinds_table[uint]==POSINT < NEGINT,
         *      so 0 is returned, as required.
         *
         */
        neededscalar = _npy_scalar_kinds_table[neededtype];
        if (neededscalar >= scalar) {
            return 1;
        }
        if (!PyTypeNum_ISUSERDEF(thistype)) {
            return 0;
        }
    }

    from = PyArray_DescrFromType(thistype);
    if (PyDataType_GetArrFuncs(from)->cancastscalarkindto
        && (castlist = PyDataType_GetArrFuncs(from)->cancastscalarkindto[scalar])) {
        while (*castlist != NPY_NOTYPE) {
            if (*castlist++ == neededtype) {
                Py_DECREF(from);
                return 1;
            }
        }
    }
    Py_DECREF(from);

    return 0;
}

/* Could perhaps be redone to not make contiguous arrays */

/*NUMPY_API
 * Numeric.innerproduct(a,v)
 */
NPY_NO_EXPORT PyObject *
PyArray_InnerProduct(PyObject *op1, PyObject *op2)
{
    PyArrayObject *ap1 = NULL;
    PyArrayObject *ap2 = NULL;
    int typenum;
    PyArray_Descr *typec = NULL;
    PyObject* ap2t = NULL;
    npy_intp dims[NPY_MAXDIMS];
    PyArray_Dims newaxes = {dims, 0};
    int i;
    PyObject* ret = NULL;

    typenum = PyArray_ObjectType(op1, NPY_NOTYPE);
    if (typenum == NPY_NOTYPE) {
        return NULL;
    }
    typenum = PyArray_ObjectType(op2, typenum);
    if (typenum == NPY_NOTYPE) {
        return NULL;
    }

    typec = PyArray_DescrFromType(typenum);
    if (typec == NULL) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError,
                            "Cannot find a common data type.");
        }
        goto fail;
    }

    Py_INCREF(typec);
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, typec, 0, 0,
                                           NPY_ARRAY_ALIGNED, NULL);
    if (ap1 == NULL) {
        Py_DECREF(typec);
        goto fail;
    }
    ap2 = (PyArrayObject *)PyArray_FromAny(op2, typec, 0, 0,
                                           NPY_ARRAY_ALIGNED, NULL);
    if (ap2 == NULL) {
        goto fail;
    }

    newaxes.len = PyArray_NDIM(ap2);
    if ((PyArray_NDIM(ap1) >= 1) && (newaxes.len >= 2)) {
        for (i = 0; i < newaxes.len - 2; i++) {
            dims[i] = (npy_intp)i;
        }
        dims[newaxes.len - 2] = newaxes.len - 1;
        dims[newaxes.len - 1] = newaxes.len - 2;

        ap2t = PyArray_Transpose(ap2, &newaxes);
        if (ap2t == NULL) {
            goto fail;
        }
    }
    else {
        ap2t = (PyObject *)ap2;
        Py_INCREF(ap2);
    }

    ret = PyArray_MatrixProduct2((PyObject *)ap1, ap2t, NULL);
    if (ret == NULL) {
        goto fail;
    }


    Py_DECREF(ap1);
    Py_DECREF(ap2);
    Py_DECREF(ap2t);
    return ret;

fail:
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    Py_XDECREF(ap2t);
    Py_XDECREF(ret);
    return NULL;
}

/*NUMPY_API
 * Numeric.matrixproduct(a,v)
 * just like inner product but does the swapaxes stuff on the fly
 */
NPY_NO_EXPORT PyObject *
PyArray_MatrixProduct(PyObject *op1, PyObject *op2)
{
    return PyArray_MatrixProduct2(op1, op2, NULL);
}

/*NUMPY_API
 * Numeric.matrixproduct2(a,v,out)
 * just like inner product but does the swapaxes stuff on the fly
 */
NPY_NO_EXPORT PyObject *
PyArray_MatrixProduct2(PyObject *op1, PyObject *op2, PyArrayObject* out)
{
    PyArrayObject *ap1, *ap2, *out_buf = NULL, *result = NULL;
    PyArrayIterObject *it1, *it2;
    npy_intp i, j, l;
    int typenum, nd, axis, matchDim;
    npy_intp is1, is2, os;
    char *op;
    npy_intp dimensions[NPY_MAXDIMS];
    PyArray_DotFunc *dot;
    PyArray_Descr *typec = NULL;
    NPY_BEGIN_THREADS_DEF;

    typenum = PyArray_ObjectType(op1, NPY_NOTYPE);
    if (typenum == NPY_NOTYPE) {
        return NULL;
    }
    typenum = PyArray_ObjectType(op2, typenum);
    if (typenum == NPY_NOTYPE) {
        return NULL;
    }

    typec = PyArray_DescrFromType(typenum);
    if (typec == NULL) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError,
                            "Cannot find a common data type.");
        }
        return NULL;
    }

    Py_INCREF(typec);
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, typec, 0, 0,
                                        NPY_ARRAY_ALIGNED, NULL);
    if (ap1 == NULL) {
        Py_DECREF(typec);
        return NULL;
    }
    ap2 = (PyArrayObject *)PyArray_FromAny(op2, typec, 0, 0,
                                        NPY_ARRAY_ALIGNED, NULL);
    if (ap2 == NULL) {
        Py_DECREF(ap1);
        return NULL;
    }

#if defined(HAVE_CBLAS)
    if (PyArray_NDIM(ap1) <= 2 && PyArray_NDIM(ap2) <= 2 &&
            (NPY_DOUBLE == typenum || NPY_CDOUBLE == typenum ||
             NPY_FLOAT == typenum || NPY_CFLOAT == typenum)) {
        return cblas_matrixproduct(typenum, ap1, ap2, out);
    }
#endif

    if (PyArray_NDIM(ap1) == 0 || PyArray_NDIM(ap2) == 0) {
        PyObject *mul_res = PyObject_CallFunctionObjArgs(
                n_ops.multiply, ap1, ap2, out, NULL);
        Py_DECREF(ap1);
        Py_DECREF(ap2);
        return mul_res;
    }
    l = PyArray_DIMS(ap1)[PyArray_NDIM(ap1) - 1];
    if (PyArray_NDIM(ap2) > 1) {
        matchDim = PyArray_NDIM(ap2) - 2;
    }
    else {
        matchDim = 0;
    }
    if (PyArray_DIMS(ap2)[matchDim] != l) {
        dot_alignment_error(ap1, PyArray_NDIM(ap1) - 1, ap2, matchDim);
        goto fail;
    }
    nd = PyArray_NDIM(ap1) + PyArray_NDIM(ap2) - 2;
    if (nd > NPY_MAXDIMS) {
        PyErr_SetString(PyExc_ValueError, "dot: too many dimensions in result");
        goto fail;
    }
    j = 0;
    for (i = 0; i < PyArray_NDIM(ap1) - 1; i++) {
        dimensions[j++] = PyArray_DIMS(ap1)[i];
    }
    for (i = 0; i < PyArray_NDIM(ap2) - 2; i++) {
        dimensions[j++] = PyArray_DIMS(ap2)[i];
    }
    if (PyArray_NDIM(ap2) > 1) {
        dimensions[j++] = PyArray_DIMS(ap2)[PyArray_NDIM(ap2)-1];
    }

    is1 = PyArray_STRIDES(ap1)[PyArray_NDIM(ap1)-1];
    is2 = PyArray_STRIDES(ap2)[matchDim];
    /* Choose which subtype to return */
    out_buf = new_array_for_sum(ap1, ap2, out, nd, dimensions, typenum, &result);
    if (out_buf == NULL) {
        goto fail;
    }
    /* Ensure that multiarray.dot(<Nx0>,<0xM>) -> zeros((N,M)) */
    if (PyArray_SIZE(ap1) == 0 && PyArray_SIZE(ap2) == 0) {
        if (PyArray_AssignZero(out_buf, NULL) < 0) {
            goto fail;
        }
    }

    dot = PyDataType_GetArrFuncs(PyArray_DESCR(out_buf))->dotfunc;
    if (dot == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "dot not available for this type");
        goto fail;
    }

    op = PyArray_DATA(out_buf);
    os = PyArray_ITEMSIZE(out_buf);
    axis = PyArray_NDIM(ap1)-1;
    it1 = (PyArrayIterObject *)
        PyArray_IterAllButAxis((PyObject *)ap1, &axis);
    if (it1 == NULL) {
        goto fail;
    }
    it2 = (PyArrayIterObject *)
        PyArray_IterAllButAxis((PyObject *)ap2, &matchDim);
    if (it2 == NULL) {
        Py_DECREF(it1);
        goto fail;
    }

    npy_clear_floatstatus_barrier((char *) result);
    NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(ap2));
    while (it1->index < it1->size) {
        while (it2->index < it2->size) {
            dot(it1->dataptr, is1, it2->dataptr, is2, op, l, NULL);
            op += os;
            PyArray_ITER_NEXT(it2);
        }
        PyArray_ITER_NEXT(it1);
        PyArray_ITER_RESET(it2);
    }
    NPY_END_THREADS_DESCR(PyArray_DESCR(ap2));
    Py_DECREF(it1);
    Py_DECREF(it2);
    if (PyErr_Occurred()) {
        /* only for OBJECT arrays */
        goto fail;
    }

    int fpes = npy_get_floatstatus_barrier((char *) result);
    if (fpes && PyUFunc_GiveFloatingpointErrors("dot", fpes) < 0) {
        goto fail;
    }
    Py_DECREF(ap1);
    Py_DECREF(ap2);

    /* Trigger possible copy-back into `result` */
    PyArray_ResolveWritebackIfCopy(out_buf);
    Py_DECREF(out_buf);

    return (PyObject *)result;

fail:
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    Py_XDECREF(out_buf);
    Py_XDECREF(result);
    return NULL;
}


/*
 * Implementation which is common between PyArray_Correlate
 * and PyArray_Correlate2.
 *
 * inverted is set to 1 if computed correlate(ap2, ap1), 0 otherwise
 */
static PyArrayObject*
_pyarray_correlate(PyArrayObject *ap1, PyArrayObject *ap2, int typenum,
                   int mode, int *inverted)
{
    PyArrayObject *ret;
    npy_intp length;
    npy_intp i, n1, n2, n, n_left, n_right;
    npy_intp is1, is2, os;
    char *ip1, *ip2, *op;
    PyArray_DotFunc *dot;

    NPY_BEGIN_THREADS_DEF;

    n1 = PyArray_DIMS(ap1)[0];
    n2 = PyArray_DIMS(ap2)[0];
    if (n1 == 0) {
        PyErr_SetString(PyExc_ValueError, "first array argument cannot be empty");
        return NULL;
    }
    if (n2 == 0) {
        PyErr_SetString(PyExc_ValueError, "second array argument cannot be empty");
        return NULL;
    }
    if (n1 < n2) {
        ret = ap1;
        ap1 = ap2;
        ap2 = ret;
        ret = NULL;
        i = n1;
        n1 = n2;
        n2 = i;
        *inverted = 1;
    } else {
        *inverted = 0;
    }

    length = n1;
    n = n2;
    switch(mode) {
    case 0:
        length = length - n + 1;
        n_left = n_right = 0;
        break;
    case 1:
        n_left = (npy_intp)(n/2);
        n_right = n - n_left - 1;
        break;
    case 2:
        n_right = n - 1;
        n_left = n - 1;
        length = length + n - 1;
        break;
    default:
        PyErr_SetString(PyExc_ValueError, "mode must be 0, 1, or 2");
        return NULL;
    }

    /*
     * Need to choose an output array that can hold a sum
     * -- use priority to determine which subtype.
     */
    ret = new_array_for_sum(ap1, ap2, NULL, 1, &length, typenum, NULL);
    if (ret == NULL) {
        return NULL;
    }
    dot = PyDataType_GetArrFuncs(PyArray_DESCR(ret))->dotfunc;
    if (dot == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "function not available for this data type");
        goto clean_ret;
    }

    int needs_pyapi = PyDataType_FLAGCHK(PyArray_DESCR(ret), NPY_NEEDS_PYAPI);
    NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(ret));
    is1 = PyArray_STRIDES(ap1)[0];
    is2 = PyArray_STRIDES(ap2)[0];
    op = PyArray_DATA(ret);
    os = PyArray_ITEMSIZE(ret);
    ip1 = PyArray_DATA(ap1);
    ip2 = PyArray_BYTES(ap2) + n_left*is2;
    n = n - n_left;
    for (i = 0; i < n_left; i++) {
        dot(ip1, is1, ip2, is2, op, n, ret);
        if (needs_pyapi && PyErr_Occurred()) {
            goto done;
        }
        n++;
        ip2 -= is2;
        op += os;
    }
    if (small_correlate(ip1, is1, n1 - n2 + 1, PyArray_TYPE(ap1),
                        ip2, is2, n, PyArray_TYPE(ap2),
                        op, os)) {
        ip1 += is1 * (n1 - n2 + 1);
        op += os * (n1 - n2 + 1);
    }
    else {
        for (i = 0; i < (n1 - n2 + 1) && (!needs_pyapi || !PyErr_Occurred());
             i++) {
            dot(ip1, is1, ip2, is2, op, n, ret);
            ip1 += is1;
            op += os;
        }
    }
    for (i = 0; i < n_right && (!needs_pyapi || !PyErr_Occurred()); i++) {
        n--;
        dot(ip1, is1, ip2, is2, op, n, ret);
        ip1 += is1;
        op += os;
    }

done:
    NPY_END_THREADS_DESCR(PyArray_DESCR(ret));
    if (PyErr_Occurred()) {
        goto clean_ret;
    }

    return ret;

clean_ret:
    Py_DECREF(ret);
    return NULL;
}

/*
 * Revert a one dimensional array in-place
 *
 * Return 0 on success, other value on failure
 */
static int
_pyarray_revert(PyArrayObject *ret)
{
    npy_intp length = PyArray_DIM(ret, 0);
    npy_intp os = PyArray_ITEMSIZE(ret);
    char *op = PyArray_DATA(ret);
    char *sw1 = op;
    char *sw2;

    if (PyArray_ISNUMBER(ret) && !PyArray_ISCOMPLEX(ret)) {
        /* Optimization for unstructured dtypes */
        PyArray_CopySwapNFunc *copyswapn = PyDataType_GetArrFuncs(PyArray_DESCR(ret))->copyswapn;
        sw2 = op + length * os - 1;
        /* First reverse the whole array byte by byte... */
        while(sw1 < sw2) {
            const char tmp = *sw1;
            *sw1++ = *sw2;
            *sw2-- = tmp;
        }
        /* ...then swap in place every item */
        copyswapn(op, os, NULL, 0, length, 1, NULL);
    }
    else {
        char *tmp = PyArray_malloc(PyArray_ITEMSIZE(ret));
        if (tmp == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        sw2 = op + (length - 1) * os;
        while (sw1 < sw2) {
            memcpy(tmp, sw1, os);
            memcpy(sw1, sw2, os);
            memcpy(sw2, tmp, os);
            sw1 += os;
            sw2 -= os;
        }
        PyArray_free(tmp);
    }

    return 0;
}

/*NUMPY_API
 * correlate(a1,a2,mode)
 *
 * This function computes the usual correlation (correlate(a1, a2) !=
 * correlate(a2, a1), and conjugate the second argument for complex inputs
 */
NPY_NO_EXPORT PyObject *
PyArray_Correlate2(PyObject *op1, PyObject *op2, int mode)
{
    PyArrayObject *ap1, *ap2, *ret = NULL;
    int typenum;
    PyArray_Descr *typec;
    int inverted;
    int st;

    typenum = PyArray_ObjectType(op1, NPY_NOTYPE);
    if (typenum == NPY_NOTYPE) {
        return NULL;
    }
    typenum = PyArray_ObjectType(op2, typenum);
    if (typenum == NPY_NOTYPE) {
        return NULL;
    }

    typec = PyArray_DescrFromType(typenum);
    Py_INCREF(typec);
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, typec, 1, 1,
                                        NPY_ARRAY_DEFAULT, NULL);
    if (ap1 == NULL) {
        Py_DECREF(typec);
        return NULL;
    }
    ap2 = (PyArrayObject *)PyArray_FromAny(op2, typec, 1, 1,
                                        NPY_ARRAY_DEFAULT, NULL);
    if (ap2 == NULL) {
        goto clean_ap1;
    }

    if (PyArray_ISCOMPLEX(ap2)) {
        PyArrayObject *cap2;
        cap2 = (PyArrayObject *)PyArray_Conjugate(ap2, NULL);
        if (cap2 == NULL) {
            goto clean_ap2;
        }
        Py_DECREF(ap2);
        ap2 = cap2;
    }

    ret = _pyarray_correlate(ap1, ap2, typenum, mode, &inverted);
    if (ret == NULL) {
        goto clean_ap2;
    }

    /*
     * If we inverted input orders, we need to reverse the output array (i.e.
     * ret = ret[::-1])
     */
    if (inverted) {
        st = _pyarray_revert(ret);
        if (st) {
            goto clean_ret;
        }
    }

    Py_DECREF(ap1);
    Py_DECREF(ap2);
    return (PyObject *)ret;

clean_ret:
    Py_DECREF(ret);
clean_ap2:
    Py_DECREF(ap2);
clean_ap1:
    Py_DECREF(ap1);
    return NULL;
}

/*NUMPY_API
 * Numeric.correlate(a1,a2,mode)
 */
NPY_NO_EXPORT PyObject *
PyArray_Correlate(PyObject *op1, PyObject *op2, int mode)
{
    PyArrayObject *ap1, *ap2, *ret = NULL;
    int typenum;
    int unused;
    PyArray_Descr *typec;

    typenum = PyArray_ObjectType(op1, NPY_NOTYPE);
    if (typenum == NPY_NOTYPE) {
        return NULL;
    }
    typenum = PyArray_ObjectType(op2, typenum);
    if (typenum == NPY_NOTYPE) {
        return NULL;
    }

    typec = PyArray_DescrFromType(typenum);
    Py_INCREF(typec);
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, typec, 1, 1,
                                            NPY_ARRAY_DEFAULT, NULL);
    if (ap1 == NULL) {
        Py_DECREF(typec);
        return NULL;
    }
    ap2 = (PyArrayObject *)PyArray_FromAny(op2, typec, 1, 1,
                                           NPY_ARRAY_DEFAULT, NULL);
    if (ap2 == NULL) {
        goto fail;
    }

    ret = _pyarray_correlate(ap1, ap2, typenum, mode, &unused);
    if (ret == NULL) {
        goto fail;
    }
    Py_DECREF(ap1);
    Py_DECREF(ap2);
    return (PyObject *)ret;

fail:
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    Py_XDECREF(ret);
    return NULL;
}

static PyObject *
array_putmask(PyObject *NPY_UNUSED(module), PyObject *const *args,
                Py_ssize_t len_args, PyObject *kwnames )
{
    PyObject *mask, *values;
    PyObject *array;

    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("putmask", args, len_args, kwnames,
            "", NULL, &array,
            "mask", NULL, &mask,
            "values", NULL, &values,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }
    if (!PyArray_Check(array)) {
        PyErr_SetString(PyExc_TypeError,
                "argument a of putmask must be a numpy array");
    }

    return PyArray_PutMask((PyArrayObject *)array, values, mask);
}


/*NUMPY_API
 *
 * This function returns true if the two typecodes are
 * equivalent (same basic kind and same itemsize).
 */
NPY_NO_EXPORT unsigned char
PyArray_EquivTypes(PyArray_Descr *type1, PyArray_Descr *type2)
{
    if (type1 == type2) {
        return 1;
    }

    /*
     * Do not use PyArray_CanCastTypeTo because it supports legacy flexible
     * dtypes as input.
     */
    npy_intp view_offset;
    NPY_CASTING safety = PyArray_GetCastInfo(type1, type2, NULL, &view_offset);
    if (safety < 0) {
        PyErr_Clear();
        return 0;
    }
    /* If casting is "no casting" this dtypes are considered equivalent. */
    return PyArray_MinCastSafety(safety, NPY_NO_CASTING) == NPY_NO_CASTING;
}


/*NUMPY_API*/
NPY_NO_EXPORT unsigned char
PyArray_EquivTypenums(int typenum1, int typenum2)
{
    PyArray_Descr *d1, *d2;
    npy_bool ret;

    if (typenum1 == typenum2) {
        return NPY_SUCCEED;
    }

    d1 = PyArray_DescrFromType(typenum1);
    d2 = PyArray_DescrFromType(typenum2);
    ret = PyArray_EquivTypes(d1, d2);
    Py_DECREF(d1);
    Py_DECREF(d2);
    return ret;
}

/*** END C-API FUNCTIONS **/
/*
 * NOTE: The order specific stride setting is not necessary to preserve
 *       contiguity and could be removed.  However, this way the resulting
 *       strides strides look better for fortran order inputs.
 */
static NPY_STEALS_REF_TO_ARG(1) PyObject *
_prepend_ones(PyArrayObject *arr, int nd, int ndmin, NPY_ORDER order)
{
    npy_intp newdims[NPY_MAXDIMS];
    npy_intp newstrides[NPY_MAXDIMS];
    npy_intp newstride;
    int i, k, num;
    PyObject *ret;
    PyArray_Descr *dtype;

    if (order == NPY_FORTRANORDER || PyArray_ISFORTRAN(arr) || PyArray_NDIM(arr) == 0) {
        newstride = PyArray_ITEMSIZE(arr);
    }
    else {
        newstride = PyArray_STRIDES(arr)[0] * PyArray_DIMS(arr)[0];
    }

    num = ndmin - nd;
    for (i = 0; i < num; i++) {
        newdims[i] = 1;
        newstrides[i] = newstride;
    }
    for (i = num; i < ndmin; i++) {
        k = i - num;
        newdims[i] = PyArray_DIMS(arr)[k];
        newstrides[i] = PyArray_STRIDES(arr)[k];
    }
    dtype = PyArray_DESCR(arr);
    Py_INCREF(dtype);
    ret = PyArray_NewFromDescrAndBase(
            Py_TYPE(arr), dtype,
            ndmin, newdims, newstrides, PyArray_DATA(arr),
            PyArray_FLAGS(arr), (PyObject *)arr, (PyObject *)arr);
    Py_DECREF(arr);

    return ret;
}

#define STRIDING_OK(op, order) \
                ((order) == NPY_ANYORDER || \
                 (order) == NPY_KEEPORDER || \
                 ((order) == NPY_CORDER && PyArray_IS_C_CONTIGUOUS(op)) || \
                 ((order) == NPY_FORTRANORDER && PyArray_IS_F_CONTIGUOUS(op)))


static inline PyObject *
_array_fromobject_generic(
        PyObject *op, PyArray_Descr *in_descr, PyArray_DTypeMeta *in_DType,
        NPY_COPYMODE copy, NPY_ORDER order, npy_bool subok, int ndmin)
{
    PyArrayObject *oparr = NULL, *ret = NULL;
    PyArray_Descr *oldtype = NULL;
    int nd, flags = 0;

    /* Hold on to `in_descr` as `dtype`, since we may also set it below. */
    Py_XINCREF(in_descr);
    PyArray_Descr *dtype = in_descr;

    if (ndmin > NPY_MAXDIMS) {
        PyErr_Format(PyExc_ValueError,
                "ndmin bigger than allowable number of dimensions "
                "NPY_MAXDIMS (=%d)", NPY_MAXDIMS);
        goto finish;
    }
    /* fast exit if simple call */
    if (PyArray_CheckExact(op) || (subok && PyArray_Check(op))) {
        oparr = (PyArrayObject *)op;

        if (dtype == NULL && in_DType == NULL) {
            /*
             * User did not ask for a specific dtype instance or class. So
             * we can return either self or a copy.
             */
            if (copy != NPY_COPY_ALWAYS && STRIDING_OK(oparr, order)) {
                ret = oparr;
                Py_INCREF(ret);
                goto finish;
            }
            else {
                if (copy == NPY_COPY_NEVER) {
                    PyErr_SetString(PyExc_ValueError, npy_no_copy_err_msg);
                    goto finish;
                }
                ret = (PyArrayObject *)PyArray_NewCopy(oparr, order);
                goto finish;
            }
        }
        else if (dtype == NULL) {
            /*
             * If the user passed a DType class but not a dtype instance,
             * we must use `PyArray_AdaptDescriptorToArray` to find the
             * correct dtype instance.
             * Even if the fast-path doesn't work we will use this.
             */
            dtype = PyArray_AdaptDescriptorToArray(oparr, in_DType, NULL);
            if (dtype == NULL) {
                goto finish;
            }
        }

        /* One more chance for faster exit if user specified the dtype. */
        oldtype = PyArray_DESCR(oparr);
        npy_intp view_offset;
        npy_intp is_safe = PyArray_SafeCast(oldtype, dtype, &view_offset, NPY_NO_CASTING, 1);
        npy_intp view_safe = (is_safe && (view_offset != NPY_MIN_INTP));
        if (view_safe) {
            if (copy != NPY_COPY_ALWAYS && STRIDING_OK(oparr, order)) {
                if (oldtype == dtype) {
                    Py_INCREF(op);
                    ret = oparr;
                }
                else {
                    /* Create a new PyArrayObject from the caller's
                     * PyArray_Descr. Use the reference `op` as the base
                     * object. */
                    Py_INCREF(dtype);
                    ret = (PyArrayObject *)PyArray_NewFromDescrAndBase(
                            Py_TYPE(op),
                            dtype,
                            PyArray_NDIM(oparr),
                            PyArray_DIMS(oparr),
                            PyArray_STRIDES(oparr),
                            PyArray_DATA(oparr),
                            PyArray_FLAGS(oparr),
                            op,
                            op
                            );
                }
                goto finish;
            }
            else {
                if (copy == NPY_COPY_NEVER) {
                    PyErr_SetString(PyExc_ValueError,
                            "Unable to avoid copy while creating a new array.");
                    goto finish;
                }
                ret = (PyArrayObject *)PyArray_NewCopy(oparr, order);
                if (oldtype == dtype || ret == NULL) {
                    goto finish;
                }
                Py_INCREF(oldtype);
                Py_DECREF(PyArray_DESCR(ret));
                ((PyArrayObject_fields *)ret)->descr = oldtype;
                goto finish;
            }
        }
    }

    if (copy == NPY_COPY_ALWAYS) {
        flags = NPY_ARRAY_ENSURECOPY;
    }
    else if (copy == NPY_COPY_NEVER) {
        flags = NPY_ARRAY_ENSURENOCOPY;
    }
    if (order == NPY_CORDER) {
        flags |= NPY_ARRAY_C_CONTIGUOUS;
    }
    else if ((order == NPY_FORTRANORDER)
                 /* order == NPY_ANYORDER && */
                 || (PyArray_Check(op) &&
                     PyArray_ISFORTRAN((PyArrayObject *)op))) {
        flags |= NPY_ARRAY_F_CONTIGUOUS;
    }
    if (!subok) {
        flags |= NPY_ARRAY_ENSUREARRAY;
    }

    flags |= NPY_ARRAY_FORCECAST;

    ret = (PyArrayObject *)PyArray_CheckFromAny_int(
            op, dtype, in_DType, 0, 0, flags, NULL);

finish:
    Py_XDECREF(dtype);

    if (ret == NULL) {
        return NULL;
    }

    nd = PyArray_NDIM(ret);
    if (nd >= ndmin) {
        return (PyObject *)ret;
    }
    /*
     * create a new array from the same data with ones in the shape
     * steals a reference to ret
     */
    return _prepend_ones(ret, nd, ndmin, order);
}

#undef STRIDING_OK


static PyObject *
array_array(PyObject *NPY_UNUSED(ignored),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *op;
    npy_bool subok = NPY_FALSE;
    NPY_COPYMODE copy = NPY_COPY_ALWAYS;
    int ndmin = 0;
    npy_dtype_info dt_info = {NULL, NULL};
    NPY_ORDER order = NPY_KEEPORDER;
    PyObject *like = Py_None;
    NPY_PREPARE_ARGPARSER;

    if (len_args != 1 || (kwnames != NULL)) {
        if (npy_parse_arguments("array", args, len_args, kwnames,
                "object", NULL, &op,
                "|dtype", &PyArray_DTypeOrDescrConverterOptional, &dt_info,
                "$copy", &PyArray_CopyConverter, &copy,
                "$order", &PyArray_OrderConverter, &order,
                "$subok", &PyArray_BoolConverter, &subok,
                "$ndmin", &PyArray_PythonPyIntFromInt, &ndmin,
                "$like", NULL, &like,
                NULL, NULL, NULL) < 0) {
            Py_XDECREF(dt_info.descr);
            Py_XDECREF(dt_info.dtype);
            return NULL;
        }
        if (like != Py_None) {
            PyObject *deferred = array_implement_c_array_function_creation(
                    "array", like, NULL, NULL, args, len_args, kwnames);
            if (deferred != Py_NotImplemented) {
                Py_XDECREF(dt_info.descr);
                Py_XDECREF(dt_info.dtype);
                return deferred;
            }
        }
    }
    else {
        /* Fast path for symmetry (we copy by default which is slow) */
        op = args[0];
    }

    PyObject *res = _array_fromobject_generic(
            op, dt_info.descr, dt_info.dtype, copy, order, subok, ndmin);
    Py_XDECREF(dt_info.descr);
    Py_XDECREF(dt_info.dtype);
    return res;
}

static PyObject *
array_asarray(PyObject *NPY_UNUSED(ignored),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *op;
    NPY_COPYMODE copy = NPY_COPY_IF_NEEDED;
    npy_dtype_info dt_info = {NULL, NULL};
    NPY_ORDER order = NPY_KEEPORDER;
    NPY_DEVICE device = NPY_DEVICE_CPU;
    PyObject *like = Py_None;
    NPY_PREPARE_ARGPARSER;

    if (len_args != 1 || (kwnames != NULL)) {
        if (npy_parse_arguments("asarray", args, len_args, kwnames,
                "a", NULL, &op,
                "|dtype", &PyArray_DTypeOrDescrConverterOptional, &dt_info,
                "|order", &PyArray_OrderConverter, &order,
                "$device", &PyArray_DeviceConverterOptional, &device,
                "$copy", &PyArray_CopyConverter, &copy,
                "$like", NULL, &like,
                NULL, NULL, NULL) < 0) {
            Py_XDECREF(dt_info.descr);
            Py_XDECREF(dt_info.dtype);
            return NULL;
        }
        if (like != Py_None) {
            PyObject *deferred = array_implement_c_array_function_creation(
                    "asarray", like, NULL, NULL, args, len_args, kwnames);
            if (deferred != Py_NotImplemented) {
                Py_XDECREF(dt_info.descr);
                Py_XDECREF(dt_info.dtype);
                return deferred;
            }
        }
    }
    else {
        op = args[0];
    }

    PyObject *res = _array_fromobject_generic(
            op, dt_info.descr, dt_info.dtype, copy, order, NPY_FALSE, 0);
    Py_XDECREF(dt_info.descr);
    Py_XDECREF(dt_info.dtype);
    return res;
}

static PyObject *
array_asanyarray(PyObject *NPY_UNUSED(ignored),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *op;
    NPY_COPYMODE copy = NPY_COPY_IF_NEEDED;
    npy_dtype_info dt_info = {NULL, NULL};
    NPY_ORDER order = NPY_KEEPORDER;
    NPY_DEVICE device = NPY_DEVICE_CPU;
    PyObject *like = Py_None;
    NPY_PREPARE_ARGPARSER;

    if (len_args != 1 || (kwnames != NULL)) {
        if (npy_parse_arguments("asanyarray", args, len_args, kwnames,
                "a", NULL, &op,
                "|dtype", &PyArray_DTypeOrDescrConverterOptional, &dt_info,
                "|order", &PyArray_OrderConverter, &order,
                "$device", &PyArray_DeviceConverterOptional, &device,
                "$copy", &PyArray_CopyConverter, &copy,
                "$like", NULL, &like,
                NULL, NULL, NULL) < 0) {
            Py_XDECREF(dt_info.descr);
            Py_XDECREF(dt_info.dtype);
            return NULL;
        }
        if (like != Py_None) {
            PyObject *deferred = array_implement_c_array_function_creation(
                    "asanyarray", like, NULL, NULL, args, len_args, kwnames);
            if (deferred != Py_NotImplemented) {
                Py_XDECREF(dt_info.descr);
                Py_XDECREF(dt_info.dtype);
                return deferred;
            }
        }
    }
    else {
        op = args[0];
    }

    PyObject *res = _array_fromobject_generic(
            op, dt_info.descr, dt_info.dtype, copy, order, NPY_TRUE, 0);
    Py_XDECREF(dt_info.descr);
    Py_XDECREF(dt_info.dtype);
    return res;
}


static PyObject *
array_ascontiguousarray(PyObject *NPY_UNUSED(ignored),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *op;
    npy_dtype_info dt_info = {NULL, NULL};
    PyObject *like = Py_None;
    NPY_PREPARE_ARGPARSER;

    if (len_args != 1 || (kwnames != NULL)) {
        if (npy_parse_arguments("ascontiguousarray", args, len_args, kwnames,
                "a", NULL, &op,
                "|dtype", &PyArray_DTypeOrDescrConverterOptional, &dt_info,
                "$like", NULL, &like,
                NULL, NULL, NULL) < 0) {
            Py_XDECREF(dt_info.descr);
            Py_XDECREF(dt_info.dtype);
            return NULL;
        }
        if (like != Py_None) {
            PyObject *deferred = array_implement_c_array_function_creation(
                    "ascontiguousarray", like, NULL, NULL, args, len_args, kwnames);
            if (deferred != Py_NotImplemented) {
                Py_XDECREF(dt_info.descr);
                Py_XDECREF(dt_info.dtype);
                return deferred;
            }
        }
    }
    else {
        op = args[0];
    }

    PyObject *res = _array_fromobject_generic(
            op, dt_info.descr, dt_info.dtype, NPY_COPY_IF_NEEDED, NPY_CORDER, NPY_FALSE,
            1);
    Py_XDECREF(dt_info.descr);
    Py_XDECREF(dt_info.dtype);
    return res;
}


static PyObject *
array_asfortranarray(PyObject *NPY_UNUSED(ignored),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *op;
    npy_dtype_info dt_info = {NULL, NULL};
    PyObject *like = Py_None;
    NPY_PREPARE_ARGPARSER;

    if (len_args != 1 || (kwnames != NULL)) {
        if (npy_parse_arguments("asfortranarray", args, len_args, kwnames,
                "a", NULL, &op,
                "|dtype", &PyArray_DTypeOrDescrConverterOptional, &dt_info,
                "$like", NULL, &like,
                NULL, NULL, NULL) < 0) {
            Py_XDECREF(dt_info.descr);
            Py_XDECREF(dt_info.dtype);
            return NULL;
        }
        if (like != Py_None) {
            PyObject *deferred = array_implement_c_array_function_creation(
                    "asfortranarray", like, NULL, NULL, args, len_args, kwnames);
            if (deferred != Py_NotImplemented) {
                Py_XDECREF(dt_info.descr);
                Py_XDECREF(dt_info.dtype);
                return deferred;
            }
        }
    }
    else {
        op = args[0];
    }

    PyObject *res = _array_fromobject_generic(
            op, dt_info.descr, dt_info.dtype, NPY_COPY_IF_NEEDED, NPY_FORTRANORDER,
            NPY_FALSE, 1);
    Py_XDECREF(dt_info.descr);
    Py_XDECREF(dt_info.dtype);
    return res;
}


static PyObject *
array_copyto(PyObject *NPY_UNUSED(ignored),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *dst_obj, *src_obj, *wheremask_in = NULL;
    PyArrayObject *src = NULL, *wheremask = NULL;
    NPY_CASTING casting = NPY_SAME_KIND_CASTING;
    NPY_PREPARE_ARGPARSER;

    if (npy_parse_arguments("copyto", args, len_args, kwnames,
            "dst", NULL, &dst_obj,
            "src", NULL, &src_obj,
            "|casting", &PyArray_CastingConverter, &casting,
            "|where", NULL, &wheremask_in,
            NULL, NULL, NULL) < 0) {
        goto fail;
    }

    if (!PyArray_Check(dst_obj)) {
        PyErr_Format(PyExc_TypeError,
            "copyto() argument 1 must be a numpy.ndarray, not %s",
            Py_TYPE(dst_obj)->tp_name);
        goto fail;
    }
    PyArrayObject *dst = (PyArrayObject *)dst_obj;

    src = (PyArrayObject *)PyArray_FromAny(src_obj, NULL, 0, 0, 0, NULL);
    if (src == NULL) {
        goto fail;
    }
    PyArray_DTypeMeta *DType = NPY_DTYPE(PyArray_DESCR(src));
    Py_INCREF(DType);
    if (npy_mark_tmp_array_if_pyscalar(src_obj, src, &DType)) {
        /* The user passed a Python scalar */
        PyArray_Descr *descr = npy_find_descr_for_scalar(
                src_obj, PyArray_DESCR(src), DType,
                NPY_DTYPE(PyArray_DESCR(dst)));
        Py_DECREF(DType);
        if (descr == NULL) {
            goto fail;
        }
        int res = npy_update_operand_for_scalar(&src, src_obj, descr, casting);
        Py_DECREF(descr);
        if (res < 0) {
            goto fail;
        }
    }
    else {
        Py_DECREF(DType);
    }

    if (wheremask_in != NULL) {
        /* Get the boolean where mask */
        PyArray_Descr *descr = PyArray_DescrFromType(NPY_BOOL);
        if (descr == NULL) {
            goto fail;
        }
        wheremask = (PyArrayObject *)PyArray_FromAny(wheremask_in,
                                        descr, 0, 0, 0, NULL);
        if (wheremask == NULL) {
            goto fail;
        }
    }

    if (PyArray_AssignArray(dst, src, wheremask, casting) < 0) {
        goto fail;
    }

    Py_XDECREF(src);
    Py_XDECREF(wheremask);

    Py_RETURN_NONE;

fail:
    Py_XDECREF(src);
    Py_XDECREF(wheremask);
    return NULL;
}

static PyObject *
array_empty(PyObject *NPY_UNUSED(ignored),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    npy_dtype_info dt_info = {NULL, NULL};
    PyArray_Dims shape = {NULL, 0};
    NPY_ORDER order = NPY_CORDER;
    npy_bool is_f_order;
    PyArrayObject *ret = NULL;
    NPY_DEVICE device = NPY_DEVICE_CPU;
    PyObject *like = Py_None;
    NPY_PREPARE_ARGPARSER;

    if (npy_parse_arguments("empty", args, len_args, kwnames,
            "shape", &PyArray_IntpConverter, &shape,
            "|dtype", &PyArray_DTypeOrDescrConverterOptional, &dt_info,
            "|order", &PyArray_OrderConverter, &order,
            "$device", &PyArray_DeviceConverterOptional, &device,
            "$like", NULL, &like,
            NULL, NULL, NULL) < 0) {
        goto fail;
    }

    if (like != Py_None) {
        PyObject *deferred = array_implement_c_array_function_creation(
                "empty", like, NULL, NULL, args, len_args, kwnames);
        if (deferred != Py_NotImplemented) {
            Py_XDECREF(dt_info.descr);
            Py_XDECREF(dt_info.dtype);
            npy_free_cache_dim_obj(shape);
            return deferred;
        }
    }

    switch (order) {
        case NPY_CORDER:
            is_f_order = NPY_FALSE;
            break;
        case NPY_FORTRANORDER:
            is_f_order = NPY_TRUE;
            break;
        default:
            PyErr_SetString(PyExc_ValueError,
                            "only 'C' or 'F' order is permitted");
            goto fail;
    }

    ret = (PyArrayObject *)PyArray_Empty_int(
        shape.len, shape.ptr, dt_info.descr, dt_info.dtype, is_f_order);

fail:
    Py_XDECREF(dt_info.descr);
    Py_XDECREF(dt_info.dtype);
    npy_free_cache_dim_obj(shape);
    return (PyObject *)ret;
}

static PyObject *
array_empty_like(PyObject *NPY_UNUSED(ignored),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyArrayObject *prototype = NULL;
    npy_dtype_info dt_info = {NULL, NULL};
    NPY_ORDER order = NPY_KEEPORDER;
    PyArrayObject *ret = NULL;
    int subok = 1;
    /* -1 is a special value meaning "not specified" */
    PyArray_Dims shape = {NULL, -1};
    NPY_DEVICE device = NPY_DEVICE_CPU;

    NPY_PREPARE_ARGPARSER;

    if (npy_parse_arguments("empty_like", args, len_args, kwnames,
            "prototype", &PyArray_Converter, &prototype,
            "|dtype", &PyArray_DTypeOrDescrConverterOptional, &dt_info,
            "|order", &PyArray_OrderConverter, &order,
            "|subok", &PyArray_PythonPyIntFromInt, &subok,
            "|shape", &PyArray_OptionalIntpConverter, &shape,
            "$device", &PyArray_DeviceConverterOptional, &device,
            NULL, NULL, NULL) < 0) {
        goto fail;
    }
    /* steals the reference to dt_info.descr if it's not NULL */
    if (dt_info.descr != NULL) {
        Py_INCREF(dt_info.descr);
    }
    ret = (PyArrayObject *)PyArray_NewLikeArrayWithShape(
            prototype, order, dt_info.descr, dt_info.dtype,
            shape.len, shape.ptr, subok);
    npy_free_cache_dim_obj(shape);

fail:
    Py_XDECREF(prototype);
    Py_XDECREF(dt_info.dtype);
    Py_XDECREF(dt_info.descr);
    return (PyObject *)ret;
}

/*
 * This function is needed for supporting Pickles of
 * numpy scalar objects.
 */
static PyObject *
array_scalar(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{

    static char *kwlist[] = {"dtype", "obj", NULL};
    PyArray_Descr *typecode;
    PyObject *obj = NULL, *tmpobj = NULL;
    int alloc = 0;
    void *dptr;
    PyObject *ret;
    PyObject *base = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|O:scalar", kwlist,
                &PyArrayDescr_Type, &typecode, &obj)) {
        return NULL;
    }
    if (PyDataType_FLAGCHK(typecode, NPY_LIST_PICKLE)) {
        if (typecode->type_num == NPY_OBJECT) {
            PyErr_SetString(PyExc_TypeError,
                    "Cannot unpickle a scalar with object dtype.");
            return NULL;
        }
        if (typecode->type_num == NPY_VSTRING) {
            // TODO: if we ever add a StringDType scalar, this might need to change
            PyErr_SetString(PyExc_TypeError,
                            "Cannot unpickle a StringDType scalar");
            return NULL;
        }
        /* We store the full array to unpack it here: */
        if (!PyArray_CheckExact(obj)) {
            /* We pickle structured voids as arrays currently */
            PyErr_SetString(PyExc_RuntimeError,
                    "Unpickling NPY_LIST_PICKLE (structured void) scalar "
                    "requires an array.  The pickle file may be corrupted?");
            return NULL;
        }
        if (!PyArray_EquivTypes(PyArray_DESCR((PyArrayObject *)obj), typecode)) {
            PyErr_SetString(PyExc_RuntimeError,
                    "Pickled array is not compatible with requested scalar "
                    "dtype.  The pickle file may be corrupted?");
            return NULL;
        }
        base = obj;
        dptr = PyArray_BYTES((PyArrayObject *)obj);
    }

    else if (PyDataType_FLAGCHK(typecode, NPY_ITEM_IS_POINTER)) {
        if (obj == NULL) {
            obj = Py_None;
        }
        dptr = &obj;
    }
    else {
        if (obj == NULL) {
            if (typecode->elsize == 0) {
                typecode->elsize = 1;
            }
            dptr = PyArray_malloc(typecode->elsize);
            if (dptr == NULL) {
                return PyErr_NoMemory();
            }
            memset(dptr, '\0', typecode->elsize);
            alloc = 1;
        }
        else {
            /* Backward compatibility with Python 2 NumPy pickles */
            if (PyUnicode_Check(obj)) {
                tmpobj = PyUnicode_AsLatin1String(obj);
                obj = tmpobj;
                if (tmpobj == NULL) {
                    /* More informative error message */
                    PyErr_SetString(PyExc_ValueError,
                            "Failed to encode Numpy scalar data string to "
                            "latin1,\npickle.load(a, encoding='latin1') is "
                            "assumed if unpickling.");
                    return NULL;
                }
            }
            if (!PyBytes_Check(obj)) {
                PyErr_SetString(PyExc_TypeError,
                        "initializing object must be a bytes object");
                Py_XDECREF(tmpobj);
                return NULL;
            }
            if (PyBytes_GET_SIZE(obj) < typecode->elsize) {
                PyErr_SetString(PyExc_ValueError,
                        "initialization string is too small");
                Py_XDECREF(tmpobj);
                return NULL;
            }
            dptr = PyBytes_AS_STRING(obj);
        }
    }
    ret = PyArray_Scalar(dptr, typecode, base);

    /* free dptr which contains zeros */
    if (alloc) {
        PyArray_free(dptr);
    }
    Py_XDECREF(tmpobj);
    return ret;
}

static PyObject *
array_zeros(PyObject *NPY_UNUSED(ignored),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    npy_dtype_info dt_info = {NULL, NULL};
    PyArray_Dims shape = {NULL, 0};
    NPY_ORDER order = NPY_CORDER;
    npy_bool is_f_order = NPY_FALSE;
    PyArrayObject *ret = NULL;
    NPY_DEVICE device = NPY_DEVICE_CPU;
    PyObject *like = Py_None;
    NPY_PREPARE_ARGPARSER;

    if (npy_parse_arguments("zeros", args, len_args, kwnames,
            "shape", &PyArray_IntpConverter, &shape,
            "|dtype", &PyArray_DTypeOrDescrConverterOptional, &dt_info,
            "|order", &PyArray_OrderConverter, &order,
            "$device", &PyArray_DeviceConverterOptional, &device,
            "$like", NULL, &like,
            NULL, NULL, NULL) < 0) {
        goto finish;
    }


    if (like != Py_None) {
        PyObject *deferred = array_implement_c_array_function_creation(
                "zeros", like, NULL, NULL, args, len_args, kwnames);
        if (deferred != Py_NotImplemented) {
            Py_XDECREF(dt_info.descr);
            Py_XDECREF(dt_info.dtype);
            npy_free_cache_dim_obj(shape);
            return deferred;
        }
    }

    switch (order) {
        case NPY_CORDER:
            is_f_order = NPY_FALSE;
            break;
        case NPY_FORTRANORDER:
            is_f_order = NPY_TRUE;
            break;
        default:
            PyErr_SetString(PyExc_ValueError,
                            "only 'C' or 'F' order is permitted");
            goto finish;
    }

    ret = (PyArrayObject *)PyArray_Zeros_int(
        shape.len, shape.ptr, dt_info.descr, dt_info.dtype, (int) is_f_order);

finish:
    npy_free_cache_dim_obj(shape);
    Py_XDECREF(dt_info.descr);
    Py_XDECREF(dt_info.dtype);
    return (PyObject *)ret;
}

static PyObject *
array_count_nonzero(PyObject *NPY_UNUSED(self), PyObject *const *args, Py_ssize_t len_args)
{
    PyArrayObject *array;
    npy_intp count;

    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("count_nonzero", args, len_args, NULL,
            "", PyArray_Converter, &array,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }

    count = PyArray_CountNonzero(array);
    Py_DECREF(array);

    if (count == -1) {
        return NULL;
    }

    PyArray_Descr *descr = PyArray_DescrFromType(NPY_INTP);
    if (descr == NULL) {
        return NULL;
    }
    return PyArray_Scalar(&count, descr, NULL);
}

static PyObject *
array_fromstring(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *keywds)
{
    char *data;
    Py_ssize_t nin = -1;
    char *sep = NULL;
    Py_ssize_t s;
    static char *kwlist[] = {"string", "dtype", "count", "sep", "like", NULL};
    PyObject *like = Py_None;
    PyArray_Descr *descr = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, keywds,
                "s#|O&" NPY_SSIZE_T_PYFMT "s$O:fromstring", kwlist,
                &data, &s, PyArray_DescrConverter, &descr, &nin, &sep, &like)) {
        Py_XDECREF(descr);
        return NULL;
    }
    if (like != Py_None) {
        PyObject *deferred = array_implement_c_array_function_creation(
                "fromstring", like, args, keywds, NULL, 0, NULL);
        if (deferred != Py_NotImplemented) {
            Py_XDECREF(descr);
            return deferred;
        }
    }

    /* binary mode, condition copied from PyArray_FromString */
    if (sep == NULL || strlen(sep) == 0) {
        PyErr_SetString(PyExc_ValueError,
            "The binary mode of fromstring is removed, use frombuffer instead");
        return NULL;
    }
    return PyArray_FromString(data, (npy_intp)s, descr, (npy_intp)nin, sep);
}



static PyObject *
array_fromfile(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *keywds)
{
    PyObject *file = NULL, *ret = NULL;
    PyObject *err_type = NULL, *err_value = NULL, *err_traceback = NULL;
    char *sep = "";
    Py_ssize_t nin = -1;
    static char *kwlist[] = {"file", "dtype", "count", "sep", "offset", "like", NULL};
    PyObject *like = Py_None;
    PyArray_Descr *type = NULL;
    int own;
    npy_off_t orig_pos = 0, offset = 0;
    FILE *fp;

    if (!PyArg_ParseTupleAndKeywords(args, keywds,
                "O|O&" NPY_SSIZE_T_PYFMT "s" NPY_OFF_T_PYFMT "$O:fromfile", kwlist,
                &file, PyArray_DescrConverter, &type, &nin, &sep, &offset, &like)) {
        Py_XDECREF(type);
        return NULL;
    }

    if (like != Py_None) {
        PyObject *deferred = array_implement_c_array_function_creation(
                "fromfile", like, args, keywds, NULL, 0, NULL);
        if (deferred != Py_NotImplemented) {
            Py_XDECREF(type);
            return deferred;
        }
    }

    file = NpyPath_PathlikeToFspath(file);
    if (file == NULL) {
        Py_XDECREF(type);
        return NULL;
    }

    if (offset != 0 && strcmp(sep, "") != 0) {
        PyErr_SetString(PyExc_TypeError, "'offset' argument only permitted for binary files");
        Py_XDECREF(type);
        Py_DECREF(file);
        return NULL;
    }
    if (PyBytes_Check(file) || PyUnicode_Check(file)) {
        Py_SETREF(file, npy_PyFile_OpenFile(file, "rb"));
        if (file == NULL) {
            Py_XDECREF(type);
            return NULL;
        }
        own = 1;
    }
    else {
        own = 0;
    }
    fp = npy_PyFile_Dup2(file, "rb", &orig_pos);
    if (fp == NULL) {
        Py_DECREF(file);
        Py_XDECREF(type);
        return NULL;
    }
    if (npy_fseek(fp, offset, SEEK_CUR) != 0) {
        PyErr_SetFromErrno(PyExc_OSError);
        goto cleanup;
    }
    if (type == NULL) {
        type = PyArray_DescrFromType(NPY_DEFAULT_TYPE);
    }
    ret = PyArray_FromFile(fp, type, (npy_intp) nin, sep);

    /* If an exception is thrown in the call to PyArray_FromFile
     * we need to clear it, and restore it later to ensure that
     * we can cleanup the duplicated file descriptor properly.
     */
cleanup:
    PyErr_Fetch(&err_type, &err_value, &err_traceback);
    if (npy_PyFile_DupClose2(file, fp, orig_pos) < 0) {
        npy_PyErr_ChainExceptions(err_type, err_value, err_traceback);
        goto fail;
    }
    if (own && npy_PyFile_CloseFile(file) < 0) {
        npy_PyErr_ChainExceptions(err_type, err_value, err_traceback);
        goto fail;
    }
    PyErr_Restore(err_type, err_value, err_traceback);
    Py_DECREF(file);
    return ret;

fail:
    Py_DECREF(file);
    Py_XDECREF(ret);
    return NULL;
}

static PyObject *
array_fromiter(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *keywds)
{
    PyObject *iter;
    Py_ssize_t nin = -1;
    static char *kwlist[] = {"iter", "dtype", "count", "like", NULL};
    PyObject *like = Py_None;
    PyArray_Descr *descr = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, keywds,
                "OO&|" NPY_SSIZE_T_PYFMT "$O:fromiter", kwlist,
                &iter, PyArray_DescrConverter, &descr, &nin, &like)) {
        Py_XDECREF(descr);
        return NULL;
    }
    if (like != Py_None) {
        PyObject *deferred = array_implement_c_array_function_creation(
                "fromiter", like, args, keywds, NULL, 0, NULL);
        if (deferred != Py_NotImplemented) {
            Py_XDECREF(descr);
            return deferred;
        }
    }

    return PyArray_FromIter(iter, descr, (npy_intp)nin);
}

static PyObject *
array_frombuffer(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *keywds)
{
    PyObject *obj = NULL;
    Py_ssize_t nin = -1, offset = 0;
    static char *kwlist[] = {"buffer", "dtype", "count", "offset", "like", NULL};
    PyObject *like = Py_None;
    PyArray_Descr *type = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, keywds,
                "O|O&" NPY_SSIZE_T_PYFMT NPY_SSIZE_T_PYFMT "$O:frombuffer", kwlist,
                &obj, PyArray_DescrConverter, &type, &nin, &offset, &like)) {
        Py_XDECREF(type);
        return NULL;
    }

    if (like != Py_None) {
        PyObject *deferred = array_implement_c_array_function_creation(
                "frombuffer", like, args, keywds, NULL, 0, NULL);
        if (deferred != Py_NotImplemented) {
            Py_XDECREF(type);
            return deferred;
        }
    }

    if (type == NULL) {
        type = PyArray_DescrFromType(NPY_DEFAULT_TYPE);
    }
    return PyArray_FromBuffer(obj, type, (npy_intp)nin, (npy_intp)offset);
}

static PyObject *
array_concatenate(PyObject *NPY_UNUSED(dummy),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *a0;
    PyObject *out = NULL;
    PyArray_Descr *dtype = NULL;
    NPY_CASTING casting = NPY_SAME_KIND_CASTING;
    PyObject *casting_obj = NULL;
    PyObject *res;
    int axis = 0;

    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("concatenate", args, len_args, kwnames,
            "seq", NULL, &a0,
            "|axis", &PyArray_AxisConverter, &axis,
            "|out", NULL, &out,
            "$dtype", &PyArray_DescrConverter2, &dtype,
            "$casting", NULL, &casting_obj,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }
    int casting_not_passed = 0;
    if (casting_obj == NULL) {
        /*
         * Casting was not passed in, needed for deprecation only.
         * This should be simplified once the deprecation is finished.
         */
        casting_not_passed = 1;
    }
    else if (!PyArray_CastingConverter(casting_obj, &casting)) {
        Py_XDECREF(dtype);
        return NULL;
    }
    if (out != NULL) {
        if (out == Py_None) {
            out = NULL;
        }
        else if (!PyArray_Check(out)) {
            PyErr_SetString(PyExc_TypeError, "'out' must be an array");
            Py_XDECREF(dtype);
            return NULL;
        }
    }
    res = PyArray_ConcatenateInto(a0, axis, (PyArrayObject *)out, dtype,
            casting, casting_not_passed);
    Py_XDECREF(dtype);
    return res;
}

static PyObject *
array_innerproduct(PyObject *NPY_UNUSED(dummy), PyObject *const *args, Py_ssize_t len_args)
{
    PyObject *b0, *a0;

    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("innerproduct", args, len_args, NULL,
            "", NULL, &a0,
            "", NULL, &b0,
            NULL, NULL, NULL) < 0) {
    return NULL;
    }

    return PyArray_Return((PyArrayObject *)PyArray_InnerProduct(a0, b0));
}

static PyObject *
array_matrixproduct(PyObject *NPY_UNUSED(dummy),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *v, *a, *o = NULL;
    PyArrayObject *ret;

    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("dot", args, len_args, kwnames,
            "a", NULL, &a,
            "b", NULL, &v,
            "|out", NULL, &o,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }
    if (o != NULL) {
        if (o == Py_None) {
            o = NULL;
        }
        else if (!PyArray_Check(o)) {
            PyErr_SetString(PyExc_TypeError, "'out' must be an array");
            return NULL;
        }
    }
    ret = (PyArrayObject *)PyArray_MatrixProduct2(a, v, (PyArrayObject *)o);
    return PyArray_Return(ret);
}


static PyObject *
array_vdot(PyObject *NPY_UNUSED(dummy), PyObject *const *args, Py_ssize_t len_args)
{
    int typenum;
    char *ip1, *ip2, *op;
    npy_intp n, stride1, stride2;
    PyObject *op1, *op2;
    npy_intp newdimptr[1] = {-1};
    PyArray_Dims newdims = {newdimptr, 1};
    PyArrayObject *ap1 = NULL, *ap2  = NULL, *ret = NULL;
    PyArray_Descr *type;
    PyArray_DotFunc *vdot;
    NPY_BEGIN_THREADS_DEF;

    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("vdot", args, len_args, NULL,
            "", NULL, &op1,
            "", NULL, &op2,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }

    /*
     * Conjugating dot product using the BLAS for vectors.
     * Flattens both op1 and op2 before dotting.
     */
    typenum = PyArray_ObjectType(op1, NPY_NOTYPE);
    if (typenum == NPY_NOTYPE) {
        return NULL;
    }
    typenum = PyArray_ObjectType(op2, typenum);
    if (typenum == NPY_NOTYPE) {
        return NULL;
    }

    type = PyArray_DescrFromType(typenum);
    Py_INCREF(type);
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, type, 0, 0, 0, NULL);
    if (ap1 == NULL) {
        Py_DECREF(type);
        goto fail;
    }

    op1 = PyArray_Newshape(ap1, &newdims, NPY_CORDER);
    if (op1 == NULL) {
        Py_DECREF(type);
        goto fail;
    }
    Py_DECREF(ap1);
    ap1 = (PyArrayObject *)op1;

    ap2 = (PyArrayObject *)PyArray_FromAny(op2, type, 0, 0, 0, NULL);
    if (ap2 == NULL) {
        goto fail;
    }
    op2 = PyArray_Newshape(ap2, &newdims, NPY_CORDER);
    if (op2 == NULL) {
        goto fail;
    }
    Py_DECREF(ap2);
    ap2 = (PyArrayObject *)op2;

    if (PyArray_DIM(ap2, 0) != PyArray_DIM(ap1, 0)) {
        PyErr_SetString(PyExc_ValueError,
                "vectors have different lengths");
        goto fail;
    }

    /* array scalar output */
    ret = new_array_for_sum(ap1, ap2, NULL, 0, (npy_intp *)NULL, typenum, NULL);
    if (ret == NULL) {
        goto fail;
    }

    n = PyArray_DIM(ap1, 0);
    stride1 = PyArray_STRIDE(ap1, 0);
    stride2 = PyArray_STRIDE(ap2, 0);
    ip1 = PyArray_DATA(ap1);
    ip2 = PyArray_DATA(ap2);
    op = PyArray_DATA(ret);

    switch (typenum) {
        case NPY_CFLOAT:
            vdot = (PyArray_DotFunc *)CFLOAT_vdot;
            break;
        case NPY_CDOUBLE:
            vdot = (PyArray_DotFunc *)CDOUBLE_vdot;
            break;
        case NPY_CLONGDOUBLE:
            vdot = (PyArray_DotFunc *)CLONGDOUBLE_vdot;
            break;
        case NPY_OBJECT:
            vdot = (PyArray_DotFunc *)OBJECT_vdot;
            break;
        default:
            vdot = PyDataType_GetArrFuncs(type)->dotfunc;
            if (vdot == NULL) {
                PyErr_SetString(PyExc_ValueError,
                        "function not available for this data type");
                goto fail;
            }
    }

    if (n < 500) {
        vdot(ip1, stride1, ip2, stride2, op, n, NULL);
    }
    else {
        NPY_BEGIN_THREADS_DESCR(type);
        vdot(ip1, stride1, ip2, stride2, op, n, NULL);
        NPY_END_THREADS_DESCR(type);
    }

    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    return PyArray_Return(ret);
fail:
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    Py_XDECREF(ret);
    return NULL;
}

static int
einsum_sub_op_from_str(
        Py_ssize_t nargs, PyObject *const *args,
        PyObject **str_obj, char **subscripts, PyArrayObject **op)
{
    Py_ssize_t nop = nargs - 1;
    PyObject *subscripts_str;

    if (nop <= 0) {
        PyErr_SetString(PyExc_ValueError,
                        "must specify the einstein sum subscripts string "
                        "and at least one operand");
        return -1;
    }
    else if (nop >= NPY_MAXARGS) {
        PyErr_SetString(PyExc_ValueError, "too many operands");
        return -1;
    }

    /* Get the subscripts string */
    subscripts_str = args[0];
    if (PyUnicode_Check(subscripts_str)) {
        *str_obj = PyUnicode_AsASCIIString(subscripts_str);
        if (*str_obj == NULL) {
            return -1;
        }
        subscripts_str = *str_obj;
    }

    *subscripts = PyBytes_AsString(subscripts_str);
    if (*subscripts == NULL) {
        Py_XDECREF(*str_obj);
        *str_obj = NULL;
        return -1;
    }

    /* Set the operands to NULL */
    for (Py_ssize_t i = 0; i < nop; ++i) {
        op[i] = NULL;
    }

    /* Get the operands */
    for (Py_ssize_t i = 0; i < nop; ++i) {
        op[i] = (PyArrayObject *)PyArray_FROM_OF(args[i+1], NPY_ARRAY_ENSUREARRAY);
        if (op[i] == NULL) {
            goto fail;
        }
    }

    return nop;

fail:
    for (Py_ssize_t i = 0; i < nop; ++i) {
        Py_XDECREF(op[i]);
        op[i] = NULL;
    }

    return -1;
}

/*
 * Converts a list of subscripts to a string.
 *
 * Returns -1 on error, the number of characters placed in subscripts
 * otherwise.
 */
static int
einsum_list_to_subscripts(PyObject *obj, char *subscripts, int subsize)
{
    int ellipsis = 0, subindex = 0;
    npy_intp i, size;
    PyObject *item;

    obj = PySequence_Fast(obj, "the subscripts for each operand must "
                               "be a list or a tuple");
    if (obj == NULL) {
        return -1;
    }
    size = PySequence_Size(obj);

    for (i = 0; i < size; ++i) {
        item = PySequence_Fast_GET_ITEM(obj, i);
        /* Ellipsis */
        if (item == Py_Ellipsis) {
            if (ellipsis) {
                PyErr_SetString(PyExc_ValueError,
                        "each subscripts list may have only one ellipsis");
                Py_DECREF(obj);
                return -1;
            }
            if (subindex + 3 >= subsize) {
                PyErr_SetString(PyExc_ValueError,
                        "subscripts list is too long");
                Py_DECREF(obj);
                return -1;
            }
            subscripts[subindex++] = '.';
            subscripts[subindex++] = '.';
            subscripts[subindex++] = '.';
            ellipsis = 1;
        }
        /* Subscript */
        else {
            npy_intp s = PyArray_PyIntAsIntp(item);
            /* Invalid */
            if (error_converting(s)) {
                PyErr_SetString(PyExc_TypeError,
                        "each subscript must be either an integer "
                        "or an ellipsis");
                Py_DECREF(obj);
                return -1;
            }
            npy_bool bad_input = 0;

            if (subindex + 1 >= subsize) {
                PyErr_SetString(PyExc_ValueError,
                        "subscripts list is too long");
                Py_DECREF(obj);
                return -1;
            }

            if (s < 0) {
                bad_input = 1;
            }
            else if (s < 26) {
                subscripts[subindex++] = 'A' + (char)s;
            }
            else if (s < 2*26) {
                subscripts[subindex++] = 'a' + (char)s - 26;
            }
            else {
                bad_input = 1;
            }

            if (bad_input) {
                PyErr_SetString(PyExc_ValueError,
                        "subscript is not within the valid range [0, 52)");
                Py_DECREF(obj);
                return -1;
            }
        }

    }

    Py_DECREF(obj);

    return subindex;
}

/*
 * Fills in the subscripts, with maximum size subsize, and op,
 * with the values in the tuple 'args'.
 *
 * Returns -1 on error, number of operands placed in op otherwise.
 */
static int
einsum_sub_op_from_lists(Py_ssize_t nargs, PyObject *const *args,
                         char *subscripts, int subsize, PyArrayObject **op)
{
    int subindex = 0;

    Py_ssize_t nop = nargs / 2;

    if (nop == 0) {
        PyErr_SetString(PyExc_ValueError, "must provide at least an "
                        "operand and a subscripts list to einsum");
        return -1;
    }
    else if (nop >= NPY_MAXARGS) {
        PyErr_SetString(PyExc_ValueError, "too many operands");
        return -1;
    }

    /* Set the operands to NULL */
    for (Py_ssize_t i = 0; i < nop; ++i) {
        op[i] = NULL;
    }

    /* Get the operands and build the subscript string */
    for (Py_ssize_t i = 0; i < nop; ++i) {
        /* Comma between the subscripts for each operand */
        if (i != 0) {
            subscripts[subindex++] = ',';
            if (subindex >= subsize) {
                PyErr_SetString(PyExc_ValueError,
                        "subscripts list is too long");
                goto fail;
            }
        }

        op[i] = (PyArrayObject *)PyArray_FROM_OF(args[2*i], NPY_ARRAY_ENSUREARRAY);
        if (op[i] == NULL) {
            goto fail;
        }

        int n = einsum_list_to_subscripts(
                args[2*i + 1], subscripts+subindex, subsize-subindex);
        if (n < 0) {
            goto fail;
        }
        subindex += n;
    }

    /* Add the '->' to the string if provided */
    if (nargs == 2*nop+1) {
        if (subindex + 2 >= subsize) {
            PyErr_SetString(PyExc_ValueError,
                    "subscripts list is too long");
            goto fail;
        }
        subscripts[subindex++] = '-';
        subscripts[subindex++] = '>';

        int n = einsum_list_to_subscripts(
                args[2*nop], subscripts+subindex, subsize-subindex);
        if (n < 0) {
            goto fail;
        }
        subindex += n;
    }

    /* NULL-terminate the subscripts string */
    subscripts[subindex] = '\0';

    return nop;

fail:
    for (Py_ssize_t i = 0; i < nop; ++i) {
        Py_XDECREF(op[i]);
        op[i] = NULL;
    }

    return -1;
}

static PyObject *
array_einsum(PyObject *NPY_UNUSED(dummy),
        PyObject *const *args, Py_ssize_t nargsf, PyObject *kwnames)
{
    char *subscripts = NULL, subscripts_buffer[256];
    PyObject *str_obj = NULL, *str_key_obj = NULL;
    int nop;
    PyArrayObject *op[NPY_MAXARGS];
    NPY_ORDER order = NPY_KEEPORDER;
    NPY_CASTING casting = NPY_SAFE_CASTING;
    PyObject *out_obj = NULL;
    PyArrayObject *out = NULL;
    PyArray_Descr *dtype = NULL;
    PyObject *ret = NULL;
    NPY_PREPARE_ARGPARSER;

    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);

    if (nargs < 1) {
        PyErr_SetString(PyExc_ValueError,
                        "must specify the einstein sum subscripts string "
                        "and at least one operand, or at least one operand "
                        "and its corresponding subscripts list");
        return NULL;
    }

    /* einsum('i,j', a, b), einsum('i,j->ij', a, b) */
    if (PyBytes_Check(args[0]) || PyUnicode_Check(args[0])) {
        nop = einsum_sub_op_from_str(nargs, args, &str_obj, &subscripts, op);
    }
    /* einsum(a, [0], b, [1]), einsum(a, [0], b, [1], [0,1]) */
    else {
        nop = einsum_sub_op_from_lists(nargs, args, subscripts_buffer,
                                       sizeof(subscripts_buffer), op);
        subscripts = subscripts_buffer;
    }
    if (nop <= 0) {
        goto finish;
    }

    /* Get the keyword arguments */
    if (kwnames != NULL) {
        if (npy_parse_arguments("einsum", args+nargs, 0, kwnames,
                "$out", NULL, &out_obj,
                "$order", &PyArray_OrderConverter, &order,
                "$casting", &PyArray_CastingConverter, &casting,
                "$dtype", &PyArray_DescrConverter2, &dtype,
                NULL, NULL, NULL) < 0) {
            goto finish;
        }
        if (out_obj != NULL && !PyArray_Check(out_obj)) {
            PyErr_SetString(PyExc_TypeError,
                        "keyword parameter out must be an "
                        "array for einsum");
            goto finish;
        }
        out = (PyArrayObject *)out_obj;
    }

    ret = (PyObject *)PyArray_EinsteinSum(subscripts, nop, op, dtype,
                                          order, casting, out);

    /* If no output was supplied, possibly convert to a scalar */
    if (ret != NULL && out == NULL) {
        ret = PyArray_Return((PyArrayObject *)ret);
    }

finish:
    for (Py_ssize_t i = 0; i < nop; ++i) {
        Py_XDECREF(op[i]);
    }
    Py_XDECREF(dtype);
    Py_XDECREF(str_obj);
    Py_XDECREF(str_key_obj);
    /* out is a borrowed reference */

    return ret;
}


static PyObject *
array_correlate(PyObject *NPY_UNUSED(dummy),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *shape, *a0;
    int mode = 0;
    NPY_PREPARE_ARGPARSER;

    if (npy_parse_arguments("correlate", args, len_args, kwnames,
            "a", NULL, &a0,
            "v", NULL, &shape,
            "|mode", &PyArray_CorrelatemodeConverter, &mode,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }
    return PyArray_Correlate(a0, shape, mode);
}

static PyObject*
array_correlate2(PyObject *NPY_UNUSED(dummy),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *shape, *a0;
    int mode = 0;
    NPY_PREPARE_ARGPARSER;

    if (npy_parse_arguments("correlate2", args, len_args, kwnames,
            "a", NULL, &a0,
            "v", NULL, &shape,
            "|mode", &PyArray_CorrelatemodeConverter, &mode,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }
    return PyArray_Correlate2(a0, shape, mode);
}

static PyObject *
array_arange(PyObject *NPY_UNUSED(ignored),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *o_start = NULL, *o_stop = NULL, *o_step = NULL, *range=NULL;
    PyArray_Descr *typecode = NULL;
    NPY_DEVICE device = NPY_DEVICE_CPU;
    PyObject *like = Py_None;
    NPY_PREPARE_ARGPARSER;

    if (npy_parse_arguments("arange", args, len_args, kwnames,
            "|start", NULL, &o_start,
            "|stop", NULL, &o_stop,
            "|step", NULL, &o_step,
            "|dtype", &PyArray_DescrConverter2, &typecode,
            "$device", &PyArray_DeviceConverterOptional, &device,
            "$like", NULL, &like,
            NULL, NULL, NULL) < 0) {
        Py_XDECREF(typecode);
        return NULL;
    }
    if (like != Py_None) {
        PyObject *deferred = array_implement_c_array_function_creation(
                "arange", like, NULL, NULL, args, len_args, kwnames);
        if (deferred != Py_NotImplemented) {
            Py_XDECREF(typecode);
            return deferred;
        }
    }

    if (o_stop == NULL) {
        if (len_args == 0){
            PyErr_SetString(PyExc_TypeError,
                "arange() requires stop to be specified.");
            Py_XDECREF(typecode);
            return NULL;
        }
    }
    else if (o_start == NULL) {
        o_start = o_stop;
        o_stop = NULL;
    }

    range = PyArray_ArangeObj(o_start, o_stop, o_step, typecode);
    Py_XDECREF(typecode);

    return range;
}

/*NUMPY_API
 *
 * Included at the very first so not auto-grabbed and thus not labeled.
 */
NPY_NO_EXPORT unsigned int
PyArray_GetNDArrayCVersion(void)
{
    return (unsigned int)NPY_ABI_VERSION;
}

/*NUMPY_API
 * Returns the built-in (at compilation time) C API version
 */
NPY_NO_EXPORT unsigned int
PyArray_GetNDArrayCFeatureVersion(void)
{
    return (unsigned int)NPY_API_VERSION;
}

static PyObject *
array__get_ndarray_c_version(
        PyObject *NPY_UNUSED(dummy), PyObject *NPY_UNUSED(arg))
{
    return PyLong_FromLong( (long) PyArray_GetNDArrayCVersion() );
}

/*NUMPY_API
*/
NPY_NO_EXPORT int
PyArray_GetEndianness(void)
{
    const union {
        npy_uint32 i;
        char c[4];
    } bint = {0x01020304};

    if (bint.c[0] == 1) {
        return NPY_CPU_BIG;
    }
    else if (bint.c[0] == 4) {
        return NPY_CPU_LITTLE;
    }
    else {
        return NPY_CPU_UNKNOWN_ENDIAN;
    }
}

static PyObject *
array__reconstruct(PyObject *NPY_UNUSED(dummy), PyObject *args)
{

    PyObject *ret;
    PyTypeObject *subtype;
    PyArray_Dims shape = {NULL, 0};
    PyArray_Descr *dtype = NULL;

    evil_global_disable_warn_O4O8_flag = 1;

    if (!PyArg_ParseTuple(args, "O!O&O&:_reconstruct",
                &PyType_Type, &subtype,
                PyArray_IntpConverter, &shape,
                PyArray_DescrConverter, &dtype)) {
        goto fail;
    }
    if (!PyType_IsSubtype(subtype, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError,
                "_reconstruct: First argument must be a sub-type of ndarray");
        goto fail;
    }
    ret = PyArray_NewFromDescr(subtype, dtype,
            (int)shape.len, shape.ptr, NULL, NULL, 0, NULL);
    npy_free_cache_dim_obj(shape);

    evil_global_disable_warn_O4O8_flag = 0;

    return ret;

fail:
    evil_global_disable_warn_O4O8_flag = 0;

    Py_XDECREF(dtype);
    npy_free_cache_dim_obj(shape);
    return NULL;
}

static PyObject *
array_set_datetimeparse_function(PyObject *NPY_UNUSED(self),
        PyObject *NPY_UNUSED(args), PyObject *NPY_UNUSED(kwds))
{
    PyErr_SetString(PyExc_RuntimeError, "This function has been removed");
    return NULL;
}

/*
 * inner loop with constant size memcpy arguments
 * this allows the compiler to replace function calls while still handling the
 * alignment requirements of the platform.
 */
#define INNER_WHERE_LOOP(size) \
    do { \
        npy_intp i; \
        for (i = 0; i < n; i++) { \
            if (*csrc) { \
                memcpy(dst, xsrc, size); \
            } \
            else { \
                memcpy(dst, ysrc, size); \
            } \
            dst += size; \
            xsrc += xstride; \
            ysrc += ystride; \
            csrc += cstride; \
        } \
    } while(0)


/*NUMPY_API
 * Where
 */
NPY_NO_EXPORT PyObject *
PyArray_Where(PyObject *condition, PyObject *x, PyObject *y)
{
    PyArrayObject *arr = NULL, *ax = NULL, *ay = NULL;
    PyObject *ret = NULL;
    PyArray_Descr *common_dt = NULL;

    arr = (PyArrayObject *)PyArray_FROM_O(condition);
    if (arr == NULL) {
        return NULL;
    }
    if ((x == NULL) && (y == NULL)) {
        ret = PyArray_Nonzero(arr);
        Py_DECREF(arr);
        return ret;
    }
    if ((x == NULL) || (y == NULL)) {
        Py_DECREF(arr);
        PyErr_SetString(PyExc_ValueError,
                "either both or neither of x and y should be given");
        return NULL;
    }

    NPY_cast_info x_cast_info = {.func = NULL};
    NPY_cast_info y_cast_info = {.func = NULL};

    ax = (PyArrayObject*)PyArray_FROM_O(x);
    if (ax == NULL) {
        goto fail;
    }
    ay = (PyArrayObject*)PyArray_FROM_O(y);
    if (ay == NULL) {
        goto fail;
    }
    npy_mark_tmp_array_if_pyscalar(x, ax, NULL);
    npy_mark_tmp_array_if_pyscalar(y, ay, NULL);

    npy_uint32 flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_BUFFERED |
                        NPY_ITER_REFS_OK | NPY_ITER_ZEROSIZE_OK;
    PyArrayObject * op_in[4] = {
        NULL, arr, ax, ay
    };
    npy_uint32 op_flags[4] = {
        NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE | NPY_ITER_NO_SUBTYPE,
        NPY_ITER_READONLY,
        NPY_ITER_READONLY | NPY_ITER_ALIGNED,
        NPY_ITER_READONLY | NPY_ITER_ALIGNED
    };

    common_dt = PyArray_ResultType(2, &op_in[2], 0, NULL);
    if (common_dt == NULL) {
        goto fail;
    }
    npy_intp itemsize = common_dt->elsize;

    // If x and y don't have references, we ask the iterator to create buffers
    // using the common data type of x and y and then do fast trivial copies
    // in the loop below.
    // Otherwise trivial copies aren't possible and we handle the cast item by item
    // in the loop.
    PyArray_Descr *x_dt, *y_dt;
    int trivial_copy_loop = !PyDataType_REFCHK(common_dt) &&
            ((itemsize == 16) || (itemsize == 8) || (itemsize == 4) ||
             (itemsize == 2) || (itemsize == 1));
    if (trivial_copy_loop) {
        x_dt = common_dt;
        y_dt = common_dt;
    }
    else {
        x_dt = PyArray_DESCR(op_in[2]);
        y_dt = PyArray_DESCR(op_in[3]);
    }
    /* `PyArray_DescrFromType` cannot fail for simple builtin types: */
    PyArray_Descr * op_dt[4] = {common_dt, PyArray_DescrFromType(NPY_BOOL), x_dt, y_dt};

    NpyIter * iter;
    NPY_BEGIN_THREADS_DEF;

    iter =  NpyIter_MultiNew(
            4, op_in, flags, NPY_KEEPORDER, NPY_UNSAFE_CASTING,
            op_flags, op_dt);
    Py_DECREF(op_dt[1]);
    if (iter == NULL) {
        goto fail;
    }

    /* Get the result from the iterator object array */
    ret = (PyObject*)NpyIter_GetOperandArray(iter)[0];
    PyArray_Descr *ret_dt = PyArray_DESCR((PyArrayObject *)ret);

    NPY_ARRAYMETHOD_FLAGS transfer_flags = 0;

    npy_intp x_strides[2] = {x_dt->elsize, itemsize};
    npy_intp y_strides[2] = {y_dt->elsize, itemsize};
    npy_intp one = 1;

    if (!trivial_copy_loop) {
        // The iterator has NPY_ITER_ALIGNED flag so no need to check alignment
        // of the input arrays.
        if (PyArray_GetDTypeTransferFunction(
                    1, x_strides[0], x_strides[1],
                    PyArray_DESCR(op_in[2]), ret_dt, 0,
                    &x_cast_info, &transfer_flags) != NPY_SUCCEED) {
            goto fail;
        }
        if (PyArray_GetDTypeTransferFunction(
                    1, y_strides[0], y_strides[1],
                    PyArray_DESCR(op_in[3]), ret_dt, 0,
                    &y_cast_info, &transfer_flags) != NPY_SUCCEED) {
            goto fail;
        }
    }

    transfer_flags = PyArrayMethod_COMBINED_FLAGS(
        transfer_flags, NpyIter_GetTransferFlags(iter));

    if (!(transfer_flags & NPY_METH_REQUIRES_PYAPI)) {
        NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(iter));
    }

    if (NpyIter_GetIterSize(iter) != 0) {
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        npy_intp * innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);
        char **dataptrarray = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);

        do {
            npy_intp n = (*innersizeptr);
            char * dst = dataptrarray[0];
            char * csrc = dataptrarray[1];
            char * xsrc = dataptrarray[2];
            char * ysrc = dataptrarray[3];

            // the iterator might mutate these pointers,
            // so need to update them every iteration
            npy_intp cstride = strides[1];
            npy_intp xstride = strides[2];
            npy_intp ystride = strides[3];

            /* constant sizes so compiler replaces memcpy */
            if (trivial_copy_loop && itemsize == 16) {
                INNER_WHERE_LOOP(16);
            }
            else if (trivial_copy_loop && itemsize == 8) {
                INNER_WHERE_LOOP(8);
            }
            else if (trivial_copy_loop && itemsize == 4) {
                INNER_WHERE_LOOP(4);
            }
            else if (trivial_copy_loop && itemsize == 2) {
                INNER_WHERE_LOOP(2);
            }
            else if (trivial_copy_loop && itemsize == 1) {
                INNER_WHERE_LOOP(1);
            }
            else {
                npy_intp i;
                for (i = 0; i < n; i++) {
                    if (*csrc) {
                        char *args[2] = {xsrc, dst};

                        if (x_cast_info.func(
                                &x_cast_info.context, args, &one,
                                x_strides, x_cast_info.auxdata) < 0) {
                            goto fail;
                        }
                    }
                    else {
                        char *args[2] = {ysrc, dst};

                        if (y_cast_info.func(
                                &y_cast_info.context, args, &one,
                                y_strides, y_cast_info.auxdata) < 0) {
                            goto fail;
                        }
                    }
                    dst += itemsize;
                    xsrc += xstride;
                    ysrc += ystride;
                    csrc += cstride;
                }
            }
        } while (iternext(iter));
    }

    NPY_END_THREADS;

    Py_INCREF(ret);
    Py_DECREF(arr);
    Py_DECREF(ax);
    Py_DECREF(ay);
    Py_DECREF(common_dt);
    NPY_cast_info_xfree(&x_cast_info);
    NPY_cast_info_xfree(&y_cast_info);

    if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
        Py_DECREF(ret);
        return NULL;
    }

    return ret;

fail:
    Py_DECREF(arr);
    Py_XDECREF(ax);
    Py_XDECREF(ay);
    Py_XDECREF(common_dt);
    NPY_cast_info_xfree(&x_cast_info);
    NPY_cast_info_xfree(&y_cast_info);
    return NULL;
}

#undef INNER_WHERE_LOOP

static PyObject *
array_where(PyObject *NPY_UNUSED(ignored), PyObject *const *args, Py_ssize_t len_args)
{
    PyObject *obj = NULL, *x = NULL, *y = NULL;

    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("where", args, len_args, NULL,
            "", NULL, &obj,
            "|x", NULL, &x,
            "|y", NULL, &y,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }

    return PyArray_Where(obj, x, y);
}

static PyObject *
array_lexsort(PyObject *NPY_UNUSED(ignored), PyObject *const *args, Py_ssize_t len_args,
                             PyObject *kwnames)
{
    int axis = -1;
    PyObject *obj;

    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("lexsort", args, len_args, kwnames,
            "keys", NULL, &obj,
            "|axis", PyArray_PythonPyIntFromInt, &axis,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }
    return PyArray_Return((PyArrayObject *)PyArray_LexSort(obj, axis));
}

static PyObject *
array_can_cast_safely(PyObject *NPY_UNUSED(self),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *from_obj = NULL;
    PyArray_Descr *d1 = NULL;
    PyArray_Descr *d2 = NULL;
    int ret;
    PyObject *retobj = NULL;
    NPY_CASTING casting = NPY_SAFE_CASTING;

    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("can_cast", args, len_args, kwnames,
            "from_", NULL, &from_obj,
            "to", &PyArray_DescrConverter2, &d2,
            "|casting", &PyArray_CastingConverter, &casting,
            NULL, NULL, NULL) < 0) {
        goto finish;
    }
    if (d2 == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "did not understand one of the types; 'None' not accepted");
        goto finish;
    }

    /* If the first parameter is an object or scalar, use CanCastArrayTo */
    if (PyArray_Check(from_obj)) {
        ret = PyArray_CanCastArrayTo((PyArrayObject *)from_obj, d2, casting);
    }
    else if (PyArray_IsScalar(from_obj, Generic)) {
        /*
         * TODO: `PyArray_IsScalar` should not be required for new dtypes.
         *       weak-promotion branch is in practice identical to dtype one.
         */
        PyObject *descr = PyObject_GetAttr(from_obj, npy_interned_str.dtype);
        if (descr == NULL) {
            goto finish;
        }
        if (!PyArray_DescrCheck(descr)) {
            Py_DECREF(descr);
            PyErr_SetString(PyExc_TypeError,
                "numpy_scalar.dtype did not return a dtype instance.");
            goto finish;
        }
        ret = PyArray_CanCastTypeTo((PyArray_Descr *)descr, d2, casting);
        Py_DECREF(descr);
    }
    else if (PyArray_IsPythonNumber(from_obj)) {
        PyErr_SetString(PyExc_TypeError,
                "can_cast() does not support Python ints, floats, and "
                "complex because the result used to depend on the value.\n"
                "This change was part of adopting NEP 50, we may "
                "explicitly allow them again in the future.");
        goto finish;
    }
    /* Otherwise use CanCastTypeTo */
    else {
        if (!PyArray_DescrConverter2(from_obj, &d1) || d1 == NULL) {
            PyErr_SetString(PyExc_TypeError,
                    "did not understand one of the types; 'None' not accepted");
            goto finish;
        }
        ret = PyArray_CanCastTypeTo(d1, d2, casting);
    }

    retobj = ret ? Py_True : Py_False;
    Py_INCREF(retobj);

 finish:
    Py_XDECREF(d1);
    Py_XDECREF(d2);
    return retobj;
}

static PyObject *
array_promote_types(PyObject *NPY_UNUSED(dummy), PyObject *const *args, Py_ssize_t len_args)
{
    PyArray_Descr *d1 = NULL;
    PyArray_Descr *d2 = NULL;
    PyObject *ret = NULL;

    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("promote_types", args, len_args, NULL,
            "", PyArray_DescrConverter2, &d1,
            "", PyArray_DescrConverter2, &d2,
            NULL, NULL, NULL) < 0) {
        goto finish;
    }

    if (d1 == NULL || d2 == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "did not understand one of the types");
        goto finish;
    }

    ret = (PyObject *)PyArray_PromoteTypes(d1, d2);

 finish:
    Py_XDECREF(d1);
    Py_XDECREF(d2);
    return ret;
}

static PyObject *
array_min_scalar_type(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyObject *array_in = NULL;
    PyArrayObject *array;
    PyObject *ret = NULL;

    if (!PyArg_ParseTuple(args, "O:min_scalar_type", &array_in)) {
        return NULL;
    }

    array = (PyArrayObject *)PyArray_FROM_O(array_in);
    if (array == NULL) {
        return NULL;
    }

    ret = (PyObject *)PyArray_MinScalarType(array);
    Py_DECREF(array);
    return ret;
}

static PyObject *
array_result_type(PyObject *NPY_UNUSED(dummy), PyObject *const *args, Py_ssize_t len)
{
    npy_intp i, narr = 0, ndtypes = 0;
    PyObject *ret = NULL;

    if (len == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "at least one array or dtype is required");
        return NULL;
    }

    NPY_ALLOC_WORKSPACE(arr, PyArrayObject *, 2 * 3, 2 * len);
    if (arr == NULL) {
        return NULL;
    }
    PyArray_Descr **dtypes = (PyArray_Descr**)&arr[len];

    PyObject *previous_obj = NULL;

    for (i = 0; i < len; ++i) {
        PyObject *obj = args[i];
        if (obj == previous_obj) {
            continue;
        }

        if (PyArray_Check(obj)) {
            Py_INCREF(obj);
            arr[narr] = (PyArrayObject *)obj;
            ++narr;
        }
        else if (PyArray_IsScalar(obj, Generic) ||
                                    PyArray_IsPythonNumber(obj)) {
            arr[narr] = (PyArrayObject *)PyArray_FROM_O(obj);
            if (arr[narr] == NULL) {
                goto finish;
            }
            /*
             * Mark array if it was a python scalar (we do not need the actual
             * DType here yet, this is figured out inside ResultType.
             */
            npy_mark_tmp_array_if_pyscalar(obj, arr[narr], NULL);
            ++narr;
        }
        else {
            if (!PyArray_DescrConverter(obj, &dtypes[ndtypes])) {
                goto finish;
            }
            ++ndtypes;
        }
    }

    ret = (PyObject *)PyArray_ResultType(narr, arr, ndtypes, dtypes);

finish:
    for (i = 0; i < narr; ++i) {
        Py_DECREF(arr[i]);
    }
    for (i = 0; i < ndtypes; ++i) {
        Py_DECREF(dtypes[i]);
    }
    npy_free_workspace(arr);
    return ret;
}

static PyObject *
array_datetime_data(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyArray_Descr *dtype;
    PyArray_DatetimeMetaData *meta;

    if (!PyArg_ParseTuple(args, "O&:datetime_data",
                PyArray_DescrConverter, &dtype)) {
        return NULL;
    }

    meta = get_datetime_metadata_from_dtype(dtype);
    if (meta == NULL) {
        Py_DECREF(dtype);
        return NULL;
    }

    PyObject *res = convert_datetime_metadata_to_tuple(meta);
    Py_DECREF(dtype);
    return res;
}


static int
trimmode_converter(PyObject *obj, TrimMode *trim)
{
    if (!PyUnicode_Check(obj) || PyUnicode_GetLength(obj) != 1) {
        goto error;
    }
    const char *trimstr = PyUnicode_AsUTF8AndSize(obj, NULL);

    if (trimstr != NULL) {
        if (trimstr[0] == 'k') {
            *trim = TrimMode_None;
        }
        else if (trimstr[0] == '.') {
            *trim = TrimMode_Zeros;
        }
        else if (trimstr[0] ==  '0') {
            *trim = TrimMode_LeaveOneZero;
        }
        else if (trimstr[0] ==  '-') {
            *trim = TrimMode_DptZeros;
        }
        else {
            goto error;
        }
    }
    return NPY_SUCCEED;

error:
    PyErr_Format(PyExc_TypeError,
            "if supplied, trim must be 'k', '.', '0' or '-' found `%100S`",
            obj);
    return NPY_FAIL;
}


/*
 * Prints floating-point scalars using the Dragon4 algorithm, scientific mode.
 * See docstring of `np.format_float_scientific` for description of arguments.
 * The differences is that a value of -1 is valid for pad_left, exp_digits,
 * precision, which is equivalent to `None`.
 */
static PyObject *
dragon4_scientific(PyObject *NPY_UNUSED(dummy),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *obj;
    int precision=-1, pad_left=-1, exp_digits=-1, min_digits=-1;
    DigitMode digit_mode;
    TrimMode trim = TrimMode_None;
    int sign=0, unique=1;
    NPY_PREPARE_ARGPARSER;

    if (npy_parse_arguments("dragon4_scientific", args, len_args, kwnames,
            "x", NULL , &obj,
            "|precision", &PyArray_PythonPyIntFromInt, &precision,
            "|unique", &PyArray_PythonPyIntFromInt, &unique,
            "|sign", &PyArray_PythonPyIntFromInt, &sign,
            "|trim", &trimmode_converter, &trim,
            "|pad_left", &PyArray_PythonPyIntFromInt, &pad_left,
            "|exp_digits", &PyArray_PythonPyIntFromInt, &exp_digits,
            "|min_digits", &PyArray_PythonPyIntFromInt, &min_digits,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }

    digit_mode = unique ? DigitMode_Unique : DigitMode_Exact;

    if (unique == 0 && precision < 0) {
        PyErr_SetString(PyExc_TypeError,
            "in non-unique mode `precision` must be supplied");
        return NULL;
    }

    return Dragon4_Scientific(obj, digit_mode, precision, min_digits, sign, trim,
                              pad_left, exp_digits);
}

/*
 * Prints floating-point scalars using the Dragon4 algorithm, positional mode.
 * See docstring of `np.format_float_positional` for description of arguments.
 * The differences is that a value of -1 is valid for pad_left, pad_right,
 * precision, which is equivalent to `None`.
 */
static PyObject *
dragon4_positional(PyObject *NPY_UNUSED(dummy),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *obj;
    int precision=-1, pad_left=-1, pad_right=-1, min_digits=-1;
    CutoffMode cutoff_mode;
    DigitMode digit_mode;
    TrimMode trim = TrimMode_None;
    int sign=0, unique=1, fractional=0;
    NPY_PREPARE_ARGPARSER;

    if (npy_parse_arguments("dragon4_positional", args, len_args, kwnames,
            "x", NULL , &obj,
            "|precision", &PyArray_PythonPyIntFromInt, &precision,
            "|unique", &PyArray_PythonPyIntFromInt, &unique,
            "|fractional", &PyArray_PythonPyIntFromInt, &fractional,
            "|sign", &PyArray_PythonPyIntFromInt, &sign,
            "|trim", &trimmode_converter, &trim,
            "|pad_left", &PyArray_PythonPyIntFromInt, &pad_left,
            "|pad_right", &PyArray_PythonPyIntFromInt, &pad_right,
            "|min_digits", &PyArray_PythonPyIntFromInt, &min_digits,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }

    digit_mode = unique ? DigitMode_Unique : DigitMode_Exact;
    cutoff_mode = fractional ? CutoffMode_FractionLength :
                               CutoffMode_TotalLength;

    if (unique == 0 && precision < 0) {
        PyErr_SetString(PyExc_TypeError,
            "in non-unique mode `precision` must be supplied");
        return NULL;
    }

    return Dragon4_Positional(obj, digit_mode, cutoff_mode, precision,
                              min_digits, sign, trim, pad_left, pad_right);
}

static PyObject *
format_longfloat(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    PyObject *obj;
    unsigned int precision;
    static char *kwlist[] = {"x", "precision", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OI:format_longfloat", kwlist,
                &obj, &precision)) {
        return NULL;
    }
    if (!PyArray_IsScalar(obj, LongDouble)) {
        PyErr_SetString(PyExc_TypeError,
                "not a longfloat");
        return NULL;
    }
    return Dragon4_Scientific(obj, DigitMode_Unique, precision, -1, 0,
                              TrimMode_LeaveOneZero, -1, -1);
}

/*
 * returns 1 if array is a user-defined string dtype, sets an error and
 * returns 0 otherwise
 */
static int _is_user_defined_string_array(PyArrayObject* array)
{
    if (NPY_DT_is_user_defined(PyArray_DESCR(array))) {
        PyTypeObject* scalar_type = NPY_DTYPE(PyArray_DESCR(array))->scalar_type;
        if (PyType_IsSubtype(scalar_type, &PyBytes_Type) ||
            PyType_IsSubtype(scalar_type, &PyUnicode_Type)) {
            return 1;
        }
        else {
            PyErr_SetString(
                PyExc_TypeError,
                "string comparisons are only allowed for dtypes with a "
                "scalar type that is a subtype of str or bytes.");
            return 0;
        }
    }
    else {
        PyErr_SetString(
            PyExc_TypeError,
            "string operation on non-string array");
        return 0;
    }
}


/*
 * The only purpose of this function is that it allows the "rstrip".
 * From my (@seberg's) perspective, this function should be deprecated
 * and I do not think it matters if it is not particularly fast.
 */
static PyObject *
compare_chararrays(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    PyObject *array;
    PyObject *other;
    PyArrayObject *newarr, *newoth;
    int cmp_op;
    npy_bool rstrip;
    char *cmp_str;
    Py_ssize_t strlength;
    PyObject *res = NULL;
    static char msg[] = "comparison must be '==', '!=', '<', '>', '<=', '>='";
    static char *kwlist[] = {"a1", "a2", "cmp", "rstrip", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOs#O&:compare_chararrays",
                kwlist,
                &array, &other, &cmp_str, &strlength,
                PyArray_BoolConverter, &rstrip)) {
        return NULL;
    }
    if (strlength < 1 || strlength > 2) {
        goto err;
    }
    if (strlength > 1) {
        if (cmp_str[1] != '=') {
            goto err;
        }
        if (cmp_str[0] == '=') {
            cmp_op = Py_EQ;
        }
        else if (cmp_str[0] == '!') {
            cmp_op = Py_NE;
        }
        else if (cmp_str[0] == '<') {
            cmp_op = Py_LE;
        }
        else if (cmp_str[0] == '>') {
            cmp_op = Py_GE;
        }
        else {
            goto err;
        }
    }
    else {
        if (cmp_str[0] == '<') {
            cmp_op = Py_LT;
        }
        else if (cmp_str[0] == '>') {
            cmp_op = Py_GT;
        }
        else {
            goto err;
        }
    }

    newarr = (PyArrayObject *)PyArray_FROM_O(array);
    if (newarr == NULL) {
        return NULL;
    }
    newoth = (PyArrayObject *)PyArray_FROM_O(other);
    if (newoth == NULL) {
        Py_DECREF(newarr);
        return NULL;
    }
    if (PyArray_ISSTRING(newarr) && PyArray_ISSTRING(newoth)) {
        res = _umath_strings_richcompare(newarr, newoth, cmp_op, rstrip != 0);
    }
    else {
        PyErr_SetString(PyExc_TypeError,
                "comparison of non-string arrays");
        Py_DECREF(newarr);
        Py_DECREF(newoth);
        return NULL;
    }
    Py_DECREF(newarr);
    Py_DECREF(newoth);
    return res;

 err:
    PyErr_SetString(PyExc_ValueError, msg);
    return NULL;
}

static PyObject *
_vec_string_with_args(PyArrayObject* char_array, PyArray_Descr* type,
                      PyObject* method, PyObject* args)
{
    PyObject* broadcast_args[NPY_MAXARGS];
    PyArrayMultiIterObject* in_iter = NULL;
    PyArrayObject* result = NULL;
    PyArrayIterObject* out_iter = NULL;
    Py_ssize_t i, n, nargs;

    nargs = PySequence_Size(args) + 1;
    if (nargs == -1 || nargs > NPY_MAXARGS) {
        PyErr_Format(PyExc_ValueError,
                "len(args) must be < %d", NPY_MAXARGS - 1);
        Py_DECREF(type);
        goto err;
    }

    broadcast_args[0] = (PyObject*)char_array;
    for (i = 1; i < nargs; i++) {
        PyObject* item = PySequence_GetItem(args, i-1);
        if (item == NULL) {
            Py_DECREF(type);
            goto err;
        }
        broadcast_args[i] = item;
        Py_DECREF(item);
    }
    in_iter = (PyArrayMultiIterObject*)PyArray_MultiIterFromObjects
        (broadcast_args, nargs, 0);
    if (in_iter == NULL) {
        Py_DECREF(type);
        goto err;
    }
    n = in_iter->numiter;

    result = (PyArrayObject*)PyArray_SimpleNewFromDescr(in_iter->nd,
            in_iter->dimensions, type);
    if (result == NULL) {
        goto err;
    }

    out_iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)result);
    if (out_iter == NULL) {
        goto err;
    }

    while (PyArray_MultiIter_NOTDONE(in_iter)) {
        PyObject* item_result;
        PyObject* args_tuple = PyTuple_New(n);
        if (args_tuple == NULL) {
            goto err;
        }

        for (i = 0; i < n; i++) {
            PyArrayIterObject* it = in_iter->iters[i];
            PyObject* arg = PyArray_ToScalar(PyArray_ITER_DATA(it), it->ao);
            if (arg == NULL) {
                Py_DECREF(args_tuple);
                goto err;
            }
            /* Steals ref to arg */
            PyTuple_SetItem(args_tuple, i, arg);
        }

        item_result = PyObject_CallObject(method, args_tuple);
        Py_DECREF(args_tuple);
        if (item_result == NULL) {
            goto err;
        }

        if (PyArray_SETITEM(result, PyArray_ITER_DATA(out_iter), item_result)) {
            Py_DECREF(item_result);
            PyErr_SetString( PyExc_TypeError,
                    "result array type does not match underlying function");
            goto err;
        }
        Py_DECREF(item_result);

        PyArray_MultiIter_NEXT(in_iter);
        PyArray_ITER_NEXT(out_iter);
    }

    Py_DECREF(in_iter);
    Py_DECREF(out_iter);

    return (PyObject*)result;

 err:
    Py_XDECREF(in_iter);
    Py_XDECREF(out_iter);
    Py_XDECREF(result);

    return 0;
}

static PyObject *
_vec_string_no_args(PyArrayObject* char_array,
                                   PyArray_Descr* type, PyObject* method)
{
    /*
     * This is a faster version of _vec_string_args to use when there
     * are no additional arguments to the string method.  This doesn't
     * require a broadcast iterator (and broadcast iterators don't work
     * with 1 argument anyway).
     */
    PyArrayIterObject* in_iter = NULL;
    PyArrayObject* result = NULL;
    PyArrayIterObject* out_iter = NULL;

    in_iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)char_array);
    if (in_iter == NULL) {
        Py_DECREF(type);
        goto err;
    }

    result = (PyArrayObject*)PyArray_SimpleNewFromDescr(
            PyArray_NDIM(char_array), PyArray_DIMS(char_array), type);
    if (result == NULL) {
        goto err;
    }

    out_iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)result);
    if (out_iter == NULL) {
        goto err;
    }

    while (PyArray_ITER_NOTDONE(in_iter)) {
        PyObject* item_result;
        PyObject* item = PyArray_ToScalar(in_iter->dataptr, in_iter->ao);
        if (item == NULL) {
            goto err;
        }

        item_result = PyObject_CallFunctionObjArgs(method, item, NULL);
        Py_DECREF(item);
        if (item_result == NULL) {
            goto err;
        }

        if (PyArray_SETITEM(result, PyArray_ITER_DATA(out_iter), item_result)) {
            Py_DECREF(item_result);
            PyErr_SetString( PyExc_TypeError,
                "result array type does not match underlying function");
            goto err;
        }
        Py_DECREF(item_result);

        PyArray_ITER_NEXT(in_iter);
        PyArray_ITER_NEXT(out_iter);
    }

    Py_DECREF(in_iter);
    Py_DECREF(out_iter);

    return (PyObject*)result;

 err:
    Py_XDECREF(in_iter);
    Py_XDECREF(out_iter);
    Py_XDECREF(result);

    return 0;
}

static PyObject *
_vec_string(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *NPY_UNUSED(kwds))
{
    PyArrayObject* char_array = NULL;
    PyArray_Descr *type;
    PyObject* method_name;
    PyObject* args_seq = NULL;

    PyObject* method = NULL;
    PyObject* result = NULL;

    if (!PyArg_ParseTuple(args, "O&O&O|O",
                PyArray_Converter, &char_array,
                PyArray_DescrConverter, &type,
                &method_name, &args_seq)) {
        goto err;
    }

    if (PyArray_TYPE(char_array) == NPY_STRING) {
        method = PyObject_GetAttr((PyObject *)&PyBytes_Type, method_name);
    }
    else if (PyArray_TYPE(char_array) == NPY_UNICODE ||
             NPY_DTYPE(PyArray_DTYPE(char_array)) == &PyArray_StringDType) {
        method = PyObject_GetAttr((PyObject *)&PyUnicode_Type, method_name);
    }
    else {
        if (_is_user_defined_string_array(char_array)) {
            PyTypeObject* scalar_type =
                NPY_DTYPE(PyArray_DESCR(char_array))->scalar_type;
            method = PyObject_GetAttr((PyObject*)scalar_type, method_name);
        }
        else {
            Py_DECREF(type);
            goto err;
        }
    }
    if (method == NULL) {
        Py_DECREF(type);
        goto err;
    }

    if (args_seq == NULL
            || (PySequence_Check(args_seq) && PySequence_Size(args_seq) == 0)) {
        result = _vec_string_no_args(char_array, type, method);
    }
    else if (PySequence_Check(args_seq)) {
        result = _vec_string_with_args(char_array, type, method, args_seq);
    }
    else {
        Py_DECREF(type);
        PyErr_SetString(PyExc_TypeError,
                "'args' must be a sequence of arguments");
        goto err;
    }
    if (result == NULL) {
        goto err;
    }

    Py_DECREF(char_array);
    Py_DECREF(method);

    return (PyObject*)result;

 err:
    Py_XDECREF(char_array);
    Py_XDECREF(method);

    return 0;
}


static PyObject *
array_shares_memory_impl(PyObject *args, PyObject *kwds, Py_ssize_t default_max_work,
                         int raise_exceptions)
{
    PyObject * self_obj = NULL;
    PyObject * other_obj = NULL;
    PyArrayObject * self = NULL;
    PyArrayObject * other = NULL;
    PyObject *max_work_obj = NULL;
    static char *kwlist[] = {"self", "other", "max_work", NULL};

    mem_overlap_t result;
    Py_ssize_t max_work;
    NPY_BEGIN_THREADS_DEF;

    max_work = default_max_work;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O:shares_memory_impl", kwlist,
                                     &self_obj, &other_obj, &max_work_obj)) {
        return NULL;
    }

    if (PyArray_Check(self_obj)) {
        self = (PyArrayObject*)self_obj;
        Py_INCREF(self);
    }
    else {
        /* Use FromAny to enable checking overlap for objects exposing array
           interfaces etc. */
        self = (PyArrayObject*)PyArray_FROM_O(self_obj);
        if (self == NULL) {
            goto fail;
        }
    }

    if (PyArray_Check(other_obj)) {
        other = (PyArrayObject*)other_obj;
        Py_INCREF(other);
    }
    else {
        other = (PyArrayObject*)PyArray_FROM_O(other_obj);
        if (other == NULL) {
            goto fail;
        }
    }

    if (max_work_obj == NULL || max_work_obj == Py_None) {
        /* noop */
    }
    else if (PyLong_Check(max_work_obj)) {
        max_work = PyLong_AsSsize_t(max_work_obj);
        if (PyErr_Occurred()) {
            goto fail;
        }
    }
    else {
        PyErr_SetString(PyExc_ValueError, "max_work must be an integer");
        goto fail;
    }

    if (max_work < -2) {
        PyErr_SetString(PyExc_ValueError, "Invalid value for max_work");
        goto fail;
    }

    NPY_BEGIN_THREADS;
    result = solve_may_share_memory(self, other, max_work);
    NPY_END_THREADS;

    Py_XDECREF(self);
    Py_XDECREF(other);

    if (result == MEM_OVERLAP_NO) {
        Py_RETURN_FALSE;
    }
    else if (result == MEM_OVERLAP_YES) {
        Py_RETURN_TRUE;
    }
    else if (result == MEM_OVERLAP_OVERFLOW) {
        if (raise_exceptions) {
            PyErr_SetString(PyExc_OverflowError,
                            "Integer overflow in computing overlap");
            return NULL;
        }
        else {
            /* Don't know, so say yes */
            Py_RETURN_TRUE;
        }
    }
    else if (result == MEM_OVERLAP_TOO_HARD) {
        if (raise_exceptions) {
            PyErr_SetString(npy_static_pydata.TooHardError,
                            "Exceeded max_work");
            return NULL;
        }
        else {
            /* Don't know, so say yes */
            Py_RETURN_TRUE;
        }
    }
    else {
        /* Doesn't happen usually */
        PyErr_SetString(PyExc_RuntimeError,
                        "Error in computing overlap");
        return NULL;
    }

fail:
    Py_XDECREF(self);
    Py_XDECREF(other);
    return NULL;
}


static PyObject *
array_shares_memory(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{
    return array_shares_memory_impl(args, kwds, NPY_MAY_SHARE_EXACT, 1);
}


static PyObject *
array_may_share_memory(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{
    return array_shares_memory_impl(args, kwds, NPY_MAY_SHARE_BOUNDS, 0);
}

static PyObject *
normalize_axis_index(PyObject *NPY_UNUSED(self),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    int axis;
    int ndim;
    PyObject *msg_prefix = Py_None;
    NPY_PREPARE_ARGPARSER;

    if (npy_parse_arguments("normalize_axis_index", args, len_args, kwnames,
            "axis", &PyArray_PythonPyIntFromInt, &axis,
            "ndim", &PyArray_PythonPyIntFromInt, &ndim,
            "|msg_prefix", NULL, &msg_prefix,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }
    if (check_and_adjust_axis_msg(&axis, ndim, msg_prefix) < 0) {
        return NULL;
    }

    return PyLong_FromLong(axis);
}


static PyObject *
_set_numpy_warn_if_no_mem_policy(PyObject *NPY_UNUSED(self), PyObject *arg)
{
    int res = PyObject_IsTrue(arg);
    if (res < 0) {
        return NULL;
    }
    int old_value = npy_thread_unsafe_state.warn_if_no_mem_policy;
    npy_thread_unsafe_state.warn_if_no_mem_policy = res;
    if (old_value) {
        Py_RETURN_TRUE;
    }
    else {
        Py_RETURN_FALSE;
    }
}


static PyObject *
_reload_guard(PyObject *NPY_UNUSED(self), PyObject *NPY_UNUSED(args)) {
#if !defined(PYPY_VERSION)
    if (PyThreadState_Get()->interp != PyInterpreterState_Main()) {
        if (PyErr_WarnEx(PyExc_UserWarning,
                "NumPy was imported from a Python sub-interpreter but "
                "NumPy does not properly support sub-interpreters. "
                "This will likely work for most users but might cause hard to "
                "track down issues or subtle bugs. "
                "A common user of the rare sub-interpreter feature is wsgi "
                "which also allows single-interpreter mode.\n"
                "Improvements in the case of bugs are welcome, but is not "
                "on the NumPy roadmap, and full support may require "
                "significant effort to achieve.", 2) < 0) {
            return NULL;
        }
        /* No need to give the other warning in a sub-interpreter as well... */
        npy_thread_unsafe_state.reload_guard_initialized = 1;
        Py_RETURN_NONE;
    }
#endif
    if (npy_thread_unsafe_state.reload_guard_initialized) {
        if (PyErr_WarnEx(PyExc_UserWarning,
                "The NumPy module was reloaded (imported a second time). "
                "This can in some cases result in small but subtle issues "
                "and is discouraged.", 2) < 0) {
            return NULL;
        }
    }
    npy_thread_unsafe_state.reload_guard_initialized = 1;
    Py_RETURN_NONE;
}


static struct PyMethodDef array_module_methods[] = {
    {"_get_implementing_args",
        (PyCFunction)array__get_implementing_args,
        METH_VARARGS, NULL},
    {"_get_ndarray_c_version",
        (PyCFunction)array__get_ndarray_c_version,
        METH_NOARGS, NULL},
    {"_reconstruct",
        (PyCFunction)array__reconstruct,
        METH_VARARGS, NULL},
    {"set_datetimeparse_function",
        (PyCFunction)array_set_datetimeparse_function,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"set_typeDict",
        (PyCFunction)array_set_typeDict,
        METH_VARARGS, NULL},
    {"array",
        (PyCFunction)array_array,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"asarray",
        (PyCFunction)array_asarray,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"asanyarray",
        (PyCFunction)array_asanyarray,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"ascontiguousarray",
        (PyCFunction)array_ascontiguousarray,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"asfortranarray",
        (PyCFunction)array_asfortranarray,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"copyto",
        (PyCFunction)array_copyto,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"nested_iters",
        (PyCFunction)NpyIter_NestedIters,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"arange",
        (PyCFunction)array_arange,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"zeros",
        (PyCFunction)array_zeros,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"count_nonzero",
        (PyCFunction)array_count_nonzero,
        METH_FASTCALL, NULL},
    {"empty",
        (PyCFunction)array_empty,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"empty_like",
        (PyCFunction)array_empty_like,
        METH_FASTCALL|METH_KEYWORDS, NULL},
    {"scalar",
        (PyCFunction)array_scalar,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"where",
        (PyCFunction)array_where,
        METH_FASTCALL, NULL},
    {"lexsort",
        (PyCFunction)array_lexsort,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"putmask",
        (PyCFunction)array_putmask,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"fromstring",
        (PyCFunction)array_fromstring,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"fromiter",
        (PyCFunction)array_fromiter,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"concatenate",
        (PyCFunction)array_concatenate,
        METH_FASTCALL|METH_KEYWORDS, NULL},
    {"inner",
        (PyCFunction)array_innerproduct,
        METH_FASTCALL, NULL},
    {"dot",
        (PyCFunction)array_matrixproduct,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"vdot",
        (PyCFunction)array_vdot,
        METH_FASTCALL, NULL},
    {"c_einsum",
        (PyCFunction)array_einsum,
        METH_FASTCALL|METH_KEYWORDS, NULL},
    {"correlate",
        (PyCFunction)array_correlate,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"correlate2",
        (PyCFunction)array_correlate2,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"frombuffer",
        (PyCFunction)array_frombuffer,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"fromfile",
        (PyCFunction)array_fromfile,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"can_cast",
        (PyCFunction)array_can_cast_safely,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"promote_types",
        (PyCFunction)array_promote_types,
        METH_FASTCALL, NULL},
    {"min_scalar_type",
        (PyCFunction)array_min_scalar_type,
        METH_VARARGS, NULL},
    {"result_type",
        (PyCFunction)array_result_type,
        METH_FASTCALL, NULL},
    {"shares_memory",
        (PyCFunction)array_shares_memory,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"may_share_memory",
        (PyCFunction)array_may_share_memory,
        METH_VARARGS | METH_KEYWORDS, NULL},
    /* Datetime-related functions */
    {"datetime_data",
        (PyCFunction)array_datetime_data,
        METH_VARARGS, NULL},
    {"datetime_as_string",
        (PyCFunction)array_datetime_as_string,
        METH_VARARGS | METH_KEYWORDS, NULL},
    /* Datetime business-day API */
    {"busday_offset",
        (PyCFunction)array_busday_offset,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"busday_count",
        (PyCFunction)array_busday_count,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"is_busday",
        (PyCFunction)array_is_busday,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"format_longfloat",
        (PyCFunction)format_longfloat,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"dragon4_positional",
        (PyCFunction)dragon4_positional,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"dragon4_scientific",
        (PyCFunction)dragon4_scientific,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"compare_chararrays",
        (PyCFunction)compare_chararrays,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"_vec_string",
        (PyCFunction)_vec_string,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"_place", (PyCFunction)arr_place,
        METH_VARARGS | METH_KEYWORDS,
        "Insert vals sequentially into equivalent 1-d positions "
        "indicated by mask."},
    {"bincount", (PyCFunction)arr_bincount,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"_monotonicity", (PyCFunction)arr__monotonicity,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"interp", (PyCFunction)arr_interp,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"interp_complex", (PyCFunction)arr_interp_complex,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"ravel_multi_index", (PyCFunction)arr_ravel_multi_index,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"unravel_index", (PyCFunction)arr_unravel_index,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"add_docstring", (PyCFunction)arr_add_docstring,
        METH_FASTCALL, NULL},
    {"packbits", (PyCFunction)io_pack,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"unpackbits", (PyCFunction)io_unpack,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"normalize_axis_index", (PyCFunction)normalize_axis_index,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"_discover_array_parameters", (PyCFunction)_discover_array_parameters,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"_get_castingimpl",  (PyCFunction)_get_castingimpl,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"_load_from_filelike", (PyCFunction)_load_from_filelike,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    /* from umath */
    {"frompyfunc",
        (PyCFunction) ufunc_frompyfunc,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"_make_extobj",
        (PyCFunction)extobj_make_extobj,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"_get_extobj_dict",
        (PyCFunction)extobj_get_extobj_dict,
        METH_NOARGS, NULL},
    {"get_handler_name",
        (PyCFunction) get_handler_name,
        METH_VARARGS, NULL},
    {"get_handler_version",
        (PyCFunction) get_handler_version,
        METH_VARARGS, NULL},
    {"_set_numpy_warn_if_no_mem_policy",
         (PyCFunction)_set_numpy_warn_if_no_mem_policy,
         METH_O, "Change the warn if no mem policy flag for testing."},
    {"_add_newdoc_ufunc", (PyCFunction)add_newdoc_ufunc,
        METH_VARARGS, NULL},
    {"_get_sfloat_dtype",
        get_sfloat_dtype, METH_NOARGS, NULL},
    {"_get_madvise_hugepage", (PyCFunction)_get_madvise_hugepage,
        METH_NOARGS, NULL},
    {"_set_madvise_hugepage", (PyCFunction)_set_madvise_hugepage,
        METH_O, NULL},
    {"_reload_guard", (PyCFunction)_reload_guard,
        METH_NOARGS,
        "Give a warning on reload and big warning in sub-interpreters."},
    {"from_dlpack", (PyCFunction)from_dlpack,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"_unique_hash",  (PyCFunction)array__unique_hash,
        METH_O, "Collect unique values via a hash map."},
    {NULL, NULL, 0, NULL}                /* sentinel */
};

#include "__multiarray_api.c"
#include "array_method.h"

/* Establish scalar-type hierarchy
 *
 *  For dual inheritance we need to make sure that the objects being
 *  inherited from have the tp->mro object initialized.  This is
 *  not necessarily true for the basic type objects of Python (it is
 *  checked for single inheritance but not dual in PyType_Ready).
 *
 *  Thus, we call PyType_Ready on the standard Python Types, here.
 */
static int
setup_scalartypes(PyObject *NPY_UNUSED(dict))
{
    if (PyType_Ready(&PyBool_Type) < 0) {
        return -1;
    }
    if (PyType_Ready(&PyFloat_Type) < 0) {
        return -1;
    }
    if (PyType_Ready(&PyComplex_Type) < 0) {
        return -1;
    }
    if (PyType_Ready(&PyBytes_Type) < 0) {
        return -1;
    }
    if (PyType_Ready(&PyUnicode_Type) < 0) {
        return -1;
    }

#define SINGLE_INHERIT(child, parent)                                   \
    Py##child##ArrType_Type.tp_base = &Py##parent##ArrType_Type;        \
    if (PyType_Ready(&Py##child##ArrType_Type) < 0) {                   \
        PyErr_Print();                                                  \
        PyErr_Format(PyExc_SystemError,                                 \
                     "could not initialize Py%sArrType_Type",           \
                     #child);                                           \
        return -1;                                                      \
    }

    if (PyType_Ready(&PyGenericArrType_Type) < 0) {
        return -1;
    }
    SINGLE_INHERIT(Number, Generic);
    SINGLE_INHERIT(Integer, Number);
    SINGLE_INHERIT(Inexact, Number);
    SINGLE_INHERIT(SignedInteger, Integer);
    SINGLE_INHERIT(UnsignedInteger, Integer);
    SINGLE_INHERIT(Floating, Inexact);
    SINGLE_INHERIT(ComplexFloating, Inexact);
    SINGLE_INHERIT(Flexible, Generic);
    SINGLE_INHERIT(Character, Flexible);

#define DUAL_INHERIT(child, parent1, parent2)                           \
    Py##child##ArrType_Type.tp_base = &Py##parent2##ArrType_Type;       \
    Py##child##ArrType_Type.tp_bases =                                  \
        Py_BuildValue("(OO)", &Py##parent2##ArrType_Type,               \
                      &Py##parent1##_Type);                             \
    Py##child##ArrType_Type.tp_hash = Py##parent1##_Type.tp_hash;       \
    if (PyType_Ready(&Py##child##ArrType_Type) < 0) {                   \
        PyErr_Print();                                                  \
        PyErr_Format(PyExc_SystemError,                                 \
                     "could not initialize Py%sArrType_Type",           \
                     #child);                                           \
        return -1;                                                      \
    }

#define DUAL_INHERIT2(child, parent1, parent2)                          \
    Py##child##ArrType_Type.tp_base = &Py##parent1##_Type;              \
    Py##child##ArrType_Type.tp_bases =                                  \
        Py_BuildValue("(OO)", &Py##parent1##_Type,                      \
                      &Py##parent2##ArrType_Type);                      \
    Py##child##ArrType_Type.tp_richcompare =                            \
        Py##parent1##_Type.tp_richcompare;                              \
    Py##child##ArrType_Type.tp_hash = Py##parent1##_Type.tp_hash;       \
    if (PyType_Ready(&Py##child##ArrType_Type) < 0) {                   \
        PyErr_Print();                                                  \
        PyErr_Format(PyExc_SystemError,                                 \
                     "could not initialize Py%sArrType_Type",           \
                     #child);                                           \
        return -1;                                                      \
    }

    SINGLE_INHERIT(Bool, Generic);
    SINGLE_INHERIT(Byte, SignedInteger);
    SINGLE_INHERIT(Short, SignedInteger);
    SINGLE_INHERIT(Int, SignedInteger);
    SINGLE_INHERIT(Long, SignedInteger);
    SINGLE_INHERIT(LongLong, SignedInteger);

    /* Datetime doesn't fit in any category */
    SINGLE_INHERIT(Datetime, Generic);
    /* Timedelta is an integer with an associated unit */
    SINGLE_INHERIT(Timedelta, SignedInteger);

    SINGLE_INHERIT(UByte, UnsignedInteger);
    SINGLE_INHERIT(UShort, UnsignedInteger);
    SINGLE_INHERIT(UInt, UnsignedInteger);
    SINGLE_INHERIT(ULong, UnsignedInteger);
    SINGLE_INHERIT(ULongLong, UnsignedInteger);

    SINGLE_INHERIT(Half, Floating);
    SINGLE_INHERIT(Float, Floating);
    DUAL_INHERIT(Double, Float, Floating);
    SINGLE_INHERIT(LongDouble, Floating);

    SINGLE_INHERIT(CFloat, ComplexFloating);
    DUAL_INHERIT(CDouble, Complex, ComplexFloating);
    SINGLE_INHERIT(CLongDouble, ComplexFloating);

    DUAL_INHERIT2(String, Bytes, Character);
    DUAL_INHERIT2(Unicode, Unicode, Character);

    SINGLE_INHERIT(Void, Flexible);

    SINGLE_INHERIT(Object, Generic);

    return 0;

#undef SINGLE_INHERIT
#undef DUAL_INHERIT
#undef DUAL_INHERIT2

    /*
     * Clean up string and unicode array types so they act more like
     * strings -- get their tables from the standard types.
     */
}

/* place a flag dictionary in d */

static void
set_flaginfo(PyObject *d)
{
    PyObject *s;
    PyObject *newd;

    newd = PyDict_New();

#define _addnew(key, val, one)                                       \
    PyDict_SetItemString(newd, #key, s=PyLong_FromLong(val));    \
    Py_DECREF(s);                                               \
    PyDict_SetItemString(newd, #one, s=PyLong_FromLong(val));    \
    Py_DECREF(s)

#define _addone(key, val)                                            \
    PyDict_SetItemString(newd, #key, s=PyLong_FromLong(val));    \
    Py_DECREF(s)

    _addnew(OWNDATA, NPY_ARRAY_OWNDATA, O);
    _addnew(FORTRAN, NPY_ARRAY_F_CONTIGUOUS, F);
    _addnew(CONTIGUOUS, NPY_ARRAY_C_CONTIGUOUS, C);
    _addnew(ALIGNED, NPY_ARRAY_ALIGNED, A);
    _addnew(WRITEBACKIFCOPY, NPY_ARRAY_WRITEBACKIFCOPY, X);
    _addnew(WRITEABLE, NPY_ARRAY_WRITEABLE, W);
    _addone(C_CONTIGUOUS, NPY_ARRAY_C_CONTIGUOUS);
    _addone(F_CONTIGUOUS, NPY_ARRAY_F_CONTIGUOUS);

#undef _addone
#undef _addnew

    PyDict_SetItemString(d, "_flagdict", newd);
    Py_DECREF(newd);
    return;
}

// static variables are automatically zero-initialized
NPY_VISIBILITY_HIDDEN npy_thread_unsafe_state_struct npy_thread_unsafe_state;

static int
initialize_thread_unsafe_state(void) {
    char *env = getenv("NUMPY_WARN_IF_NO_MEM_POLICY");
    if ((env != NULL) && (strncmp(env, "1", 1) == 0)) {
        npy_thread_unsafe_state.warn_if_no_mem_policy = 1;
    }
    else {
        npy_thread_unsafe_state.warn_if_no_mem_policy = 0;
    }

    return 0;
}

static int module_loaded = 0;

static int
_multiarray_umath_exec(PyObject *m) {
    PyObject *d, *s, *c_api;

    // https://docs.python.org/3/howto/isolating-extensions.html#opt-out-limiting-to-one-module-object-per-process
    if (module_loaded) {
        PyErr_SetString(PyExc_ImportError,
                        "cannot load module more than once per process");
        return -1;
    }
    module_loaded = 1;

    /* Initialize CPU features */
    if (npy_cpu_init() < 0) {
        return -1;
    }
    /* Initialize CPU dispatch tracer */
    if (npy_cpu_dispatch_tracer_init(m) < 0) {
        return -1;
    }

#if defined(MS_WIN64) && defined(__GNUC__)
  PyErr_WarnEx(PyExc_Warning,
        "Numpy built with MINGW-W64 on Windows 64 bits is experimental, " \
        "and only available for \n" \
        "testing. You are advised not to use it for production. \n\n" \
        "CRASHES ARE TO BE EXPECTED - PLEASE REPORT THEM TO NUMPY DEVELOPERS",
        1);
#endif

    /* Initialize access to the PyDateTime API */
    numpy_pydatetime_import();

    if (PyErr_Occurred()) {
        return -1;
    }

    /* Add some symbolic constants to the module */
    d = PyModule_GetDict(m);
    if (!d) {
        return -1;
    }

    if (intern_strings() < 0) {
        return -1;
    }

    if (initialize_static_globals() < 0) {
        return -1;
    }

    if (initialize_thread_unsafe_state() < 0) {
        return -1;
    }

    if (init_import_mutex() < 0) {
        return -1;
    }

    if (init_extobj() < 0) {
        return -1;
    }

    if (PyType_Ready(&PyUFunc_Type) < 0) {
        return -1;
    }

    PyArrayDTypeMeta_Type.tp_base = &PyType_Type;
    if (PyType_Ready(&PyArrayDTypeMeta_Type) < 0) {
        return -1;
    }

    PyArrayDescr_Type.tp_hash = PyArray_DescrHash;
    Py_SET_TYPE(&PyArrayDescr_Type, &PyArrayDTypeMeta_Type);
    if (PyType_Ready(&PyArrayDescr_Type) < 0) {
        return -1;
    }

    initialize_casting_tables();
    initialize_numeric_types();

    if (initscalarmath(m) < 0) {
        return -1;
    }

    if (PyType_Ready(&PyArray_Type) < 0) {
        return -1;
    }
    if (setup_scalartypes(d) < 0) {
        return -1;
    }

    PyArrayIter_Type.tp_iter = PyObject_SelfIter;
    NpyIter_Type.tp_iter = PyObject_SelfIter;
    PyArrayMultiIter_Type.tp_iter = PyObject_SelfIter;
    PyArrayMultiIter_Type.tp_free = PyArray_free;
    if (PyType_Ready(&PyArrayIter_Type) < 0) {
        return -1;
    }
    if (PyType_Ready(&PyArrayMapIter_Type) < 0) {
        return -1;
    }
    if (PyType_Ready(&PyArrayMultiIter_Type) < 0) {
        return -1;
    }
    PyArrayNeighborhoodIter_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready(&PyArrayNeighborhoodIter_Type) < 0) {
        return -1;
    }
    if (PyType_Ready(&NpyIter_Type) < 0) {
        return -1;
    }

    if (PyType_Ready(&PyArrayFlags_Type) < 0) {
        return -1;
    }
    NpyBusDayCalendar_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready(&NpyBusDayCalendar_Type) < 0) {
        return -1;
    }

    /*
     * PyExc_Exception should catch all the standard errors that are
     * now raised instead of the string exception "multiarray.error"

     * This is for backward compatibility with existing code.
     */
    PyDict_SetItemString (d, "error", PyExc_Exception);

    s = PyLong_FromLong(NPY_TRACE_DOMAIN);
    PyDict_SetItemString(d, "tracemalloc_domain", s);
    Py_DECREF(s);

    s = PyUnicode_FromString("3.1");
    PyDict_SetItemString(d, "__version__", s);
    Py_DECREF(s);

    s = npy_cpu_features_dict();
    if (s == NULL) {
        return -1;
    }
    if (PyDict_SetItemString(d, "__cpu_features__", s) < 0) {
        Py_DECREF(s);
        return -1;
    }
    Py_DECREF(s);

    s = npy_cpu_baseline_list();
    if (s == NULL) {
        return -1;
    }
    if (PyDict_SetItemString(d, "__cpu_baseline__", s) < 0) {
        Py_DECREF(s);
        return -1;
    }
    Py_DECREF(s);

    s = npy_cpu_dispatch_list();
    if (s == NULL) {
        return -1;
    }
    if (PyDict_SetItemString(d, "__cpu_dispatch__", s) < 0) {
        Py_DECREF(s);
        return -1;
    }
    Py_DECREF(s);

    s = PyCapsule_New((void *)_datetime_strings, NULL, NULL);
    if (s == NULL) {
        return -1;
    }
    PyDict_SetItemString(d, "DATETIMEUNITS", s);
    Py_DECREF(s);

#define ADDCONST(NAME)                          \
    s = PyLong_FromLong(NPY_##NAME);            \
    PyDict_SetItemString(d, #NAME, s);          \
    Py_DECREF(s)


    ADDCONST(ALLOW_THREADS);
    ADDCONST(BUFSIZE);
    ADDCONST(CLIP);

    ADDCONST(ITEM_HASOBJECT);
    ADDCONST(LIST_PICKLE);
    ADDCONST(ITEM_IS_POINTER);
    ADDCONST(NEEDS_INIT);
    ADDCONST(NEEDS_PYAPI);
    ADDCONST(USE_GETITEM);
    ADDCONST(USE_SETITEM);

    ADDCONST(RAISE);
    ADDCONST(WRAP);
    ADDCONST(MAXDIMS);

    ADDCONST(MAY_SHARE_BOUNDS);
    ADDCONST(MAY_SHARE_EXACT);
#undef ADDCONST

    PyDict_SetItemString(d, "ndarray", (PyObject *)&PyArray_Type);
    PyDict_SetItemString(d, "flatiter", (PyObject *)&PyArrayIter_Type);
    PyDict_SetItemString(d, "nditer", (PyObject *)&NpyIter_Type);
    PyDict_SetItemString(d, "broadcast",
                         (PyObject *)&PyArrayMultiIter_Type);
    PyDict_SetItemString(d, "dtype", (PyObject *)&PyArrayDescr_Type);
    PyDict_SetItemString(d, "flagsobj", (PyObject *)&PyArrayFlags_Type);

    /* Business day calendar object */
    PyDict_SetItemString(d, "busdaycalendar",
                            (PyObject *)&NpyBusDayCalendar_Type);
    set_flaginfo(d);

    /* Finalize scalar types and expose them via namespace or typeinfo dict */
    if (set_typeinfo(d) != 0) {
        return -1;
    }

    if (PyType_Ready(&PyArrayFunctionDispatcher_Type) < 0) {
        return -1;
    }
    PyDict_SetItemString(
            d, "_ArrayFunctionDispatcher",
            (PyObject *)&PyArrayFunctionDispatcher_Type);

    if (PyType_Ready(&PyArrayArrayConverter_Type) < 0) {
        return -1;
    }
    PyDict_SetItemString(
            d, "_array_converter",
            (PyObject *)&PyArrayArrayConverter_Type);

    if (PyType_Ready(&PyArrayMethod_Type) < 0) {
        return -1;
    }
    if (PyType_Ready(&PyBoundArrayMethod_Type) < 0) {
        return -1;
    }
    if (initialize_and_map_pytypes_to_dtypes() < 0) {
        return -1;
    }

    if (PyArray_InitializeCasts() < 0) {
        return -1;
    }

    if (init_string_dtype() < 0) {
        return -1;
    }

    /*
     * Initialize the default PyDataMem_Handler capsule singleton.
     */
    PyDataMem_DefaultHandler = PyCapsule_New(
            &default_handler, MEM_HANDLER_CAPSULE_NAME, NULL);
    if (PyDataMem_DefaultHandler == NULL) {
        return -1;
    }

    /*
     * Initialize the context-local current handler
     * with the default PyDataMem_Handler capsule.
     */
    current_handler = PyContextVar_New("current_allocator", PyDataMem_DefaultHandler);
    if (current_handler == NULL) {
        return -1;
    }

    if (initumath(m) != 0) {
        return -1;
    }

    if (set_matmul_flags(d) < 0) {
        return -1;
    }

    // initialize static references to ndarray.__array_*__ special methods
    npy_static_pydata.ndarray_array_finalize = PyObject_GetAttrString(
            (PyObject *)&PyArray_Type, "__array_finalize__");
    if (npy_static_pydata.ndarray_array_finalize == NULL) {
        return -1;
    }
    npy_static_pydata.ndarray_array_ufunc = PyObject_GetAttrString(
            (PyObject *)&PyArray_Type, "__array_ufunc__");
    if (npy_static_pydata.ndarray_array_ufunc == NULL) {
        return -1;
    }
    npy_static_pydata.ndarray_array_function = PyObject_GetAttrString(
            (PyObject *)&PyArray_Type, "__array_function__");
    if (npy_static_pydata.ndarray_array_function == NULL) {
        return -1;
    }

    /*
     * Initialize np.dtypes.StringDType
     *
     * Note that this happens here after initializing
     * the legacy built-in DTypes to avoid a circular dependency
     * during NumPy setup, since this needs to happen after
     * init_string_dtype() but that needs to happen after
     * the legacy dtypemeta classes are available.
     */

    if (npy_cache_import_runtime(
            "numpy.dtypes", "_add_dtype_helper",
            &npy_runtime_imports._add_dtype_helper) == -1) {
        return -1;
    }

    if (PyObject_CallFunction(
            npy_runtime_imports._add_dtype_helper,
            "Os", (PyObject *)&PyArray_StringDType, NULL) == NULL) {
        return -1;
    }
    PyDict_SetItemString(d, "StringDType", (PyObject *)&PyArray_StringDType);

    // initialize static reference to a zero-like array
    npy_static_pydata.zero_pyint_like_arr = PyArray_ZEROS(
            0, NULL, NPY_DEFAULT_INT, NPY_FALSE);
    if (npy_static_pydata.zero_pyint_like_arr == NULL) {
        return -1;
    }
    ((PyArrayObject_fields *)npy_static_pydata.zero_pyint_like_arr)->flags |=
            (NPY_ARRAY_WAS_PYTHON_INT|NPY_ARRAY_WAS_INT_AND_REPLACED);

    if (verify_static_structs_initialized() < 0) {
        return -1;
    }

    /*
     * Export the API tables
     */
    c_api = PyCapsule_New((void *)PyArray_API, NULL, NULL);
    /* The dtype API is not auto-filled/generated via Python scripts: */
    _fill_dtype_api(PyArray_API);
    if (c_api == NULL) {
        return -1;
    }
    PyDict_SetItemString(d, "_ARRAY_API", c_api);
    Py_DECREF(c_api);

    c_api = PyCapsule_New((void *)PyUFunc_API, NULL, NULL);
    if (c_api == NULL) {
        return -1;
    }
    PyDict_SetItemString(d, "_UFUNC_API", c_api);
    Py_DECREF(c_api);
    if (PyErr_Occurred()) {
        return -1;
    }

    return 0;
}

static struct PyModuleDef_Slot _multiarray_umath_slots[] = {
    {Py_mod_exec, _multiarray_umath_exec},
#if PY_VERSION_HEX >= 0x030c00f0  // Python 3.12+
    {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},
#endif
#if PY_VERSION_HEX >= 0x030d00f0  // Python 3.13+
    // signal that this module supports running without an active GIL
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL},
};

static struct PyModuleDef moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_multiarray_umath",
    .m_size = 0,
    .m_methods = array_module_methods,
    .m_slots = _multiarray_umath_slots,
};

PyMODINIT_FUNC PyInit__multiarray_umath(void) {
    return PyModuleDef_Init(&moduledef);
}
