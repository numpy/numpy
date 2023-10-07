/*
 * This module corresponds to the `Special functions for NPY_OBJECT`
 * section in the numpy reference for C-API.
 */
#include "array_method.h"
#include "dtype_traversal.h"
#include "lowlevel_strided_loops.h"
#include "pyerrors.h"
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"
#include "iterators.h"
#include "dtypemeta.h"
#include "refcount.h"

#include "npy_config.h"

#include "npy_pycompat.h"

/*
 * Helper function to clear a strided memory (normally or always contiguous)
 * from all Python (or other) references.  The function does nothing if the
 * array dtype does not indicate holding references.
 *
 * It is safe to call this function more than once, failing here is usually
 * critical (during cleanup) and should be set up to minimize the risk or
 * avoid it fully.
 */
NPY_NO_EXPORT int
PyArray_ClearBuffer(
        PyArray_Descr *descr, char *data,
        npy_intp stride, npy_intp size, int aligned)
{
    if (!PyDataType_REFCHK(descr)) {
        return 0;
    }

    NPY_traverse_info clear_info;
    /* Flags unused: float errors do not matter and we do not release GIL */
    NPY_ARRAYMETHOD_FLAGS flags_unused;
    if (PyArray_GetClearFunction(
            aligned, stride, descr, &clear_info, &flags_unused) < 0) {
        return -1;
    }

    int res = clear_info.func(
            NULL, clear_info.descr, data, size, stride, clear_info.auxdata);
    NPY_traverse_info_xfree(&clear_info);
    return res;
}


/*
 * Helper function to clear whole array.  It seems plausible that we should
 * be able to get away with assuming the the array is contiguous.
 *
 * Must only be called on arrays which own their data (and asserts this).
 */
 NPY_NO_EXPORT int
 PyArray_ClearArray(PyArrayObject *arr)
 {
    assert(PyArray_FLAGS(arr) & NPY_ARRAY_OWNDATA);

    PyArray_Descr *descr = PyArray_DESCR(arr);
    if (!PyDataType_REFCHK(descr)) {
        return 0;
    }
    /*
     * The contiguous path should cover practically all important cases since
     * it is difficult to create a non-contiguous array which owns its memory
     * and only arrays which own their memory should clear it.
     */
    int aligned = PyArray_ISALIGNED(arr);
    if (PyArray_ISCONTIGUOUS(arr)) {
        return PyArray_ClearBuffer(
                descr, PyArray_BYTES(arr), descr->elsize,
                PyArray_SIZE(arr), aligned);
    }
    int idim, ndim;
    npy_intp shape_it[NPY_MAXDIMS], strides_it[NPY_MAXDIMS];
    npy_intp coord[NPY_MAXDIMS];
    char *data_it;
    if (PyArray_PrepareOneRawArrayIter(
                    PyArray_NDIM(arr), PyArray_DIMS(arr),
                    PyArray_BYTES(arr), PyArray_STRIDES(arr),
                    &ndim, shape_it, &data_it, strides_it) < 0) {
        return -1;
    }
    npy_intp inner_stride = strides_it[0];
    npy_intp inner_shape = shape_it[0];
    NPY_traverse_info clear_info;
    /* Flags unused: float errors do not matter and we do not release GIL */
    NPY_ARRAYMETHOD_FLAGS flags_unused;
    if (PyArray_GetClearFunction(
            aligned, inner_stride, descr, &clear_info, &flags_unused) < 0) {
        return -1;
    }
    NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {
        /* Process the innermost dimension */
        if (clear_info.func(NULL, clear_info.descr,
                data_it, inner_shape, inner_stride, clear_info.auxdata) < 0) {
            return -1;
        }
    } NPY_RAW_ITER_ONE_NEXT(idim, ndim, coord,
                            shape_it, data_it, strides_it);
    return 0;
}


/*NUMPY_API
 * XINCREF all objects in a single array item. This is complicated for
 * structured datatypes where the position of objects needs to be extracted.
 * The function is execute recursively for each nested field or subarrays dtype
 * such as as `np.dtype([("field1", "O"), ("field2", "f,O", (3,2))])`
 */
NPY_NO_EXPORT void
PyArray_Item_INCREF(char *data, PyArray_Descr *descr)
{
    PyObject *temp;

    if (!PyDataType_REFCHK(descr)) {
        return;
    }
    if (descr->type_num == NPY_OBJECT) {
        memcpy(&temp, data, sizeof(temp));
        Py_XINCREF(temp);
    }
    else if (PyDataType_HASFIELDS(descr)) {
        PyObject *key, *value, *title = NULL;
        PyArray_Descr *new;
        int offset;
        Py_ssize_t pos = 0;

        while (PyDict_Next(descr->fields, &pos, &key, &value)) {
            if (NPY_TITLE_KEY(key, value)) {
                continue;
            }
            if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset,
                                  &title)) {
                return;
            }
            PyArray_Item_INCREF(data + offset, new);
        }
    }
    else if (PyDataType_HASSUBARRAY(descr)) {
        int size, i, inner_elsize;

        inner_elsize = descr->subarray->base->elsize;
        if (inner_elsize == 0) {
            /* There cannot be any elements, so return */
            return;
        }
        /* Subarrays are always contiguous in memory */
        size = descr->elsize / inner_elsize;

        for (i = 0; i < size; i++){
            /* Recursively increment the reference count of subarray elements */
            PyArray_Item_INCREF(data + i * inner_elsize,
                                descr->subarray->base);
        }
    }
    else {
        /* This path should not be reachable. */
        assert(0);
    }
    return;
}


/*NUMPY_API
 *
 * XDECREF all objects in a single array item. This is complicated for
 * structured datatypes where the position of objects needs to be extracted.
 * The function is execute recursively for each nested field or subarrays dtype
 * such as as `np.dtype([("field1", "O"), ("field2", "f,O", (3,2))])`
 */
NPY_NO_EXPORT void
PyArray_Item_XDECREF(char *data, PyArray_Descr *descr)
{
    PyObject *temp;

    if (!PyDataType_REFCHK(descr)) {
        return;
    }

    if (descr->type_num == NPY_OBJECT) {
        memcpy(&temp, data, sizeof(temp));
        Py_XDECREF(temp);
    }
    else if (PyDataType_HASFIELDS(descr)) {
            PyObject *key, *value, *title = NULL;
            PyArray_Descr *new;
            int offset;
            Py_ssize_t pos = 0;

            while (PyDict_Next(descr->fields, &pos, &key, &value)) {
                if (NPY_TITLE_KEY(key, value)) {
                    continue;
                }
                if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset,
                                      &title)) {
                    return;
                }
                PyArray_Item_XDECREF(data + offset, new);
            }
        }
    else if (PyDataType_HASSUBARRAY(descr)) {
        int size, i, inner_elsize;

        inner_elsize = descr->subarray->base->elsize;
        if (inner_elsize == 0) {
            /* There cannot be any elements, so return */
            return;
        }
        /* Subarrays are always contiguous in memory */
        size = descr->elsize / inner_elsize;

        for (i = 0; i < size; i++){
            /* Recursively decrement the reference count of subarray elements */
            PyArray_Item_XDECREF(data + i * inner_elsize,
                                 descr->subarray->base);
        }
    }
    else {
        /* This path should not be reachable. */
        assert(0);
    }
    return;
}

/* Used for arrays of python objects to increment the reference count of */
/* every python object in the array. */
/*NUMPY_API
  For object arrays, increment all internal references.
*/
NPY_NO_EXPORT int
PyArray_INCREF(PyArrayObject *mp)
{
    npy_intp i, n;
    PyObject **data;
    PyObject *temp;
    PyArrayIterObject *it;

    if (!PyDataType_REFCHK(PyArray_DESCR(mp))) {
        return 0;
    }
    if (PyArray_DESCR(mp)->type_num != NPY_OBJECT) {
        it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)mp);
        if (it == NULL) {
            return -1;
        }
        while(it->index < it->size) {
            PyArray_Item_INCREF(it->dataptr, PyArray_DESCR(mp));
            PyArray_ITER_NEXT(it);
        }
        Py_DECREF(it);
        return 0;
    }

    if (PyArray_ISONESEGMENT(mp)) {
        data = (PyObject **)PyArray_DATA(mp);
        n = PyArray_SIZE(mp);
        if (PyArray_ISALIGNED(mp)) {
            for (i = 0; i < n; i++, data++) {
                Py_XINCREF(*data);
            }
        }
        else {
            for( i = 0; i < n; i++, data++) {
                memcpy(&temp, data, sizeof(temp));
                Py_XINCREF(temp);
            }
        }
    }
    else { /* handles misaligned data too */
        it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)mp);
        if (it == NULL) {
            return -1;
        }
        while(it->index < it->size) {
            memcpy(&temp, it->dataptr, sizeof(temp));
            Py_XINCREF(temp);
            PyArray_ITER_NEXT(it);
        }
        Py_DECREF(it);
    }
    return 0;
}

/*NUMPY_API
  Decrement all internal references for object arrays.
  (or arrays with object fields)

  The use of this function is strongly discouraged, within NumPy
  use PyArray_Clear, which DECREF's and sets everything to NULL and can
  work with any dtype.
*/
NPY_NO_EXPORT int
PyArray_XDECREF(PyArrayObject *mp)
{
    npy_intp i, n;
    PyObject **data;
    PyObject *temp;
    /*
     * statically allocating it allows this function to not modify the
     * reference count of the array for use during dealloc.
     * (statically is not necessary as such)
     */
    PyArrayIterObject it;

    if (!PyDataType_REFCHK(PyArray_DESCR(mp))) {
        return 0;
    }
    if (PyArray_DESCR(mp)->type_num != NPY_OBJECT) {
        PyArray_RawIterBaseInit(&it, mp);
        while(it.index < it.size) {
            PyArray_Item_XDECREF(it.dataptr, PyArray_DESCR(mp));
            PyArray_ITER_NEXT(&it);
        }
        return 0;
    }

    if (PyArray_ISONESEGMENT(mp)) {
        data = (PyObject **)PyArray_DATA(mp);
        n = PyArray_SIZE(mp);
        if (PyArray_ISALIGNED(mp)) {
            for (i = 0; i < n; i++, data++) Py_XDECREF(*data);
        }
        else {
            for (i = 0; i < n; i++, data++) {
                memcpy(&temp, data, sizeof(temp));
                Py_XDECREF(temp);
            }
        }
    }
    else { /* handles misaligned data too */
        PyArray_RawIterBaseInit(&it, mp);
        while(it.index < it.size) {
            memcpy(&temp, it.dataptr, sizeof(temp));
            Py_XDECREF(temp);
            PyArray_ITER_NEXT(&it);
        }
    }
    return 0;
}


static void
_fillobject(char *optr, PyObject *obj, PyArray_Descr *dtype);


/*NUMPY_API
 * Assumes contiguous
 */
NPY_NO_EXPORT void
PyArray_FillObjectArray(PyArrayObject *arr, PyObject *obj)
{
    PyArray_Descr* descr = PyArray_DESCR(arr);

    // non-legacy dtypes are responsible for initializing
    // their own internal references
    if (!NPY_DT_is_legacy(NPY_DTYPE(descr))) {
        return;
    }

    npy_intp i,n;
    n = PyArray_SIZE(arr);
    if (descr->type_num == NPY_OBJECT) {
        PyObject **optr;
        optr = (PyObject **)(PyArray_DATA(arr));
        n = PyArray_SIZE(arr);
        if (obj == NULL) {
            for (i = 0; i < n; i++) {
                *optr++ = NULL;
            }
        }
        else {
            for (i = 0; i < n; i++) {
                Py_INCREF(obj);
                *optr++ = obj;
            }
        }
    }
    else {
        char *optr;
        optr = PyArray_DATA(arr);
        for (i = 0; i < n; i++) {
            _fillobject(optr, obj, descr);
            optr += descr->elsize;
        }
    }
}

static void
_fillobject(char *optr, PyObject *obj, PyArray_Descr *dtype)
{
    if (!PyDataType_FLAGCHK(dtype, NPY_ITEM_REFCOUNT)) {
        PyObject *arr;

        if ((obj == Py_None) ||
                (PyLong_Check(obj) && PyLong_AsLong(obj) == 0)) {
            return;
        }
        /* Clear possible long conversion error */
        PyErr_Clear();
        Py_INCREF(dtype);
        arr = PyArray_NewFromDescr(&PyArray_Type, dtype,
                                   0, NULL, NULL, NULL,
                                   0, NULL);
        if (arr!=NULL) {
            dtype->f->setitem(obj, optr, arr);
        }
        Py_XDECREF(arr);
    }
    if (dtype->type_num == NPY_OBJECT) {
        Py_XINCREF(obj);
        memcpy(optr, &obj, sizeof(obj));
    }
    else if (PyDataType_HASFIELDS(dtype)) {
        PyObject *key, *value, *title = NULL;
        PyArray_Descr *new;
        int offset;
        Py_ssize_t pos = 0;

        while (PyDict_Next(dtype->fields, &pos, &key, &value)) {
            if (NPY_TITLE_KEY(key, value)) {
                continue;
            }
            if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset, &title)) {
                return;
            }
            _fillobject(optr + offset, obj, new);
        }
    }
    else if (PyDataType_HASSUBARRAY(dtype)) {
        int size, i, inner_elsize;

        inner_elsize = dtype->subarray->base->elsize;
        if (inner_elsize == 0) {
            /* There cannot be any elements, so return */
            return;
        }
        /* Subarrays are always contiguous in memory */
        size = dtype->elsize / inner_elsize;

        /* Call _fillobject on each item recursively. */
        for (i = 0; i < size; i++){
            _fillobject(optr, obj, dtype->subarray->base);
            optr += inner_elsize;
        }
    }
    else {
        /* This path should not be reachable. */
        assert(0);
    }
    return;
}
