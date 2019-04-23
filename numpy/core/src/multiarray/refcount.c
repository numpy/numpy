/*
 * This module corresponds to the `Special functions for NPY_OBJECT`
 * section in the numpy reference for C-API.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"

#include "npy_pycompat.h"

static void
_fillobject(char *optr, PyObject *obj, PyArray_Descr *dtype);


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
        NPY_COPY_PYOBJECT_PTR(&temp, data);
        Py_XINCREF(temp);
    }
    else if (PyDataType_HASFIELDS(descr)) {
        PyObject *key, *value, *title = NULL;
        PyArray_Descr *new;
        int offset;
        Py_ssize_t pos = 0;

        while (PyDict_Next(descr->fields, &pos, &key, &value)) {
            if NPY_TITLE_KEY(key, value) {
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
        NPY_COPY_PYOBJECT_PTR(&temp, data);
        Py_XDECREF(temp);
    }
    else if (PyDataType_HASFIELDS(descr)) {
            PyObject *key, *value, *title = NULL;
            PyArray_Descr *new;
            int offset;
            Py_ssize_t pos = 0;

            while (PyDict_Next(descr->fields, &pos, &key, &value)) {
                if NPY_TITLE_KEY(key, value) {
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
                NPY_COPY_PYOBJECT_PTR(&temp, data);
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
            NPY_COPY_PYOBJECT_PTR(&temp, it->dataptr);
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
*/
NPY_NO_EXPORT int
PyArray_XDECREF(PyArrayObject *mp)
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
            PyArray_Item_XDECREF(it->dataptr, PyArray_DESCR(mp));
            PyArray_ITER_NEXT(it);
        }
        Py_DECREF(it);
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
                NPY_COPY_PYOBJECT_PTR(&temp, data);
                Py_XDECREF(temp);
            }
        }
    }
    else { /* handles misaligned data too */
        it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)mp);
        if (it == NULL) {
            return -1;
        }
        while(it->index < it->size) {
            NPY_COPY_PYOBJECT_PTR(&temp, it->dataptr);
            Py_XDECREF(temp);
            PyArray_ITER_NEXT(it);
        }
        Py_DECREF(it);
    }
    return 0;
}

/*NUMPY_API
 * Assumes contiguous
 */
NPY_NO_EXPORT void
PyArray_FillObjectArray(PyArrayObject *arr, PyObject *obj)
{
    npy_intp i,n;
    n = PyArray_SIZE(arr);
    if (PyArray_DESCR(arr)->type_num == NPY_OBJECT) {
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
            _fillobject(optr, obj, PyArray_DESCR(arr));
            optr += PyArray_DESCR(arr)->elsize;
        }
    }
}

static void
_fillobject(char *optr, PyObject *obj, PyArray_Descr *dtype)
{
    if (!PyDataType_FLAGCHK(dtype, NPY_ITEM_REFCOUNT)) {
        if ((obj == Py_None) || (PyInt_Check(obj) && PyInt_AsLong(obj)==0)) {
            return;
        }
        else {
            PyObject *arr;
            Py_INCREF(dtype);
            arr = PyArray_NewFromDescr(&PyArray_Type, dtype,
                                       0, NULL, NULL, NULL,
                                       0, NULL);
            if (arr!=NULL) {
                dtype->f->setitem(obj, optr, arr);
            }
            Py_XDECREF(arr);
        }
    }
    if (dtype->type_num == NPY_OBJECT) {
        Py_XINCREF(obj);
        NPY_COPY_PYOBJECT_PTR(optr, &obj);
    }
    else if (PyDataType_HASFIELDS(dtype)) {
        PyObject *key, *value, *title = NULL;
        PyArray_Descr *new;
        int offset;
        Py_ssize_t pos = 0;

        while (PyDict_Next(dtype->fields, &pos, &key, &value)) {
            if NPY_TITLE_KEY(key, value) {
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
