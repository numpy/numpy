#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"

#include "npy_pycompat.h"

#include "common.h"
#include "mapping.h"

#include "sequence.h"

static int
array_any_nonzero(PyArrayObject *mp);

/*************************************************************************
 ****************   Implement Sequence Protocol **************************
 *************************************************************************/

/* Some of this is repeated in the array_as_mapping protocol.  But
   we fill it in here so that PySequence_XXXX calls work as expected
*/


static PyObject *
array_slice(PyArrayObject *self, Py_ssize_t ilow, Py_ssize_t ihigh)
{
    PyArrayObject *ret;
    PyArray_Descr *dtype;
    Py_ssize_t dim0;
    char *data;
    npy_intp shape[NPY_MAXDIMS];

    if (PyArray_NDIM(self) == 0) {
        PyErr_SetString(PyExc_ValueError, "cannot slice a 0-d array");
        return NULL;
    }

    dim0 = PyArray_DIM(self, 0);
    if (ilow < 0) {
        ilow = 0;
    }
    else if (ilow > dim0) {
        ilow = dim0;
    }
    if (ihigh < ilow) {
        ihigh = ilow;
    }
    else if (ihigh > dim0) {
        ihigh = dim0;
    }

    data = PyArray_DATA(self);
    if (ilow < ihigh) {
        data += ilow * PyArray_STRIDE(self, 0);
    }

    /* Same shape except dimension 0 */
    shape[0] = ihigh - ilow;
    memcpy(shape+1, PyArray_DIMS(self) + 1,
                        (PyArray_NDIM(self)-1)*sizeof(npy_intp));

    dtype = PyArray_DESCR(self);
    Py_INCREF(dtype);
    ret = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(self), dtype,
                             PyArray_NDIM(self), shape,
                             PyArray_STRIDES(self), data,
                             PyArray_FLAGS(self),
                             (PyObject *)self);
    if (ret == NULL) {
        return NULL;
    }
    Py_INCREF(self);
    if (PyArray_SetBaseObject(ret, (PyObject *)self) < 0) {
        Py_DECREF(ret);
        return NULL;
    }
    PyArray_UpdateFlags(ret, NPY_ARRAY_UPDATE_ALL);

    return (PyObject *)ret;
}


static int
array_ass_slice(PyArrayObject *self, Py_ssize_t ilow,
                Py_ssize_t ihigh, PyObject *v) {
    int ret;
    PyArrayObject *tmp;

    if (v == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "cannot delete array elements");
        return -1;
    }
    if (PyArray_FailUnlessWriteable(self, "assignment destination") < 0) {
        return -1;
    }
    tmp = (PyArrayObject *)array_slice(self, ilow, ihigh);
    if (tmp == NULL) {
        return -1;
    }
    ret = PyArray_CopyObject(tmp, v);
    Py_DECREF(tmp);

    return ret;
}

static int
array_contains(PyArrayObject *self, PyObject *el)
{
    /* equivalent to (self == el).any() */

    PyObject *res;
    int ret;

    res = PyArray_EnsureAnyArray(PyObject_RichCompare((PyObject *)self,
                                                      el, Py_EQ));
    if (res == NULL) {
        return -1;
    }
    ret = array_any_nonzero((PyArrayObject *)res);
    Py_DECREF(res);
    return ret;
}

NPY_NO_EXPORT PySequenceMethods array_as_sequence = {
    (lenfunc)array_length,                  /*sq_length*/
    (binaryfunc)NULL,                       /*sq_concat is handled by nb_add*/
    (ssizeargfunc)NULL,
    (ssizeargfunc)array_item,
    (ssizessizeargfunc)array_slice,
    (ssizeobjargproc)array_ass_item,        /*sq_ass_item*/
    (ssizessizeobjargproc)array_ass_slice,  /*sq_ass_slice*/
    (objobjproc) array_contains,            /*sq_contains */
    (binaryfunc) NULL,                      /*sg_inplace_concat */
    (ssizeargfunc)NULL,
};


/****************** End of Sequence Protocol ****************************/

/*
 * Helpers
 */

/* Array evaluates as "TRUE" if any of the elements are non-zero*/
static int
array_any_nonzero(PyArrayObject *arr)
{
    npy_intp counter;
    PyArrayIterObject *it;
    npy_bool anyTRUE = NPY_FALSE;

    it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)arr);
    if (it == NULL) {
        return anyTRUE;
    }
    counter = it->size;
    while (counter--) {
        if (PyArray_DESCR(arr)->f->nonzero(it->dataptr, arr)) {
            anyTRUE = NPY_TRUE;
            break;
        }
        PyArray_ITER_NEXT(it);
    }
    Py_DECREF(it);
    return anyTRUE;
}
