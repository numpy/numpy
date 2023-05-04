#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"

#include "npy_pycompat.h"

#include "common.h"
#include "mapping.h"

#include "sequence.h"
#include "calculation.h"

/*************************************************************************
 ****************   Implement Sequence Protocol **************************
 *************************************************************************/

/* Some of this is repeated in the array_as_mapping protocol.  But
   we fill it in here so that PySequence_XXXX calls work as expected
*/

static int
array_contains(PyArrayObject *self, PyObject *el)
{
    /* equivalent to (self == el).any() */

    int ret;
    PyObject *res, *any;

    res = PyArray_EnsureAnyArray(PyObject_RichCompare((PyObject *)self,
                                                      el, Py_EQ));
    if (res == NULL) {
        return -1;
    }

    any = PyArray_Any((PyArrayObject *)res, NPY_MAXDIMS, NULL);
    Py_DECREF(res);
    if (any == NULL) {
        return -1;
    }

    ret = PyObject_IsTrue(any);
    Py_DECREF(any);
    return ret;
}

static PyObject *
array_concat(PyObject *self, PyObject *other)
{
    /*
     * Throw a type error, when trying to concat NDArrays
     * NOTE: This error is not Thrown when running with PyPy
     */
    PyErr_SetString(PyExc_TypeError,
            "Concatenation operation is not implemented for NumPy arrays, "
            "use np.concatenate() instead. Please do not rely on this error; "
            "it may not be given on all Python implementations.");
    return NULL;
}


NPY_NO_EXPORT PySequenceMethods array_as_sequence = {
    (lenfunc)array_length,                  /*sq_length*/
    (binaryfunc)array_concat,               /*sq_concat for operator.concat*/
    (ssizeargfunc)NULL,
    (ssizeargfunc)array_item,
    (ssizessizeargfunc)NULL,
    (ssizeobjargproc)array_assign_item,     /*sq_ass_item*/
    (ssizessizeobjargproc)NULL,             /*sq_ass_slice*/
    (objobjproc) array_contains,            /*sq_contains */
    (binaryfunc) NULL,                      /*sg_inplace_concat */
    (ssizeargfunc)NULL,
};


/****************** End of Sequence Protocol ****************************/

