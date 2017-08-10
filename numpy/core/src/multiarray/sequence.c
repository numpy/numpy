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
    (ssizessizeargfunc)NULL,
    (ssizeobjargproc)array_assign_item,     /*sq_ass_item*/
    (ssizessizeobjargproc)NULL,             /*sq_ass_slice*/
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
    PyArray_NonzeroFunc *nonzero;
    int ret = NPY_FALSE;
    int needs_api = 0;

    NpyIter *iter;
    NpyIter_IterNextFunc *iternext;
    char **dataptr;
    npy_intp *strideptr, *innersizeptr;
    NPY_BEGIN_THREADS_DEF;

    nonzero = PyArray_DESCR(arr)->f->nonzero;

    iter = NpyIter_New(arr, NPY_ITER_READONLY |
                            NPY_ITER_EXTERNAL_LOOP |
                            NPY_ITER_REFS_OK,
                            NPY_KEEPORDER, NPY_NO_CASTING,
                            NULL);
    if (iter == NULL) {
        return ret;
    }
    needs_api = NpyIter_IterationNeedsAPI(iter);

    /* Get the pointers for inner loop iteration */
    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        NpyIter_Deallocate(iter);
        return -1;
    }

    NPY_BEGIN_THREADS_NDITER(iter);

    dataptr = NpyIter_GetDataPtrArray(iter);
    strideptr = NpyIter_GetInnerStrideArray(iter);
    innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    /* Iterate over all the elements to count the nonzeros */
    do {
        char *data = *dataptr;
        npy_intp stride = *strideptr;
        npy_intp count = *innersizeptr;

        while (count--) {
            if (nonzero(data, arr)) {
                ret = NPY_TRUE;
                goto finish;
            }
            if (needs_api && PyErr_Occurred()) {
                ret = -1;
                goto finish;
            }
            data += stride;
        }

    } while(iternext(iter));


finish:
    NPY_END_THREADS;

    NpyIter_Deallocate(iter);

    return ret;
}
