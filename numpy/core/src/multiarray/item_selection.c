#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "numpy/npy_math.h"
#include "numpy/npy_cpu.h"

#include "npy_config.h"

#include "npy_pycompat.h"

#include "multiarraymodule.h"
#include "common.h"
#include "arrayobject.h"
#include "ctors.h"
#include "lowlevel_strided_loops.h"
#include "array_assign.h"

#include "item_selection.h"
#include "npy_sort.h"
#include "npy_partition.h"
#include "npy_binsearch.h"
#include "alloc.h"

/*NUMPY_API
 * Take
 */
NPY_NO_EXPORT PyObject *
PyArray_TakeFrom(PyArrayObject *self0, PyObject *indices0, int axis,
                 PyArrayObject *out, NPY_CLIPMODE clipmode)
{
    PyArray_Descr *dtype;
    PyArray_FastTakeFunc *func;
    PyArrayObject *obj = NULL, *self, *indices;
    npy_intp nd, i, j, n, m, k, max_item, tmp, chunk, itemsize, nelem;
    npy_intp shape[NPY_MAXDIMS];
    char *src, *dest, *tmp_src;
    int err;
    npy_bool needs_refcounting;

    indices = NULL;
    self = (PyArrayObject *)PyArray_CheckAxis(self0, &axis,
                                    NPY_ARRAY_CARRAY_RO);
    if (self == NULL) {
        return NULL;
    }
    indices = (PyArrayObject *)PyArray_ContiguousFromAny(indices0,
                                                         NPY_INTP,
                                                         0, 0);
    if (indices == NULL) {
        goto fail;
    }

    n = m = chunk = 1;
    nd = PyArray_NDIM(self) + PyArray_NDIM(indices) - 1;
    for (i = 0; i < nd; i++) {
        if (i < axis) {
            shape[i] = PyArray_DIMS(self)[i];
            n *= shape[i];
        }
        else {
            if (i < axis+PyArray_NDIM(indices)) {
                shape[i] = PyArray_DIMS(indices)[i-axis];
                m *= shape[i];
            }
            else {
                shape[i] = PyArray_DIMS(self)[i-PyArray_NDIM(indices)+1];
                chunk *= shape[i];
            }
        }
    }
    if (!out) {
        dtype = PyArray_DESCR(self);
        Py_INCREF(dtype);
        obj = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(self),
                                                    dtype,
                                                    nd, shape,
                                                    NULL, NULL, 0,
                                                    (PyObject *)self);

        if (obj == NULL) {
            goto fail;
        }

    }
    else {
        int flags = NPY_ARRAY_CARRAY | NPY_ARRAY_WRITEBACKIFCOPY;

        if ((PyArray_NDIM(out) != nd) ||
            !PyArray_CompareLists(PyArray_DIMS(out), shape, nd)) {
            PyErr_SetString(PyExc_ValueError,
                        "output array does not match result of ndarray.take");
            goto fail;
        }

        if (clipmode == NPY_RAISE) {
            /*
             * we need to make sure and get a copy
             * so the input array is not changed
             * before the error is called
             */
            flags |= NPY_ARRAY_ENSURECOPY;
        }
        dtype = PyArray_DESCR(self);
        Py_INCREF(dtype);
        obj = (PyArrayObject *)PyArray_FromArray(out, dtype, flags);
        if (obj == NULL) {
            goto fail;
        }
    }

    max_item = PyArray_DIMS(self)[axis];
    nelem = chunk;
    itemsize = PyArray_ITEMSIZE(obj);
    chunk = chunk * itemsize;
    src = PyArray_DATA(self);
    dest = PyArray_DATA(obj);
    needs_refcounting = PyDataType_REFCHK(PyArray_DESCR(self));

    if ((max_item == 0) && (PyArray_SIZE(obj) != 0)) {
        /* Index error, since that is the usual error for raise mode */
        PyErr_SetString(PyExc_IndexError,
                    "cannot do a non-empty take from an empty axes.");
        goto fail;
    }

    func = PyArray_DESCR(self)->f->fasttake;
    if (func == NULL) {
        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(self));
        switch(clipmode) {
        case NPY_RAISE:
            for (i = 0; i < n; i++) {
                for (j = 0; j < m; j++) {
                    tmp = ((npy_intp *)(PyArray_DATA(indices)))[j];
                    if (check_and_adjust_index(&tmp, max_item, axis,
                                               _save) < 0) {
                        goto fail;
                    }
                    tmp_src = src + tmp * chunk;
                    if (needs_refcounting) {
                        for (k=0; k < nelem; k++) {
                            PyArray_Item_INCREF(tmp_src, PyArray_DESCR(self));
                            PyArray_Item_XDECREF(dest, PyArray_DESCR(self));
                            memmove(dest, tmp_src, itemsize);
                            dest += itemsize;
                            tmp_src += itemsize;
                        }
                    }
                    else {
                        memmove(dest, tmp_src, chunk);
                        dest += chunk;
                    }
                }
                src += chunk*max_item;
            }
            break;
        case NPY_WRAP:
            for (i = 0; i < n; i++) {
                for (j = 0; j < m; j++) {
                    tmp = ((npy_intp *)(PyArray_DATA(indices)))[j];
                    if (tmp < 0) {
                        while (tmp < 0) {
                            tmp += max_item;
                        }
                    }
                    else if (tmp >= max_item) {
                        while (tmp >= max_item) {
                            tmp -= max_item;
                        }
                    }
                    tmp_src = src + tmp * chunk;
                    if (needs_refcounting) {
                        for (k=0; k < nelem; k++) {
                            PyArray_Item_INCREF(tmp_src, PyArray_DESCR(self));
                            PyArray_Item_XDECREF(dest, PyArray_DESCR(self));
                            memmove(dest, tmp_src, itemsize);
                            dest += itemsize;
                            tmp_src += itemsize;
                        }
                    }
                    else {
                        memmove(dest, tmp_src, chunk);
                        dest += chunk;
                    }
                }
                src += chunk*max_item;
            }
            break;
        case NPY_CLIP:
            for (i = 0; i < n; i++) {
                for (j = 0; j < m; j++) {
                    tmp = ((npy_intp *)(PyArray_DATA(indices)))[j];
                    if (tmp < 0) {
                        tmp = 0;
                    }
                    else if (tmp >= max_item) {
                        tmp = max_item - 1;
                    }
                    tmp_src = src + tmp * chunk;
                    if (needs_refcounting) {
                        for (k=0; k < nelem; k++) {
                            PyArray_Item_INCREF(tmp_src, PyArray_DESCR(self));
                            PyArray_Item_XDECREF(dest, PyArray_DESCR(self));
                            memmove(dest, tmp_src, itemsize);
                            dest += itemsize;
                            tmp_src += itemsize;
                        }
                    }
                    else {
                        memmove(dest, tmp_src, chunk);
                        dest += chunk;
                    }
                }
                src += chunk*max_item;
            }
            break;
        }
        NPY_END_THREADS;
    }
    else {
        /* no gil release, need it for error reporting */
        err = func(dest, src, (npy_intp *)(PyArray_DATA(indices)),
                    max_item, n, m, nelem, clipmode);
        if (err) {
            goto fail;
        }
    }

    Py_XDECREF(indices);
    Py_XDECREF(self);
    if (out != NULL && out != obj) {
        Py_INCREF(out);
        PyArray_ResolveWritebackIfCopy(obj);
        Py_DECREF(obj);
        obj = out;
    }
    return (PyObject *)obj;

 fail:
    PyArray_DiscardWritebackIfCopy(obj);
    Py_XDECREF(obj);
    Py_XDECREF(indices);
    Py_XDECREF(self);
    return NULL;
}

/*NUMPY_API
 * Put values into an array
 */
NPY_NO_EXPORT PyObject *
PyArray_PutTo(PyArrayObject *self, PyObject* values0, PyObject *indices0,
              NPY_CLIPMODE clipmode)
{
    PyArrayObject  *indices, *values;
    npy_intp i, chunk, ni, max_item, nv, tmp;
    char *src, *dest;
    int copied = 0;

    indices = NULL;
    values = NULL;
    if (!PyArray_Check(self)) {
        PyErr_SetString(PyExc_TypeError,
                        "put: first argument must be an array");
        return NULL;
    }

    if (PyArray_FailUnlessWriteable(self, "put: output array") < 0) {
        return NULL;
    }

    if (!PyArray_ISCONTIGUOUS(self)) {
        PyArrayObject *obj;
        int flags = NPY_ARRAY_CARRAY | NPY_ARRAY_WRITEBACKIFCOPY;

        if (clipmode == NPY_RAISE) {
            flags |= NPY_ARRAY_ENSURECOPY;
        }
        Py_INCREF(PyArray_DESCR(self));
        obj = (PyArrayObject *)PyArray_FromArray(self,
                                                 PyArray_DESCR(self), flags);
        if (obj != self) {
            copied = 1;
        }
        self = obj;
    }
    max_item = PyArray_SIZE(self);
    dest = PyArray_DATA(self);
    chunk = PyArray_DESCR(self)->elsize;
    indices = (PyArrayObject *)PyArray_ContiguousFromAny(indices0,
                                                         NPY_INTP, 0, 0);
    if (indices == NULL) {
        goto fail;
    }
    ni = PyArray_SIZE(indices);
    Py_INCREF(PyArray_DESCR(self));
    values = (PyArrayObject *)PyArray_FromAny(values0, PyArray_DESCR(self), 0, 0,
                              NPY_ARRAY_DEFAULT | NPY_ARRAY_FORCECAST, NULL);
    if (values == NULL) {
        goto fail;
    }
    nv = PyArray_SIZE(values);
    if (nv <= 0) {
        goto finish;
    }
    if (PyDataType_REFCHK(PyArray_DESCR(self))) {
        switch(clipmode) {
        case NPY_RAISE:
            for (i = 0; i < ni; i++) {
                src = PyArray_BYTES(values) + chunk*(i % nv);
                tmp = ((npy_intp *)(PyArray_DATA(indices)))[i];
                if (check_and_adjust_index(&tmp, max_item, 0, NULL) < 0) {
                    goto fail;
                }
                PyArray_Item_INCREF(src, PyArray_DESCR(self));
                PyArray_Item_XDECREF(dest+tmp*chunk, PyArray_DESCR(self));
                memmove(dest + tmp*chunk, src, chunk);
            }
            break;
        case NPY_WRAP:
            for (i = 0; i < ni; i++) {
                src = PyArray_BYTES(values) + chunk * (i % nv);
                tmp = ((npy_intp *)(PyArray_DATA(indices)))[i];
                if (tmp < 0) {
                    while (tmp < 0) {
                        tmp += max_item;
                    }
                }
                else if (tmp >= max_item) {
                    while (tmp >= max_item) {
                        tmp -= max_item;
                    }
                }
                PyArray_Item_INCREF(src, PyArray_DESCR(self));
                PyArray_Item_XDECREF(dest+tmp*chunk, PyArray_DESCR(self));
                memmove(dest + tmp * chunk, src, chunk);
            }
            break;
        case NPY_CLIP:
            for (i = 0; i < ni; i++) {
                src = PyArray_BYTES(values) + chunk * (i % nv);
                tmp = ((npy_intp *)(PyArray_DATA(indices)))[i];
                if (tmp < 0) {
                    tmp = 0;
                }
                else if (tmp >= max_item) {
                    tmp = max_item - 1;
                }
                PyArray_Item_INCREF(src, PyArray_DESCR(self));
                PyArray_Item_XDECREF(dest+tmp*chunk, PyArray_DESCR(self));
                memmove(dest + tmp * chunk, src, chunk);
            }
            break;
        }
    }
    else {
        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(ni);
        switch(clipmode) {
        case NPY_RAISE:
            for (i = 0; i < ni; i++) {
                src = PyArray_BYTES(values) + chunk * (i % nv);
                tmp = ((npy_intp *)(PyArray_DATA(indices)))[i];
                if (check_and_adjust_index(&tmp, max_item, 0, _save) < 0) {
                    goto fail;
                }
                memmove(dest + tmp * chunk, src, chunk);
            }
            break;
        case NPY_WRAP:
            for (i = 0; i < ni; i++) {
                src = PyArray_BYTES(values) + chunk * (i % nv);
                tmp = ((npy_intp *)(PyArray_DATA(indices)))[i];
                if (tmp < 0) {
                    while (tmp < 0) {
                        tmp += max_item;
                    }
                }
                else if (tmp >= max_item) {
                    while (tmp >= max_item) {
                        tmp -= max_item;
                    }
                }
                memmove(dest + tmp * chunk, src, chunk);
            }
            break;
        case NPY_CLIP:
            for (i = 0; i < ni; i++) {
                src = PyArray_BYTES(values) + chunk * (i % nv);
                tmp = ((npy_intp *)(PyArray_DATA(indices)))[i];
                if (tmp < 0) {
                    tmp = 0;
                }
                else if (tmp >= max_item) {
                    tmp = max_item - 1;
                }
                memmove(dest + tmp * chunk, src, chunk);
            }
            break;
        }
        NPY_END_THREADS;
    }

 finish:
    Py_XDECREF(values);
    Py_XDECREF(indices);
    if (copied) {
        PyArray_ResolveWritebackIfCopy(self);
        Py_DECREF(self);
    }
    Py_RETURN_NONE;

 fail:
    Py_XDECREF(indices);
    Py_XDECREF(values);
    if (copied) {
        PyArray_DiscardWritebackIfCopy(self);
        Py_XDECREF(self);
    }
    return NULL;
}

/*NUMPY_API
 * Put values into an array according to a mask.
 */
NPY_NO_EXPORT PyObject *
PyArray_PutMask(PyArrayObject *self, PyObject* values0, PyObject* mask0)
{
    PyArray_FastPutmaskFunc *func;
    PyArrayObject *mask, *values;
    PyArray_Descr *dtype;
    npy_intp i, j, chunk, ni, max_item, nv;
    char *src, *dest;
    npy_bool *mask_data;
    int copied = 0;

    mask = NULL;
    values = NULL;
    if (!PyArray_Check(self)) {
        PyErr_SetString(PyExc_TypeError,
                        "putmask: first argument must "
                        "be an array");
        return NULL;
    }
    if (!PyArray_ISCONTIGUOUS(self)) {
        PyArrayObject *obj;

        dtype = PyArray_DESCR(self);
        Py_INCREF(dtype);
        obj = (PyArrayObject *)PyArray_FromArray(self, dtype,
                                NPY_ARRAY_CARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (obj != self) {
            copied = 1;
        }
        self = obj;
    }

    max_item = PyArray_SIZE(self);
    dest = PyArray_DATA(self);
    chunk = PyArray_DESCR(self)->elsize;
    mask = (PyArrayObject *)PyArray_FROM_OTF(mask0, NPY_BOOL,
                                NPY_ARRAY_CARRAY | NPY_ARRAY_FORCECAST);
    if (mask == NULL) {
        goto fail;
    }
    ni = PyArray_SIZE(mask);
    if (ni != max_item) {
        PyErr_SetString(PyExc_ValueError,
                        "putmask: mask and data must be "
                        "the same size");
        goto fail;
    }
    mask_data = PyArray_DATA(mask);
    dtype = PyArray_DESCR(self);
    Py_INCREF(dtype);
    values = (PyArrayObject *)PyArray_FromAny(values0, dtype,
                                    0, 0, NPY_ARRAY_CARRAY, NULL);
    if (values == NULL) {
        goto fail;
    }
    nv = PyArray_SIZE(values); /* zero if null array */
    if (nv <= 0) {
        Py_XDECREF(values);
        Py_XDECREF(mask);
        Py_RETURN_NONE;
    }
    src = PyArray_DATA(values);

    if (PyDataType_REFCHK(PyArray_DESCR(self))) {
        for (i = 0, j = 0; i < ni; i++, j++) {
            if (j >= nv) {
                j = 0;
            }
            if (mask_data[i]) {
                char *src_ptr = src + j*chunk;
                char *dest_ptr = dest + i*chunk;

                PyArray_Item_INCREF(src_ptr, PyArray_DESCR(self));
                PyArray_Item_XDECREF(dest_ptr, PyArray_DESCR(self));
                memmove(dest_ptr, src_ptr, chunk);
            }
        }
    }
    else {
        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(self));
        func = PyArray_DESCR(self)->f->fastputmask;
        if (func == NULL) {
            for (i = 0, j = 0; i < ni; i++, j++) {
                if (j >= nv) {
                    j = 0;
                }
                if (mask_data[i]) {
                    memmove(dest + i*chunk, src + j*chunk, chunk);
                }
            }
        }
        else {
            func(dest, mask_data, ni, src, nv);
        }
        NPY_END_THREADS;
    }

    Py_XDECREF(values);
    Py_XDECREF(mask);
    if (copied) {
        PyArray_ResolveWritebackIfCopy(self);
        Py_DECREF(self);
    }
    Py_RETURN_NONE;

 fail:
    Py_XDECREF(mask);
    Py_XDECREF(values);
    if (copied) {
        PyArray_DiscardWritebackIfCopy(self);
        Py_XDECREF(self);
    }
    return NULL;
}

/*NUMPY_API
 * Repeat the array.
 */
NPY_NO_EXPORT PyObject *
PyArray_Repeat(PyArrayObject *aop, PyObject *op, int axis)
{
    npy_intp *counts;
    npy_intp n, n_outer, i, j, k, chunk;
    npy_intp total = 0;
    npy_bool broadcast = NPY_FALSE;
    PyArrayObject *repeats = NULL;
    PyObject *ap = NULL;
    PyArrayObject *ret = NULL;
    char *new_data, *old_data;

    repeats = (PyArrayObject *)PyArray_ContiguousFromAny(op, NPY_INTP, 0, 1);
    if (repeats == NULL) {
        return NULL;
    }

    /*
     * Scalar and size 1 'repeat' arrays broadcast to any shape, for all
     * other inputs the dimension must match exactly.
     */
    if (PyArray_NDIM(repeats) == 0 || PyArray_SIZE(repeats) == 1) {
        broadcast = NPY_TRUE;
    }

    counts = (npy_intp *)PyArray_DATA(repeats);

    if ((ap = PyArray_CheckAxis(aop, &axis, NPY_ARRAY_CARRAY)) == NULL) {
        Py_DECREF(repeats);
        return NULL;
    }

    aop = (PyArrayObject *)ap;
    n = PyArray_DIM(aop, axis);

    if (!broadcast && PyArray_SIZE(repeats) != n) {
        PyErr_Format(PyExc_ValueError,
                     "operands could not be broadcast together "
                     "with shape (%zd,) (%zd,)", n, PyArray_DIM(repeats, 0));
        goto fail;
    }
    if (broadcast) {
        total = counts[0] * n;
    }
    else {
        for (j = 0; j < n; j++) {
            if (counts[j] < 0) {
                PyErr_SetString(PyExc_ValueError, "count < 0");
                goto fail;
            }
            total += counts[j];
        }
    }

    /* Construct new array */
    PyArray_DIMS(aop)[axis] = total;
    Py_INCREF(PyArray_DESCR(aop));
    ret = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(aop),
                                                PyArray_DESCR(aop),
                                                PyArray_NDIM(aop),
                                                PyArray_DIMS(aop),
                                                NULL, NULL, 0,
                                                (PyObject *)aop);
    PyArray_DIMS(aop)[axis] = n;
    if (ret == NULL) {
        goto fail;
    }
    new_data = PyArray_DATA(ret);
    old_data = PyArray_DATA(aop);

    chunk = PyArray_DESCR(aop)->elsize;
    for(i = axis + 1; i < PyArray_NDIM(aop); i++) {
        chunk *= PyArray_DIMS(aop)[i];
    }

    n_outer = 1;
    for (i = 0; i < axis; i++) {
        n_outer *= PyArray_DIMS(aop)[i];
    }
    for (i = 0; i < n_outer; i++) {
        for (j = 0; j < n; j++) {
            npy_intp tmp = broadcast ? counts[0] : counts[j];
            for (k = 0; k < tmp; k++) {
                memcpy(new_data, old_data, chunk);
                new_data += chunk;
            }
            old_data += chunk;
        }
    }

    Py_DECREF(repeats);
    PyArray_INCREF(ret);
    Py_XDECREF(aop);
    return (PyObject *)ret;

 fail:
    Py_DECREF(repeats);
    Py_XDECREF(aop);
    Py_XDECREF(ret);
    return NULL;
}

/*NUMPY_API
 */
NPY_NO_EXPORT PyObject *
PyArray_Choose(PyArrayObject *ip, PyObject *op, PyArrayObject *out,
               NPY_CLIPMODE clipmode)
{
    PyArrayObject *obj = NULL;
    PyArray_Descr *dtype;
    int n, elsize;
    npy_intp i;
    char *ret_data;
    PyArrayObject **mps, *ap;
    PyArrayMultiIterObject *multi = NULL;
    npy_intp mi;
    ap = NULL;

    /*
     * Convert all inputs to arrays of a common type
     * Also makes them C-contiguous
     */
    mps = PyArray_ConvertToCommonType(op, &n);
    if (mps == NULL) {
        return NULL;
    }
    for (i = 0; i < n; i++) {
        if (mps[i] == NULL) {
            goto fail;
        }
    }
    ap = (PyArrayObject *)PyArray_FROM_OT((PyObject *)ip, NPY_INTP);
    if (ap == NULL) {
        goto fail;
    }
    /* Broadcast all arrays to each other, index array at the end. */
    multi = (PyArrayMultiIterObject *)
        PyArray_MultiIterFromObjects((PyObject **)mps, n, 1, ap);
    if (multi == NULL) {
        goto fail;
    }
    /* Set-up return array */
    if (out == NULL) {
        dtype = PyArray_DESCR(mps[0]);
        Py_INCREF(dtype);
        obj = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(ap),
                                                    dtype,
                                                    multi->nd,
                                                    multi->dimensions,
                                                    NULL, NULL, 0,
                                                    (PyObject *)ap);
    }
    else {
        int flags = NPY_ARRAY_CARRAY |
                    NPY_ARRAY_WRITEBACKIFCOPY |
                    NPY_ARRAY_FORCECAST;

        if ((PyArray_NDIM(out) != multi->nd)
                    || !PyArray_CompareLists(PyArray_DIMS(out),
                                             multi->dimensions,
                                             multi->nd)) {
            PyErr_SetString(PyExc_TypeError,
                            "choose: invalid shape for output array.");
            goto fail;
        }
        if (clipmode == NPY_RAISE) {
            /*
             * we need to make sure and get a copy
             * so the input array is not changed
             * before the error is called
             */
            flags |= NPY_ARRAY_ENSURECOPY;
        }
        dtype = PyArray_DESCR(mps[0]);
        Py_INCREF(dtype);
        obj = (PyArrayObject *)PyArray_FromArray(out, dtype, flags);
    }

    if (obj == NULL) {
        goto fail;
    }
    elsize = PyArray_DESCR(obj)->elsize;
    ret_data = PyArray_DATA(obj);

    while (PyArray_MultiIter_NOTDONE(multi)) {
        mi = *((npy_intp *)PyArray_MultiIter_DATA(multi, n));
        if (mi < 0 || mi >= n) {
            switch(clipmode) {
            case NPY_RAISE:
                PyErr_SetString(PyExc_ValueError,
                        "invalid entry in choice "\
                        "array");
                goto fail;
            case NPY_WRAP:
                if (mi < 0) {
                    while (mi < 0) {
                        mi += n;
                    }
                }
                else {
                    while (mi >= n) {
                        mi -= n;
                    }
                }
                break;
            case NPY_CLIP:
                if (mi < 0) {
                    mi = 0;
                }
                else if (mi >= n) {
                    mi = n - 1;
                }
                break;
            }
        }
        memmove(ret_data, PyArray_MultiIter_DATA(multi, mi), elsize);
        ret_data += elsize;
        PyArray_MultiIter_NEXT(multi);
    }

    PyArray_INCREF(obj);
    Py_DECREF(multi);
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
    }
    Py_DECREF(ap);
    npy_free_cache(mps, n * sizeof(mps[0]));
    if (out != NULL && out != obj) {
        Py_INCREF(out);
        PyArray_ResolveWritebackIfCopy(obj);
        Py_DECREF(obj);
        obj = out;
    }
    return (PyObject *)obj;

 fail:
    Py_XDECREF(multi);
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
    }
    Py_XDECREF(ap);
    npy_free_cache(mps, n * sizeof(mps[0]));
    PyArray_DiscardWritebackIfCopy(obj);
    Py_XDECREF(obj);
    return NULL;
}

/*
 * These algorithms use special sorting.  They are not called unless the
 * underlying sort function for the type is available.  Note that axis is
 * already valid. The sort functions require 1-d contiguous and well-behaved
 * data.  Therefore, a copy will be made of the data if needed before handing
 * it to the sorting routine.  An iterator is constructed and adjusted to walk
 * over all but the desired sorting axis.
 */
static int
_new_sortlike(PyArrayObject *op, int axis, PyArray_SortFunc *sort,
              PyArray_PartitionFunc *part, npy_intp *kth, npy_intp nkth)
{
    npy_intp N = PyArray_DIM(op, axis);
    npy_intp elsize = (npy_intp)PyArray_ITEMSIZE(op);
    npy_intp astride = PyArray_STRIDE(op, axis);
    int swap = PyArray_ISBYTESWAPPED(op);
    int needcopy = !IsAligned(op) || swap || astride != elsize;
    int hasrefs = PyDataType_REFCHK(PyArray_DESCR(op));

    PyArray_CopySwapNFunc *copyswapn = PyArray_DESCR(op)->f->copyswapn;
    char *buffer = NULL;

    PyArrayIterObject *it;
    npy_intp size;

    int ret = 0;

    NPY_BEGIN_THREADS_DEF;

    /* Check if there is any sorting to do */
    if (N <= 1 || PyArray_SIZE(op) == 0) {
        return 0;
    }

    it = (PyArrayIterObject *)PyArray_IterAllButAxis((PyObject *)op, &axis);
    if (it == NULL) {
        return -1;
    }
    size = it->size;

    if (needcopy) {
        buffer = npy_alloc_cache(N * elsize);
        if (buffer == NULL) {
            ret = -1;
            goto fail;
        }
    }

    NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(op));

    while (size--) {
        char *bufptr = it->dataptr;

        if (needcopy) {
            if (hasrefs) {
                /*
                 * For dtype's with objects, copyswapn Py_XINCREF's src
                 * and Py_XDECREF's dst. This would crash if called on
                 * an uninitialized buffer, or leak a reference to each
                 * object if initialized.
                 *
                 * So, first do the copy with no refcounting...
                 */
                _unaligned_strided_byte_copy(buffer, elsize,
                                             it->dataptr, astride, N, elsize);
                /* ...then swap in-place if needed */
                if (swap) {
                    copyswapn(buffer, elsize, NULL, 0, N, swap, op);
                }
            }
            else {
                copyswapn(buffer, elsize, it->dataptr, astride, N, swap, op);
            }
            bufptr = buffer;
        }
        /*
         * TODO: If the input array is byte-swapped but contiguous and
         * aligned, it could be swapped (and later unswapped) in-place
         * rather than after copying to the buffer. Care would have to
         * be taken to ensure that, if there is an error in the call to
         * sort or part, the unswapping is still done before returning.
         */

        if (part == NULL) {
            ret = sort(bufptr, N, op);
            if (hasrefs && PyErr_Occurred()) {
                ret = -1;
            }
            if (ret < 0) {
                goto fail;
            }
        }
        else {
            npy_intp pivots[NPY_MAX_PIVOT_STACK];
            npy_intp npiv = 0;
            npy_intp i;
            for (i = 0; i < nkth; ++i) {
                ret = part(bufptr, N, kth[i], pivots, &npiv, op);
                if (hasrefs && PyErr_Occurred()) {
                    ret = -1;
                }
                if (ret < 0) {
                    goto fail;
                }
            }
        }

        if (needcopy) {
            if (hasrefs) {
                if (swap) {
                    copyswapn(buffer, elsize, NULL, 0, N, swap, op);
                }
                _unaligned_strided_byte_copy(it->dataptr, astride,
                                             buffer, elsize, N, elsize);
            }
            else {
                copyswapn(it->dataptr, astride, buffer, elsize, N, swap, op);
            }
        }

        PyArray_ITER_NEXT(it);
    }

fail:
    NPY_END_THREADS_DESCR(PyArray_DESCR(op));
    npy_free_cache(buffer, N * elsize);
    if (ret < 0 && !PyErr_Occurred()) {
        /* Out of memory during sorting or buffer creation */
        PyErr_NoMemory();
    }
    Py_DECREF(it);

    return ret;
}

static PyObject*
_new_argsortlike(PyArrayObject *op, int axis, PyArray_ArgSortFunc *argsort,
                 PyArray_ArgPartitionFunc *argpart,
                 npy_intp *kth, npy_intp nkth)
{
    npy_intp N = PyArray_DIM(op, axis);
    npy_intp elsize = (npy_intp)PyArray_ITEMSIZE(op);
    npy_intp astride = PyArray_STRIDE(op, axis);
    int swap = PyArray_ISBYTESWAPPED(op);
    int needcopy = !IsAligned(op) || swap || astride != elsize;
    int hasrefs = PyDataType_REFCHK(PyArray_DESCR(op));
    int needidxbuffer;

    PyArray_CopySwapNFunc *copyswapn = PyArray_DESCR(op)->f->copyswapn;
    char *valbuffer = NULL;
    npy_intp *idxbuffer = NULL;

    PyArrayObject *rop;
    npy_intp rstride;

    PyArrayIterObject *it, *rit;
    npy_intp size;

    int ret = 0;

    NPY_BEGIN_THREADS_DEF;

    rop = (PyArrayObject *)PyArray_NewFromDescr(
            Py_TYPE(op), PyArray_DescrFromType(NPY_INTP),
            PyArray_NDIM(op), PyArray_DIMS(op), NULL, NULL,
            0, (PyObject *)op);
    if (rop == NULL) {
        return NULL;
    }
    rstride = PyArray_STRIDE(rop, axis);
    needidxbuffer = rstride != sizeof(npy_intp);

    /* Check if there is any argsorting to do */
    if (N <= 1 || PyArray_SIZE(op) == 0) {
        memset(PyArray_DATA(rop), 0, PyArray_NBYTES(rop));
        return (PyObject *)rop;
    }

    it = (PyArrayIterObject *)PyArray_IterAllButAxis((PyObject *)op, &axis);
    rit = (PyArrayIterObject *)PyArray_IterAllButAxis((PyObject *)rop, &axis);
    if (it == NULL || rit == NULL) {
        ret = -1;
        goto fail;
    }
    size = it->size;

    if (needcopy) {
        valbuffer = npy_alloc_cache(N * elsize);
        if (valbuffer == NULL) {
            ret = -1;
            goto fail;
        }
    }

    if (needidxbuffer) {
        idxbuffer = (npy_intp *)npy_alloc_cache(N * sizeof(npy_intp));
        if (idxbuffer == NULL) {
            ret = -1;
            goto fail;
        }
    }

    NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(op));

    while (size--) {
        char *valptr = it->dataptr;
        npy_intp *idxptr = (npy_intp *)rit->dataptr;
        npy_intp *iptr, i;

        if (needcopy) {
            if (hasrefs) {
                /*
                 * For dtype's with objects, copyswapn Py_XINCREF's src
                 * and Py_XDECREF's dst. This would crash if called on
                 * an uninitialized valbuffer, or leak a reference to
                 * each object item if initialized.
                 *
                 * So, first do the copy with no refcounting...
                 */
                 _unaligned_strided_byte_copy(valbuffer, elsize,
                                              it->dataptr, astride, N, elsize);
                /* ...then swap in-place if needed */
                if (swap) {
                    copyswapn(valbuffer, elsize, NULL, 0, N, swap, op);
                }
            }
            else {
                copyswapn(valbuffer, elsize,
                          it->dataptr, astride, N, swap, op);
            }
            valptr = valbuffer;
        }

        if (needidxbuffer) {
            idxptr = idxbuffer;
        }

        iptr = idxptr;
        for (i = 0; i < N; ++i) {
            *iptr++ = i;
        }

        if (argpart == NULL) {
            ret = argsort(valptr, idxptr, N, op);
#if defined(NPY_PY3K)
            /* Object comparisons may raise an exception in Python 3 */
            if (hasrefs && PyErr_Occurred()) {
                ret = -1;
            }
#endif
            if (ret < 0) {
                goto fail;
            }
        }
        else {
            npy_intp pivots[NPY_MAX_PIVOT_STACK];
            npy_intp npiv = 0;

            for (i = 0; i < nkth; ++i) {
                ret = argpart(valptr, idxptr, N, kth[i], pivots, &npiv, op);
#if defined(NPY_PY3K)
                /* Object comparisons may raise an exception in Python 3 */
                if (hasrefs && PyErr_Occurred()) {
                    ret = -1;
                }
#endif
                if (ret < 0) {
                    goto fail;
                }
            }
        }

        if (needidxbuffer) {
            char *rptr = rit->dataptr;
            iptr = idxbuffer;

            for (i = 0; i < N; ++i) {
                *(npy_intp *)rptr = *iptr++;
                rptr += rstride;
            }
        }

        PyArray_ITER_NEXT(it);
        PyArray_ITER_NEXT(rit);
    }

fail:
    NPY_END_THREADS_DESCR(PyArray_DESCR(op));
    npy_free_cache(valbuffer, N * elsize);
    npy_free_cache(idxbuffer, N * sizeof(npy_intp));
    if (ret < 0) {
        if (!PyErr_Occurred()) {
            /* Out of memory during sorting or buffer creation */
            PyErr_NoMemory();
        }
        Py_XDECREF(rop);
        rop = NULL;
    }
    Py_XDECREF(it);
    Py_XDECREF(rit);

    return (PyObject *)rop;
}


/*NUMPY_API
 * Sort an array in-place
 */
NPY_NO_EXPORT int
PyArray_Sort(PyArrayObject *op, int axis, NPY_SORTKIND which)
{
    PyArray_SortFunc *sort;
    int n = PyArray_NDIM(op);

    if (check_and_adjust_axis(&axis, n) < 0) {
        return -1;
    }

    if (PyArray_FailUnlessWriteable(op, "sort array") < 0) {
        return -1;
    }

    if (which < 0 || which >= NPY_NSORTS) {
        PyErr_SetString(PyExc_ValueError, "not a valid sort kind");
        return -1;
    }

    sort = PyArray_DESCR(op)->f->sort[which];
    if (sort == NULL) {
        if (PyArray_DESCR(op)->f->compare) {
            switch (which) {
                default:
                case NPY_QUICKSORT:
                    sort = npy_quicksort;
                    break;
                case NPY_HEAPSORT:
                    sort = npy_heapsort;
                    break;
                case NPY_STABLESORT:
                    sort = npy_timsort;
                    break;
            }
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                            "type does not have compare function");
            return -1;
        }
    }

    return _new_sortlike(op, axis, sort, NULL, NULL, 0);
}


/*
 * make kth array positive, ravel and sort it
 */
static PyArrayObject *
partition_prep_kth_array(PyArrayObject * ktharray,
                         PyArrayObject * op,
                         int axis)
{
    const npy_intp * shape = PyArray_SHAPE(op);
    PyArrayObject * kthrvl;
    npy_intp * kth;
    npy_intp nkth, i;

    if (!PyArray_CanCastSafely(PyArray_TYPE(ktharray), NPY_INTP)) {
        PyErr_Format(PyExc_TypeError, "Partition index must be integer");
        return NULL;
    }

    if (PyArray_NDIM(ktharray) > 1) {
        PyErr_Format(PyExc_ValueError, "kth array must have dimension <= 1");
        return NULL;
    }
    kthrvl = (PyArrayObject *)PyArray_Cast(ktharray, NPY_INTP);

    if (kthrvl == NULL)
        return NULL;

    kth = PyArray_DATA(kthrvl);
    nkth = PyArray_SIZE(kthrvl);

    for (i = 0; i < nkth; i++) {
        if (kth[i] < 0) {
            kth[i] += shape[axis];
        }
        if (PyArray_SIZE(op) != 0 &&
                    (kth[i] < 0 || kth[i] >= shape[axis])) {
            PyErr_Format(PyExc_ValueError, "kth(=%zd) out of bounds (%zd)",
                         kth[i], shape[axis]);
            Py_XDECREF(kthrvl);
            return NULL;
        }
    }

    /*
     * sort the array of kths so the partitions will
     * not trample on each other
     */
    if (PyArray_SIZE(kthrvl) > 1) {
        PyArray_Sort(kthrvl, -1, NPY_QUICKSORT);
    }

    return kthrvl;
}


/*NUMPY_API
 * Partition an array in-place
 */
NPY_NO_EXPORT int
PyArray_Partition(PyArrayObject *op, PyArrayObject * ktharray, int axis,
                  NPY_SELECTKIND which)
{
    PyArrayObject *kthrvl;
    PyArray_PartitionFunc *part;
    PyArray_SortFunc *sort;
    int n = PyArray_NDIM(op);
    int ret;

    if (check_and_adjust_axis(&axis, n) < 0) {
        return -1;
    }

    if (PyArray_FailUnlessWriteable(op, "partition array") < 0) {
        return -1;
    }

    if (which < 0 || which >= NPY_NSELECTS) {
        PyErr_SetString(PyExc_ValueError, "not a valid partition kind");
        return -1;
    }
    part = get_partition_func(PyArray_TYPE(op), which);
    if (part == NULL) {
        /* Use sorting, slower but equivalent */
        if (PyArray_DESCR(op)->f->compare) {
            sort = npy_quicksort;
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                            "type does not have compare function");
            return -1;
        }
    }

    /* Process ktharray even if using sorting to do bounds checking */
    kthrvl = partition_prep_kth_array(ktharray, op, axis);
    if (kthrvl == NULL) {
        return -1;
    }

    ret = _new_sortlike(op, axis, sort, part,
                        PyArray_DATA(kthrvl), PyArray_SIZE(kthrvl));

    Py_DECREF(kthrvl);

    return ret;
}


/*NUMPY_API
 * ArgSort an array
 */
NPY_NO_EXPORT PyObject *
PyArray_ArgSort(PyArrayObject *op, int axis, NPY_SORTKIND which)
{
    PyArrayObject *op2;
    PyArray_ArgSortFunc *argsort;
    PyObject *ret;

    if (which < 0 || which >= NPY_NSORTS) {
        PyErr_SetString(PyExc_ValueError,
                        "not a valid sort kind");
        return NULL;
    }

    argsort = PyArray_DESCR(op)->f->argsort[which];
    if (argsort == NULL) {
        if (PyArray_DESCR(op)->f->compare) {
            switch (which) {
                default:
                case NPY_QUICKSORT:
                    argsort = npy_aquicksort;
                    break;
                case NPY_HEAPSORT:
                    argsort = npy_aheapsort;
                    break;
                case NPY_STABLESORT:
                    argsort = npy_atimsort;
                    break;
            }
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                            "type does not have compare function");
            return NULL;
        }
    }

    op2 = (PyArrayObject *)PyArray_CheckAxis(op, &axis, 0);
    if (op2 == NULL) {
        return NULL;
    }

    ret = _new_argsortlike(op2, axis, argsort, NULL, NULL, 0);

    Py_DECREF(op2);
    return ret;
}


/*NUMPY_API
 * ArgPartition an array
 */
NPY_NO_EXPORT PyObject *
PyArray_ArgPartition(PyArrayObject *op, PyArrayObject *ktharray, int axis,
                     NPY_SELECTKIND which)
{
    PyArrayObject *op2, *kthrvl;
    PyArray_ArgPartitionFunc *argpart;
    PyArray_ArgSortFunc *argsort;
    PyObject *ret;

    if (which < 0 || which >= NPY_NSELECTS) {
        PyErr_SetString(PyExc_ValueError,
                        "not a valid partition kind");
        return NULL;
    }

    argpart = get_argpartition_func(PyArray_TYPE(op), which);
    if (argpart == NULL) {
        /* Use sorting, slower but equivalent */
        if (PyArray_DESCR(op)->f->compare) {
            argsort = npy_aquicksort;
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                            "type does not have compare function");
            return NULL;
        }
    }

    op2 = (PyArrayObject *)PyArray_CheckAxis(op, &axis, 0);
    if (op2 == NULL) {
        return NULL;
    }

    /* Process ktharray even if using sorting to do bounds checking */
    kthrvl = partition_prep_kth_array(ktharray, op2, axis);
    if (kthrvl == NULL) {
        Py_DECREF(op2);
        return NULL;
    }

    ret = _new_argsortlike(op2, axis, argsort, argpart,
                           PyArray_DATA(kthrvl), PyArray_SIZE(kthrvl));

    Py_DECREF(kthrvl);
    Py_DECREF(op2);

    return ret;
}


/*NUMPY_API
 *LexSort an array providing indices that will sort a collection of arrays
 *lexicographically.  The first key is sorted on first, followed by the second key
 *-- requires that arg"merge"sort is available for each sort_key
 *
 *Returns an index array that shows the indexes for the lexicographic sort along
 *the given axis.
 */
NPY_NO_EXPORT PyObject *
PyArray_LexSort(PyObject *sort_keys, int axis)
{
    PyArrayObject **mps;
    PyArrayIterObject **its;
    PyArrayObject *ret = NULL;
    PyArrayIterObject *rit = NULL;
    npy_intp n, N, size, i, j;
    npy_intp astride, rstride, *iptr;
    int nd;
    int needcopy = 0;
    int elsize;
    int maxelsize;
    int object = 0;
    PyArray_ArgSortFunc *argsort;
    NPY_BEGIN_THREADS_DEF;

    if (!PySequence_Check(sort_keys)
           || ((n = PySequence_Size(sort_keys)) <= 0)) {
        PyErr_SetString(PyExc_TypeError,
                "need sequence of keys with len > 0 in lexsort");
        return NULL;
    }
    mps = (PyArrayObject **) PyArray_malloc(n * sizeof(PyArrayObject *));
    if (mps == NULL) {
        return PyErr_NoMemory();
    }
    its = (PyArrayIterObject **) PyArray_malloc(n * sizeof(PyArrayIterObject *));
    if (its == NULL) {
        PyArray_free(mps);
        return PyErr_NoMemory();
    }
    for (i = 0; i < n; i++) {
        mps[i] = NULL;
        its[i] = NULL;
    }
    for (i = 0; i < n; i++) {
        PyObject *obj;
        obj = PySequence_GetItem(sort_keys, i);
        if (obj == NULL) {
            goto fail;
        }
        mps[i] = (PyArrayObject *)PyArray_FROM_O(obj);
        Py_DECREF(obj);
        if (mps[i] == NULL) {
            goto fail;
        }
        if (i > 0) {
            if ((PyArray_NDIM(mps[i]) != PyArray_NDIM(mps[0]))
                || (!PyArray_CompareLists(PyArray_DIMS(mps[i]),
                                       PyArray_DIMS(mps[0]),
                                       PyArray_NDIM(mps[0])))) {
                PyErr_SetString(PyExc_ValueError,
                                "all keys need to be the same shape");
                goto fail;
            }
        }
        if (!PyArray_DESCR(mps[i])->f->argsort[NPY_STABLESORT]
                && !PyArray_DESCR(mps[i])->f->compare) {
            PyErr_Format(PyExc_TypeError,
                         "item %zd type does not have compare function", i);
            goto fail;
        }
        if (!object
            && PyDataType_FLAGCHK(PyArray_DESCR(mps[i]), NPY_NEEDS_PYAPI)) {
            object = 1;
        }
    }

    /* Now we can check the axis */
    nd = PyArray_NDIM(mps[0]);
    if ((nd == 0) || (PyArray_SIZE(mps[0]) == 1)) {
        /* single element case */
        ret = (PyArrayObject *)PyArray_NewFromDescr(
            &PyArray_Type, PyArray_DescrFromType(NPY_INTP),
            PyArray_NDIM(mps[0]), PyArray_DIMS(mps[0]), NULL, NULL,
            0, NULL);

        if (ret == NULL) {
            goto fail;
        }
        *((npy_intp *)(PyArray_DATA(ret))) = 0;
        goto finish;
    }
    if (check_and_adjust_axis(&axis, nd) < 0) {
        goto fail;
    }

    for (i = 0; i < n; i++) {
        its[i] = (PyArrayIterObject *)PyArray_IterAllButAxis(
                (PyObject *)mps[i], &axis);
        if (its[i] == NULL) {
            goto fail;
        }
    }

    /* Now do the sorting */
    ret = (PyArrayObject *)PyArray_NewFromDescr(
            &PyArray_Type, PyArray_DescrFromType(NPY_INTP),
            PyArray_NDIM(mps[0]), PyArray_DIMS(mps[0]), NULL, NULL,
            0, NULL);
    if (ret == NULL) {
        goto fail;
    }
    rit = (PyArrayIterObject *)
            PyArray_IterAllButAxis((PyObject *)ret, &axis);
    if (rit == NULL) {
        goto fail;
    }
    if (!object) {
        NPY_BEGIN_THREADS;
    }
    size = rit->size;
    N = PyArray_DIMS(mps[0])[axis];
    rstride = PyArray_STRIDE(ret, axis);
    maxelsize = PyArray_DESCR(mps[0])->elsize;
    needcopy = (rstride != sizeof(npy_intp));
    for (j = 0; j < n; j++) {
        needcopy = needcopy
            || PyArray_ISBYTESWAPPED(mps[j])
            || !(PyArray_FLAGS(mps[j]) & NPY_ARRAY_ALIGNED)
            || (PyArray_STRIDES(mps[j])[axis] != (npy_intp)PyArray_DESCR(mps[j])->elsize);
        if (PyArray_DESCR(mps[j])->elsize > maxelsize) {
            maxelsize = PyArray_DESCR(mps[j])->elsize;
        }
    }

    if (needcopy) {
        char *valbuffer, *indbuffer;
        int *swaps;

        valbuffer = PyDataMem_NEW(N * maxelsize);
        if (valbuffer == NULL) {
            goto fail;
        }
        indbuffer = PyDataMem_NEW(N * sizeof(npy_intp));
        if (indbuffer == NULL) {
            PyDataMem_FREE(indbuffer);
            goto fail;
        }
        swaps = malloc(n*sizeof(int));
        for (j = 0; j < n; j++) {
            swaps[j] = PyArray_ISBYTESWAPPED(mps[j]);
        }
        while (size--) {
            iptr = (npy_intp *)indbuffer;
            for (i = 0; i < N; i++) {
                *iptr++ = i;
            }
            for (j = 0; j < n; j++) {
                int rcode;
                elsize = PyArray_DESCR(mps[j])->elsize;
                astride = PyArray_STRIDES(mps[j])[axis];
                argsort = PyArray_DESCR(mps[j])->f->argsort[NPY_STABLESORT];
                if(argsort == NULL) {
                    argsort = npy_atimsort;
                }
                _unaligned_strided_byte_copy(valbuffer, (npy_intp) elsize,
                                             its[j]->dataptr, astride, N, elsize);
                if (swaps[j]) {
                    _strided_byte_swap(valbuffer, (npy_intp) elsize, N, elsize);
                }
                rcode = argsort(valbuffer, (npy_intp *)indbuffer, N, mps[j]);
#if defined(NPY_PY3K)
                if (rcode < 0 || (PyDataType_REFCHK(PyArray_DESCR(mps[j]))
                            && PyErr_Occurred())) {
#else
                if (rcode < 0) {
#endif
                    npy_free_cache(valbuffer, N * maxelsize);
                    npy_free_cache(indbuffer, N * sizeof(npy_intp));
                    free(swaps);
                    goto fail;
                }
                PyArray_ITER_NEXT(its[j]);
            }
            _unaligned_strided_byte_copy(rit->dataptr, rstride, indbuffer,
                                         sizeof(npy_intp), N, sizeof(npy_intp));
            PyArray_ITER_NEXT(rit);
        }
        PyDataMem_FREE(valbuffer);
        PyDataMem_FREE(indbuffer);
        free(swaps);
    }
    else {
        while (size--) {
            iptr = (npy_intp *)rit->dataptr;
            for (i = 0; i < N; i++) {
                *iptr++ = i;
            }
            for (j = 0; j < n; j++) {
                int rcode;
                argsort = PyArray_DESCR(mps[j])->f->argsort[NPY_STABLESORT];
                if(argsort == NULL) {
                    argsort = npy_atimsort;
                }
                rcode = argsort(its[j]->dataptr,
                        (npy_intp *)rit->dataptr, N, mps[j]);
#if defined(NPY_PY3K)
                if (rcode < 0 || (PyDataType_REFCHK(PyArray_DESCR(mps[j]))
                            && PyErr_Occurred())) {
#else
                if (rcode < 0) {
#endif
                    goto fail;
                }
                PyArray_ITER_NEXT(its[j]);
            }
            PyArray_ITER_NEXT(rit);
        }
    }

    if (!object) {
        NPY_END_THREADS;
    }

 finish:
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
        Py_XDECREF(its[i]);
    }
    Py_XDECREF(rit);
    PyArray_free(mps);
    PyArray_free(its);
    return (PyObject *)ret;

 fail:
    NPY_END_THREADS;
    if (!PyErr_Occurred()) {
        /* Out of memory during sorting or buffer creation */
        PyErr_NoMemory();
    }
    Py_XDECREF(rit);
    Py_XDECREF(ret);
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
        Py_XDECREF(its[i]);
    }
    PyArray_free(mps);
    PyArray_free(its);
    return NULL;
}


/*NUMPY_API
 *
 * Search the sorted array op1 for the location of the items in op2. The
 * result is an array of indexes, one for each element in op2, such that if
 * the item were to be inserted in op1 just before that index the array
 * would still be in sorted order.
 *
 * Parameters
 * ----------
 * op1 : PyArrayObject *
 *     Array to be searched, must be 1-D.
 * op2 : PyObject *
 *     Array of items whose insertion indexes in op1 are wanted
 * side : {NPY_SEARCHLEFT, NPY_SEARCHRIGHT}
 *     If NPY_SEARCHLEFT, return first valid insertion indexes
 *     If NPY_SEARCHRIGHT, return last valid insertion indexes
 * perm : PyObject *
 *     Permutation array that sorts op1 (optional)
 *
 * Returns
 * -------
 * ret : PyObject *
 *   New reference to npy_intp array containing indexes where items in op2
 *   could be validly inserted into op1. NULL on error.
 *
 * Notes
 * -----
 * Binary search is used to find the indexes.
 */
NPY_NO_EXPORT PyObject *
PyArray_SearchSorted(PyArrayObject *op1, PyObject *op2,
                     NPY_SEARCHSIDE side, PyObject *perm)
{
    PyArrayObject *ap1 = NULL;
    PyArrayObject *ap2 = NULL;
    PyArrayObject *ap3 = NULL;
    PyArrayObject *sorter = NULL;
    PyArrayObject *ret = NULL;
    PyArray_Descr *dtype;
    int ap1_flags = NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_ALIGNED;
    PyArray_BinSearchFunc *binsearch = NULL;
    PyArray_ArgBinSearchFunc *argbinsearch = NULL;
    NPY_BEGIN_THREADS_DEF;

    /* Find common type */
    dtype = PyArray_DescrFromObject((PyObject *)op2, PyArray_DESCR(op1));
    if (dtype == NULL) {
        return NULL;
    }
    /* refs to dtype we own = 1 */

    /* Look for binary search function */
    if (perm) {
        argbinsearch = get_argbinsearch_func(dtype, side);
    }
    else {
        binsearch = get_binsearch_func(dtype, side);
    }
    if (binsearch == NULL && argbinsearch == NULL) {
        PyErr_SetString(PyExc_TypeError, "compare not supported for type");
        /* refs to dtype we own = 1 */
        Py_DECREF(dtype);
        /* refs to dtype we own = 0 */
        return NULL;
    }

    /* need ap2 as contiguous array and of right type */
    /* refs to dtype we own = 1 */
    Py_INCREF(dtype);
    /* refs to dtype we own = 2 */
    ap2 = (PyArrayObject *)PyArray_CheckFromAny(op2, dtype,
                                0, 0,
                                NPY_ARRAY_CARRAY_RO | NPY_ARRAY_NOTSWAPPED,
                                NULL);
    /* refs to dtype we own = 1, array creation steals one even on failure */
    if (ap2 == NULL) {
        Py_DECREF(dtype);
        /* refs to dtype we own = 0 */
        return NULL;
    }

    /*
     * If the needle (ap2) is larger than the haystack (op1) we copy the
     * haystack to a contiguous array for improved cache utilization.
     */
    if (PyArray_SIZE(ap2) > PyArray_SIZE(op1)) {
        ap1_flags |= NPY_ARRAY_CARRAY_RO;
    }
    ap1 = (PyArrayObject *)PyArray_CheckFromAny((PyObject *)op1, dtype,
                                1, 1, ap1_flags, NULL);
    /* refs to dtype we own = 0, array creation steals one even on failure */
    if (ap1 == NULL) {
        goto fail;
    }

    if (perm) {
        /* need ap3 as a 1D aligned, not swapped, array of right type */
        ap3 = (PyArrayObject *)PyArray_CheckFromAny(perm, NULL,
                                    1, 1,
                                    NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED,
                                    NULL);
        if (ap3 == NULL) {
            PyErr_SetString(PyExc_TypeError,
                        "could not parse sorter argument");
            goto fail;
        }
        if (!PyArray_ISINTEGER(ap3)) {
            PyErr_SetString(PyExc_TypeError,
                        "sorter must only contain integers");
            goto fail;
        }
        /* convert to known integer size */
        sorter = (PyArrayObject *)PyArray_FromArray(ap3,
                                    PyArray_DescrFromType(NPY_INTP),
                                    NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED);
        if (sorter == NULL) {
            PyErr_SetString(PyExc_ValueError,
                        "could not parse sorter argument");
            goto fail;
        }
        if (PyArray_SIZE(sorter) != PyArray_SIZE(ap1)) {
            PyErr_SetString(PyExc_ValueError,
                        "sorter.size must equal a.size");
            goto fail;
        }
    }

    /* ret is a contiguous array of intp type to hold returned indexes */
    ret = (PyArrayObject *)PyArray_NewFromDescr(
            &PyArray_Type, PyArray_DescrFromType(NPY_INTP),
            PyArray_NDIM(ap2), PyArray_DIMS(ap2), NULL, NULL,
            0, (PyObject *)ap2);
    if (ret == NULL) {
        goto fail;
    }

    if (ap3 == NULL) {
        /* do regular binsearch */
        NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(ap2));
        binsearch((const char *)PyArray_DATA(ap1),
                  (const char *)PyArray_DATA(ap2),
                  (char *)PyArray_DATA(ret),
                  PyArray_SIZE(ap1), PyArray_SIZE(ap2),
                  PyArray_STRIDES(ap1)[0], PyArray_DESCR(ap2)->elsize,
                  NPY_SIZEOF_INTP, ap2);
        NPY_END_THREADS_DESCR(PyArray_DESCR(ap2));
    }
    else {
        /* do binsearch with a sorter array */
        int error = 0;
        NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(ap2));
        error = argbinsearch((const char *)PyArray_DATA(ap1),
                             (const char *)PyArray_DATA(ap2),
                             (const char *)PyArray_DATA(sorter),
                             (char *)PyArray_DATA(ret),
                             PyArray_SIZE(ap1), PyArray_SIZE(ap2),
                             PyArray_STRIDES(ap1)[0],
                             PyArray_DESCR(ap2)->elsize,
                             PyArray_STRIDES(sorter)[0], NPY_SIZEOF_INTP, ap2);
        NPY_END_THREADS_DESCR(PyArray_DESCR(ap2));
        if (error < 0) {
            PyErr_SetString(PyExc_ValueError,
                        "Sorter index out of range.");
            goto fail;
        }
        Py_DECREF(ap3);
        Py_DECREF(sorter);
    }
    Py_DECREF(ap1);
    Py_DECREF(ap2);
    return (PyObject *)ret;

 fail:
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    Py_XDECREF(ap3);
    Py_XDECREF(sorter);
    Py_XDECREF(ret);
    return NULL;
}

/*NUMPY_API
 * Diagonal
 *
 * In NumPy versions prior to 1.7,  this function always returned a copy of
 * the diagonal array. In 1.7, the code has been updated to compute a view
 * onto 'self', but it still copies this array before returning, as well as
 * setting the internal WARN_ON_WRITE flag. In a future version, it will
 * simply return a view onto self.
 */
NPY_NO_EXPORT PyObject *
PyArray_Diagonal(PyArrayObject *self, int offset, int axis1, int axis2)
{
    int i, idim, ndim = PyArray_NDIM(self);
    npy_intp *strides;
    npy_intp stride1, stride2, offset_stride;
    npy_intp *shape, dim1, dim2;

    char *data;
    npy_intp diag_size;
    PyArray_Descr *dtype;
    PyObject *ret;
    npy_intp ret_shape[NPY_MAXDIMS], ret_strides[NPY_MAXDIMS];

    if (ndim < 2) {
        PyErr_SetString(PyExc_ValueError,
                        "diag requires an array of at least two dimensions");
        return NULL;
    }

    /* Handle negative axes with standard Python indexing rules */
    if (check_and_adjust_axis_msg(&axis1, ndim, npy_ma_str_axis1) < 0) {
        return NULL;
    }
    if (check_and_adjust_axis_msg(&axis2, ndim, npy_ma_str_axis2) < 0) {
        return NULL;
    }
    if (axis1 == axis2) {
        PyErr_SetString(PyExc_ValueError,
                    "axis1 and axis2 cannot be the same");
        return NULL;
    }

    /* Get the shape and strides of the two axes */
    shape = PyArray_SHAPE(self);
    dim1 = shape[axis1];
    dim2 = shape[axis2];
    strides = PyArray_STRIDES(self);
    stride1 = strides[axis1];
    stride2 = strides[axis2];

    /* Compute the data pointers and diag_size for the view */
    data = PyArray_DATA(self);
    if (offset >= 0) {
        offset_stride = stride2;
        dim2 -= offset;
    }
    else {
        offset = -offset;
        offset_stride = stride1;
        dim1 -= offset;
    }
    diag_size = dim2 < dim1 ? dim2 : dim1;
    if (diag_size < 0) {
        diag_size = 0;
    }
    else {
        data += offset * offset_stride;
    }

    /* Build the new shape and strides for the main data */
    i = 0;
    for (idim = 0; idim < ndim; ++idim) {
        if (idim != axis1 && idim != axis2) {
            ret_shape[i] = shape[idim];
            ret_strides[i] = strides[idim];
            ++i;
        }
    }
    ret_shape[ndim-2] = diag_size;
    ret_strides[ndim-2] = stride1 + stride2;

    /* Create the diagonal view */
    dtype = PyArray_DTYPE(self);
    Py_INCREF(dtype);
    ret = PyArray_NewFromDescrAndBase(
            Py_TYPE(self), dtype,
            ndim-1, ret_shape, ret_strides, data,
            PyArray_FLAGS(self), (PyObject *)self, (PyObject *)self);
    if (ret == NULL) {
        return NULL;
    }

    /*
     * For numpy 1.9 the diagonal view is not writeable.
     * This line needs to be removed in 1.10.
     */
    PyArray_CLEARFLAGS((PyArrayObject *)ret, NPY_ARRAY_WRITEABLE);

    return ret;
}

/*NUMPY_API
 * Compress
 */
NPY_NO_EXPORT PyObject *
PyArray_Compress(PyArrayObject *self, PyObject *condition, int axis,
                 PyArrayObject *out)
{
    PyArrayObject *cond;
    PyObject *res, *ret;

    if (PyArray_Check(condition)) {
        cond = (PyArrayObject *)condition;
        Py_INCREF(cond);
    }
    else {
        PyArray_Descr *dtype = PyArray_DescrFromType(NPY_BOOL);
        if (dtype == NULL) {
            return NULL;
        }
        cond = (PyArrayObject *)PyArray_FromAny(condition, dtype,
                                    0, 0, 0, NULL);
        if (cond == NULL) {
            return NULL;
        }
    }

    if (PyArray_NDIM(cond) != 1) {
        Py_DECREF(cond);
        PyErr_SetString(PyExc_ValueError,
                        "condition must be a 1-d array");
        return NULL;
    }

    res = PyArray_Nonzero(cond);
    Py_DECREF(cond);
    if (res == NULL) {
        return res;
    }
    ret = PyArray_TakeFrom(self, PyTuple_GET_ITEM(res, 0), axis,
                           out, NPY_RAISE);
    Py_DECREF(res);
    return ret;
}

/*
 * count number of nonzero bytes in 48 byte block
 * w must be aligned to 8 bytes
 *
 * even though it uses 64 bit types its faster than the bytewise sum on 32 bit
 * but a 32 bit type version would make it even faster on these platforms
 */
static NPY_INLINE npy_intp
count_nonzero_bytes_384(const npy_uint64 * w)
{
    const npy_uint64 w1 = w[0];
    const npy_uint64 w2 = w[1];
    const npy_uint64 w3 = w[2];
    const npy_uint64 w4 = w[3];
    const npy_uint64 w5 = w[4];
    const npy_uint64 w6 = w[5];
    npy_intp r;

    /*
     * last part of sideways add popcount, first three bisections can be
     * skipped as we are dealing with bytes.
     * multiplication equivalent to (x + (x>>8) + (x>>16) + (x>>24)) & 0xFF
     * multiplication overflow well defined for unsigned types.
     * w1 + w2 guaranteed to not overflow as we only have 0 and 1 data.
     */
    r = ((w1 + w2 + w3 + w4 + w5 + w6) * 0x0101010101010101ULL) >> 56ULL;

    /*
     * bytes not exclusively 0 or 1, sum them individually.
     * should only happen if one does weird stuff with views or external
     * buffers.
     * Doing this after the optimistic computation allows saving registers and
     * better pipelining
     */
    if (NPY_UNLIKELY(
             ((w1 | w2 | w3 | w4 | w5 | w6) & 0xFEFEFEFEFEFEFEFEULL) != 0)) {
        /* reload from pointer to avoid a unnecessary stack spill with gcc */
        const char * c = (const char *)w;
        npy_uintp i, count = 0;
        for (i = 0; i < 48; i++) {
            count += (c[i] != 0);
        }
        return count;
    }

    return r;
}

/*
 * Counts the number of True values in a raw boolean array. This
 * is a low-overhead function which does no heap allocations.
 *
 * Returns -1 on error.
 */
NPY_NO_EXPORT npy_intp
count_boolean_trues(int ndim, char *data, npy_intp *ashape, npy_intp *astrides)
{
    int idim;
    npy_intp shape[NPY_MAXDIMS], strides[NPY_MAXDIMS];
    npy_intp i, coord[NPY_MAXDIMS];
    npy_intp count = 0;
    NPY_BEGIN_THREADS_DEF;

    /* Use raw iteration with no heap memory allocation */
    if (PyArray_PrepareOneRawArrayIter(
                    ndim, ashape,
                    data, astrides,
                    &ndim, shape,
                    &data, strides) < 0) {
        return -1;
    }

    /* Handle zero-sized array */
    if (shape[0] == 0) {
        return 0;
    }

    NPY_BEGIN_THREADS_THRESHOLDED(shape[0]);

    /* Special case for contiguous inner loop */
    if (strides[0] == 1) {
        NPY_RAW_ITER_START(idim, ndim, coord, shape) {
            /* Process the innermost dimension */
            const char *d = data;
            const char *e = data + shape[0];
            if (NPY_CPU_HAVE_UNALIGNED_ACCESS ||
                    npy_is_aligned(d, sizeof(npy_uint64))) {
                npy_uintp stride = 6 * sizeof(npy_uint64);
                for (; d < e - (shape[0] % stride); d += stride) {
                    count += count_nonzero_bytes_384((const npy_uint64 *)d);
                }
            }
            for (; d < e; ++d) {
                count += (*d != 0);
            }
        } NPY_RAW_ITER_ONE_NEXT(idim, ndim, coord, shape, data, strides);
    }
    /* General inner loop */
    else {
        NPY_RAW_ITER_START(idim, ndim, coord, shape) {
            char *d = data;
            /* Process the innermost dimension */
            for (i = 0; i < shape[0]; ++i, d += strides[0]) {
                count += (*d != 0);
            }
        } NPY_RAW_ITER_ONE_NEXT(idim, ndim, coord, shape, data, strides);
    }

    NPY_END_THREADS;

    return count;
}

/*NUMPY_API
 * Counts the number of non-zero elements in the array.
 *
 * Returns -1 on error.
 */
NPY_NO_EXPORT npy_intp
PyArray_CountNonzero(PyArrayObject *self)
{
    PyArray_NonzeroFunc *nonzero;
    char *data;
    npy_intp stride, count;
    npy_intp nonzero_count = 0;
    int needs_api = 0;
    PyArray_Descr *dtype;

    NpyIter *iter;
    NpyIter_IterNextFunc *iternext;
    char **dataptr;
    npy_intp *strideptr, *innersizeptr;
    NPY_BEGIN_THREADS_DEF;

    /* Special low-overhead version specific to the boolean type */
    dtype = PyArray_DESCR(self);
    if (dtype->type_num == NPY_BOOL) {
        return count_boolean_trues(PyArray_NDIM(self), PyArray_DATA(self),
                        PyArray_DIMS(self), PyArray_STRIDES(self));
    }
    nonzero = PyArray_DESCR(self)->f->nonzero;

    /* If it's a trivial one-dimensional loop, don't use an iterator */
    if (PyArray_TRIVIALLY_ITERABLE(self)) {
        needs_api = PyDataType_FLAGCHK(dtype, NPY_NEEDS_PYAPI);
        PyArray_PREPARE_TRIVIAL_ITERATION(self, count, data, stride);

        if (needs_api){
            while (count--) {
                if (nonzero(data, self)) {
                    ++nonzero_count;
                }
                if (PyErr_Occurred()) {
                    return -1;
                }
                data += stride;
            }
        }
        else {
            NPY_BEGIN_THREADS_THRESHOLDED(count);
            while (count--) {
                if (nonzero(data, self)) {
                    ++nonzero_count;
                }
                data += stride;
            }
            NPY_END_THREADS;
        }

        return nonzero_count;
    }

    /*
     * If the array has size zero, return zero (the iterator rejects
     * size zero arrays)
     */
    if (PyArray_SIZE(self) == 0) {
        return 0;
    }

    /*
     * Otherwise create and use an iterator to count the nonzeros.
     */
    iter = NpyIter_New(self, NPY_ITER_READONLY |
                             NPY_ITER_EXTERNAL_LOOP |
                             NPY_ITER_REFS_OK,
                        NPY_KEEPORDER, NPY_NO_CASTING,
                        NULL);
    if (iter == NULL) {
        return -1;
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
        data = *dataptr;
        stride = *strideptr;
        count = *innersizeptr;

        while (count--) {
            if (nonzero(data, self)) {
                ++nonzero_count;
            }
            if (needs_api && PyErr_Occurred()) {
                nonzero_count = -1;
                goto finish;
            }
            data += stride;
        }

    } while(iternext(iter));

finish:
    NPY_END_THREADS;

    NpyIter_Deallocate(iter);

    return nonzero_count;
}

/*NUMPY_API
 * Nonzero
 *
 * TODO: In NumPy 2.0, should make the iteration order a parameter.
 */
NPY_NO_EXPORT PyObject *
PyArray_Nonzero(PyArrayObject *self)
{
    int i, ndim = PyArray_NDIM(self);
    PyArrayObject *ret = NULL;
    PyObject *ret_tuple;
    npy_intp ret_dims[2];
    PyArray_NonzeroFunc *nonzero = PyArray_DESCR(self)->f->nonzero;
    npy_intp nonzero_count;

    NpyIter *iter;
    NpyIter_IterNextFunc *iternext;
    NpyIter_GetMultiIndexFunc *get_multi_index;
    char **dataptr;
    int is_empty = 0;

    /*
     * First count the number of non-zeros in 'self'.
     */
    nonzero_count = PyArray_CountNonzero(self);
    if (nonzero_count < 0) {
        return NULL;
    }

    /* Allocate the result as a 2D array */
    ret_dims[0] = nonzero_count;
    ret_dims[1] = (ndim == 0) ? 1 : ndim;
    ret = (PyArrayObject *)PyArray_NewFromDescr(
            &PyArray_Type, PyArray_DescrFromType(NPY_INTP),
            2, ret_dims, NULL, NULL,
            0, NULL);
    if (ret == NULL) {
        return NULL;
    }

    /* If it's a one-dimensional result, don't use an iterator */
    if (ndim <= 1) {
        npy_intp * multi_index = (npy_intp *)PyArray_DATA(ret);
        char * data = PyArray_BYTES(self);
        npy_intp stride = (ndim == 0) ? 0 : PyArray_STRIDE(self, 0);
        npy_intp count = (ndim == 0) ? 1 : PyArray_DIM(self, 0);
        NPY_BEGIN_THREADS_DEF;

        /* nothing to do */
        if (nonzero_count == 0) {
            goto finish;
        }

        NPY_BEGIN_THREADS_THRESHOLDED(count);

        /* avoid function call for bool */
        if (PyArray_ISBOOL(self)) {
            /*
             * use fast memchr variant for sparse data, see gh-4370
             * the fast bool count is followed by this sparse path is faster
             * than combining the two loops, even for larger arrays
             */
            if (((double)nonzero_count / count) <= 0.1) {
                npy_intp subsize;
                npy_intp j = 0;
                while (1) {
                    npy_memchr(data + j * stride, 0, stride, count - j,
                               &subsize, 1);
                    j += subsize;
                    if (j >= count) {
                        break;
                    }
                    *multi_index++ = j++;
                }
            }
            else {
                npy_intp j;
                for (j = 0; j < count; ++j) {
                    if (*data != 0) {
                        *multi_index++ = j;
                    }
                    data += stride;
                }
            }
        }
        else {
            npy_intp j;
            for (j = 0; j < count; ++j) {
                if (nonzero(data, self)) {
                    *multi_index++ = j;
                }
                data += stride;
            }
        }

        NPY_END_THREADS;

        goto finish;
    }

    /*
     * Build an iterator tracking a multi-index, in C order.
     */
    iter = NpyIter_New(self, NPY_ITER_READONLY |
                             NPY_ITER_MULTI_INDEX |
                             NPY_ITER_ZEROSIZE_OK |
                             NPY_ITER_REFS_OK,
                        NPY_CORDER, NPY_NO_CASTING,
                        NULL);

    if (iter == NULL) {
        Py_DECREF(ret);
        return NULL;
    }

    if (NpyIter_GetIterSize(iter) != 0) {
        npy_intp * multi_index;
        NPY_BEGIN_THREADS_DEF;
        /* Get the pointers for inner loop iteration */
        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(ret);
            return NULL;
        }
        get_multi_index = NpyIter_GetGetMultiIndex(iter, NULL);
        if (get_multi_index == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(ret);
            return NULL;
        }

        NPY_BEGIN_THREADS_NDITER(iter);

        dataptr = NpyIter_GetDataPtrArray(iter);

        multi_index = (npy_intp *)PyArray_DATA(ret);

        /* Get the multi-index for each non-zero element */
        if (PyArray_ISBOOL(self)) {
            /* avoid function call for bool */
            do {
                if (**dataptr != 0) {
                    get_multi_index(iter, multi_index);
                    multi_index += ndim;
                }
            } while(iternext(iter));
        }
        else {
            do {
                if (nonzero(*dataptr, self)) {
                    get_multi_index(iter, multi_index);
                    multi_index += ndim;
                }
            } while(iternext(iter));
        }

        NPY_END_THREADS;
    }

    NpyIter_Deallocate(iter);

finish:
    /* Treat zero-dimensional as shape (1,) */
    if (ndim == 0) {
        ndim = 1;
    }

    ret_tuple = PyTuple_New(ndim);
    if (ret_tuple == NULL) {
        Py_DECREF(ret);
        return NULL;
    }

    for (i = 0; i < PyArray_NDIM(ret); ++i) {
        if (PyArray_DIMS(ret)[i] == 0) {
            is_empty = 1;
            break;
        }
    }

    /* Create views into ret, one for each dimension */
    for (i = 0; i < ndim; ++i) {
        npy_intp stride = ndim * NPY_SIZEOF_INTP;
        /* the result is an empty array, the view must point to valid memory */
        npy_intp data_offset = is_empty ? 0 : i * NPY_SIZEOF_INTP;

        PyArrayObject *view = (PyArrayObject *)PyArray_NewFromDescrAndBase(
            Py_TYPE(ret), PyArray_DescrFromType(NPY_INTP),
            1, &nonzero_count, &stride, PyArray_BYTES(ret) + data_offset,
            PyArray_FLAGS(ret), (PyObject *)ret, (PyObject *)ret);
        if (view == NULL) {
            Py_DECREF(ret);
            Py_DECREF(ret_tuple);
            return NULL;
        }
        PyTuple_SET_ITEM(ret_tuple, i, (PyObject *)view);
    }
    Py_DECREF(ret);

    return ret_tuple;
}

/*
 * Gets a single item from the array, based on a single multi-index
 * array of values, which must be of length PyArray_NDIM(self).
 */
NPY_NO_EXPORT PyObject *
PyArray_MultiIndexGetItem(PyArrayObject *self, npy_intp *multi_index)
{
    int idim, ndim = PyArray_NDIM(self);
    char *data = PyArray_DATA(self);
    npy_intp *shape = PyArray_SHAPE(self);
    npy_intp *strides = PyArray_STRIDES(self);

    /* Get the data pointer */
    for (idim = 0; idim < ndim; ++idim) {
        npy_intp shapevalue = shape[idim];
        npy_intp ind = multi_index[idim];

        if (check_and_adjust_index(&ind, shapevalue, idim, NULL) < 0) {
          return NULL;
        }
        data += ind * strides[idim];
    }

    return PyArray_GETITEM(self, data);
}

/*
 * Sets a single item in the array, based on a single multi-index
 * array of values, which must be of length PyArray_NDIM(self).
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_MultiIndexSetItem(PyArrayObject *self, npy_intp *multi_index,
                                                PyObject *obj)
{
    int idim, ndim = PyArray_NDIM(self);
    char *data = PyArray_DATA(self);
    npy_intp *shape = PyArray_SHAPE(self);
    npy_intp *strides = PyArray_STRIDES(self);

    /* Get the data pointer */
    for (idim = 0; idim < ndim; ++idim) {
        npy_intp shapevalue = shape[idim];
        npy_intp ind = multi_index[idim];

        if (check_and_adjust_index(&ind, shapevalue, idim, NULL) < 0) {
            return -1;
        }
        data += ind * strides[idim];
    }

    return PyArray_SETITEM(self, data, obj);
}
