#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "numpy/npy_math.h"

#include "npy_config.h"

#include "npy_pycompat.h"

#include "common.h"
#include "arrayobject.h"
#include "ctors.h"
#include "lowlevel_strided_loops.h"

#include "item_selection.h"
#include "npy_sort.h"
#include "npy_partition.h"

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
                                    NPY_ARRAY_CARRAY);
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
        int flags = NPY_ARRAY_CARRAY |
                    NPY_ARRAY_UPDATEIFCOPY;

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
        switch(clipmode) {
        case NPY_RAISE:
            for (i = 0; i < n; i++) {
                for (j = 0; j < m; j++) {
                    tmp = ((npy_intp *)(PyArray_DATA(indices)))[j];
                    if (check_and_adjust_index(&tmp, max_item, axis) < 0) {
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
    }
    else {
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
        Py_DECREF(obj);
        obj = out;
    }
    return (PyObject *)obj;

 fail:
    PyArray_XDECREF_ERR(obj);
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
    if (!PyArray_ISCONTIGUOUS(self)) {
        PyArrayObject *obj;
        int flags = NPY_ARRAY_CARRAY | NPY_ARRAY_UPDATEIFCOPY;

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
                if (check_and_adjust_index(&tmp, max_item, 0) < 0) {
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
        switch(clipmode) {
        case NPY_RAISE:
            for (i = 0; i < ni; i++) {
                src = PyArray_BYTES(values) + chunk * (i % nv);
                tmp = ((npy_intp *)(PyArray_DATA(indices)))[i];
                if (check_and_adjust_index(&tmp, max_item, 0) < 0) {
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
    }

 finish:
    Py_XDECREF(values);
    Py_XDECREF(indices);
    if (copied) {
        Py_DECREF(self);
    }
    Py_INCREF(Py_None);
    return Py_None;

 fail:
    Py_XDECREF(indices);
    Py_XDECREF(values);
    if (copied) {
        PyArray_XDECREF_ERR(self);
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
    PyArrayObject  *mask, *values;
    PyArray_Descr *dtype;
    npy_intp i, chunk, ni, max_item, nv, tmp;
    char *src, *dest;
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
                                NPY_ARRAY_CARRAY | NPY_ARRAY_UPDATEIFCOPY);
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
        Py_INCREF(Py_None);
        return Py_None;
    }
    if (PyDataType_REFCHK(PyArray_DESCR(self))) {
        for (i = 0; i < ni; i++) {
            tmp = ((npy_bool *)(PyArray_DATA(mask)))[i];
            if (tmp) {
                src = PyArray_BYTES(values) + chunk * (i % nv);
                PyArray_Item_INCREF(src, PyArray_DESCR(self));
                PyArray_Item_XDECREF(dest+i*chunk, PyArray_DESCR(self));
                memmove(dest + i * chunk, src, chunk);
            }
        }
    }
    else {
        func = PyArray_DESCR(self)->f->fastputmask;
        if (func == NULL) {
            for (i = 0; i < ni; i++) {
                tmp = ((npy_bool *)(PyArray_DATA(mask)))[i];
                if (tmp) {
                    src = PyArray_BYTES(values) + chunk*(i % nv);
                    memmove(dest + i*chunk, src, chunk);
                }
            }
        }
        else {
            func(dest, PyArray_DATA(mask), ni, PyArray_DATA(values), nv);
        }
    }

    Py_XDECREF(values);
    Py_XDECREF(mask);
    if (copied) {
        Py_DECREF(self);
    }
    Py_INCREF(Py_None);
    return Py_None;

 fail:
    Py_XDECREF(mask);
    Py_XDECREF(values);
    if (copied) {
        PyArray_XDECREF_ERR(self);
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
    npy_intp n, n_outer, i, j, k, chunk, total;
    npy_intp tmp;
    int nd;
    PyArrayObject *repeats = NULL;
    PyObject *ap = NULL;
    PyArrayObject *ret = NULL;
    char *new_data, *old_data;

    repeats = (PyArrayObject *)PyArray_ContiguousFromAny(op, NPY_INTP, 0, 1);
    if (repeats == NULL) {
        return NULL;
    }
    nd = PyArray_NDIM(repeats);
    counts = (npy_intp *)PyArray_DATA(repeats);

    if ((ap=PyArray_CheckAxis(aop, &axis, NPY_ARRAY_CARRAY))==NULL) {
        Py_DECREF(repeats);
        return NULL;
    }

    aop = (PyArrayObject *)ap;
    if (nd == 1) {
        n = PyArray_DIMS(repeats)[0];
    }
    else {
        /* nd == 0 */
        n = PyArray_DIMS(aop)[axis];
    }
    if (PyArray_DIMS(aop)[axis] != n) {
        PyErr_SetString(PyExc_ValueError,
                        "a.shape[axis] != len(repeats)");
        goto fail;
    }

    if (nd == 0) {
        total = counts[0]*n;
    }
    else {

        total = 0;
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
            tmp = nd ? counts[j] : counts[0];
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
                    NPY_ARRAY_UPDATEIFCOPY |
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
    PyDataMem_FREE(mps);
    if (out != NULL && out != obj) {
        Py_INCREF(out);
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
    PyDataMem_FREE(mps);
    PyArray_XDECREF_ERR(obj);
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
_new_sortlike(PyArrayObject *op, int axis,
              NPY_SORTKIND swhich,
              PyArray_PartitionFunc * part,
              NPY_SELECTKIND pwhich,
              npy_intp * kth, npy_intp nkth)
{
    PyArrayIterObject *it;
    int needcopy = 0, swap;
    npy_intp N, size;
    int elsize;
    npy_intp astride;
    PyArray_SortFunc *sort = NULL;

    NPY_BEGIN_THREADS_DEF;

    it = (PyArrayIterObject *)PyArray_IterAllButAxis((PyObject *)op, &axis);
    swap = !PyArray_ISNOTSWAPPED(op);
    if (it == NULL) {
        return -1;
    }

    NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(op));
    if (part == NULL) {
        sort = PyArray_DESCR(op)->f->sort[swhich];
    }

    size = it->size;
    N = PyArray_DIMS(op)[axis];
    elsize = PyArray_DESCR(op)->elsize;
    astride = PyArray_STRIDES(op)[axis];

    needcopy = !(PyArray_FLAGS(op) & NPY_ARRAY_ALIGNED) ||
                (astride != (npy_intp) elsize) || swap;
    if (needcopy) {
        char *buffer = PyDataMem_NEW(N*elsize);
        if (buffer == NULL) {
            goto fail;
        }

        while (size--) {
            _unaligned_strided_byte_copy(buffer, (npy_intp) elsize, it->dataptr,
                                         astride, N, elsize);
            if (swap) {
                _strided_byte_swap(buffer, (npy_intp) elsize, N, elsize);
            }
            if (part == NULL) {
                if (sort(buffer, N, op) < 0) {
                    PyDataMem_FREE(buffer);
                    goto fail;
                }
            }
            else {
                npy_intp pivots[NPY_MAX_PIVOT_STACK];
                npy_intp npiv = 0;
                npy_intp i;
                for (i = 0; i < nkth; i++) {
                    if (part(buffer, N, kth[i], pivots, &npiv, op) < 0) {
                        PyDataMem_FREE(buffer);
                        goto fail;
                    }
                }
            }
            if (swap) {
                _strided_byte_swap(buffer, (npy_intp) elsize, N, elsize);
            }
            _unaligned_strided_byte_copy(it->dataptr, astride, buffer,
                                         (npy_intp) elsize, N, elsize);
            PyArray_ITER_NEXT(it);
        }
        PyDataMem_FREE(buffer);
    }
    else {
        while (size--) {
            if (part == NULL) {
                if (sort(it->dataptr, N, op) < 0) {
                    goto fail;
                }
            }
            else {
                npy_intp pivots[NPY_MAX_PIVOT_STACK];
                npy_intp npiv = 0;
                npy_intp i;
                for (i = 0; i < nkth; i++) {
                    if (part(it->dataptr, N, kth[i], pivots, &npiv, op) < 0) {
                        goto fail;
                    }
                }
            }
            PyArray_ITER_NEXT(it);
        }
    }
    NPY_END_THREADS_DESCR(PyArray_DESCR(op));
    Py_DECREF(it);
    return 0;

 fail:
    /* Out of memory during sorting or buffer creation */
    NPY_END_THREADS;
    PyErr_NoMemory();
    Py_DECREF(it);
    return -1;
}

static PyObject*
_new_argsortlike(PyArrayObject *op, int axis,
                 NPY_SORTKIND swhich,
                 PyArray_ArgPartitionFunc * argpart,
                 NPY_SELECTKIND pwhich,
                 npy_intp * kth, npy_intp nkth)
{

    PyArrayIterObject *it = NULL;
    PyArrayIterObject *rit = NULL;
    PyArrayObject *ret;
    npy_intp N, size, i;
    npy_intp astride, rstride, *iptr;
    int elsize;
    int needcopy = 0, swap;
    PyArray_ArgSortFunc *argsort = NULL;

    NPY_BEGIN_THREADS_DEF;

    ret = (PyArrayObject *)PyArray_New(Py_TYPE(op),
                            PyArray_NDIM(op),
                            PyArray_DIMS(op),
                            NPY_INTP,
                            NULL, NULL, 0, 0, (PyObject *)op);
    if (ret == NULL) {
        return NULL;
    }
    it = (PyArrayIterObject *)PyArray_IterAllButAxis((PyObject *)op, &axis);
    rit = (PyArrayIterObject *)PyArray_IterAllButAxis((PyObject *)ret, &axis);
    if (rit == NULL || it == NULL) {
        goto fail;
    }
    swap = !PyArray_ISNOTSWAPPED(op);

    NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(op));
    if (argpart == NULL) {
        argsort = PyArray_DESCR(op)->f->argsort[swhich];
    }
    else {
        argpart = get_argpartition_func(PyArray_TYPE(op), pwhich);
    }
    size = it->size;
    N = PyArray_DIMS(op)[axis];
    elsize = PyArray_DESCR(op)->elsize;
    astride = PyArray_STRIDES(op)[axis];
    rstride = PyArray_STRIDE(ret,axis);

    needcopy = swap || !(PyArray_FLAGS(op) & NPY_ARRAY_ALIGNED) ||
                         (astride != (npy_intp) elsize) ||
            (rstride != sizeof(npy_intp));
    if (needcopy) {
        char *valbuffer, *indbuffer;

        valbuffer = PyDataMem_NEW(N*elsize);
        if (valbuffer == NULL) {
            goto fail;
        }
        indbuffer = PyDataMem_NEW(N*sizeof(npy_intp));
        if (indbuffer == NULL) {
            PyDataMem_FREE(valbuffer);
            goto fail;
        }
        while (size--) {
            _unaligned_strided_byte_copy(valbuffer, (npy_intp) elsize, it->dataptr,
                                         astride, N, elsize);
            if (swap) {
                _strided_byte_swap(valbuffer, (npy_intp) elsize, N, elsize);
            }
            iptr = (npy_intp *)indbuffer;
            for (i = 0; i < N; i++) {
                *iptr++ = i;
            }
            if (argpart == NULL) {
                if (argsort(valbuffer, (npy_intp *)indbuffer, N, op) < 0) {
                    PyDataMem_FREE(valbuffer);
                    PyDataMem_FREE(indbuffer);
                    goto fail;
                }
            }
            else {
                npy_intp pivots[NPY_MAX_PIVOT_STACK];
                npy_intp npiv = 0;
                npy_intp i;
                for (i = 0; i < nkth; i++) {
                    if (argpart(valbuffer, (npy_intp *)indbuffer,
                                N, kth[i], pivots, &npiv, op) < 0) {
                        PyDataMem_FREE(valbuffer);
                        PyDataMem_FREE(indbuffer);
                        goto fail;
                    }
                }
            }
            _unaligned_strided_byte_copy(rit->dataptr, rstride, indbuffer,
                                         sizeof(npy_intp), N, sizeof(npy_intp));
            PyArray_ITER_NEXT(it);
            PyArray_ITER_NEXT(rit);
        }
        PyDataMem_FREE(valbuffer);
        PyDataMem_FREE(indbuffer);
    }
    else {
        while (size--) {
            iptr = (npy_intp *)rit->dataptr;
            for (i = 0; i < N; i++) {
                *iptr++ = i;
            }
            if (argpart == NULL) {
                if (argsort(it->dataptr, (npy_intp *)rit->dataptr,
                            N, op) < 0) {
                    goto fail;
                }
            }
            else {
                npy_intp pivots[NPY_MAX_PIVOT_STACK];
                npy_intp npiv = 0;
                npy_intp i;
                for (i = 0; i < nkth; i++) {
                    if (argpart(it->dataptr, (npy_intp *)rit->dataptr,
                                N, kth[i], pivots, &npiv, op) < 0) {
                        goto fail;
                    }
                }
            }
            PyArray_ITER_NEXT(it);
            PyArray_ITER_NEXT(rit);
        }
    }

    NPY_END_THREADS_DESCR(PyArray_DESCR(op));

    Py_DECREF(it);
    Py_DECREF(rit);
    return (PyObject *)ret;

 fail:
    NPY_END_THREADS;
    if (!PyErr_Occurred()) {
        /* Out of memory during sorting or buffer creation */
        PyErr_NoMemory();
    }
    Py_DECREF(ret);
    Py_XDECREF(it);
    Py_XDECREF(rit);
    return NULL;
}

/* Be sure to save this global_compare when necessary */
static PyArrayObject *global_obj;

static int
sortCompare (const void *a, const void *b)
{
    return PyArray_DESCR(global_obj)->f->compare(a,b,global_obj);
}

/*
 * Consumes reference to ap (op gets it) op contains a version of
 * the array with axes swapped if local variable axis is not the
 * last dimension.  Origin must be defined locally.
 */
#define SWAPAXES(op, ap) { \
        orign = PyArray_NDIM(ap)-1; \
        if (axis != orign) { \
            (op) = (PyArrayObject *)PyArray_SwapAxes((ap), axis, orign); \
            Py_DECREF((ap)); \
            if ((op) == NULL) return NULL; \
        } \
        else (op) = (ap); \
    }

/*
 * Consumes reference to ap (op gets it) origin must be previously
 * defined locally.  SWAPAXES must have been called previously.
 * op contains the swapped version of the array.
 */
#define SWAPBACK(op, ap) {                                      \
        if (axis != orign) {                                    \
            (op) = (PyArrayObject *)PyArray_SwapAxes((ap), axis, orign); \
            Py_DECREF((ap));                                    \
            if ((op) == NULL) return NULL;                      \
        }                                                       \
        else (op) = (ap);                                       \
    }

/* These swap axes in-place if necessary */
#define SWAPINTP(a,b) {npy_intp c; c=(a); (a) = (b); (b) = c;}
#define SWAPAXES2(ap) { \
        orign = PyArray_NDIM(ap)-1; \
        if (axis != orign) { \
            SWAPINTP(PyArray_DIMS(ap)[axis], PyArray_DIMS(ap)[orign]); \
            SWAPINTP(PyArray_STRIDES(ap)[axis], PyArray_STRIDES(ap)[orign]); \
            PyArray_UpdateFlags(ap, NPY_ARRAY_C_CONTIGUOUS | \
                                    NPY_ARRAY_F_CONTIGUOUS); \
        } \
    }

#define SWAPBACK2(ap) { \
        if (axis != orign) { \
            SWAPINTP(PyArray_DIMS(ap)[axis], PyArray_DIMS(ap)[orign]); \
            SWAPINTP(PyArray_STRIDES(ap)[axis], PyArray_STRIDES(ap)[orign]); \
            PyArray_UpdateFlags(ap, NPY_ARRAY_C_CONTIGUOUS | \
                                    NPY_ARRAY_F_CONTIGUOUS); \
        } \
    }

/*NUMPY_API
 * Sort an array in-place
 */
NPY_NO_EXPORT int
PyArray_Sort(PyArrayObject *op, int axis, NPY_SORTKIND which)
{
    PyArrayObject *ap = NULL, *store_arr = NULL;
    char *ip;
    npy_intp i, n, m;
    int elsize, orign;
    int res = 0;
    int axis_orig = axis;
    int (*sort)(void *, size_t, size_t, npy_comparator);

    n = PyArray_NDIM(op);
    if ((n == 0) || (PyArray_SIZE(op) == 1)) {
        return 0;
    }
    if (axis < 0) {
        axis += n;
    }
    if ((axis < 0) || (axis >= n)) {
        PyErr_Format(PyExc_ValueError, "axis(=%d) out of bounds", axis_orig);
        return -1;
    }
    if (PyArray_FailUnlessWriteable(op, "sort array") < 0) {
        return -1;
    }

    /* Determine if we should use type-specific algorithm or not */
    if (PyArray_DESCR(op)->f->sort[which] != NULL) {
        return _new_sortlike(op, axis, which, NULL, 0, NULL, 0);
    }

    if (PyArray_DESCR(op)->f->compare == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "type does not have compare function");
        return -1;
    }

    SWAPAXES2(op);

    switch (which) {
        case NPY_QUICKSORT :
            sort = npy_quicksort;
            break;
        case NPY_HEAPSORT :
            sort = npy_heapsort;
            break;
        case NPY_MERGESORT :
            sort = npy_mergesort;
            break;
        default:
            PyErr_SetString(PyExc_TypeError,
                    "requested sort kind is not supported");
            goto fail;
    }

    ap = (PyArrayObject *)PyArray_FromAny((PyObject *)op,
                          NULL, 1, 0,
                          NPY_ARRAY_DEFAULT | NPY_ARRAY_UPDATEIFCOPY, NULL);
    if (ap == NULL) {
        goto fail;
    }
    elsize = PyArray_DESCR(ap)->elsize;
    m = PyArray_DIMS(ap)[PyArray_NDIM(ap)-1];
    if (m == 0) {
        goto finish;
    }
    n = PyArray_SIZE(ap)/m;

    /* Store global -- allows re-entry -- restore before leaving*/
    store_arr = global_obj;
    global_obj = ap;
    for (ip = PyArray_DATA(ap), i = 0; i < n; i++, ip += elsize*m) {
        res = sort(ip, m, elsize, sortCompare);
        if (res < 0) {
            break;
        }
    }
    global_obj = store_arr;

    if (PyErr_Occurred()) {
        goto fail;
    }
    else if (res == -NPY_ENOMEM) {
        PyErr_NoMemory();
        goto fail;
    }
    else if (res == -NPY_ECOMP) {
        PyErr_SetString(PyExc_TypeError,
                "sort comparison failed");
        goto fail;
    }


 finish:
    Py_DECREF(ap);  /* Should update op if needed */
    SWAPBACK2(op);
    return 0;

 fail:
    Py_XDECREF(ap);
    SWAPBACK2(op);
    return -1;
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
        if (DEPRECATE("Calling partition with a non integer index"
                      " will result in an error in the future") < 0) {
            return NULL;
        }
    }

    if (PyArray_NDIM(ktharray) > 1) {
        PyErr_Format(PyExc_ValueError, "kth array must have dimension <= 1");
        return NULL;
    }
    kthrvl = PyArray_Cast(ktharray, NPY_INTP);

    if (kthrvl == NULL)
        return NULL;

    kth = PyArray_DATA(kthrvl);
    nkth = PyArray_SIZE(kthrvl);

    for (i = 0; i < nkth; i++) {
        if (kth[i] < 0) {
            kth[i] += shape[axis];
        }
        if (PyArray_SIZE(op) != 0 && ((kth[i] < 0) || (kth[i] >= shape[axis]))) {
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
    PyArray_Sort(kthrvl, -1, NPY_QUICKSORT);

    return kthrvl;
}



/*NUMPY_API
 * Partition an array in-place
 */
NPY_NO_EXPORT int
PyArray_Partition(PyArrayObject *op, PyArrayObject * ktharray, int axis, NPY_SELECTKIND which)
{
    PyArrayObject *ap = NULL, *store_arr = NULL;
    char *ip;
    npy_intp i, n, m;
    int elsize, orign;
    int res = 0;
    int axis_orig = axis;
    int (*sort)(void *, size_t, size_t, npy_comparator);
    PyArray_PartitionFunc * part = get_partition_func(PyArray_TYPE(op), which);

    n = PyArray_NDIM(op);
    if ((n == 0)) {
        return 0;
    }
    if (axis < 0) {
        axis += n;
    }
    if ((axis < 0) || (axis >= n)) {
        PyErr_Format(PyExc_ValueError, "axis(=%d) out of bounds", axis_orig);
        return -1;
    }
    if (PyArray_FailUnlessWriteable(op, "sort array") < 0) {
        return -1;
    }

    if (part) {
        PyArrayObject * kthrvl = partition_prep_kth_array(ktharray, op, axis);
        if (kthrvl == NULL)
            return -1;

        res = _new_sortlike(op, axis, 0,
                            part, which,
                            PyArray_DATA(kthrvl),
                            PyArray_SIZE(kthrvl));
        Py_DECREF(kthrvl);
        return res;
    }


    if (PyArray_DESCR(op)->f->compare == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "type does not have compare function");
        return -1;
    }

    SWAPAXES2(op);

    /* select not implemented, use quicksort, slower but equivalent */
    switch (which) {
        case NPY_INTROSELECT :
            sort = npy_quicksort;
            break;
        default:
            PyErr_SetString(PyExc_TypeError,
                    "requested sort kind is not supported");
            goto fail;
    }

    ap = (PyArrayObject *)PyArray_FromAny((PyObject *)op,
                          NULL, 1, 0,
                          NPY_ARRAY_DEFAULT | NPY_ARRAY_UPDATEIFCOPY, NULL);
    if (ap == NULL) {
        goto fail;
    }
    elsize = PyArray_DESCR(ap)->elsize;
    m = PyArray_DIMS(ap)[PyArray_NDIM(ap)-1];
    if (m == 0) {
        goto finish;
    }
    n = PyArray_SIZE(ap)/m;

    /* Store global -- allows re-entry -- restore before leaving*/
    store_arr = global_obj;
    global_obj = ap;
    /* we don't need to care about kth here as we are using a full sort */
    for (ip = PyArray_DATA(ap), i = 0; i < n; i++, ip += elsize*m) {
        res = sort(ip, m, elsize, sortCompare);
        if (res < 0) {
            break;
        }
    }
    global_obj = store_arr;

    if (PyErr_Occurred()) {
        goto fail;
    }
    else if (res == -NPY_ENOMEM) {
        PyErr_NoMemory();
        goto fail;
    }
    else if (res == -NPY_ECOMP) {
        PyErr_SetString(PyExc_TypeError,
                "sort comparison failed");
        goto fail;
    }


 finish:
    Py_DECREF(ap);  /* Should update op if needed */
    SWAPBACK2(op);
    return 0;

 fail:
    Py_XDECREF(ap);
    SWAPBACK2(op);
    return -1;
}


static char *global_data;

static int
argsort_static_compare(const void *ip1, const void *ip2)
{
    int isize = PyArray_DESCR(global_obj)->elsize;
    const npy_intp *ipa = ip1;
    const npy_intp *ipb = ip2;
    return PyArray_DESCR(global_obj)->f->compare(global_data + (isize * *ipa),
                                         global_data + (isize * *ipb),
                                         global_obj);
}

/*NUMPY_API
 * ArgSort an array
 */
NPY_NO_EXPORT PyObject *
PyArray_ArgSort(PyArrayObject *op, int axis, NPY_SORTKIND which)
{
    PyArrayObject *ap = NULL, *ret = NULL, *store, *op2;
    npy_intp *ip;
    npy_intp i, j, n, m, orign;
    int argsort_elsize;
    char *store_ptr;
    int res = 0;
    int (*sort)(void *, size_t, size_t, npy_comparator);

    n = PyArray_NDIM(op);
    if ((n == 0) || (PyArray_SIZE(op) == 1)) {
        ret = (PyArrayObject *)PyArray_New(Py_TYPE(op), PyArray_NDIM(op),
                                           PyArray_DIMS(op),
                                           NPY_INTP,
                                           NULL, NULL, 0, 0,
                                           (PyObject *)op);
        if (ret == NULL) {
            return NULL;
        }
        *((npy_intp *)PyArray_DATA(ret)) = 0;
        return (PyObject *)ret;
    }

    /* Creates new reference op2 */
    if ((op2=(PyArrayObject *)PyArray_CheckAxis(op, &axis, 0)) == NULL) {
        return NULL;
    }
    /* Determine if we should use new algorithm or not */
    if (PyArray_DESCR(op2)->f->argsort[which] != NULL) {
        ret = (PyArrayObject *)_new_argsortlike(op2, axis, which,
                                                NULL, 0, NULL, 0);
        Py_DECREF(op2);
        return (PyObject *)ret;
    }

    if (PyArray_DESCR(op2)->f->compare == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "type does not have compare function");
        Py_DECREF(op2);
        op = NULL;
        goto fail;
    }

    switch (which) {
        case NPY_QUICKSORT :
            sort = npy_quicksort;
            break;
        case NPY_HEAPSORT :
            sort = npy_heapsort;
            break;
        case NPY_MERGESORT :
            sort = npy_mergesort;
            break;
        default:
            PyErr_SetString(PyExc_TypeError,
                    "requested sort kind is not supported");
            Py_DECREF(op2);
            op = NULL;
            goto fail;
    }

    /* ap will contain the reference to op2 */
    SWAPAXES(ap, op2);
    op = (PyArrayObject *)PyArray_ContiguousFromAny((PyObject *)ap,
                                                    NPY_NOTYPE,
                                                    1, 0);
    Py_DECREF(ap);
    if (op == NULL) {
        return NULL;
    }
    ret = (PyArrayObject *)PyArray_New(Py_TYPE(op), PyArray_NDIM(op),
                                       PyArray_DIMS(op), NPY_INTP,
                                       NULL, NULL, 0, 0, (PyObject *)op);
    if (ret == NULL) {
        goto fail;
    }
    ip = (npy_intp *)PyArray_DATA(ret);
    argsort_elsize = PyArray_DESCR(op)->elsize;
    m = PyArray_DIMS(op)[PyArray_NDIM(op)-1];
    if (m == 0) {
        goto finish;
    }
    n = PyArray_SIZE(op)/m;
    store_ptr = global_data;
    global_data = PyArray_DATA(op);
    store = global_obj;
    global_obj = op;
    for (i = 0; i < n; i++, ip += m, global_data += m*argsort_elsize) {
        for (j = 0; j < m; j++) {
            ip[j] = j;
        }
        res = sort((char *)ip, m, sizeof(npy_intp), argsort_static_compare);
        if (res < 0) {
            break;
        }
    }
    global_data = store_ptr;
    global_obj = store;

    if (PyErr_Occurred()) {
        goto fail;
    }
    else if (res == -NPY_ENOMEM) {
        PyErr_NoMemory();
        goto fail;
    }
    else if (res == -NPY_ECOMP) {
        PyErr_SetString(PyExc_TypeError,
                "sort comparison failed");
        goto fail;
    }

 finish:
    Py_DECREF(op);
    SWAPBACK(op, ret);
    return (PyObject *)op;

 fail:
    Py_XDECREF(op);
    Py_XDECREF(ret);
    return NULL;
}


/*NUMPY_API
 * ArgPartition an array
 */
NPY_NO_EXPORT PyObject *
PyArray_ArgPartition(PyArrayObject *op, PyArrayObject * ktharray, int axis, NPY_SELECTKIND which)
{
    PyArrayObject *ap = NULL, *ret = NULL, *store, *op2;
    npy_intp *ip;
    npy_intp i, j, n, m, orign;
    int argsort_elsize;
    char *store_ptr;
    int res = 0;
    int (*sort)(void *, size_t, size_t, npy_comparator);
    PyArray_ArgPartitionFunc * argpart =
        get_argpartition_func(PyArray_TYPE(op), which);

    n = PyArray_NDIM(op);
    if ((n == 0) || (PyArray_SIZE(op) == 1)) {
        ret = (PyArrayObject *)PyArray_New(Py_TYPE(op), PyArray_NDIM(op),
                                           PyArray_DIMS(op),
                                           NPY_INTP,
                                           NULL, NULL, 0, 0,
                                           (PyObject *)op);
        if (ret == NULL) {
            return NULL;
        }
        *((npy_intp *)PyArray_DATA(ret)) = 0;
        return (PyObject *)ret;
    }

    /* Creates new reference op2 */
    if ((op2=(PyArrayObject *)PyArray_CheckAxis(op, &axis, 0)) == NULL) {
        return NULL;
    }

    /* Determine if we should use new algorithm or not */
    if (argpart) {
        PyArrayObject * kthrvl = partition_prep_kth_array(ktharray, op2, axis);
        if (kthrvl == NULL) {
            Py_DECREF(op2);
            return NULL;
        }

        ret = (PyArrayObject *)_new_argsortlike(op2, axis, 0,
                                                argpart, which,
                                                PyArray_DATA(kthrvl),
                                                PyArray_SIZE(kthrvl));
        Py_DECREF(kthrvl);
        Py_DECREF(op2);
        return (PyObject *)ret;
    }

    if (PyArray_DESCR(op2)->f->compare == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "type does not have compare function");
        Py_DECREF(op2);
        op = NULL;
        goto fail;
    }

    /* select not implemented, use quicksort, slower but equivalent */
    switch (which) {
        case NPY_INTROSELECT :
            sort = npy_quicksort;
            break;
        default:
            PyErr_SetString(PyExc_TypeError,
                    "requested sort kind is not supported");
            Py_DECREF(op2);
            op = NULL;
            goto fail;
    }

    /* ap will contain the reference to op2 */
    SWAPAXES(ap, op2);
    op = (PyArrayObject *)PyArray_ContiguousFromAny((PyObject *)ap,
                                                    NPY_NOTYPE,
                                                    1, 0);
    Py_DECREF(ap);
    if (op == NULL) {
        return NULL;
    }
    ret = (PyArrayObject *)PyArray_New(Py_TYPE(op), PyArray_NDIM(op),
                                       PyArray_DIMS(op), NPY_INTP,
                                       NULL, NULL, 0, 0, (PyObject *)op);
    if (ret == NULL) {
        goto fail;
    }
    ip = (npy_intp *)PyArray_DATA(ret);
    argsort_elsize = PyArray_DESCR(op)->elsize;
    m = PyArray_DIMS(op)[PyArray_NDIM(op)-1];
    if (m == 0) {
        goto finish;
    }
    n = PyArray_SIZE(op)/m;
    store_ptr = global_data;
    global_data = PyArray_DATA(op);
    store = global_obj;
    global_obj = op;
    /* we don't need to care about kth here as we are using a full sort */
    for (i = 0; i < n; i++, ip += m, global_data += m*argsort_elsize) {
        for (j = 0; j < m; j++) {
            ip[j] = j;
        }
        res = sort((char *)ip, m, sizeof(npy_intp), argsort_static_compare);
        if (res < 0) {
            break;
        }
    }
    global_data = store_ptr;
    global_obj = store;

    if (PyErr_Occurred()) {
        goto fail;
    }
    else if (res == -NPY_ENOMEM) {
        PyErr_NoMemory();
        goto fail;
    }
    else if (res == -NPY_ECOMP) {
        PyErr_SetString(PyExc_TypeError,
                "sort comparison failed");
        goto fail;
    }

 finish:
    Py_DECREF(op);
    SWAPBACK(op, ret);
    return (PyObject *)op;

 fail:
    Py_XDECREF(op);
    Py_XDECREF(ret);
    return NULL;
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
        if (!PyArray_DESCR(mps[i])->f->argsort[NPY_MERGESORT]) {
            PyErr_Format(PyExc_TypeError,
                         "merge sort not available for item %zd", i);
            goto fail;
        }
        if (!object
            && PyDataType_FLAGCHK(PyArray_DESCR(mps[i]), NPY_NEEDS_PYAPI)) {
            object = 1;
        }
        its[i] = (PyArrayIterObject *)PyArray_IterAllButAxis(
                (PyObject *)mps[i], &axis);
        if (its[i] == NULL) {
            goto fail;
        }
    }

    /* Now we can check the axis */
    nd = PyArray_NDIM(mps[0]);
    if ((nd == 0) || (PyArray_SIZE(mps[0]) == 1)) {
        /* single element case */
        ret = (PyArrayObject *)PyArray_New(&PyArray_Type, PyArray_NDIM(mps[0]),
                                           PyArray_DIMS(mps[0]),
                                           NPY_INTP,
                                           NULL, NULL, 0, 0, NULL);

        if (ret == NULL) {
            goto fail;
        }
        *((npy_intp *)(PyArray_DATA(ret))) = 0;
        goto finish;
    }
    if (axis < 0) {
        axis += nd;
    }
    if ((axis < 0) || (axis >= nd)) {
        PyErr_Format(PyExc_ValueError,
                "axis(=%d) out of bounds", axis);
        goto fail;
    }

    /* Now do the sorting */
    ret = (PyArrayObject *)PyArray_New(&PyArray_Type, PyArray_NDIM(mps[0]),
                                       PyArray_DIMS(mps[0]), NPY_INTP,
                                       NULL, NULL, 0, 0, NULL);
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

        valbuffer = PyDataMem_NEW(N*maxelsize);
        if (valbuffer == NULL) {
            goto fail;
        }
        indbuffer = PyDataMem_NEW(N*sizeof(npy_intp));
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
                elsize = PyArray_DESCR(mps[j])->elsize;
                astride = PyArray_STRIDES(mps[j])[axis];
                argsort = PyArray_DESCR(mps[j])->f->argsort[NPY_MERGESORT];
                _unaligned_strided_byte_copy(valbuffer, (npy_intp) elsize,
                                             its[j]->dataptr, astride, N, elsize);
                if (swaps[j]) {
                    _strided_byte_swap(valbuffer, (npy_intp) elsize, N, elsize);
                }
                if (argsort(valbuffer, (npy_intp *)indbuffer, N, mps[j]) < 0) {
                    PyDataMem_FREE(valbuffer);
                    PyDataMem_FREE(indbuffer);
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
                argsort = PyArray_DESCR(mps[j])->f->argsort[NPY_MERGESORT];
                if (argsort(its[j]->dataptr, (npy_intp *)rit->dataptr,
                            N, mps[j]) < 0) {
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


/** @brief Use bisection of sorted array to find first entries >= keys.
 *
 * For each key use bisection to find the first index i s.t. key <= arr[i].
 * When there is no such index i, set i = len(arr). Return the results in ret.
 * Both arr and key must be of the same comparable type.
 *
 * @param arr 1d, strided, sorted array to be searched.
 * @param key contiguous array of keys.
 * @param ret contiguous array of intp for returned indices.
 * @return void
 */
static void
local_search_left(PyArrayObject *arr, PyArrayObject *key, PyArrayObject *ret)
{
    PyArray_CompareFunc *compare = PyArray_DESCR(key)->f->compare;
    npy_intp nelts = PyArray_DIMS(arr)[PyArray_NDIM(arr) - 1];
    npy_intp nkeys = PyArray_SIZE(key);
    char *parr = PyArray_DATA(arr);
    char *pkey = PyArray_DATA(key);
    npy_intp *pret = (npy_intp *)PyArray_DATA(ret);
    int elsize = PyArray_DESCR(key)->elsize;
    npy_intp arrstride = *PyArray_STRIDES(arr);
    npy_intp i;

    for (i = 0; i < nkeys; ++i) {
        npy_intp imin = 0;
        npy_intp imax = nelts;
        while (imin < imax) {
            npy_intp imid = imin + ((imax - imin) >> 1);
            if (compare(parr + arrstride*imid, pkey, key) < 0) {
                imin = imid + 1;
            }
            else {
                imax = imid;
            }
        }
        *pret = imin;
        pret += 1;
        pkey += elsize;
    }
}


/** @brief Use bisection of sorted array to find first entries > keys.
 *
 * For each key use bisection to find the first index i s.t. key < arr[i].
 * When there is no such index i, set i = len(arr). Return the results in ret.
 * Both arr and key must be of the same comparable type.
 *
 * @param arr 1d, strided, sorted array to be searched.
 * @param key contiguous array of keys.
 * @param ret contiguous array of intp for returned indices.
 * @return void
 */
static void
local_search_right(PyArrayObject *arr, PyArrayObject *key, PyArrayObject *ret)
{
    PyArray_CompareFunc *compare = PyArray_DESCR(key)->f->compare;
    npy_intp nelts = PyArray_DIMS(arr)[PyArray_NDIM(arr) - 1];
    npy_intp nkeys = PyArray_SIZE(key);
    char *parr = PyArray_DATA(arr);
    char *pkey = PyArray_DATA(key);
    npy_intp *pret = (npy_intp *)PyArray_DATA(ret);
    int elsize = PyArray_DESCR(key)->elsize;
    npy_intp arrstride = *PyArray_STRIDES(arr);
    npy_intp i;

    for(i = 0; i < nkeys; ++i) {
        npy_intp imin = 0;
        npy_intp imax = nelts;
        while (imin < imax) {
            npy_intp imid = imin + ((imax - imin) >> 1);
            if (compare(parr + arrstride*imid, pkey, key) <= 0) {
                imin = imid + 1;
            }
            else {
                imax = imid;
            }
        }
        *pret = imin;
        pret += 1;
        pkey += elsize;
    }
}

/** @brief Use bisection of sorted array to find first entries >= keys.
 *
 * For each key use bisection to find the first index i s.t. key <= arr[i].
 * When there is no such index i, set i = len(arr). Return the results in ret.
 * Both arr and key must be of the same comparable type.
 *
 * @param arr 1d, strided array to be searched.
 * @param key contiguous array of keys.
 * @param sorter 1d, strided array of intp that sorts arr.
 * @param ret contiguous array of intp for returned indices.
 * @return int
 */
static int
local_argsearch_left(PyArrayObject *arr, PyArrayObject *key,
                     PyArrayObject *sorter, PyArrayObject *ret)
{
    PyArray_CompareFunc *compare = PyArray_DESCR(key)->f->compare;
    npy_intp nelts = PyArray_DIMS(arr)[PyArray_NDIM(arr) - 1];
    npy_intp nkeys = PyArray_SIZE(key);
    char *parr = PyArray_DATA(arr);
    char *pkey = PyArray_DATA(key);
    char *psorter = PyArray_DATA(sorter);
    npy_intp *pret = (npy_intp *)PyArray_DATA(ret);
    int elsize = PyArray_DESCR(key)->elsize;
    npy_intp arrstride = *PyArray_STRIDES(arr);
    npy_intp sorterstride = *PyArray_STRIDES(sorter);
    npy_intp i;

    for (i = 0; i < nkeys; ++i) {
        npy_intp imin = 0;
        npy_intp imax = nelts;
        while (imin < imax) {
            npy_intp imid = imin + ((imax - imin) >> 1);
            npy_intp indx = *(npy_intp *)(psorter + sorterstride * imid);

            if (indx < 0 || indx >= nelts) {
                return -1;
            }
            if (compare(parr + arrstride*indx, pkey, key) < 0) {
                imin = imid + 1;
            }
            else {
                imax = imid;
            }
        }
        *pret = imin;
        pret += 1;
        pkey += elsize;
    }
    return 0;
}


/** @brief Use bisection of sorted array to find first entries > keys.
 *
 * For each key use bisection to find the first index i s.t. key < arr[i].
 * When there is no such index i, set i = len(arr). Return the results in ret.
 * Both arr and key must be of the same comparable type.
 *
 * @param arr 1d, strided array to be searched.
 * @param key contiguous array of keys.
 * @param sorter 1d, strided array of intp that sorts arr.
 * @param ret contiguous array of intp for returned indices.
 * @return int
 */
static int
local_argsearch_right(PyArrayObject *arr, PyArrayObject *key,
                      PyArrayObject *sorter, PyArrayObject *ret)
{
    PyArray_CompareFunc *compare = PyArray_DESCR(key)->f->compare;
    npy_intp nelts = PyArray_DIMS(arr)[PyArray_NDIM(arr) - 1];
    npy_intp nkeys = PyArray_SIZE(key);
    char *parr = PyArray_DATA(arr);
    char *pkey = PyArray_DATA(key);
    char *psorter = PyArray_DATA(sorter);
    npy_intp *pret = (npy_intp *)PyArray_DATA(ret);
    int elsize = PyArray_DESCR(key)->elsize;
    npy_intp arrstride = *PyArray_STRIDES(arr);
    npy_intp sorterstride = *PyArray_STRIDES(sorter);
    npy_intp i;

    for(i = 0; i < nkeys; ++i) {
        npy_intp imin = 0;
        npy_intp imax = nelts;
        while (imin < imax) {
            npy_intp imid = imin + ((imax - imin) >> 1);
            npy_intp indx = *(npy_intp *)(psorter + sorterstride * imid);

            if (indx < 0 || indx >= nelts) {
                return -1;
            }
            if (compare(parr + arrstride*indx, pkey, key) <= 0) {
                imin = imid + 1;
            }
            else {
                imax = imid;
            }
        }
        *pret = imin;
        pret += 1;
        pkey += elsize;
    }
    return 0;
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
    NPY_BEGIN_THREADS_DEF;

    /* Find common type */
    dtype = PyArray_DescrFromObject((PyObject *)op2, PyArray_DESCR(op1));
    if (dtype == NULL) {
        return NULL;
    }

    /* need ap2 as contiguous array and of right type */
    Py_INCREF(dtype);
    ap2 = (PyArrayObject *)PyArray_CheckFromAny(op2, dtype,
                                0, 0,
                                NPY_ARRAY_CARRAY_RO | NPY_ARRAY_NOTSWAPPED,
                                NULL);
    if (ap2 == NULL) {
        Py_DECREF(dtype);
        return NULL;
    }

    /*
     * If the needle (ap2) is larger than the haystack (op1) we copy the
     * haystack to a continuous array for improved cache utilization.
     */
    if (PyArray_SIZE(ap2) > PyArray_SIZE(op1)) {
        ap1_flags |= NPY_ARRAY_CARRAY_RO;
    }

    ap1 = (PyArrayObject *)PyArray_CheckFromAny((PyObject *)op1, dtype,
                                1, 1, ap1_flags, NULL);
    if (ap1 == NULL) {
        goto fail;
    }
    /* check that comparison function exists */
    if (PyArray_DESCR(ap2)->f->compare == NULL) {
        PyErr_SetString(PyExc_TypeError,
                        "compare not supported for type");
        goto fail;
    }

    if (perm) {
        /* need ap3 as contiguous array and of right type */
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

    /* ret is a contiguous array of intp type to hold returned indices */
    ret = (PyArrayObject *)PyArray_New(Py_TYPE(ap2), PyArray_NDIM(ap2),
                                       PyArray_DIMS(ap2), NPY_INTP,
                                       NULL, NULL, 0, 0, (PyObject *)ap2);
    if (ret == NULL) {
        goto fail;
    }

    if (ap3 == NULL) {
        if (side == NPY_SEARCHLEFT) {
            NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(ap2));
            local_search_left(ap1, ap2, ret);
            NPY_END_THREADS_DESCR(PyArray_DESCR(ap2));
        }
        else if (side == NPY_SEARCHRIGHT) {
            NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(ap2));
            local_search_right(ap1, ap2, ret);
            NPY_END_THREADS_DESCR(PyArray_DESCR(ap2));
        }
    }
    else {
        int err=0;

        if (side == NPY_SEARCHLEFT) {
            NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(ap2));
            err = local_argsearch_left(ap1, ap2, sorter, ret);
            NPY_END_THREADS_DESCR(PyArray_DESCR(ap2));
        }
        else if (side == NPY_SEARCHRIGHT) {
            NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(ap2));
            err = local_argsearch_right(ap1, ap2, sorter, ret);
            NPY_END_THREADS_DESCR(PyArray_DESCR(ap2));
        }
        if (err < 0) {
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
    npy_intp stride1, stride2;
    npy_intp *shape, dim1, dim2;

    char *data;
    npy_intp diag_size;
    PyArrayObject *ret;
    PyArray_Descr *dtype;
    npy_intp ret_shape[NPY_MAXDIMS], ret_strides[NPY_MAXDIMS];
    PyObject *copy;

    if (ndim < 2) {
        PyErr_SetString(PyExc_ValueError,
                        "diag requires an array of at least two dimensions");
        return NULL;
    }

    /* Handle negative axes with standard Python indexing rules */
    if (axis1 < 0) {
        axis1 += ndim;
    }
    if (axis2 < 0) {
        axis2 += ndim;
    }

    /* Error check the two axes */
    if (axis1 == axis2) {
        PyErr_SetString(PyExc_ValueError,
                    "axis1 and axis2 cannot be the same");
        return NULL;
    }
    else if (axis1 < 0 || axis1 >= ndim || axis2 < 0 || axis2 >= ndim) {
        PyErr_Format(PyExc_ValueError,
                    "axis1(=%d) and axis2(=%d) "
                    "must be within range (ndim=%d)",
                    axis1, axis2, ndim);
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
    if (offset > 0) {
        if (offset >= dim2) {
            diag_size = 0;
        }
        else {
            data += offset * stride2;

            diag_size = dim2 - offset;
            if (dim1 < diag_size) {
                diag_size = dim1;
            }
        }
    }
    else if (offset < 0) {
        offset = -offset;
        if (offset >= dim1) {
            diag_size = 0;
        }
        else {
            data += offset * stride1;

            diag_size = dim1 - offset;
            if (dim2 < diag_size) {
                diag_size = dim2;
            }
        }
    }
    else {
        diag_size = dim1 < dim2 ? dim1 : dim2;
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
    ret = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(self),
                               dtype,
                               ndim-1, ret_shape,
                               ret_strides,
                               data,
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

    /* For backwards compatibility, during the deprecation period: */
    copy = PyArray_NewCopy(ret, NPY_KEEPORDER);
    Py_DECREF(ret);
    if (!copy) {
        return NULL;
    }
    PyArray_ENABLEFLAGS((PyArrayObject *)copy, NPY_ARRAY_WARN_ON_WRITE);
    return copy;
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

    /* Special case for contiguous inner loop */
    if (strides[0] == 1) {
        NPY_RAW_ITER_START(idim, ndim, coord, shape) {
            char *d = data;
            /* Process the innermost dimension */
            for (i = 0; i < shape[0]; ++i, ++d) {
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

    NpyIter *iter;
    NpyIter_IterNextFunc *iternext;
    char **dataptr;
    npy_intp *strideptr, *innersizeptr;

    /* Special low-overhead version specific to the boolean type */
    if (PyArray_DESCR(self)->type_num == NPY_BOOL) {
        return count_boolean_trues(PyArray_NDIM(self), PyArray_DATA(self),
                        PyArray_DIMS(self), PyArray_STRIDES(self));
    }

    nonzero = PyArray_DESCR(self)->f->nonzero;

    /* If it's a trivial one-dimensional loop, don't use an iterator */
    if (PyArray_TRIVIALLY_ITERABLE(self)) {
        PyArray_PREPARE_TRIVIAL_ITERATION(self, count, data, stride);

        while (count--) {
            if (nonzero(data, self)) {
                ++nonzero_count;
            }
            data += stride;
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

    /* Get the pointers for inner loop iteration */
    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        NpyIter_Deallocate(iter);
        return -1;
    }
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
            data += stride;
        }

    } while(iternext(iter));

    NpyIter_Deallocate(iter);

    return PyErr_Occurred() ? -1 : nonzero_count;
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
    char *data;
    npy_intp stride, count;
    npy_intp nonzero_count;
    npy_intp *multi_index;

    NpyIter *iter;
    NpyIter_IterNextFunc *iternext;
    NpyIter_GetMultiIndexFunc *get_multi_index;
    char **dataptr;

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
    ret = (PyArrayObject *)PyArray_New(&PyArray_Type, 2, ret_dims,
                       NPY_INTP, NULL, NULL, 0, 0,
                       NULL);
    if (ret == NULL) {
        return NULL;
    }

    /* If it's a one-dimensional result, don't use an iterator */
    if (ndim <= 1) {
        npy_intp j;

        multi_index = (npy_intp *)PyArray_DATA(ret);
        data = PyArray_BYTES(self);
        stride = (ndim == 0) ? 0 : PyArray_STRIDE(self, 0);
        count = (ndim == 0) ? 1 : PyArray_DIM(self, 0);

        for (j = 0; j < count; ++j) {
            if (nonzero(data, self)) {
                *multi_index++ = j;
            }
            data += stride;
        }

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
        dataptr = NpyIter_GetDataPtrArray(iter);

        multi_index = (npy_intp *)PyArray_DATA(ret);

        /* Get the multi-index for each non-zero element */
        do {
            if (nonzero(*dataptr, self)) {
                get_multi_index(iter, multi_index);
                multi_index += ndim;
            }
        } while(iternext(iter));
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

    /* Create views into ret, one for each dimension */
    if (ndim == 1) {
        /* Directly switch to one dimensions (dimension 1 is 1 anyway) */
        ((PyArrayObject_fields *)ret)->nd = 1;
        PyTuple_SET_ITEM(ret_tuple, 0, (PyObject *)ret);
    }
    else {
        for (i = 0; i < ndim; ++i) {
            PyArrayObject *view;
            stride = ndim*NPY_SIZEOF_INTP;

            view = (PyArrayObject *)PyArray_New(Py_TYPE(self), 1,
                                &nonzero_count,
                                NPY_INTP, &stride,
                                PyArray_BYTES(ret) + i*NPY_SIZEOF_INTP,
                                0, 0, (PyObject *)self);
            if (view == NULL) {
                Py_DECREF(ret);
                Py_DECREF(ret_tuple);
                return NULL;
            }
            Py_INCREF(ret);
            if (PyArray_SetBaseObject(view, (PyObject *)ret) < 0) {
                Py_DECREF(ret);
                Py_DECREF(ret_tuple);
            }
            PyTuple_SET_ITEM(ret_tuple, i, (PyObject *)view);
        }

        Py_DECREF(ret);
    }

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

        if (check_and_adjust_index(&ind, shapevalue, idim) < 0) {
          return NULL;
        }
        data += ind * strides[idim];
    }

    return PyArray_DESCR(self)->f->getitem(data, self);
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

        if (check_and_adjust_index(&ind, shapevalue, idim) < 0) {
            return -1;
        }
        data += ind * strides[idim];
    }

    return PyArray_DESCR(self)->f->setitem(obj, data, self);
}
