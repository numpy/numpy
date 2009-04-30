#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "numpy/npy_math.h"

#include "config.h"

#include "common.h"
#include "ctors.h"

#define PyAO PyArrayObject
#define _check_axis PyArray_CheckAxis

/*NUMPY_API
 * Take
 */
NPY_NO_EXPORT PyObject *
PyArray_TakeFrom(PyArrayObject *self0, PyObject *indices0, int axis,
                 PyArrayObject *ret, NPY_CLIPMODE clipmode)
{
    PyArray_FastTakeFunc *func;
    PyArrayObject *self, *indices;
    intp nd, i, j, n, m, max_item, tmp, chunk, nelem;
    intp shape[MAX_DIMS];
    char *src, *dest;
    int copyret = 0;
    int err;

    indices = NULL;
    self = (PyAO *)_check_axis(self0, &axis, CARRAY);
    if (self == NULL) {
        return NULL;
    }
    indices = (PyArrayObject *)PyArray_ContiguousFromAny(indices0,
                                                         PyArray_INTP,
                                                         1, 0);
    if (indices == NULL) {
        goto fail;
    }
    n = m = chunk = 1;
    nd = self->nd + indices->nd - 1;
    for (i = 0; i < nd; i++) {
        if (i < axis) {
            shape[i] = self->dimensions[i];
            n *= shape[i];
        }
        else {
            if (i < axis+indices->nd) {
                shape[i] = indices->dimensions[i-axis];
                m *= shape[i];
            }
            else {
                shape[i] = self->dimensions[i-indices->nd+1];
                chunk *= shape[i];
            }
        }
    }
    Py_INCREF(self->descr);
    if (!ret) {
        ret = (PyArrayObject *)PyArray_NewFromDescr(self->ob_type,
                                                    self->descr,
                                                    nd, shape,
                                                    NULL, NULL, 0,
                                                    (PyObject *)self);

        if (ret == NULL) {
            goto fail;
        }
    }
    else {
        PyArrayObject *obj;
        int flags = NPY_CARRAY | NPY_UPDATEIFCOPY;

        if ((ret->nd != nd) ||
            !PyArray_CompareLists(ret->dimensions, shape, nd)) {
            PyErr_SetString(PyExc_ValueError,
                            "bad shape in output array");
            ret = NULL;
            Py_DECREF(self->descr);
            goto fail;
        }

        if (clipmode == NPY_RAISE) {
            /*
             * we need to make sure and get a copy
             * so the input array is not changed
             * before the error is called
             */
            flags |= NPY_ENSURECOPY;
        }
        obj = (PyArrayObject *)PyArray_FromArray(ret, self->descr,
                                                 flags);
        if (obj != ret) {
            copyret = 1;
        }
        ret = obj;
	if (ret == NULL) {
            goto fail;
        }
    }

    max_item = self->dimensions[axis];
    nelem = chunk;
    chunk = chunk * ret->descr->elsize;
    src = self->data;
    dest = ret->data;

    func = self->descr->f->fasttake;
    if (func == NULL) {
        switch(clipmode) {
        case NPY_RAISE:
            for (i = 0; i < n; i++) {
                for (j = 0; j < m; j++) {
                    tmp = ((intp *)(indices->data))[j];
                    if (tmp < 0) {
                        tmp = tmp + max_item;
                    }
                    if ((tmp < 0) || (tmp >= max_item)) {
                        PyErr_SetString(PyExc_IndexError,
                                "index out of range "\
                                "for array");
                        goto fail;
                    }
                    memmove(dest, src + tmp*chunk, chunk);
                    dest += chunk;
                }
                src += chunk*max_item;
            }
            break;
        case NPY_WRAP:
            for (i = 0; i < n; i++) {
                for (j = 0; j < m; j++) {
                    tmp = ((intp *)(indices->data))[j];
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
                    memmove(dest, src + tmp*chunk, chunk);
                    dest += chunk;
                }
                src += chunk*max_item;
            }
            break;
        case NPY_CLIP:
            for (i = 0; i < n; i++) {
                for (j = 0; j < m; j++) {
                    tmp = ((intp *)(indices->data))[j];
                    if (tmp < 0) {
                        tmp = 0;
                    }
                    else if (tmp >= max_item) {
                        tmp = max_item - 1;
                    }
                    memmove(dest, src+tmp*chunk, chunk);
                    dest += chunk;
                }
                src += chunk*max_item;
            }
            break;
        }
    }
    else {
        err = func(dest, src, (intp *)(indices->data),
                    max_item, n, m, nelem, clipmode);
        if (err) {
            goto fail;
        }
    }

    PyArray_INCREF(ret);
    Py_XDECREF(indices);
    Py_XDECREF(self);
    if (copyret) {
        PyObject *obj;
        obj = ret->base;
        Py_INCREF(obj);
        Py_DECREF(ret);
        ret = (PyArrayObject *)obj;
    }
    return (PyObject *)ret;

 fail:
    PyArray_XDECREF_ERR(ret);
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
    int i, chunk, ni, max_item, nv, tmp;
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
        int flags = NPY_CARRAY | NPY_UPDATEIFCOPY;

        if (clipmode == NPY_RAISE) {
            flags |= NPY_ENSURECOPY;
        }
        Py_INCREF(self->descr);
        obj = (PyArrayObject *)PyArray_FromArray(self,
                                                 self->descr, flags);
        if (obj != self) {
            copied = 1;
        }
        self = obj;
    }
    max_item = PyArray_SIZE(self);
    dest = self->data;
    chunk = self->descr->elsize;
    indices = (PyArrayObject *)PyArray_ContiguousFromAny(indices0,
                                                         PyArray_INTP, 0, 0);
    if (indices == NULL) {
        goto fail;
    }
    ni = PyArray_SIZE(indices);
    Py_INCREF(self->descr);
    values = (PyArrayObject *)PyArray_FromAny(values0, self->descr, 0, 0,
                                              DEFAULT | FORCECAST, NULL);
    if (values == NULL) {
        goto fail;
    }
    nv = PyArray_SIZE(values);
    if (nv <= 0) {
        goto finish;
    }
    if (PyDataType_REFCHK(self->descr)) {
        switch(clipmode) {
        case NPY_RAISE:
            for (i = 0; i < ni; i++) {
                src = values->data + chunk*(i % nv);
                tmp = ((intp *)(indices->data))[i];
                if (tmp < 0) {
                    tmp = tmp + max_item;
                }
                if ((tmp < 0) || (tmp >= max_item)) {
                    PyErr_SetString(PyExc_IndexError,
                            "index out of " \
                            "range for array");
                    goto fail;
                }
                PyArray_Item_INCREF(src, self->descr);
                PyArray_Item_XDECREF(dest+tmp*chunk, self->descr);
                memmove(dest + tmp*chunk, src, chunk);
            }
            break;
        case NPY_WRAP:
            for (i = 0; i < ni; i++) {
                src = values->data + chunk * (i % nv);
                tmp = ((intp *)(indices->data))[i];
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
                PyArray_Item_INCREF(src, self->descr);
                PyArray_Item_XDECREF(dest+tmp*chunk, self->descr);
                memmove(dest + tmp * chunk, src, chunk);
            }
            break;
        case NPY_CLIP:
            for (i = 0; i < ni; i++) {
                src = values->data + chunk * (i % nv);
                tmp = ((intp *)(indices->data))[i];
                if (tmp < 0) {
                    tmp = 0;
                }
                else if (tmp >= max_item) {
                    tmp = max_item - 1;
                }
                PyArray_Item_INCREF(src, self->descr);
                PyArray_Item_XDECREF(dest+tmp*chunk, self->descr);
                memmove(dest + tmp * chunk, src, chunk);
            }
            break;
        }
    }
    else {
        switch(clipmode) {
        case NPY_RAISE:
            for (i = 0; i < ni; i++) {
                src = values->data + chunk * (i % nv);
                tmp = ((intp *)(indices->data))[i];
                if (tmp < 0) {
                    tmp = tmp + max_item;
                }
                if ((tmp < 0) || (tmp >= max_item)) {
                    PyErr_SetString(PyExc_IndexError,
                            "index out of " \
                            "range for array");
                    goto fail;
                }
                memmove(dest + tmp * chunk, src, chunk);
            }
            break;
        case NPY_WRAP:
            for (i = 0; i < ni; i++) {
                src = values->data + chunk * (i % nv);
                tmp = ((intp *)(indices->data))[i];
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
                src = values->data + chunk * (i % nv);
                tmp = ((intp *)(indices->data))[i];
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
    int i, chunk, ni, max_item, nv, tmp;
    char *src, *dest;
    int copied = 0;

    mask = NULL;
    values = NULL;
    if (!PyArray_Check(self)) {
        PyErr_SetString(PyExc_TypeError,
                        "putmask: first argument must "\
                        "be an array");
        return NULL;
    }
    if (!PyArray_ISCONTIGUOUS(self)) {
        PyArrayObject *obj;
        int flags = NPY_CARRAY | NPY_UPDATEIFCOPY;

        Py_INCREF(self->descr);
        obj = (PyArrayObject *)PyArray_FromArray(self,
                                                 self->descr, flags);
        if (obj != self) {
            copied = 1;
        }
        self = obj;
    }

    max_item = PyArray_SIZE(self);
    dest = self->data;
    chunk = self->descr->elsize;
    mask = (PyArrayObject *)\
        PyArray_FROM_OTF(mask0, PyArray_BOOL, CARRAY | FORCECAST);
    if (mask == NULL) {
        goto fail;
    }
    ni = PyArray_SIZE(mask);
    if (ni != max_item) {
        PyErr_SetString(PyExc_ValueError,
                        "putmask: mask and data must be "\
                        "the same size");
        goto fail;
    }
    Py_INCREF(self->descr);
    values = (PyArrayObject *)\
        PyArray_FromAny(values0, self->descr, 0, 0, NPY_CARRAY, NULL);
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
    if (PyDataType_REFCHK(self->descr)) {
        for (i = 0; i < ni; i++) {
            tmp = ((Bool *)(mask->data))[i];
            if (tmp) {
                src = values->data + chunk * (i % nv);
                PyArray_Item_INCREF(src, self->descr);
                PyArray_Item_XDECREF(dest+i*chunk, self->descr);
                memmove(dest + i * chunk, src, chunk);
            }
        }
    }
    else {
        func = self->descr->f->fastputmask;
        if (func == NULL) {
            for (i = 0; i < ni; i++) {
                tmp = ((Bool *)(mask->data))[i];
                if (tmp) {
                    src = values->data + chunk*(i % nv);
                    memmove(dest + i*chunk, src, chunk);
                }
            }
        }
        else {
            func(dest, mask->data, ni, values->data, nv);
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
    intp *counts;
    intp n, n_outer, i, j, k, chunk, total;
    intp tmp;
    int nd;
    PyArrayObject *repeats = NULL;
    PyObject *ap = NULL;
    PyArrayObject *ret = NULL;
    char *new_data, *old_data;

    repeats = (PyAO *)PyArray_ContiguousFromAny(op, PyArray_INTP, 0, 1);
    if (repeats == NULL) {
        return NULL;
    }
    nd = repeats->nd;
    counts = (intp *)repeats->data;

    if ((ap=_check_axis(aop, &axis, CARRAY))==NULL) {
        Py_DECREF(repeats);
        return NULL;
    }

    aop = (PyAO *)ap;
    if (nd == 1) {
        n = repeats->dimensions[0];
    }
    else {
        /* nd == 0 */
        n = aop->dimensions[axis];
    }
    if (aop->dimensions[axis] != n) {
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
    aop->dimensions[axis] = total;
    Py_INCREF(aop->descr);
    ret = (PyArrayObject *)PyArray_NewFromDescr(aop->ob_type,
                                                aop->descr,
                                                aop->nd,
                                                aop->dimensions,
                                                NULL, NULL, 0,
                                                (PyObject *)aop);
    aop->dimensions[axis] = n;
    if (ret == NULL) {
        goto fail;
    }
    new_data = ret->data;
    old_data = aop->data;

    chunk = aop->descr->elsize;
    for(i = axis + 1; i < aop->nd; i++) {
        chunk *= aop->dimensions[i];
    }

    n_outer = 1;
    for (i = 0; i < axis; i++) {
        n_outer *= aop->dimensions[i];
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
PyArray_Choose(PyArrayObject *ip, PyObject *op, PyArrayObject *ret,
               NPY_CLIPMODE clipmode)
{
    int n, elsize;
    intp i;
    char *ret_data;
    PyArrayObject **mps, *ap;
    PyArrayMultiIterObject *multi = NULL;
    intp mi;
    int copyret = 0;
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
    if (!ret) {
        Py_INCREF(mps[0]->descr);
        ret = (PyArrayObject *)PyArray_NewFromDescr(ap->ob_type,
                                                    mps[0]->descr,
                                                    multi->nd,
                                                    multi->dimensions,
                                                    NULL, NULL, 0,
                                                    (PyObject *)ap);
    }
    else {
        PyArrayObject *obj;
        int flags = NPY_CARRAY | NPY_UPDATEIFCOPY | NPY_FORCECAST;

        if ((PyArray_NDIM(ret) != multi->nd)
                || !PyArray_CompareLists(
                    PyArray_DIMS(ret), multi->dimensions, multi->nd)) {
	    PyErr_SetString(PyExc_TypeError,
                            "invalid shape for output array.");
            ret = NULL;
            goto fail;
        }
        if (clipmode == NPY_RAISE) {
            /*
             * we need to make sure and get a copy
             * so the input array is not changed
             * before the error is called
             */
            flags |= NPY_ENSURECOPY;
        }
        Py_INCREF(mps[0]->descr);
        obj = (PyArrayObject *)PyArray_FromArray(ret, mps[0]->descr, flags);
        if (obj != ret) {
            copyret = 1;
        }
        ret = obj;
    }

    if (ret == NULL) {
        goto fail;
    }
    elsize = ret->descr->elsize;
    ret_data = ret->data;

    while (PyArray_MultiIter_NOTDONE(multi)) {
	mi = *((intp *)PyArray_MultiIter_DATA(multi, n));
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

    PyArray_INCREF(ret);
    Py_DECREF(multi);
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
    }
    Py_DECREF(ap);
    PyDataMem_FREE(mps);
    if (copyret) {
        PyObject *obj;
        obj = ret->base;
        Py_INCREF(obj);
        Py_DECREF(ret);
        ret = (PyArrayObject *)obj;
    }
    return (PyObject *)ret;

 fail:
    Py_XDECREF(multi);
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
    }
    Py_XDECREF(ap);
    PyDataMem_FREE(mps);
    PyArray_XDECREF_ERR(ret);
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
_new_sort(PyArrayObject *op, int axis, NPY_SORTKIND which)
{
    PyArrayIterObject *it;
    int needcopy = 0, swap;
    intp N, size;
    int elsize;
    intp astride;
    PyArray_SortFunc *sort;
    BEGIN_THREADS_DEF;

    it = (PyArrayIterObject *)PyArray_IterAllButAxis((PyObject *)op, &axis);
    swap = !PyArray_ISNOTSWAPPED(op);
    if (it == NULL) {
        return -1;
    }

    NPY_BEGIN_THREADS_DESCR(op->descr);
    sort = op->descr->f->sort[which];
    size = it->size;
    N = op->dimensions[axis];
    elsize = op->descr->elsize;
    astride = op->strides[axis];

    needcopy = !(op->flags & ALIGNED) || (astride != (intp) elsize) || swap;
    if (needcopy) {
        char *buffer = PyDataMem_NEW(N*elsize);

        while (size--) {
            _unaligned_strided_byte_copy(buffer, (intp) elsize, it->dataptr,
                                         astride, N, elsize);
            if (swap) {
                _strided_byte_swap(buffer, (intp) elsize, N, elsize);
            }
            if (sort(buffer, N, op) < 0) {
                PyDataMem_FREE(buffer);
                goto fail;
            }
            if (swap) {
                _strided_byte_swap(buffer, (intp) elsize, N, elsize);
            }
            _unaligned_strided_byte_copy(it->dataptr, astride, buffer,
                                         (intp) elsize, N, elsize);
            PyArray_ITER_NEXT(it);
        }
        PyDataMem_FREE(buffer);
    }
    else {
        while (size--) {
            if (sort(it->dataptr, N, op) < 0) {
                goto fail;
            }
            PyArray_ITER_NEXT(it);
        }
    }
    NPY_END_THREADS_DESCR(op->descr);
    Py_DECREF(it);
    return 0;

 fail:
    NPY_END_THREADS;
    Py_DECREF(it);
    return 0;
}

static PyObject*
_new_argsort(PyArrayObject *op, int axis, NPY_SORTKIND which)
{

    PyArrayIterObject *it = NULL;
    PyArrayIterObject *rit = NULL;
    PyObject *ret;
    int needcopy = 0, i;
    intp N, size;
    int elsize, swap;
    intp astride, rstride, *iptr;
    PyArray_ArgSortFunc *argsort;
    BEGIN_THREADS_DEF;

    ret = PyArray_New(op->ob_type, op->nd,
                          op->dimensions, PyArray_INTP,
                          NULL, NULL, 0, 0, (PyObject *)op);
    if (ret == NULL) {
        return NULL;
    }
    it = (PyArrayIterObject *)PyArray_IterAllButAxis((PyObject *)op, &axis);
    rit = (PyArrayIterObject *)PyArray_IterAllButAxis(ret, &axis);
    if (rit == NULL || it == NULL) {
        goto fail;
    }
    swap = !PyArray_ISNOTSWAPPED(op);

    NPY_BEGIN_THREADS_DESCR(op->descr);
    argsort = op->descr->f->argsort[which];
    size = it->size;
    N = op->dimensions[axis];
    elsize = op->descr->elsize;
    astride = op->strides[axis];
    rstride = PyArray_STRIDE(ret,axis);

    needcopy = swap || !(op->flags & ALIGNED) || (astride != (intp) elsize) ||
            (rstride != sizeof(intp));
    if (needcopy) {
        char *valbuffer, *indbuffer;

        valbuffer = PyDataMem_NEW(N*elsize);
        indbuffer = PyDataMem_NEW(N*sizeof(intp));
        while (size--) {
            _unaligned_strided_byte_copy(valbuffer, (intp) elsize, it->dataptr,
                                         astride, N, elsize);
            if (swap) {
                _strided_byte_swap(valbuffer, (intp) elsize, N, elsize);
            }
            iptr = (intp *)indbuffer;
            for (i = 0; i < N; i++) {
                *iptr++ = i;
            }
            if (argsort(valbuffer, (intp *)indbuffer, N, op) < 0) {
                PyDataMem_FREE(valbuffer);
                PyDataMem_FREE(indbuffer);
                goto fail;
            }
            _unaligned_strided_byte_copy(rit->dataptr, rstride, indbuffer,
                                         sizeof(intp), N, sizeof(intp));
            PyArray_ITER_NEXT(it);
            PyArray_ITER_NEXT(rit);
        }
        PyDataMem_FREE(valbuffer);
        PyDataMem_FREE(indbuffer);
    }
    else {
        while (size--) {
            iptr = (intp *)rit->dataptr;
            for (i = 0; i < N; i++) {
                *iptr++ = i;
            }
            if (argsort(it->dataptr, (intp *)rit->dataptr, N, op) < 0) {
                goto fail;
            }
            PyArray_ITER_NEXT(it);
            PyArray_ITER_NEXT(rit);
        }
    }

    NPY_END_THREADS_DESCR(op->descr);

    Py_DECREF(it);
    Py_DECREF(rit);
    return ret;

 fail:
    NPY_END_THREADS;
    Py_DECREF(ret);
    Py_XDECREF(it);
    Py_XDECREF(rit);
    return NULL;
}


/* Be sure to save this global_compare when necessary */
static PyArrayObject *global_obj;

static int
qsortCompare (const void *a, const void *b)
{
    return global_obj->descr->f->compare(a,b,global_obj);
}

/*
 * Consumes reference to ap (op gets it) op contains a version of
 * the array with axes swapped if local variable axis is not the
 * last dimension.  Origin must be defined locally.
 */
#define SWAPAXES(op, ap) {                                      \
        orign = (ap)->nd-1;                                     \
        if (axis != orign) {                                    \
            (op) = (PyAO *)PyArray_SwapAxes((ap), axis, orign); \
            Py_DECREF((ap));                                    \
            if ((op) == NULL) return NULL;                      \
        }                                                       \
        else (op) = (ap);                                       \
    }

/*
 * Consumes reference to ap (op gets it) origin must be previously
 * defined locally.  SWAPAXES must have been called previously.
 * op contains the swapped version of the array.
 */
#define SWAPBACK(op, ap) {                                      \
        if (axis != orign) {                                    \
            (op) = (PyAO *)PyArray_SwapAxes((ap), axis, orign); \
            Py_DECREF((ap));                                    \
            if ((op) == NULL) return NULL;                      \
        }                                                       \
        else (op) = (ap);                                       \
    }

/* These swap axes in-place if necessary */
#define SWAPINTP(a,b) {intp c; c=(a); (a) = (b); (b) = c;}
#define SWAPAXES2(ap) {                                                 \
        orign = (ap)->nd-1;                                             \
        if (axis != orign) {                                            \
            SWAPINTP(ap->dimensions[axis], ap->dimensions[orign]);      \
            SWAPINTP(ap->strides[axis], ap->strides[orign]);            \
            PyArray_UpdateFlags(ap, CONTIGUOUS | FORTRAN);              \
        }                                                               \
    }

#define SWAPBACK2(ap) {                                                 \
        if (axis != orign) {                                            \
            SWAPINTP(ap->dimensions[axis], ap->dimensions[orign]);      \
            SWAPINTP(ap->strides[axis], ap->strides[orign]);            \
            PyArray_UpdateFlags(ap, CONTIGUOUS | FORTRAN);              \
        }                                                               \
    }

/*NUMPY_API
 * Sort an array in-place
 */
NPY_NO_EXPORT int
PyArray_Sort(PyArrayObject *op, int axis, NPY_SORTKIND which)
{
    PyArrayObject *ap = NULL, *store_arr = NULL;
    char *ip;
    int i, n, m, elsize, orign;

    n = op->nd;
    if ((n == 0) || (PyArray_SIZE(op) == 1)) {
        return 0;
    }
    if (axis < 0) {
        axis += n;
    }
    if ((axis < 0) || (axis >= n)) {
        PyErr_Format(PyExc_ValueError, "axis(=%d) out of bounds", axis);
        return -1;
    }
    if (!PyArray_ISWRITEABLE(op)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "attempted sort on unwriteable array.");
        return -1;
    }

    /* Determine if we should use type-specific algorithm or not */
    if (op->descr->f->sort[which] != NULL) {
        return _new_sort(op, axis, which);
    }
    if ((which != PyArray_QUICKSORT)
        || op->descr->f->compare == NULL) {
        PyErr_SetString(PyExc_TypeError,
                        "desired sort not supported for this type");
        return -1;
    }

    SWAPAXES2(op);

    ap = (PyArrayObject *)PyArray_FromAny((PyObject *)op,
                                          NULL, 1, 0,
                                          DEFAULT | UPDATEIFCOPY, NULL);
    if (ap == NULL) {
        goto fail;
    }
    elsize = ap->descr->elsize;
    m = ap->dimensions[ap->nd-1];
    if (m == 0) {
        goto finish;
    }
    n = PyArray_SIZE(ap)/m;

    /* Store global -- allows re-entry -- restore before leaving*/
    store_arr = global_obj;
    global_obj = ap;
    for (ip = ap->data, i = 0; i < n; i++, ip += elsize*m) {
        qsort(ip, m, elsize, qsortCompare);
    }
    global_obj = store_arr;

    if (PyErr_Occurred()) {
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
    int isize = global_obj->descr->elsize;
    const intp *ipa = ip1;
    const intp *ipb = ip2;
    return global_obj->descr->f->compare(global_data + (isize * *ipa),
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
    intp *ip;
    intp i, j, n, m, orign;
    int argsort_elsize;
    char *store_ptr;

    n = op->nd;
    if ((n == 0) || (PyArray_SIZE(op) == 1)) {
        ret = (PyArrayObject *)PyArray_New(op->ob_type, op->nd,
                                           op->dimensions,
                                           PyArray_INTP,
                                           NULL, NULL, 0, 0,
                                           (PyObject *)op);
        if (ret == NULL) {
            return NULL;
        }
        *((intp *)ret->data) = 0;
        return (PyObject *)ret;
    }

    /* Creates new reference op2 */
    if ((op2=(PyAO *)_check_axis(op, &axis, 0)) == NULL) {
        return NULL;
    }
    /* Determine if we should use new algorithm or not */
    if (op2->descr->f->argsort[which] != NULL) {
        ret = (PyArrayObject *)_new_argsort(op2, axis, which);
        Py_DECREF(op2);
        return (PyObject *)ret;
    }

    if ((which != PyArray_QUICKSORT) || op2->descr->f->compare == NULL) {
        PyErr_SetString(PyExc_TypeError,
                        "requested sort not available for type");
        Py_DECREF(op2);
        op = NULL;
        goto fail;
    }

    /* ap will contain the reference to op2 */
    SWAPAXES(ap, op2);
    op = (PyArrayObject *)PyArray_ContiguousFromAny((PyObject *)ap,
                                                    PyArray_NOTYPE,
                                                    1, 0);
    Py_DECREF(ap);
    if (op == NULL) {
        return NULL;
    }
    ret = (PyArrayObject *)PyArray_New(op->ob_type, op->nd,
                                       op->dimensions, PyArray_INTP,
                                       NULL, NULL, 0, 0, (PyObject *)op);
    if (ret == NULL) {
        goto fail;
    }
    ip = (intp *)ret->data;
    argsort_elsize = op->descr->elsize;
    m = op->dimensions[op->nd-1];
    if (m == 0) {
        goto finish;
    }
    n = PyArray_SIZE(op)/m;
    store_ptr = global_data;
    global_data = op->data;
    store = global_obj;
    global_obj = op;
    for (i = 0; i < n; i++, ip += m, global_data += m*argsort_elsize) {
        for (j = 0; j < m; j++) {
            ip[j] = j;
        }
        qsort((char *)ip, m, sizeof(intp), argsort_static_compare);
    }
    global_data = store_ptr;
    global_obj = store;

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
    int n;
    int nd;
    int needcopy=0, i,j;
    intp N, size;
    int elsize;
    int maxelsize;
    intp astride, rstride, *iptr;
    int object = 0;
    PyArray_ArgSortFunc *argsort;
    NPY_BEGIN_THREADS_DEF;

    if (!PySequence_Check(sort_keys)
           || ((n=PySequence_Size(sort_keys)) <= 0)) {
        PyErr_SetString(PyExc_TypeError,
                "need sequence of keys with len > 0 in lexsort");
        return NULL;
    }
    mps = (PyArrayObject **) _pya_malloc(n*sizeof(PyArrayObject));
    if (mps == NULL) {
        return PyErr_NoMemory();
    }
    its = (PyArrayIterObject **) _pya_malloc(n*sizeof(PyArrayIterObject));
    if (its == NULL) {
        _pya_free(mps);
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
            if ((mps[i]->nd != mps[0]->nd)
                || (!PyArray_CompareLists(mps[i]->dimensions,
                                       mps[0]->dimensions,
                                       mps[0]->nd))) {
                PyErr_SetString(PyExc_ValueError,
                                "all keys need to be the same shape");
                goto fail;
            }
        }
        if (!mps[i]->descr->f->argsort[PyArray_MERGESORT]) {
            PyErr_Format(PyExc_TypeError,
                         "merge sort not available for item %d", i);
            goto fail;
        }
        if (!object
            && PyDataType_FLAGCHK(mps[i]->descr, NPY_NEEDS_PYAPI)) {
            object = 1;
        }
        its[i] = (PyArrayIterObject *)PyArray_IterAllButAxis
            ((PyObject *)mps[i], &axis);
        if (its[i] == NULL) {
            goto fail;
        }
    }

    /* Now we can check the axis */
    nd = mps[0]->nd;
    if ((nd == 0) || (PyArray_SIZE(mps[0]) == 1)) {
        ret = (PyArrayObject *)PyArray_New(&PyArray_Type, mps[0]->nd,
                                           mps[0]->dimensions,
                                           PyArray_INTP,
                                           NULL, NULL, 0, 0, NULL);

        if (ret == NULL) {
            goto fail;
        }
        *((intp *)(ret->data)) = 0;
        goto finish;
    }
    if (axis < 0) {
        axis += nd;
    }
    if ((axis < 0) || (axis >= nd)) {
        PyErr_Format(PyExc_ValueError, "axis(=%d) out of bounds", axis);
        goto fail;
    }

    /* Now do the sorting */
    ret = (PyArrayObject *)PyArray_New(&PyArray_Type, mps[0]->nd,
                                       mps[0]->dimensions, PyArray_INTP,
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
    N = mps[0]->dimensions[axis];
    rstride = PyArray_STRIDE(ret,axis);
    maxelsize = mps[0]->descr->elsize;
    needcopy = (rstride != sizeof(intp));
    for (j = 0; j < n && !needcopy; j++) {
        needcopy = PyArray_ISBYTESWAPPED(mps[j])
            || !(mps[j]->flags & ALIGNED)
            || (mps[j]->strides[axis] != (intp)mps[j]->descr->elsize);
        if (mps[j]->descr->elsize > maxelsize) {
            maxelsize = mps[j]->descr->elsize;
        }
    }

    if (needcopy) {
        char *valbuffer, *indbuffer;
        int *swaps;

        valbuffer = PyDataMem_NEW(N*maxelsize);
        indbuffer = PyDataMem_NEW(N*sizeof(intp));
        swaps = malloc(n*sizeof(int));
        for (j = 0; j < n; j++) {
            swaps[j] = PyArray_ISBYTESWAPPED(mps[j]);
        }
        while (size--) {
            iptr = (intp *)indbuffer;
            for (i = 0; i < N; i++) {
                *iptr++ = i;
            }
            for (j = 0; j < n; j++) {
                elsize = mps[j]->descr->elsize;
                astride = mps[j]->strides[axis];
                argsort = mps[j]->descr->f->argsort[PyArray_MERGESORT];
                _unaligned_strided_byte_copy(valbuffer, (intp) elsize,
                                             its[j]->dataptr, astride, N, elsize);
                if (swaps[j]) {
                    _strided_byte_swap(valbuffer, (intp) elsize, N, elsize);
                }
                if (argsort(valbuffer, (intp *)indbuffer, N, mps[j]) < 0) {
                    PyDataMem_FREE(valbuffer);
                    PyDataMem_FREE(indbuffer);
                    free(swaps);
                    goto fail;
                }
                PyArray_ITER_NEXT(its[j]);
            }
            _unaligned_strided_byte_copy(rit->dataptr, rstride, indbuffer,
                                         sizeof(intp), N, sizeof(intp));
            PyArray_ITER_NEXT(rit);
        }
        PyDataMem_FREE(valbuffer);
        PyDataMem_FREE(indbuffer);
        free(swaps);
    }
    else {
        while (size--) {
            iptr = (intp *)rit->dataptr;
            for (i = 0; i < N; i++) {
                *iptr++ = i;
            }
            for (j = 0; j < n; j++) {
                argsort = mps[j]->descr->f->argsort[PyArray_MERGESORT];
                if (argsort(its[j]->dataptr, (intp *)rit->dataptr,
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
    _pya_free(mps);
    _pya_free(its);
    return (PyObject *)ret;

 fail:
    NPY_END_THREADS;
    Py_XDECREF(rit);
    Py_XDECREF(ret);
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
        Py_XDECREF(its[i]);
    }
    _pya_free(mps);
    _pya_free(its);
    return NULL;
}


/** @brief Use bisection of sorted array to find first entries >= keys.
 *
 * For each key use bisection to find the first index i s.t. key <= arr[i].
 * When there is no such index i, set i = len(arr). Return the results in ret.
 * All arrays are assumed contiguous on entry and both arr and key must be of
 * the same comparable type.
 *
 * @param arr contiguous sorted array to be searched.
 * @param key contiguous array of keys.
 * @param ret contiguous array of intp for returned indices.
 * @return void
 */
static void
local_search_left(PyArrayObject *arr, PyArrayObject *key, PyArrayObject *ret)
{
    PyArray_CompareFunc *compare = key->descr->f->compare;
    intp nelts = arr->dimensions[arr->nd - 1];
    intp nkeys = PyArray_SIZE(key);
    char *parr = arr->data;
    char *pkey = key->data;
    intp *pret = (intp *)ret->data;
    int elsize = arr->descr->elsize;
    intp i;

    for (i = 0; i < nkeys; ++i) {
        intp imin = 0;
        intp imax = nelts;
        while (imin < imax) {
            intp imid = imin + ((imax - imin) >> 2);
            if (compare(parr + elsize*imid, pkey, key) < 0) {
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
 * All arrays are assumed contiguous on entry and both arr and key must be of
 * the same comparable type.
 *
 * @param arr contiguous sorted array to be searched.
 * @param key contiguous array of keys.
 * @param ret contiguous array of intp for returned indices.
 * @return void
 */
static void
local_search_right(PyArrayObject *arr, PyArrayObject *key, PyArrayObject *ret)
{
    PyArray_CompareFunc *compare = key->descr->f->compare;
    intp nelts = arr->dimensions[arr->nd - 1];
    intp nkeys = PyArray_SIZE(key);
    char *parr = arr->data;
    char *pkey = key->data;
    intp *pret = (intp *)ret->data;
    int elsize = arr->descr->elsize;
    intp i;

    for(i = 0; i < nkeys; ++i) {
        intp imin = 0;
        intp imax = nelts;
        while (imin < imax) {
            intp imid = imin + ((imax - imin) >> 2);
            if (compare(parr + elsize*imid, pkey, key) <= 0) {
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


/*NUMPY_API
 * Convert object to searchsorted side
 */
NPY_NO_EXPORT int
PyArray_SearchsideConverter(PyObject *obj, void *addr)
{
    NPY_SEARCHSIDE *side = (NPY_SEARCHSIDE *)addr;
    char *str = PyString_AsString(obj);

    if (!str || strlen(str) < 1) {
        PyErr_SetString(PyExc_ValueError,
                        "expected nonempty string for keyword 'side'");
        return PY_FAIL;
    }

    if (str[0] == 'l' || str[0] == 'L') {
        *side = NPY_SEARCHLEFT;
    }
    else if (str[0] == 'r' || str[0] == 'R') {
        *side = NPY_SEARCHRIGHT;
    }
    else {
        PyErr_Format(PyExc_ValueError,
                     "'%s' is an invalid value for keyword 'side'", str);
        return PY_FAIL;
    }
    return PY_SUCCEED;
}


/*NUMPY_API
 * Numeric.searchsorted(a,v)
 */
NPY_NO_EXPORT PyObject *
PyArray_SearchSorted(PyArrayObject *op1, PyObject *op2, NPY_SEARCHSIDE side)
{
    PyArrayObject *ap1 = NULL;
    PyArrayObject *ap2 = NULL;
    PyArrayObject *ret = NULL;
    PyArray_Descr *dtype;
    NPY_BEGIN_THREADS_DEF;

    dtype = PyArray_DescrFromObject((PyObject *)op2, op1->descr);
    /* need ap1 as contiguous array and of right type */
    Py_INCREF(dtype);
    ap1 = (PyArrayObject *)PyArray_FromAny((PyObject *)op1, dtype,
					   1, 1, NPY_DEFAULT, NULL);
    if (ap1 == NULL) {
        Py_DECREF(dtype);
        return NULL;
    }

    /* need ap2 as contiguous array and of right type */
    ap2 = (PyArrayObject *)PyArray_FromAny(op2, dtype,
                                          0, 0, NPY_DEFAULT, NULL);
    if (ap2 == NULL) {
        goto fail;
    }
    /* ret is a contiguous array of intp type to hold returned indices */
    ret = (PyArrayObject *)PyArray_New(ap2->ob_type, ap2->nd,
                                       ap2->dimensions, PyArray_INTP,
                                       NULL, NULL, 0, 0, (PyObject *)ap2);
    if (ret == NULL) {
        goto fail;
    }
    /* check that comparison function exists */
    if (ap2->descr->f->compare == NULL) {
        PyErr_SetString(PyExc_TypeError,
                        "compare not supported for type");
        goto fail;
    }

    if (side == NPY_SEARCHLEFT) {
        NPY_BEGIN_THREADS_DESCR(ap2->descr);
        local_search_left(ap1, ap2, ret);
        NPY_END_THREADS_DESCR(ap2->descr);
    }
    else if (side == NPY_SEARCHRIGHT) {
        NPY_BEGIN_THREADS_DESCR(ap2->descr);
        local_search_right(ap1, ap2, ret);
        NPY_END_THREADS_DESCR(ap2->descr);
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

/*NUMPY_API
 * Diagonal
 */
NPY_NO_EXPORT PyObject *
PyArray_Diagonal(PyArrayObject *self, int offset, int axis1, int axis2)
{
    int n = self->nd;
    PyObject *new;
    PyArray_Dims newaxes;
    intp dims[MAX_DIMS];
    int i, pos;

    newaxes.ptr = dims;
    if (n < 2) {
        PyErr_SetString(PyExc_ValueError,
                        "array.ndim must be >= 2");
        return NULL;
    }
    if (axis1 < 0) {
        axis1 += n;
    }
    if (axis2 < 0) {
        axis2 += n;
    }
    if ((axis1 == axis2) || (axis1 < 0) || (axis1 >= n) ||
        (axis2 < 0) || (axis2 >= n)) {
        PyErr_Format(PyExc_ValueError, "axis1(=%d) and axis2(=%d) "\
                     "must be different and within range (nd=%d)",
                     axis1, axis2, n);
        return NULL;
    }

    newaxes.len = n;
    /* insert at the end */
    newaxes.ptr[n-2] = axis1;
    newaxes.ptr[n-1] = axis2;
    pos = 0;
    for (i = 0; i < n; i++) {
        if ((i==axis1) || (i==axis2)) {
            continue;
        }
        newaxes.ptr[pos++] = i;
    }
    new = PyArray_Transpose(self, &newaxes);
    if (new == NULL) {
        return NULL;
    }
    self = (PyAO *)new;

    if (n == 2) {
        PyObject *a = NULL, *indices= NULL, *ret = NULL;
        intp n1, n2, start, stop, step, count;
        intp *dptr;

        n1 = self->dimensions[0];
        n2 = self->dimensions[1];
        step = n2 + 1;
        if (offset < 0) {
            start = -n2 * offset;
            stop = MIN(n2, n1+offset)*(n2+1) - n2*offset;
        }
        else {
            start = offset;
            stop = MIN(n1, n2-offset)*(n2+1) + offset;
        }

        /* count = ceil((stop-start)/step) */
        count = ((stop-start) / step) + (((stop-start) % step) != 0);
        indices = PyArray_New(&PyArray_Type, 1, &count,
                              PyArray_INTP, NULL, NULL, 0, 0, NULL);
        if (indices == NULL) {
            Py_DECREF(self);
            return NULL;
        }
        dptr = (intp *)PyArray_DATA(indices);
        for (n1 = start; n1 < stop; n1 += step) {
            *dptr++ = n1;
        }
        a = PyArray_IterNew((PyObject *)self);
        Py_DECREF(self);
        if (a == NULL) {
            Py_DECREF(indices);
            return NULL;
        }
        ret = PyObject_GetItem(a, indices);
        Py_DECREF(a);
        Py_DECREF(indices);
        return ret;
    }

    else {
        /*
         * my_diagonal = []
         * for i in range (s [0]) :
         * my_diagonal.append (diagonal (a [i], offset))
         * return array (my_diagonal)
         */
        PyObject *mydiagonal = NULL, *new = NULL, *ret = NULL, *sel = NULL;
        intp i, n1;
        int res;
        PyArray_Descr *typecode;

        typecode = self->descr;
        mydiagonal = PyList_New(0);
        if (mydiagonal == NULL) {
            Py_DECREF(self);
            return NULL;
        }
        n1 = self->dimensions[0];
        for (i = 0; i < n1; i++) {
            new = PyInt_FromLong((long) i);
            sel = PyArray_EnsureAnyArray(PyObject_GetItem((PyObject *)self, new));
            Py_DECREF(new);
            if (sel == NULL) {
                Py_DECREF(self);
                Py_DECREF(mydiagonal);
                return NULL;
            }
            new = PyArray_Diagonal((PyAO *)sel, offset, n-3, n-2);
            Py_DECREF(sel);
            if (new == NULL) {
                Py_DECREF(self);
                Py_DECREF(mydiagonal);
                return NULL;
            }
            res = PyList_Append(mydiagonal, new);
            Py_DECREF(new);
            if (res < 0) {
                Py_DECREF(self);
                Py_DECREF(mydiagonal);
                return NULL;
            }
        }
        Py_DECREF(self);
        Py_INCREF(typecode);
        ret =  PyArray_FromAny(mydiagonal, typecode, 0, 0, 0, NULL);
        Py_DECREF(mydiagonal);
        return ret;
    }
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

    cond = (PyAO *)PyArray_FROM_O(condition);
    if (cond == NULL) {
        return NULL;
    }
    if (cond->nd != 1) {
        Py_DECREF(cond);
        PyErr_SetString(PyExc_ValueError,
                        "condition must be 1-d array");
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

/*NUMPY_API
 * Nonzero
 */
NPY_NO_EXPORT PyObject *
PyArray_Nonzero(PyArrayObject *self)
{
    int n = self->nd, j;
    intp count = 0, i, size;
    PyArrayIterObject *it = NULL;
    PyObject *ret = NULL, *item;
    intp *dptr[MAX_DIMS];

    it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)self);
    if (it == NULL) {
        return NULL;
    }
    size = it->size;
    for (i = 0; i < size; i++) {
        if (self->descr->f->nonzero(it->dataptr, self)) {
            count++;
        }
        PyArray_ITER_NEXT(it);
    }

    PyArray_ITER_RESET(it);
    ret = PyTuple_New(n);
    if (ret == NULL) {
        goto fail;
    }
    for (j = 0; j < n; j++) {
        item = PyArray_New(self->ob_type, 1, &count,
                           PyArray_INTP, NULL, NULL, 0, 0,
                           (PyObject *)self);
        if (item == NULL) {
            goto fail;
        }
        PyTuple_SET_ITEM(ret, j, item);
        dptr[j] = (intp *)PyArray_DATA(item);
    }
    if (n == 1) {
        for (i = 0; i < size; i++) {
            if (self->descr->f->nonzero(it->dataptr, self)) {
                *(dptr[0])++ = i;
            }
            PyArray_ITER_NEXT(it);
        }
    }
    else {
        /* reset contiguous so that coordinates gets updated */
        it->contiguous = 0;
        for (i = 0; i < size; i++) {
            if (self->descr->f->nonzero(it->dataptr, self)) {
                for (j = 0; j < n; j++) {
                    *(dptr[j])++ = it->coordinates[j];
                }
            }
            PyArray_ITER_NEXT(it);
        }
    }

    Py_DECREF(it);
    return ret;

 fail:
    Py_XDECREF(ret);
    Py_XDECREF(it);
    return NULL;

}

