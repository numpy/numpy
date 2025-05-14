#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "numpy/npy_math.h"
#include "numpy/npy_cpu.h"

#include "npy_config.h"



#include "npy_static_data.h"
#include "common.h"
#include "dtype_transfer.h"
#include "dtypemeta.h"
#include "arrayobject.h"
#include "ctors.h"
#include "lowlevel_strided_loops.h"
#include "array_assign.h"
#include "refcount.h"

#include "npy_sort.h"
#include "npy_partition.h"
#include "npy_binsearch.h"
#include "alloc.h"
#include "arraytypes.h"
#include "array_coercion.h"
#include "simd/simd.h"

static NPY_GCC_OPT_3 inline int
npy_fasttake_impl(
        char *dest, char *src, const npy_intp *indices,
        npy_intp n, npy_intp m, npy_intp max_item,
        npy_intp nelem, npy_intp chunk,
        NPY_CLIPMODE clipmode, npy_intp itemsize, int needs_refcounting,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype, int axis)
{
    NPY_BEGIN_THREADS_DEF;

    NPY_cast_info cast_info;
    NPY_ARRAYMETHOD_FLAGS flags;
    NPY_cast_info_init(&cast_info);

    if (!needs_refcounting) {
        /* if "refcounting" is not needed memcpy is safe for a simple copy  */
        NPY_BEGIN_THREADS;
    }
    else {
        if (PyArray_GetDTypeTransferFunction(
                1, itemsize, itemsize, src_dtype, dst_dtype, 0,
                &cast_info, &flags) < 0) {
            return -1;
        }
        if (!(flags & NPY_METH_REQUIRES_PYAPI)) {
            NPY_BEGIN_THREADS;
        }
    }

    switch (clipmode) {
        case NPY_RAISE:
            for (npy_intp i = 0; i < n; i++) {
                for (npy_intp j = 0; j < m; j++) {
                    npy_intp tmp = indices[j];
                    if (check_and_adjust_index(&tmp, max_item, axis,
                                               _save) < 0) {
                        goto fail;
                    }
                    char *tmp_src = src + tmp * chunk;
                    if (needs_refcounting) {
                        char *data[2] = {tmp_src, dest};
                        npy_intp strides[2] = {itemsize, itemsize};
                        if (cast_info.func(
                                &cast_info.context, data, &nelem, strides,
                                cast_info.auxdata) < 0) {
                            NPY_END_THREADS;
                            goto fail;
                        }
                    }
                    else {
                        memcpy(dest, tmp_src, chunk);
                    }
                    dest += chunk;
                }
                src += chunk*max_item;
            }
            break;
        case NPY_WRAP:
            for (npy_intp i = 0; i < n; i++) {
                for (npy_intp j = 0; j < m; j++) {
                    npy_intp tmp = indices[j];
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
                    char *tmp_src = src + tmp * chunk;
                    if (needs_refcounting) {
                        char *data[2] = {tmp_src, dest};
                        npy_intp strides[2] = {itemsize, itemsize};
                        if (cast_info.func(
                                &cast_info.context, data, &nelem, strides,
                                cast_info.auxdata) < 0) {
                            NPY_END_THREADS;
                            goto fail;
                        }
                    }
                    else {
                        memcpy(dest, tmp_src, chunk);
                    }
                    dest += chunk;
                }
                src += chunk*max_item;
            }
            break;
        case NPY_CLIP:
            for (npy_intp i = 0; i < n; i++) {
                for (npy_intp j = 0; j < m; j++) {
                    npy_intp tmp = indices[j];
                    if (tmp < 0) {
                        tmp = 0;
                    }
                    else if (tmp >= max_item) {
                        tmp = max_item - 1;
                    }
                    char *tmp_src = src + tmp * chunk;
                    if (needs_refcounting) {
                        char *data[2] = {tmp_src, dest};
                        npy_intp strides[2] = {itemsize, itemsize};
                        if (cast_info.func(
                                &cast_info.context, data, &nelem, strides,
                                cast_info.auxdata) < 0) {
                            NPY_END_THREADS;
                            goto fail;
                        }
                    }
                    else {
                        memcpy(dest, tmp_src, chunk);
                    }
                    dest += chunk;
                }
                src += chunk*max_item;
            }
            break;
    }

    NPY_END_THREADS;
    NPY_cast_info_xfree(&cast_info);
    return 0;

  fail:
    /* NPY_END_THREADS already ensured. */
    NPY_cast_info_xfree(&cast_info);
    return -1;
}


/*
 * Helper function instantiating npy_fasttake_impl in different branches
 * to allow the compiler to optimize each to the specific itemsize.
 */
static NPY_GCC_OPT_3 int
npy_fasttake(
        char *dest, char *src, const npy_intp *indices,
        npy_intp n, npy_intp m, npy_intp max_item,
        npy_intp nelem, npy_intp chunk,
        NPY_CLIPMODE clipmode, npy_intp itemsize, int needs_refcounting,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype, int axis)
{
    if (!needs_refcounting) {
        if (chunk == 1) {
            return npy_fasttake_impl(
                    dest, src, indices, n, m, max_item, nelem, chunk,
                    clipmode, itemsize, needs_refcounting, src_dtype,
                    dst_dtype, axis);
        }
        if (chunk == 2) {
            return npy_fasttake_impl(
                    dest, src, indices, n, m, max_item, nelem, chunk,
                    clipmode, itemsize, needs_refcounting, src_dtype,
                    dst_dtype, axis);
        }
        if (chunk == 4) {
            return npy_fasttake_impl(
                    dest, src, indices, n, m, max_item, nelem, chunk,
                    clipmode, itemsize, needs_refcounting, src_dtype,
                    dst_dtype, axis);
        }
        if (chunk == 8) {
            return npy_fasttake_impl(
                    dest, src, indices, n, m, max_item, nelem, chunk,
                    clipmode, itemsize, needs_refcounting, src_dtype,
                    dst_dtype, axis);
        }
        if (chunk == 16) {
            return npy_fasttake_impl(
                    dest, src, indices, n, m, max_item, nelem, chunk,
                    clipmode, itemsize, needs_refcounting, src_dtype,
                    dst_dtype, axis);
        }
        if (chunk == 32) {
            return npy_fasttake_impl(
                    dest, src, indices, n, m, max_item, nelem, chunk,
                    clipmode, itemsize, needs_refcounting, src_dtype,
                    dst_dtype, axis);
        }
    }

    return npy_fasttake_impl(
            dest, src, indices, n, m, max_item, nelem, chunk,
            clipmode, itemsize, needs_refcounting, src_dtype,
            dst_dtype, axis);
}


/*NUMPY_API
 * Take
 */
NPY_NO_EXPORT PyObject *
PyArray_TakeFrom(PyArrayObject *self0, PyObject *indices0, int axis,
                 PyArrayObject *out, NPY_CLIPMODE clipmode)
{
    PyArray_Descr *dtype;
    PyArrayObject *obj = NULL, *self, *indices;
    npy_intp nd, i, n, m, max_item, chunk, itemsize, nelem;
    npy_intp shape[NPY_MAXDIMS];

    npy_bool needs_refcounting;

    indices = NULL;
    self = (PyArrayObject *)PyArray_CheckAxis(self0, &axis,
                                    NPY_ARRAY_CARRAY_RO);
    if (self == NULL) {
        return NULL;
    }

    indices = (PyArrayObject *)PyArray_FromAny(indices0,
                PyArray_DescrFromType(NPY_INTP),
                0, 0,
                NPY_ARRAY_SAME_KIND_CASTING | NPY_ARRAY_DEFAULT,
                NULL);
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

        if (arrays_overlap(out, self)) {
            flags |= NPY_ARRAY_ENSURECOPY;
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
    char *src = PyArray_DATA(self);
    char *dest = PyArray_DATA(obj);
    PyArray_Descr *src_descr = PyArray_DESCR(self);
    PyArray_Descr *dst_descr = PyArray_DESCR(obj);
    needs_refcounting = PyDataType_REFCHK(PyArray_DESCR(self));
    npy_intp *indices_data = (npy_intp *)PyArray_DATA(indices);

    if ((max_item == 0) && (PyArray_SIZE(obj) != 0)) {
        /* Index error, since that is the usual error for raise mode */
        PyErr_SetString(PyExc_IndexError,
                    "cannot do a non-empty take from an empty axes.");
        goto fail;
    }

    if (npy_fasttake(
            dest, src, indices_data, n, m, max_item, nelem, chunk,
            clipmode, itemsize, needs_refcounting, src_descr, dst_descr,
            axis) < 0) {
        goto fail;
    }

    if (out != NULL && out != obj) {
        if (PyArray_ResolveWritebackIfCopy(obj) < 0) {
            goto fail;
        }
        Py_DECREF(obj);
        Py_INCREF(out);
        obj = out;
    }
    Py_XDECREF(indices);
    Py_XDECREF(self);
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
    npy_intp i, itemsize, ni, max_item, nv, tmp;
    char *src, *dest;
    int copied = 0;
    int overlap = 0;

    NPY_BEGIN_THREADS_DEF;
    NPY_cast_info cast_info;
    NPY_ARRAYMETHOD_FLAGS flags;

    NPY_cast_info_init(&cast_info);

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

    indices = (PyArrayObject *)PyArray_ContiguousFromAny(indices0,
                                                         NPY_INTP, 0, 0);
    if (indices == NULL) {
        goto fail;
    }
    ni = PyArray_SIZE(indices);
    if ((ni > 0) && (PyArray_Size((PyObject *)self) == 0)) {
        PyErr_SetString(PyExc_IndexError,
                        "cannot replace elements of an empty array");
        goto fail;
    }
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

    overlap = arrays_overlap(self, values) || arrays_overlap(self, indices);
    if (overlap || !PyArray_ISCONTIGUOUS(self)) {
        PyArrayObject *obj;
        int flags = NPY_ARRAY_CARRAY | NPY_ARRAY_WRITEBACKIFCOPY |
                    NPY_ARRAY_ENSURECOPY;

        Py_INCREF(PyArray_DESCR(self));
        obj = (PyArrayObject *)PyArray_FromArray(self,
                                                 PyArray_DESCR(self), flags);
        copied = 1;
        assert(self != obj);
        self = obj;
    }
    max_item = PyArray_SIZE(self);
    dest = PyArray_DATA(self);
    itemsize = PyArray_ITEMSIZE(self);

    int has_references = PyDataType_REFCHK(PyArray_DESCR(self));

    if (!has_references) {
        /* if has_references is not needed memcpy is safe for a simple copy  */
        NPY_BEGIN_THREADS_THRESHOLDED(ni);
    }
    else {
        PyArray_Descr *dtype = PyArray_DESCR(self);
        if (PyArray_GetDTypeTransferFunction(
                PyArray_ISALIGNED(self), itemsize, itemsize, dtype, dtype, 0,
                &cast_info, &flags) < 0) {
            goto fail;
        }
        if (!(flags & NPY_METH_REQUIRES_PYAPI)) {
            NPY_BEGIN_THREADS_THRESHOLDED(ni);
        }
    }


    if (has_references) {
        const npy_intp one = 1;
        const npy_intp strides[2] = {itemsize, itemsize};

        switch(clipmode) {
        case NPY_RAISE:
            for (i = 0; i < ni; i++) {
                src = PyArray_BYTES(values) + itemsize*(i % nv);
                tmp = ((npy_intp *)(PyArray_DATA(indices)))[i];
                if (check_and_adjust_index(&tmp, max_item, 0, _save) < 0) {
                    goto fail;
                }
                char *data[2] = {src, dest + tmp*itemsize};
                if (cast_info.func(
                        &cast_info.context, data, &one, strides,
                        cast_info.auxdata) < 0) {
                    NPY_END_THREADS;
                    goto fail;
                }
            }
            break;
        case NPY_WRAP:
            for (i = 0; i < ni; i++) {
                src = PyArray_BYTES(values) + itemsize * (i % nv);
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
                char *data[2] = {src, dest + tmp*itemsize};
                if (cast_info.func(
                        &cast_info.context, data, &one, strides,
                        cast_info.auxdata) < 0) {
                    NPY_END_THREADS;
                    goto fail;
                }
            }
            break;
        case NPY_CLIP:
            for (i = 0; i < ni; i++) {
                src = PyArray_BYTES(values) + itemsize * (i % nv);
                tmp = ((npy_intp *)(PyArray_DATA(indices)))[i];
                if (tmp < 0) {
                    tmp = 0;
                }
                else if (tmp >= max_item) {
                    tmp = max_item - 1;
                }
                char *data[2] = {src, dest + tmp*itemsize};
                if (cast_info.func(
                        &cast_info.context, data, &one, strides,
                        cast_info.auxdata) < 0) {
                    NPY_END_THREADS;
                    goto fail;
                }
            }
            break;
        }
    }
    else {
        switch(clipmode) {
        case NPY_RAISE:
            for (i = 0; i < ni; i++) {
                src = PyArray_BYTES(values) + itemsize * (i % nv);
                tmp = ((npy_intp *)(PyArray_DATA(indices)))[i];
                if (check_and_adjust_index(&tmp, max_item, 0, _save) < 0) {
                    goto fail;
                }
                memmove(dest + tmp * itemsize, src, itemsize);
            }
            break;
        case NPY_WRAP:
            for (i = 0; i < ni; i++) {
                src = PyArray_BYTES(values) + itemsize * (i % nv);
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
                memmove(dest + tmp * itemsize, src, itemsize);
            }
            break;
        case NPY_CLIP:
            for (i = 0; i < ni; i++) {
                src = PyArray_BYTES(values) + itemsize * (i % nv);
                tmp = ((npy_intp *)(PyArray_DATA(indices)))[i];
                if (tmp < 0) {
                    tmp = 0;
                }
                else if (tmp >= max_item) {
                    tmp = max_item - 1;
                }
                memmove(dest + tmp * itemsize, src, itemsize);
            }
            break;
        }
    }
    NPY_END_THREADS;

 finish:
    NPY_cast_info_xfree(&cast_info);

    Py_XDECREF(values);
    Py_XDECREF(indices);
    if (copied) {
        PyArray_ResolveWritebackIfCopy(self);
        Py_DECREF(self);
    }
    Py_RETURN_NONE;

 fail:
    NPY_cast_info_xfree(&cast_info);

    Py_XDECREF(indices);
    Py_XDECREF(values);
    if (copied) {
        PyArray_DiscardWritebackIfCopy(self);
        Py_XDECREF(self);
    }
    return NULL;
}


static NPY_GCC_OPT_3 inline void
npy_fastputmask_impl(
        char *dest, char *src, const npy_bool *mask_data,
        npy_intp ni, npy_intp nv, npy_intp chunk)
{
    if (nv == 1) {
        for (npy_intp i = 0; i < ni; i++) {
            if (mask_data[i]) {
                memmove(dest, src, chunk);
            }
            dest += chunk;
        }
    }
    else {
        char *tmp_src = src;
        for (npy_intp i = 0, j = 0; i < ni; i++, j++) {
            if (NPY_UNLIKELY(j >= nv)) {
                j = 0;
                tmp_src = src;
            }
            if (mask_data[i]) {
                memmove(dest, tmp_src, chunk);
            }
            dest += chunk;
            tmp_src += chunk;
        }
    }
}


/*
 * Helper function instantiating npy_fastput_impl in different branches
 * to allow the compiler to optimize each to the specific itemsize.
 */
static NPY_GCC_OPT_3 void
npy_fastputmask(
        char *dest, char *src, npy_bool *mask_data,
        npy_intp ni, npy_intp nv, npy_intp chunk)
{
    if (chunk == 1) {
        return npy_fastputmask_impl(dest, src, mask_data, ni, nv, chunk);
    }
    if (chunk == 2) {
        return npy_fastputmask_impl(dest, src, mask_data, ni, nv, chunk);
    }
    if (chunk == 4) {
        return npy_fastputmask_impl(dest, src, mask_data, ni, nv, chunk);
    }
    if (chunk == 8) {
        return npy_fastputmask_impl(dest, src, mask_data, ni, nv, chunk);
    }
    if (chunk == 16) {
        return npy_fastputmask_impl(dest, src, mask_data, ni, nv, chunk);
    }
    if (chunk == 32) {
        return npy_fastputmask_impl(dest, src, mask_data, ni, nv, chunk);
    }

    return npy_fastputmask_impl(dest, src, mask_data, ni, nv, chunk);
}


/*NUMPY_API
 * Put values into an array according to a mask.
 */
NPY_NO_EXPORT PyObject *
PyArray_PutMask(PyArrayObject *self, PyObject* values0, PyObject* mask0)
{
    PyArrayObject *mask, *values;
    PyArray_Descr *dtype;
    npy_intp itemsize, ni, nv;
    char *src, *dest;
    npy_bool *mask_data;
    int copied = 0;
    int overlap = 0;
    NPY_BEGIN_THREADS_DEF;

    mask = NULL;
    values = NULL;
    if (!PyArray_Check(self)) {
        PyErr_SetString(PyExc_TypeError,
                        "putmask: first argument must "
                        "be an array");
        return NULL;
    }

    if (PyArray_FailUnlessWriteable(self, "putmask: output array") < 0) {
        return NULL;
    }

    mask = (PyArrayObject *)PyArray_FROM_OTF(mask0, NPY_BOOL,
                                NPY_ARRAY_CARRAY | NPY_ARRAY_FORCECAST);
    if (mask == NULL) {
        goto fail;
    }
    ni = PyArray_SIZE(mask);
    if (ni != PyArray_SIZE(self)) {
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

    overlap = arrays_overlap(self, values) || arrays_overlap(self, mask);
    if (overlap || !PyArray_ISCONTIGUOUS(self)) {
        int flags = NPY_ARRAY_CARRAY | NPY_ARRAY_WRITEBACKIFCOPY;
        PyArrayObject *obj;

        if (overlap) {
            flags |= NPY_ARRAY_ENSURECOPY;
        }

        dtype = PyArray_DESCR(self);
        Py_INCREF(dtype);
        obj = (PyArrayObject *)PyArray_FromArray(self, dtype, flags);
        if (obj != self) {
            copied = 1;
        }
        self = obj;
    }

    itemsize = PyArray_ITEMSIZE(self);
    dest = PyArray_DATA(self);

    if (PyDataType_REFCHK(PyArray_DESCR(self))) {
        NPY_cast_info cast_info;
        NPY_ARRAYMETHOD_FLAGS flags;
        const npy_intp one = 1;
        const npy_intp strides[2] = {itemsize, itemsize};

        NPY_cast_info_init(&cast_info);
        if (PyArray_GetDTypeTransferFunction(
                PyArray_ISALIGNED(self), itemsize, itemsize, dtype, dtype, 0,
                &cast_info, &flags) < 0) {
            goto fail;
        }
        if (!(flags & NPY_METH_REQUIRES_PYAPI)) {
            NPY_BEGIN_THREADS;
        }

        for (npy_intp i = 0, j = 0; i < ni; i++, j++) {
            if (j >= nv) {
                j = 0;
            }
            if (mask_data[i]) {
                char *data[2] = {src + j*itemsize, dest + i*itemsize};
                if (cast_info.func(
                        &cast_info.context, data, &one, strides,
                        cast_info.auxdata) < 0) {
                    NPY_END_THREADS;
                    NPY_cast_info_xfree(&cast_info);
                    goto fail;
                }
            }
        }
        NPY_cast_info_xfree(&cast_info);
    }
    else {
        NPY_BEGIN_THREADS;
        npy_fastputmask(dest, src, mask_data, ni, nv, itemsize);
    }

    NPY_END_THREADS;

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

static NPY_GCC_OPT_3 inline int
npy_fastrepeat_impl(
    npy_intp n_outer, npy_intp n, npy_intp nel, npy_intp chunk,
    npy_bool broadcast, npy_intp* counts, char* new_data, char* old_data,
    npy_intp elsize, NPY_cast_info *cast_info, int needs_custom_copy)
{
    npy_intp i, j, k;
    for (i = 0; i < n_outer; i++) {
        for (j = 0; j < n; j++) {
            npy_intp tmp = broadcast ? counts[0] : counts[j];
            for (k = 0; k < tmp; k++) {
                if (!needs_custom_copy) {
                    memcpy(new_data, old_data, chunk);
                }
                else {
                    char *data[2] = {old_data, new_data};
                    npy_intp strides[2] = {elsize, elsize};
                    if (cast_info->func(&cast_info->context, data, &nel,
                                       strides, cast_info->auxdata) < 0) {
                        return -1;
                    }
                }
                new_data += chunk;
            }
            old_data += chunk;
        }
    }
    return 0;
}


/*
 * Helper to allow the compiler to specialize for all direct element copy
 * cases (e.g. all numerical dtypes).
 */
static NPY_GCC_OPT_3 int
npy_fastrepeat(
    npy_intp n_outer, npy_intp n, npy_intp nel, npy_intp chunk,
    npy_bool broadcast, npy_intp* counts, char* new_data, char* old_data,
    npy_intp elsize, NPY_cast_info *cast_info, int needs_custom_copy)
{
    if (!needs_custom_copy) {
        if (chunk == 1) {
            return npy_fastrepeat_impl(
                n_outer, n, nel, chunk, broadcast, counts, new_data, old_data,
                elsize, cast_info, needs_custom_copy);
        }
        if (chunk == 2) {
            return npy_fastrepeat_impl(
                n_outer, n, nel, chunk, broadcast, counts, new_data, old_data,
                elsize, cast_info, needs_custom_copy);
        }
        if (chunk == 4) {
            return npy_fastrepeat_impl(
                n_outer, n, nel, chunk, broadcast, counts, new_data, old_data,
                elsize, cast_info, needs_custom_copy);
        }
        if (chunk == 8) {
            return npy_fastrepeat_impl(
                n_outer, n, nel, chunk, broadcast, counts, new_data, old_data,
                elsize, cast_info, needs_custom_copy);
        }
        if (chunk == 16) {
            return npy_fastrepeat_impl(
                n_outer, n, nel, chunk, broadcast, counts, new_data, old_data,
                elsize, cast_info, needs_custom_copy);
        }
        if (chunk == 32) {
            return npy_fastrepeat_impl(
                n_outer, n, nel, chunk, broadcast, counts, new_data, old_data,
                elsize, cast_info, needs_custom_copy);
        }
    }

    return npy_fastrepeat_impl(
        n_outer, n, nel, chunk, broadcast, counts, new_data, old_data, elsize,
        cast_info, needs_custom_copy);
}


/*NUMPY_API
 * Repeat the array.
 */
NPY_NO_EXPORT PyObject *
PyArray_Repeat(PyArrayObject *aop, PyObject *op, int axis)
{
    npy_intp *counts;
    npy_intp i, j, n, n_outer, chunk, elsize, nel;
    npy_intp total = 0;
    npy_bool broadcast = NPY_FALSE;
    PyArrayObject *repeats = NULL;
    PyObject *ap = NULL;
    PyArrayObject *ret = NULL;
    char *new_data, *old_data;
    NPY_cast_info cast_info;
    NPY_ARRAYMETHOD_FLAGS flags;

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
    NPY_cast_info_init(&cast_info);

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
                PyErr_SetString(PyExc_ValueError,
                                "repeats may not contain negative values.");
                goto fail;
            }
            total += counts[j];
        }
    }

    /* Fill in dimensions of new array */
    npy_intp dims[NPY_MAXDIMS] = {0};

    for (int i = 0; i < PyArray_NDIM(aop); i++) {
        dims[i] = PyArray_DIMS(aop)[i];
    }

    dims[axis] = total;

    /* Construct new array */
    Py_INCREF(PyArray_DESCR(aop));
    ret = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(aop),
                                                PyArray_DESCR(aop),
                                                PyArray_NDIM(aop),
                                                dims,
                                                NULL, NULL, 0,
                                                (PyObject *)aop);
    if (ret == NULL) {
        goto fail;
    }
    new_data = PyArray_DATA(ret);
    old_data = PyArray_DATA(aop);

    nel = 1;
    elsize = PyArray_ITEMSIZE(aop);
    for(i = axis + 1; i < PyArray_NDIM(aop); i++) {
        nel *= PyArray_DIMS(aop)[i];
    }
    chunk = nel*elsize;

    n_outer = 1;
    for (i = 0; i < axis; i++) {
        n_outer *= PyArray_DIMS(aop)[i];
    }

    int needs_custom_copy = 0;
    if (PyDataType_REFCHK(PyArray_DESCR(ret))) {
        needs_custom_copy = 1;
        if (PyArray_GetDTypeTransferFunction(
                1, elsize, elsize, PyArray_DESCR(aop), PyArray_DESCR(ret), 0,
                &cast_info, &flags) < 0) {
            goto fail;
        }
    }

    if (npy_fastrepeat(n_outer, n, nel, chunk, broadcast, counts, new_data,
                       old_data, elsize, &cast_info, needs_custom_copy) < 0) {
        goto fail;
    }

    Py_DECREF(repeats);
    Py_XDECREF(aop);
    NPY_cast_info_xfree(&cast_info);
    return (PyObject *)ret;

 fail:
    Py_DECREF(repeats);
    Py_XDECREF(aop);
    Py_XDECREF(ret);
    NPY_cast_info_xfree(&cast_info);
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
    NPY_cast_info cast_info = {.func = NULL};
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
    dtype = PyArray_DESCR(mps[0]);

    int copy_existing_out = 0;
    /* Set-up return array */
    if (out == NULL) {
        Py_INCREF(dtype);
        obj = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(ap),
                                                    dtype,
                                                    multi->nd,
                                                    multi->dimensions,
                                                    NULL, NULL, 0,
                                                    (PyObject *)ap);
    }
    else {
        if ((PyArray_NDIM(out) != multi->nd)
                    || !PyArray_CompareLists(PyArray_DIMS(out),
                                             multi->dimensions,
                                             multi->nd)) {
            PyErr_SetString(PyExc_TypeError,
                            "choose: invalid shape for output array.");
            goto fail;
        }

        if (PyArray_FailUnlessWriteable(out, "output array") < 0) {
            goto fail;
        }

        for (i = 0; i < n; i++) {
            if (arrays_overlap(out, mps[i])) {
                copy_existing_out = 1;
            }
        }

        if (clipmode == NPY_RAISE) {
            /*
             * we need to make sure and get a copy
             * so the input array is not changed
             * before the error is called
             */
            copy_existing_out = 1;
        }

        if (!PyArray_EquivTypes(dtype, PyArray_DESCR(out))) {
            copy_existing_out = 1;
        }

        if (copy_existing_out) {
            Py_INCREF(dtype);
            obj = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
                                                        dtype,
                                                        multi->nd,
                                                        multi->dimensions,
                                                        NULL, NULL, 0,
                                                        (PyObject *)out);
        }
        else {
            obj = (PyArrayObject *)Py_NewRef(out);
        }
    }

    if (obj == NULL) {
        goto fail;
    }
    elsize = dtype->elsize;
    ret_data = PyArray_DATA(obj);
    npy_intp transfer_strides[2] = {elsize, elsize};
    npy_intp one = 1;
    NPY_ARRAYMETHOD_FLAGS transfer_flags = 0;
    if (PyDataType_REFCHK(dtype)) {
        int is_aligned = IsUintAligned(obj);
        PyArray_Descr *obj_dtype = PyArray_DESCR(obj);
        PyArray_GetDTypeTransferFunction(
                    is_aligned,
                    dtype->elsize,
                    obj_dtype->elsize,
                    dtype,
                    obj_dtype, 0, &cast_info,
                    &transfer_flags);
    }

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
        if (cast_info.func == NULL) {
            /* We ensure memory doesn't overlap, so can use memcpy */
            memcpy(ret_data, PyArray_MultiIter_DATA(multi, mi), elsize);
        }
        else {
            char *args[2] = {PyArray_MultiIter_DATA(multi, mi), ret_data};
            if (cast_info.func(&cast_info.context, args, &one,
                                transfer_strides, cast_info.auxdata) < 0) {
                goto fail;
            }
        }
        ret_data += elsize;
        PyArray_MultiIter_NEXT(multi);
    }

    NPY_cast_info_xfree(&cast_info);
    Py_DECREF(multi);
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
    }
    Py_DECREF(ap);
    PyDataMem_FREE(mps);
    if (copy_existing_out) {
        int res = PyArray_CopyInto(out, obj);
        Py_DECREF(obj);
        if (res < 0) {
            return NULL;
        }
        return Py_NewRef(out);
    }
    return (PyObject *)obj;

 fail:
    NPY_cast_info_xfree(&cast_info);
    Py_XDECREF(multi);
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
    }
    Py_XDECREF(ap);
    PyDataMem_FREE(mps);
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
              PyArray_PartitionFunc *part, npy_intp const *kth, npy_intp nkth)
{
    npy_intp N = PyArray_DIM(op, axis);
    npy_intp elsize = (npy_intp)PyArray_ITEMSIZE(op);
    npy_intp astride = PyArray_STRIDE(op, axis);
    int swap = PyArray_ISBYTESWAPPED(op);
    int is_aligned = IsAligned(op);
    int needcopy = !is_aligned || swap || astride != elsize;
    int needs_api = PyDataType_FLAGCHK(PyArray_DESCR(op), NPY_NEEDS_PYAPI);

    char *buffer = NULL;

    PyArrayIterObject *it;
    npy_intp size;

    int ret = 0;

    PyArray_Descr *descr = PyArray_DESCR(op);
    PyArray_Descr *odescr = NULL;

    NPY_cast_info to_cast_info = {.func = NULL};
    NPY_cast_info from_cast_info = {.func = NULL};

    NPY_BEGIN_THREADS_DEF;

    /* Check if there is any sorting to do */
    if (N <= 1 || PyArray_SIZE(op) == 0) {
        return 0;
    }

    PyObject *mem_handler = PyDataMem_GetHandler();
    if (mem_handler == NULL) {
        return -1;
    }
    it = (PyArrayIterObject *)PyArray_IterAllButAxis((PyObject *)op, &axis);
    if (it == NULL) {
        Py_DECREF(mem_handler);
        return -1;
    }
    size = it->size;

    if (needcopy) {
        buffer = PyDataMem_UserNEW(N * elsize, mem_handler);
        if (buffer == NULL) {
            ret = -1;
            goto fail;
        }
        if (PyDataType_FLAGCHK(descr, NPY_NEEDS_INIT)) {
            memset(buffer, 0, N * elsize);
        }

        if (swap) {
            odescr = PyArray_DescrNewByteorder(descr, NPY_SWAP);
        }
        else {
            odescr = descr;
            Py_INCREF(odescr);
        }

        NPY_ARRAYMETHOD_FLAGS to_transfer_flags;

        if (PyArray_GetDTypeTransferFunction(
                is_aligned, astride, elsize, descr, odescr, 0, &to_cast_info,
                &to_transfer_flags) != NPY_SUCCEED) {
            goto fail;
        }

        NPY_ARRAYMETHOD_FLAGS from_transfer_flags;

        if (PyArray_GetDTypeTransferFunction(
                is_aligned, elsize, astride, odescr, descr, 0, &from_cast_info,
                &from_transfer_flags) != NPY_SUCCEED) {
            goto fail;
        }
    }

    NPY_BEGIN_THREADS_DESCR(descr);

    while (size--) {
        char *bufptr = it->dataptr;

        if (needcopy) {
            char *args[2] = {it->dataptr, buffer};
            npy_intp strides[2] = {astride, elsize};

            if (NPY_UNLIKELY(to_cast_info.func(
                                 &to_cast_info.context, args, &N, strides,
                                 to_cast_info.auxdata) < 0)) {
                goto fail;
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
            if (needs_api && PyErr_Occurred()) {
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
                ret = part(bufptr, N, kth[i], pivots, &npiv, nkth, op);
                if (needs_api && PyErr_Occurred()) {
                    ret = -1;
                }
                if (ret < 0) {
                    goto fail;
                }
            }
        }

        if (needcopy) {
            char *args[2] = {buffer, it->dataptr};
            npy_intp strides[2] = {elsize, astride};

            if (NPY_UNLIKELY(from_cast_info.func(
                                 &from_cast_info.context, args, &N, strides,
                                 from_cast_info.auxdata) < 0)) {
                goto fail;
            }
        }

        PyArray_ITER_NEXT(it);
    }

fail:
    NPY_END_THREADS_DESCR(descr);
    /* cleanup internal buffer */
    if (needcopy) {
        PyArray_ClearBuffer(odescr, buffer, elsize, N, 1);
        PyDataMem_UserFREE(buffer, N * elsize, mem_handler);
        Py_DECREF(odescr);
    }
    if (ret < 0 && !PyErr_Occurred()) {
        /* Out of memory during sorting or buffer creation */
        PyErr_NoMemory();
    }
    // if an error happened with a dtype that doesn't hold the GIL, need
    // to make sure we return an error value from this function.
    // note: only the first error is ever reported, subsequent errors
    // must *not* set the error handler.
    if (PyErr_Occurred() && ret == 0) {
        ret = -1;
    }
    Py_DECREF(it);
    Py_DECREF(mem_handler);
    NPY_cast_info_xfree(&to_cast_info);
    NPY_cast_info_xfree(&from_cast_info);

    return ret;
}

static PyObject*
_new_argsortlike(PyArrayObject *op, int axis, PyArray_ArgSortFunc *argsort,
                 PyArray_ArgPartitionFunc *argpart,
                 npy_intp const *kth, npy_intp nkth)
{
    npy_intp N = PyArray_DIM(op, axis);
    npy_intp elsize = (npy_intp)PyArray_ITEMSIZE(op);
    npy_intp astride = PyArray_STRIDE(op, axis);
    int swap = PyArray_ISBYTESWAPPED(op);
    int is_aligned = IsAligned(op);
    int needcopy = !is_aligned || swap || astride != elsize;
    int needs_api = PyDataType_FLAGCHK(PyArray_DESCR(op), NPY_NEEDS_PYAPI);
    int needidxbuffer;

    char *valbuffer = NULL;
    npy_intp *idxbuffer = NULL;

    PyArrayObject *rop;
    npy_intp rstride;

    PyArrayIterObject *it, *rit;
    npy_intp size;

    int ret = 0;

    PyArray_Descr *descr = PyArray_DESCR(op);
    PyArray_Descr *odescr = NULL;

    NPY_ARRAYMETHOD_FLAGS transfer_flags;
    NPY_cast_info cast_info = {.func = NULL};

    NPY_BEGIN_THREADS_DEF;

    PyObject *mem_handler = PyDataMem_GetHandler();
    if (mem_handler == NULL) {
        return NULL;
    }
    rop = (PyArrayObject *)PyArray_NewFromDescr(
            Py_TYPE(op), PyArray_DescrFromType(NPY_INTP),
            PyArray_NDIM(op), PyArray_DIMS(op), NULL, NULL,
            0, (PyObject *)op);
    if (rop == NULL) {
        Py_DECREF(mem_handler);
        return NULL;
    }
    rstride = PyArray_STRIDE(rop, axis);
    needidxbuffer = rstride != sizeof(npy_intp);

    /* Check if there is any argsorting to do */
    if (N <= 1 || PyArray_SIZE(op) == 0) {
        Py_DECREF(mem_handler);
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
        valbuffer = PyDataMem_UserNEW(N * elsize, mem_handler);
        if (valbuffer == NULL) {
            ret = -1;
            goto fail;
        }
        if (PyDataType_FLAGCHK(descr, NPY_NEEDS_INIT)) {
            memset(valbuffer, 0, N * elsize);
        }

        if (swap) {
            odescr = PyArray_DescrNewByteorder(descr, NPY_SWAP);
        }
        else {
            odescr = descr;
            Py_INCREF(odescr);
        }

        if (PyArray_GetDTypeTransferFunction(
                is_aligned, astride, elsize, descr, odescr, 0, &cast_info,
                &transfer_flags) != NPY_SUCCEED) {
            goto fail;
        }
    }

    if (needidxbuffer) {
        idxbuffer = (npy_intp *)PyDataMem_UserNEW(N * sizeof(npy_intp),
                                                  mem_handler);
        if (idxbuffer == NULL) {
            ret = -1;
            goto fail;
        }
    }

    NPY_BEGIN_THREADS_DESCR(descr);

    while (size--) {
        char *valptr = it->dataptr;
        npy_intp *idxptr = (npy_intp *)rit->dataptr;
        npy_intp *iptr, i;

        if (needcopy) {
            char *args[2] = {it->dataptr, valbuffer};
            npy_intp strides[2] = {astride, elsize};

            if (NPY_UNLIKELY(cast_info.func(
                                 &cast_info.context, args, &N, strides,
                                 cast_info.auxdata) < 0)) {
                goto fail;
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
            /* Object comparisons may raise an exception */
            if (needs_api && PyErr_Occurred()) {
                ret = -1;
            }
            if (ret < 0) {
                goto fail;
            }
        }
        else {
            npy_intp pivots[NPY_MAX_PIVOT_STACK];
            npy_intp npiv = 0;

            for (i = 0; i < nkth; ++i) {
                ret = argpart(valptr, idxptr, N, kth[i], pivots, &npiv, nkth, op);
                /* Object comparisons may raise an exception */
                if (needs_api && PyErr_Occurred()) {
                    ret = -1;
                }
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
    NPY_END_THREADS_DESCR(descr);
    /* cleanup internal buffers */
    if (needcopy) {
        PyArray_ClearBuffer(odescr, valbuffer, elsize, N, 1);
        PyDataMem_UserFREE(valbuffer, N * elsize, mem_handler);
        Py_DECREF(odescr);
    }
    PyDataMem_UserFREE(idxbuffer, N * sizeof(npy_intp), mem_handler);
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
    Py_DECREF(mem_handler);
    NPY_cast_info_xfree(&cast_info);

    return (PyObject *)rop;
}


/*NUMPY_API
 * Sort an array in-place
 */
NPY_NO_EXPORT int
PyArray_Sort(PyArrayObject *op, int axis, NPY_SORTKIND which)
{
    PyArray_SortFunc *sort = NULL;
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

    sort = PyDataType_GetArrFuncs(PyArray_DESCR(op))->sort[which];

    if (sort == NULL) {
        if (PyDataType_GetArrFuncs(PyArray_DESCR(op))->compare) {
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

    if (PyArray_ISBOOL(ktharray)) {
        PyErr_SetString(PyExc_ValueError,
                "Booleans unacceptable as partition index");
        return NULL;
    }
    else if (!PyArray_ISINTEGER(ktharray)) {
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
        if (PyDataType_GetArrFuncs(PyArray_DESCR(op))->compare) {
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
    PyArray_ArgSortFunc *argsort = NULL;
    PyObject *ret;

    argsort = PyDataType_GetArrFuncs(PyArray_DESCR(op))->argsort[which];

    if (argsort == NULL) {
        if (PyDataType_GetArrFuncs(PyArray_DESCR(op))->compare) {
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

    /*
     * As a C-exported function, enum NPY_SELECTKIND loses its enum property
     * Check the values to make sure they are in range
     */
    if ((int)which < 0 || (int)which >= NPY_NSELECTS) {
        PyErr_SetString(PyExc_ValueError,
                        "not a valid partition kind");
        return NULL;
    }

    argpart = get_argpartition_func(PyArray_TYPE(op), which);
    if (argpart == NULL) {
        /* Use sorting, slower but equivalent */
        if (PyDataType_GetArrFuncs(PyArray_DESCR(op))->compare) {
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
        if (!PyDataType_GetArrFuncs(PyArray_DESCR(mps[i]))->argsort[NPY_STABLESORT]
                && !PyDataType_GetArrFuncs(PyArray_DESCR(mps[i]))->compare) {
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
    /*
    * Special case letting axis={-1,0} slip through for scalars,
    * for backwards compatibility reasons.
    */
    if (nd == 0 && (axis == 0 || axis == -1)) {
        /* TODO: can we deprecate this? */
    }
    else if (check_and_adjust_axis(&axis, nd) < 0) {
        goto fail;
    }
    if ((nd == 0) || (PyArray_SIZE(mps[0]) <= 1)) {
        /* empty/single element case */
        ret = (PyArrayObject *)PyArray_NewFromDescr(
            &PyArray_Type, PyArray_DescrFromType(NPY_INTP),
            PyArray_NDIM(mps[0]), PyArray_DIMS(mps[0]), NULL, NULL,
            0, NULL);

        if (ret == NULL) {
            goto fail;
        }
        if (PyArray_SIZE(mps[0]) > 0) {
            *((npy_intp *)(PyArray_DATA(ret))) = 0;
        }
        goto finish;
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
    maxelsize = PyArray_ITEMSIZE(mps[0]);
    needcopy = (rstride != sizeof(npy_intp));
    for (j = 0; j < n; j++) {
        needcopy = needcopy
            || PyArray_ISBYTESWAPPED(mps[j])
            || !(PyArray_FLAGS(mps[j]) & NPY_ARRAY_ALIGNED)
            || (PyArray_STRIDES(mps[j])[axis] != (npy_intp)PyArray_ITEMSIZE(mps[j]));
        if (PyArray_ITEMSIZE(mps[j]) > maxelsize) {
            maxelsize = PyArray_ITEMSIZE(mps[j]);
        }
    }

    if (needcopy) {
        char *valbuffer, *indbuffer;
        int *swaps;

        assert(N > 0);  /* Guaranteed and assumed by indbuffer */
        npy_intp valbufsize = N * maxelsize;
        if (NPY_UNLIKELY(valbufsize) == 0) {
            valbufsize = 1;  /* Ensure allocation is not empty */
        }

        valbuffer = PyDataMem_NEW(valbufsize);
        if (valbuffer == NULL) {
            goto fail;
        }
        indbuffer = PyDataMem_NEW(N * sizeof(npy_intp));
        if (indbuffer == NULL) {
            PyDataMem_FREE(valbuffer);
            goto fail;
        }
        swaps = malloc(NPY_LIKELY(n > 0) ? n * sizeof(int) : 1);
        if (swaps == NULL) {
            PyDataMem_FREE(valbuffer);
            PyDataMem_FREE(indbuffer);
            goto fail;
        }

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
                elsize = PyArray_ITEMSIZE(mps[j]);
                astride = PyArray_STRIDES(mps[j])[axis];
                argsort = PyDataType_GetArrFuncs(PyArray_DESCR(mps[j]))->argsort[NPY_STABLESORT];
                if(argsort == NULL) {
                    argsort = npy_atimsort;
                }
                _unaligned_strided_byte_copy(valbuffer, (npy_intp) elsize,
                                             its[j]->dataptr, astride, N, elsize);
                if (swaps[j]) {
                    _strided_byte_swap(valbuffer, (npy_intp) elsize, N, elsize);
                }
                rcode = argsort(valbuffer, (npy_intp *)indbuffer, N, mps[j]);
                if (rcode < 0 || (PyDataType_REFCHK(PyArray_DESCR(mps[j]))
                            && PyErr_Occurred())) {
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
                int rcode;
                argsort = PyDataType_GetArrFuncs(PyArray_DESCR(mps[j]))->argsort[NPY_STABLESORT];
                if(argsort == NULL) {
                    argsort = npy_atimsort;
                }
                rcode = argsort(its[j]->dataptr,
                        (npy_intp *)rit->dataptr, N, mps[j]);
                if (rcode < 0 || (object && PyErr_Occurred())) {
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

    /* Look for binary search function */
    if (perm) {
        argbinsearch = get_argbinsearch_func(dtype, side);
    }
    else {
        binsearch = get_binsearch_func(dtype, side);
    }
    if (binsearch == NULL && argbinsearch == NULL) {
        PyErr_SetString(PyExc_TypeError, "compare not supported for type");
        Py_DECREF(dtype);
        return NULL;
    }

    /* need ap2 as contiguous array and of right dtype (note: steals dtype reference) */
    ap2 = (PyArrayObject *)PyArray_CheckFromAny(op2, dtype,
                                0, 0,
                                NPY_ARRAY_CARRAY_RO | NPY_ARRAY_NOTSWAPPED,
                                NULL);
    if (ap2 == NULL) {
        return NULL;
    }
    /*
     * The dtype reference we had was used for creating ap2, which may have
     * replaced it with another. So here we copy the dtype of ap2 and use it for `ap1`.
     */
     dtype = (PyArray_Descr *)Py_NewRef(PyArray_DESCR(ap2));

    /*
     * If the needle (ap2) is larger than the haystack (op1) we copy the
     * haystack to a contiguous array for improved cache utilization.
     */
    if (PyArray_SIZE(ap2) > PyArray_SIZE(op1)) {
        ap1_flags |= NPY_ARRAY_CARRAY_RO;
    }
    /* dtype is stolen, after this we have no reference */
    ap1 = (PyArrayObject *)PyArray_CheckFromAny((PyObject *)op1, dtype,
                                1, 1, ap1_flags, NULL);
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
                  PyArray_STRIDES(ap1)[0], PyArray_ITEMSIZE(ap2),
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
                             PyArray_ITEMSIZE(ap2),
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
    if (check_and_adjust_axis_msg(&axis1, ndim, npy_interned_str.axis1) < 0) {
        return NULL;
    }
    if (check_and_adjust_axis_msg(&axis2, ndim, npy_interned_str.axis2) < 0) {
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
#if !NPY_SIMD
static inline npy_intp
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
#endif

#if NPY_SIMD
/* Count the zero bytes between `*d` and `end`, updating `*d` to point to where to keep counting from. */
NPY_FINLINE NPY_GCC_OPT_3 npyv_u8
count_zero_bytes_u8(const npy_uint8 **d, const npy_uint8 *end, npy_uint8 max_count)
{
    const npyv_u8 vone = npyv_setall_u8(1);
    const npyv_u8 vzero = npyv_zero_u8();

    npy_intp lane_max = 0;
    npyv_u8 vsum8 = npyv_zero_u8();
    while (*d < end && lane_max <= max_count - 1) {
        // we count zeros because `cmpeq` cheaper than `cmpneq` for most archs
        npyv_u8 vt = npyv_cvt_u8_b8(npyv_cmpeq_u8(npyv_load_u8(*d), vzero));
        vt = npyv_and_u8(vt, vone);
        vsum8 = npyv_add_u8(vsum8, vt);
        *d += npyv_nlanes_u8;
        lane_max += 1;
    }
    return vsum8;
}

NPY_FINLINE NPY_GCC_OPT_3 npyv_u16x2
count_zero_bytes_u16(const npy_uint8 **d, const npy_uint8 *end, npy_uint16 max_count)
{
    npyv_u16x2 vsum16;
    vsum16.val[0] = vsum16.val[1] = npyv_zero_u16();
    npy_intp lane_max = 0;
    while (*d < end && lane_max <= max_count - NPY_MAX_UINT8) {
        npyv_u8 vsum8 = count_zero_bytes_u8(d, end, NPY_MAX_UINT8);
        npyv_u16x2 part = npyv_expand_u16_u8(vsum8);
        vsum16.val[0] = npyv_add_u16(vsum16.val[0], part.val[0]);
        vsum16.val[1] = npyv_add_u16(vsum16.val[1], part.val[1]);
        lane_max += NPY_MAX_UINT8;
    }
    return vsum16;
}
#endif // NPY_SIMD
/*
 * Counts the number of non-zero values in a raw array.
 * The one loop process is shown below(take SSE2 with 128bits vector for example):
 *          |------------16 lanes---------|
 *[vsum8]   255 255 255 ... 255 255 255 255 count_zero_bytes_u8: counting 255*16 elements
 *                          !!
 *           |------------8 lanes---------|
 *[vsum16]   65535 65535 65535 ...   65535  count_zero_bytes_u16: counting (2*16-1)*16 elements
 *           65535 65535 65535 ...   65535
 *                          !!
 *           |------------4 lanes---------|
 *[sum_32_0] 65535    65535   65535   65535  count_nonzero_bytes
 *           65535    65535   65535   65535
 *[sum_32_1] 65535    65535   65535   65535
 *           65535    65535   65535   65535
 *                          !!
 *                     (2*16-1)*16
*/
static inline NPY_GCC_OPT_3 npy_intp
count_nonzero_u8(const char *data, npy_intp bstride, npy_uintp len)
{
    npy_intp count = 0;
    if (bstride == 1) {
    #if NPY_SIMD
        npy_uintp len_m = len & -npyv_nlanes_u8;
        npy_uintp zcount = 0;
        for (const char *end = data + len_m; data < end;) {
            npyv_u16x2 vsum16 = count_zero_bytes_u16((const npy_uint8**)&data, (const npy_uint8*)end, NPY_MAX_UINT16);
            npyv_u32x2 sum_32_0 = npyv_expand_u32_u16(vsum16.val[0]);
            npyv_u32x2 sum_32_1 = npyv_expand_u32_u16(vsum16.val[1]);
            zcount += npyv_sum_u32(npyv_add_u32(
                    npyv_add_u32(sum_32_0.val[0], sum_32_0.val[1]),
                    npyv_add_u32(sum_32_1.val[0], sum_32_1.val[1])
            ));
        }
        len  -= len_m;
        count = len_m - zcount;
    #else
        if (!NPY_ALIGNMENT_REQUIRED || npy_is_aligned(data, sizeof(npy_uint64))) {
            int step = 6 * sizeof(npy_uint64);
            int left_bytes = len % step;
            for (const char *end = data + len; data < end - left_bytes; data += step) {
                 count += count_nonzero_bytes_384((const npy_uint64 *)data);
            }
            len = left_bytes;
        }
    #endif // NPY_SIMD
    }
    for (; len > 0; --len, data += bstride) {
        count += (*data != 0);
    }
    return count;
}

static inline NPY_GCC_OPT_3 npy_intp
count_nonzero_u16(const char *data, npy_intp bstride, npy_uintp len)
{
    npy_intp count = 0;
#if NPY_SIMD
    if (bstride == sizeof(npy_uint16)) {
        npy_uintp zcount = 0, len_m = len & -npyv_nlanes_u16;
        const npyv_u16 vone  = npyv_setall_u16(1);
        const npyv_u16 vzero = npyv_zero_u16();

        for (npy_uintp lenx = len_m; lenx > 0;) {
            npyv_u16 vsum16 = npyv_zero_u16();
            npy_uintp max16 = PyArray_MIN(lenx, NPY_MAX_UINT16*npyv_nlanes_u16);

            for (const char *end = data + max16*bstride; data < end; data += NPY_SIMD_WIDTH) {
                npyv_u16 mask = npyv_cvt_u16_b16(npyv_cmpeq_u16(npyv_load_u16((npy_uint16*)data), vzero));
                         mask = npyv_and_u16(mask, vone);
                       vsum16 = npyv_add_u16(vsum16, mask);
            }
            lenx   -= max16;
            zcount += npyv_sumup_u16(vsum16);
        }
        len  -= len_m;
        count = len_m - zcount;
    }
#endif
    for (; len > 0; --len, data += bstride) {
        count += (*(npy_uint16*)data != 0);
    }
    return count;
}

static inline NPY_GCC_OPT_3 npy_intp
count_nonzero_u32(const char *data, npy_intp bstride, npy_uintp len)
{
    npy_intp count = 0;
#if NPY_SIMD
    if (bstride == sizeof(npy_uint32)) {
        const npy_uintp max_iter = NPY_MAX_UINT32*npyv_nlanes_u32;
        const npy_uintp len_m = (len > max_iter ? max_iter : len) & -npyv_nlanes_u32;
        const npyv_u32 vone   = npyv_setall_u32(1);
        const npyv_u32 vzero  = npyv_zero_u32();

        npyv_u32 vsum32 = npyv_zero_u32();
        for (const char *end = data + len_m*bstride; data < end; data += NPY_SIMD_WIDTH) {
            npyv_u32 mask = npyv_cvt_u32_b32(npyv_cmpeq_u32(npyv_load_u32((npy_uint32*)data), vzero));
                     mask = npyv_and_u32(mask, vone);
                   vsum32 = npyv_add_u32(vsum32, mask);
        }
        const npyv_u32 maskevn = npyv_reinterpret_u32_u64(npyv_setall_u64(0xffffffffULL));
        npyv_u64 odd  = npyv_shri_u64(npyv_reinterpret_u64_u32(vsum32), 32);
        npyv_u64 even = npyv_reinterpret_u64_u32(npyv_and_u32(vsum32, maskevn));
        count = len_m - npyv_sum_u64(npyv_add_u64(odd, even));
        len  -= len_m;
    }
#endif
    for (; len > 0; --len, data += bstride) {
        count += (*(npy_uint32*)data != 0);
    }
    return count;
}

static inline NPY_GCC_OPT_3 npy_intp
count_nonzero_u64(const char *data, npy_intp bstride, npy_uintp len)
{
    npy_intp count = 0;
#if NPY_SIMD
    if (bstride == sizeof(npy_uint64)) {
        const npy_uintp len_m = len & -npyv_nlanes_u64;
        const npyv_u64 vone   = npyv_setall_u64(1);
        const npyv_u64 vzero  = npyv_zero_u64();

        npyv_u64 vsum64 = npyv_zero_u64();
        for (const char *end = data + len_m*bstride; data < end; data += NPY_SIMD_WIDTH) {
            npyv_u64 mask = npyv_cvt_u64_b64(npyv_cmpeq_u64(npyv_load_u64((npy_uint64*)data), vzero));
                     mask = npyv_and_u64(mask, vone);
                   vsum64 = npyv_add_u64(vsum64, mask);
        }
        len  -= len_m;
        count = len_m - npyv_sum_u64(vsum64);
    }
#endif
    for (; len > 0; --len, data += bstride) {
        count += (*(npy_uint64*)data != 0);
    }
    return count;
}
/*
 * Counts the number of non-zero values in a raw int array. This
 * is a low-overhead function which does no heap allocations.
 *
 * Returns -1 on error.
 */
static NPY_GCC_OPT_3 npy_intp
count_nonzero_int(int ndim, char *data, const npy_intp *ashape, const npy_intp *astrides, int elsize)
{
    assert(elsize <= 8);
    int idim;
    npy_intp shape[NPY_MAXDIMS], strides[NPY_MAXDIMS];
    npy_intp coord[NPY_MAXDIMS];

    // Use raw iteration with no heap memory allocation
    if (PyArray_PrepareOneRawArrayIter(
                    ndim, ashape,
                    data, astrides,
                    &ndim, shape,
                    &data, strides) < 0) {
        return -1;
    }

    // Handle zero-sized array
    if (shape[0] == 0) {
        return 0;
    }

    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS_THRESHOLDED(shape[0]);

    #define NONZERO_CASE(LEN, SFX) \
        case LEN: \
            NPY_RAW_ITER_START(idim, ndim, coord, shape) { \
                count += count_nonzero_##SFX(data, strides[0], shape[0]); \
            } NPY_RAW_ITER_ONE_NEXT(idim, ndim, coord, shape, data, strides); \
            break

    npy_intp count = 0;
    switch(elsize) {
        NONZERO_CASE(1, u8);
        NONZERO_CASE(2, u16);
        NONZERO_CASE(4, u32);
        NONZERO_CASE(8, u64);
    }
    #undef NONZERO_CASE

    NPY_END_THREADS;
    return count;
}
/*
 * Counts the number of True values in a raw boolean array. This
 * is a low-overhead function which does no heap allocations.
 *
 * Returns -1 on error.
 */
NPY_NO_EXPORT NPY_GCC_OPT_3 npy_intp
count_boolean_trues(int ndim, char *data, npy_intp const *ashape, npy_intp const *astrides)
{
    return count_nonzero_int(ndim, data, ashape, astrides, 1);
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

    dtype = PyArray_DESCR(self);
    /* Special low-overhead version specific to the boolean/int types */
    if (PyArray_ISALIGNED(self) && (
            PyDataType_ISBOOL(dtype) || PyDataType_ISINTEGER(dtype))) {
        return count_nonzero_int(
            PyArray_NDIM(self), PyArray_BYTES(self), PyArray_DIMS(self),
            PyArray_STRIDES(self), dtype->elsize
        );
    }

    nonzero = PyDataType_GetArrFuncs(PyArray_DESCR(self))->nonzero;
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
            /* Special low-overhead version specific to the float types (and some others) */
            if (PyArray_ISNOTSWAPPED(self) && PyArray_ISALIGNED(self)) {
                npy_intp dispatched_nonzero_count = count_nonzero_trivial_dispatcher(count,
                                                        data, stride, dtype->type_num);
                if (dispatched_nonzero_count >= 0) {
                    return dispatched_nonzero_count;
                }
            }

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
    /* IterationNeedsAPI also checks dtype for whether `nonzero` may need it */
    needs_api = NpyIter_IterationNeedsAPI(iter);

    /* Get the pointers for inner loop iteration */
    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        NpyIter_Deallocate(iter);
        return -1;
    }

    if (!needs_api) {
        NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(iter));
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
    if (ndim == 0) {
        char const* msg;
        if (PyArray_ISBOOL(self)) {
            msg =
                "Calling nonzero on 0d arrays is not allowed. "
                "Use np.atleast_1d(scalar).nonzero() instead. "
                "If the context of this error is of the form "
                "`arr[nonzero(cond)]`, just use `arr[cond]`.";
        } else {
            msg =
                "Calling nonzero on 0d arrays is not allowed. "
                "Use np.atleast_1d(scalar).nonzero() instead.";
        }
        PyErr_SetString(PyExc_ValueError, msg);
        return NULL;
    }

    PyArrayObject *ret = NULL;
    PyObject *ret_tuple;
    npy_intp ret_dims[2];

    PyArray_NonzeroFunc *nonzero;
    PyArray_Descr *dtype;

    npy_intp nonzero_count;
    npy_intp added_count = 0;
    int needs_api;
    int is_bool;

    NpyIter *iter;
    NpyIter_IterNextFunc *iternext;
    NpyIter_GetMultiIndexFunc *get_multi_index;
    char **dataptr;

    dtype = PyArray_DESCR(self);
    nonzero = PyDataType_GetArrFuncs(dtype)->nonzero;
    needs_api = PyDataType_FLAGCHK(dtype, NPY_NEEDS_PYAPI);

    /*
     * First count the number of non-zeros in 'self'.
     */
    nonzero_count = PyArray_CountNonzero(self);
    if (nonzero_count < 0) {
        return NULL;
    }

    is_bool = PyArray_ISBOOL(self);

    /* Allocate the result as a 2D array */
    ret_dims[0] = nonzero_count;
    ret_dims[1] = ndim;
    ret = (PyArrayObject *)PyArray_NewFromDescr(
            &PyArray_Type, PyArray_DescrFromType(NPY_INTP),
            2, ret_dims, NULL, NULL,
            0, NULL);
    if (ret == NULL) {
        return NULL;
    }

    /* If it's a one-dimensional result, don't use an iterator */
    if (ndim == 1) {
        npy_intp * multi_index = (npy_intp *)PyArray_DATA(ret);
        char * data = PyArray_BYTES(self);
        npy_intp stride = PyArray_STRIDE(self, 0);
        npy_intp count = PyArray_DIM(self, 0);
        NPY_BEGIN_THREADS_DEF;

        /* nothing to do */
        if (nonzero_count == 0) {
            goto finish;
        }

        if (!needs_api) {
            NPY_BEGIN_THREADS_THRESHOLDED(count);
        }

        /* avoid function call for bool */
        if (is_bool) {
            /*
             * use fast memchr variant for sparse data, see gh-4370
             * the fast bool count is followed by this sparse path is faster
             * than combining the two loops, even for larger arrays
             */
            npy_intp * multi_index_end = multi_index + nonzero_count;
            if (((double)nonzero_count / count) <= 0.1) {
                npy_intp subsize;
                npy_intp j = 0;
                while (multi_index < multi_index_end) {
                    npy_memchr(data + j * stride, 0, stride, count - j,
                               &subsize, 1);
                    j += subsize;
                    if (j >= count) {
                        break;
                    }
                    *multi_index++ = j++;
                }
            }
            /*
             * Fallback to a branchless strategy to avoid branch misprediction
             * stalls that are very expensive on most modern processors.
             */
            else {
                npy_intp j = 0;

                /* Manually unroll for GCC and maybe other compilers */
                while (multi_index + 4 < multi_index_end && (j < count - 4) ) {
                    *multi_index = j;
                    multi_index += data[0] != 0;
                    *multi_index = j + 1;
                    multi_index += data[stride] != 0;
                    *multi_index = j + 2;
                    multi_index += data[stride * 2] != 0;
                    *multi_index = j + 3;
                    multi_index += data[stride * 3] != 0;
                    data += stride * 4;
                    j += 4;
                }

                while (multi_index < multi_index_end && (j < count) ) {
                    *multi_index = j;
                    multi_index += *data != 0;
                    data += stride;
                    ++j;
                }
            }
        }
        else {
            npy_intp j;
            for (j = 0; j < count; ++j) {
                if (nonzero(data, self)) {
                    if (++added_count > nonzero_count) {
                        break;
                    }
                    *multi_index++ = j;
                }
                if (needs_api && PyErr_Occurred()) {
                    break;
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

        /* IterationNeedsAPI also checks dtype for whether `nonzero` may need it */
        needs_api = NpyIter_IterationNeedsAPI(iter);

        if (!needs_api) {
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(iter));
        }

        dataptr = NpyIter_GetDataPtrArray(iter);

        multi_index = (npy_intp *)PyArray_DATA(ret);

        /* Get the multi-index for each non-zero element */
        if (is_bool) {
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
                    if (++added_count > nonzero_count) {
                        break;
                    }
                    get_multi_index(iter, multi_index);
                    multi_index += ndim;
                }
                if (needs_api && PyErr_Occurred()) {
                    break;
                }
            } while(iternext(iter));
        }

        NPY_END_THREADS;
    }

    NpyIter_Deallocate(iter);

finish:
    if (PyErr_Occurred()) {
        Py_DECREF(ret);
        return NULL;
    }

    /* if executed `nonzero()` check for miscount due to side-effect */
    if (!is_bool && added_count != nonzero_count) {
        PyErr_SetString(PyExc_RuntimeError,
            "number of non-zero array elements "
            "changed during function execution.");
        Py_DECREF(ret);
        return NULL;
    }

    ret_tuple = PyTuple_New(ndim);
    if (ret_tuple == NULL) {
        Py_DECREF(ret);
        return NULL;
    }

    /* Create views into ret, one for each dimension */
    for (i = 0; i < ndim; ++i) {
        npy_intp stride = ndim * NPY_SIZEOF_INTP;
        /* the result is an empty array, the view must point to valid memory */
        npy_intp data_offset = nonzero_count == 0 ? 0 : i * NPY_SIZEOF_INTP;

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
PyArray_MultiIndexGetItem(PyArrayObject *self, const npy_intp *multi_index)
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
PyArray_MultiIndexSetItem(PyArrayObject *self, const npy_intp *multi_index,
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

    return PyArray_Pack(PyArray_DESCR(self), data, obj);
}
