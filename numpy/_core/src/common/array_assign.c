/*
 * This file implements some helper functions for the array assignment
 * routines. The actual assignment routines are in array_assign_*.c
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <numpy/ndarraytypes.h>
#include "npy_config.h"
#include "npy_pycompat.h"

#include "shape.h"

#include "array_assign.h"
#include "common.h"
#include "lowlevel_strided_loops.h"
#include "mem_overlap.h"

/* See array_assign.h for parameter documentation */
NPY_NO_EXPORT int
broadcast_strides(int ndim, npy_intp const *shape,
                int strides_ndim, npy_intp const *strides_shape, npy_intp const *strides,
                char const *strides_name,
                npy_intp *out_strides)
{
    int idim, idim_start = ndim - strides_ndim;

    /* Can't broadcast to fewer dimensions */
    if (idim_start < 0) {
        goto broadcast_error;
    }

    /*
     * Process from the end to the start, so that 'strides' and 'out_strides'
     * can point to the same memory.
     */
    for (idim = ndim - 1; idim >= idim_start; --idim) {
        npy_intp strides_shape_value = strides_shape[idim - idim_start];
        /* If it doesn't have dimension one, it must match */
        if (strides_shape_value == 1) {
            out_strides[idim] = 0;
        }
        else if (strides_shape_value != shape[idim]) {
            goto broadcast_error;
        }
        else {
            out_strides[idim] = strides[idim - idim_start];
        }
    }

    /* New dimensions get a zero stride */
    for (idim = 0; idim < idim_start; ++idim) {
        out_strides[idim] = 0;
    }

    return 0;

broadcast_error: {
        PyObject *shape1 = convert_shape_to_string(strides_ndim,
                                                   strides_shape, "");
        if (shape1 == NULL) {
            return -1;
        }

        PyObject *shape2 = convert_shape_to_string(ndim, shape, "");
        if (shape2 == NULL) {
            Py_DECREF(shape1);
            return -1;
        }
        PyErr_Format(PyExc_ValueError,
                "could not broadcast %s from shape %S into shape %S",
                strides_name, shape1, shape2);
        Py_DECREF(shape1);
        Py_DECREF(shape2);
        return -1;
    }
}

/* See array_assign.h for parameter documentation */
NPY_NO_EXPORT int
raw_array_is_aligned(int ndim, npy_intp const *shape,
                     char *data, npy_intp const *strides, int alignment)
{

    /*
     * The code below expects the following:
     *  * that alignment is a power of two, as required by the C standard.
     *  * that casting from pointer to uintp gives a sensible representation
     *    we can use bitwise operations on (perhaps *not* req. by C std,
     *    but assumed by glibc so it should be fine)
     *  * that casting stride from intp to uintp (to avoid dependence on the
     *    signed int representation) preserves remainder wrt alignment, so
     *    stride%a is the same as ((unsigned intp)stride)%a. Req. by C std.
     *
     *  The code checks whether the lowest log2(alignment) bits of `data`
     *  and all `strides` are 0, as this implies that
     *  (data + n*stride)%alignment == 0 for all integers n.
     */

    if (alignment > 1) {
        npy_uintp align_check = (npy_uintp)data;
        int i;

        for (i = 0; i < ndim; i++) {
            /* skip dim == 1 as it is not required to have stride 0 */
            if (shape[i] > 1) {
                /* if shape[i] == 1, the stride is never used */
                align_check |= (npy_uintp)strides[i];
            }
            else if (shape[i] == 0) {
                /* an array with zero elements is always aligned */
                return 1;
            }
        }

        return npy_is_aligned((void *)align_check, alignment);
    }
    else if (alignment == 1) {
        return 1;
    }
    else {
        /* always return false for alignment == 0, which means cannot-be-aligned */
        return 0;
    }
}

NPY_NO_EXPORT int
IsAligned(PyArrayObject *ap)
{
    return raw_array_is_aligned(PyArray_NDIM(ap), PyArray_DIMS(ap),
                                PyArray_DATA(ap), PyArray_STRIDES(ap),
                                PyArray_DESCR(ap)->alignment);
}

NPY_NO_EXPORT int
IsUintAligned(PyArrayObject *ap)
{
    return raw_array_is_aligned(PyArray_NDIM(ap), PyArray_DIMS(ap),
                                PyArray_DATA(ap), PyArray_STRIDES(ap),
                                npy_uint_alignment(PyArray_DESCR(ap)->elsize));
}



/* Returns 1 if the arrays have overlapping data, 0 otherwise */
NPY_NO_EXPORT int
arrays_overlap(PyArrayObject *arr1, PyArrayObject *arr2)
{
    mem_overlap_t result;

    result = solve_may_share_memory(arr1, arr2, NPY_MAY_SHARE_BOUNDS);
    if (result == MEM_OVERLAP_NO) {
        return 0;
    }
    else {
        return 1;
    }
}
