/*
 * This file implements some helper functions for the array assignment
 * routines. The actual assignment routines are in array_assign_*.c
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include <numpy/ndarraytypes.h>

#include "npy_config.h"
#include "npy_pycompat.h"

#include "shape.h"

#include "array_assign.h"

/* See array_assign.h for parameter documentation */
NPY_NO_EXPORT int
broadcast_strides(int ndim, npy_intp *shape,
                int strides_ndim, npy_intp *strides_shape, npy_intp *strides,
                char *strides_name,
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
        PyObject *errmsg;

        errmsg = PyUString_FromFormat("could not broadcast %s from shape ",
                                strides_name);
        PyUString_ConcatAndDel(&errmsg,
                build_shape_string(strides_ndim, strides_shape));
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromString(" into shape "));
        PyUString_ConcatAndDel(&errmsg,
                build_shape_string(ndim, shape));
        PyErr_SetObject(PyExc_ValueError, errmsg);
        Py_DECREF(errmsg);

        return -1;
   }
}

/* See array_assign.h for parameter documentation */
NPY_NO_EXPORT int
raw_array_is_aligned(int ndim, char *data, npy_intp *strides, int alignment)
{
    if (alignment > 1) {
        npy_intp align_check = (npy_intp)data;
        int idim;

        for (idim = 0; idim < ndim; ++idim) {
            align_check |= strides[idim];
        }

        return ((align_check & (alignment - 1)) == 0);
    }
    else {
        return 1;
    }
}


/* Gets a half-open range [start, end) which contains the array data */
NPY_NO_EXPORT void
get_array_memory_extents(PyArrayObject *arr,
                    npy_uintp *out_start, npy_uintp *out_end)
{
    npy_uintp start, end;
    npy_intp idim, ndim = PyArray_NDIM(arr);
    npy_intp *dimensions = PyArray_DIMS(arr),
            *strides = PyArray_STRIDES(arr);

    /* Calculate with a closed range [start, end] */
    start = end = (npy_uintp)PyArray_DATA(arr);
    for (idim = 0; idim < ndim; ++idim) {
        npy_intp stride = strides[idim], dim = dimensions[idim];
        /* If the array size is zero, return an empty range */
        if (dim == 0) {
            *out_start = *out_end = (npy_uintp)PyArray_DATA(arr);
            return;
        }
        /* Expand either upwards or downwards depending on stride */
        else {
            if (stride > 0) {
                end += stride*(dim-1);
            }
            else if (stride < 0) {
                start += stride*(dim-1);
            }
        }
    }

    /* Return a half-open range */
    *out_start = start;
    *out_end = end + PyArray_DESCR(arr)->elsize;
}

/* Returns 1 if the arrays have overlapping data, 0 otherwise */
NPY_NO_EXPORT int
arrays_overlap(PyArrayObject *arr1, PyArrayObject *arr2)
{
    npy_uintp start1 = 0, start2 = 0, end1 = 0, end2 = 0;

    get_array_memory_extents(arr1, &start1, &end1);
    get_array_memory_extents(arr2, &start2, &end2);

    return (start1 < end2) && (start2 < end1);
}
