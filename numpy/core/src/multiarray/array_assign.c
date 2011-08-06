/*
 * This file implements several array assignment routines.
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API
#define _MULTIARRAYMODULE
#include <numpy/ndarraytypes.h>

#include "npy_config.h"
#include "numpy/npy_3kcompat.h"

#include "convert_datatype.h"
#include "methods.h"
#include "shape.h"
#include "lowlevel_strided_loops.h"
#include "array_assign.h"

/*
 * Broadcasts strides to match the given dimensions. Can be used,
 * for instance, to set up a raw iteration.
 *
 * 'strides_name' is used to produce an error message if the strides
 * cannot be broadcast.
 *
 * Returns 0 on success, -1 on failure.
 */
static int
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
    for (idim = ndim - 1; idim >= idim_start; ++idim) {
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

        return -1;
   }
}

/*
 * Checks whether a data pointer + set of strides refers to a raw
 * array which is fully aligned data.
 */
static int
strides_are_aligned(int ndim, char *data, npy_intp *strides, int alignment)
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

/*
 * Assigns the scalar value to every element of the destination raw array.
 *
 * Returns 0 on success, -1 on failure.
 */
static int
raw_array_assign_scalar(int ndim, npy_intp *shape,
        PyArray_Descr *dst_dtype, char *dst_data, npy_intp *dst_strides,
        PyArray_Descr *src_dtype, char *src_data)
{
    int idim;
    npy_intp shape_it[NPY_MAXDIMS], dst_strides_it[NPY_MAXDIMS];
    npy_intp coord[NPY_MAXDIMS];
    NPY_BEGIN_THREADS_DEF;

    PyArray_StridedTransferFn *stransfer = NULL;
    NpyAuxData *transferdata = NULL;
    int aligned, needs_api = 0;
    npy_intp src_itemsize = src_dtype->elsize;

    /* Check alignment */
    aligned = strides_are_aligned(ndim, dst_data, dst_strides,
                                    dst_dtype->alignment);
    if (((npy_intp)src_data & (src_dtype->alignment - 1)) != 0) {
        aligned = 0;
    }

    /* Use raw iteration with no heap allocation */
    if (PyArray_PrepareOneRawArrayIter(
                    ndim, shape,
                    dst_data, dst_strides,
                    &ndim, shape_it,
                    &dst_data, dst_strides_it) < 0) {
        return -1;
    }

    /* Get the function to do the casting */
    if (PyArray_GetDTypeTransferFunction(aligned,
                        0, dst_strides_it[0],
                        src_dtype, dst_dtype,
                        0,
                        &stransfer, &transferdata,
                        &needs_api) != NPY_SUCCEED) {
        return -1;
    }

    if (!needs_api) {
        NPY_BEGIN_THREADS;
    }

    NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {
        /* Process the innermost dimension */
        stransfer(dst_data, dst_strides_it[0], src_data, 0,
                    shape_it[0], src_itemsize, transferdata);
    } NPY_RAW_ITER_ONE_NEXT(idim, ndim, coord,
                            shape_it, dst_data, dst_strides_it);

    if (!needs_api) {
        NPY_END_THREADS;
    }

    NPY_AUXDATA_FREE(transferdata);

    return (needs_api && PyErr_Occurred()) ? -1 : 0;
}

/*
 * Assigns the scalar value to every element of the destination raw array
 * where the 'wheremask' value is True.
 *
 * Returns 0 on success, -1 on failure.
 */
static int
raw_array_wheremasked_assign_scalar(int ndim, npy_intp *shape,
        PyArray_Descr *dst_dtype, char *dst_data, npy_intp *dst_strides,
        PyArray_Descr *src_dtype, char *src_data,
        PyArray_Descr *wheremask_dtype, char *wheremask_data,
        npy_intp *wheremask_strides)
{
    int idim;
    npy_intp shape_it[NPY_MAXDIMS], dst_strides_it[NPY_MAXDIMS];
    npy_intp wheremask_strides_it[NPY_MAXDIMS];
    npy_intp coord[NPY_MAXDIMS];
    NPY_BEGIN_THREADS_DEF;

    PyArray_MaskedStridedTransferFn *stransfer = NULL;
    NpyAuxData *transferdata = NULL;
    int aligned, needs_api = 0;
    npy_intp src_itemsize = src_dtype->elsize;

    /* Check alignment */
    aligned = strides_are_aligned(ndim, dst_data, dst_strides,
                                    dst_dtype->alignment);
    if (((npy_intp)src_data & (src_dtype->alignment - 1)) != 0) {
        aligned = 0;
    }

    /* Use raw iteration with no heap allocation */
    if (PyArray_PrepareTwoRawArrayIter(
                    ndim, shape,
                    dst_data, dst_strides,
                    wheremask_data, wheremask_strides,
                    &ndim, shape_it,
                    &dst_data, dst_strides_it,
                    &wheremask_data, wheremask_strides_it) < 0) {
        return -1;
    }

    /* Get the function to do the casting */
    if (PyArray_GetMaskedDTypeTransferFunction(aligned,
                        0, dst_strides_it[0], wheremask_strides_it[0],
                        src_dtype, dst_dtype, wheremask_dtype,
                        0,
                        &stransfer, &transferdata,
                        &needs_api) != NPY_SUCCEED) {
        return -1;
    }

    if (!needs_api) {
        NPY_BEGIN_THREADS;
    }

    NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {
        /* Process the innermost dimension */
        stransfer(dst_data, dst_strides_it[0], src_data, 0,
                    (npy_mask *)wheremask_data, wheremask_strides_it[0],
                    shape_it[0], src_itemsize, transferdata);
    } NPY_RAW_ITER_TWO_NEXT(idim, ndim, coord, shape_it,
                            dst_data, dst_strides_it,
                            wheremask_data, wheremask_strides_it);

    if (!needs_api) {
        NPY_END_THREADS;
    }

    NPY_AUXDATA_FREE(transferdata);

    return (needs_api && PyErr_Occurred()) ? -1 : 0;
}

/* See array_assign.h for documentation */
NPY_NO_EXPORT int
array_assign_scalar(PyArrayObject *dst,
                    PyArray_Descr *src_dtype, char *src_data,
                    PyArrayObject *wheremask,
                    NPY_CASTING casting, npy_bool overwritena)
{
    int copied_src_data = 0, dst_has_maskna = PyArray_HASMASKNA(dst);

    /* Check the casting rule */
    if (!can_cast_scalar_to(src_dtype, src_data,
                            PyArray_DESCR(dst), casting)) {
        PyObject *errmsg;
        errmsg = PyUString_FromString("Cannot cast scalar from ");
        PyUString_ConcatAndDel(&errmsg,
                PyObject_Repr((PyObject *)src_dtype));
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromString(" to "));
        PyUString_ConcatAndDel(&errmsg,
                PyObject_Repr((PyObject *)PyArray_DESCR(dst)));
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromFormat(" according to the rule %s",
                        npy_casting_to_string(casting)));
        PyErr_SetObject(PyExc_TypeError, errmsg);
        return -1;
    }

    /*
     * Make a copy of the src data if it's a different dtype than 'dst'
     * or isn't aligned, and the destination we're copying to has
     * more than one element.
     */
    if ((!PyArray_EquivTypes(PyArray_DESCR(dst), src_dtype) ||
                ((npy_intp)src_data & (src_dtype->alignment - 1)) != 0) &&
                    PyArray_SIZE(dst) > 1) {
        char *tmp_src_data;

        /* Allocate a new buffer to store the copied src data */
        tmp_src_data = PyArray_malloc(PyArray_DESCR(dst)->elsize);
        copied_src_data = 1;
        if (PyArray_CastRawArrays(1, src_data, tmp_src_data, 0, 0,
                            src_dtype, PyArray_DESCR(dst), 0) != NPY_SUCCEED) {
            goto fail;
        }

        /* Replace src_data/src_dtype */
        src_data = tmp_src_data;
        src_dtype = PyArray_DESCR(dst);
    }

    if (wheremask == NULL) {
        /* This is the case of a straightforward value assignment */
        if (overwritena || !dst_has_maskna) {
            /* If we're assigning to an array with a mask, set to all exposed */
            if (dst_has_maskna) {
                if (PyArray_AssignMaskNA(dst, 1) < 0) {
                    goto fail;
                }
            }

            /* Do the assignment with raw array iteration */
            if (raw_array_assign_scalar(PyArray_NDIM(dst), PyArray_DIMS(dst),
                    PyArray_DESCR(dst), PyArray_DATA(dst), PyArray_STRIDES(dst),
                    src_dtype, src_data) < 0) {
                goto fail;
            }
        }
        /* This is value assignment without overwriting NA values */
        else {
        }
    }
    else {
        /* This is the case of a straightforward where-masked assignment */
        if (overwritena || !dst_has_maskna) {
            npy_intp wheremask_strides[NPY_MAXDIMS];

            /* Broadcast the wheremask to 'dst' for raw iteration */
            if (broadcast_strides(PyArray_NDIM(dst), PyArray_DIMS(dst),
                        PyArray_NDIM(wheremask), PyArray_DIMS(wheremask),
                        PyArray_STRIDES(wheremask), "where mask",
                        wheremask_strides) < 0) {
                goto fail;
            }

            /* Do the masked assignment with raw array iteration */
            if (raw_array_wheremasked_assign_scalar(
                    PyArray_NDIM(dst), PyArray_DIMS(dst),
                    PyArray_DESCR(dst), PyArray_DATA(dst), PyArray_STRIDES(dst),
                    src_dtype, src_data,
                    PyArray_DESCR(wheremask), PyArray_DATA(wheremask),
                    wheremask_strides) < 0) {
                goto fail;
            }
        }
        /* This is masked value assignment without overwriting NA values */
        else {
        }
    }

    if (copied_src_data) {
        PyArray_free(src_data);
    }

    return 0;

fail:
    if (copied_src_data) {
        PyArray_free(src_data);
    }

    return -1;
}

