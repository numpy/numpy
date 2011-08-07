/*
 * This file implements assignment from a scalar to an ndarray.
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
#include "na_mask.h"

#include "array_assign.h"

/*
 * Assigns the scalar value to every element of the destination raw array.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
raw_array_assign_scalar(int ndim, npy_intp *shape,
        PyArray_Descr *dst_dtype, char *dst_data, npy_intp *dst_strides,
        PyArray_Descr *src_dtype, char *src_data)
{
    int idim;
    npy_intp shape_it[NPY_MAXDIMS], dst_strides_it[NPY_MAXDIMS];
    npy_intp coord[NPY_MAXDIMS];
    NPY_BEGIN_THREADS_DEF;

    PyArray_StridedUnaryOp *stransfer = NULL;
    NpyAuxData *transferdata = NULL;
    int aligned, needs_api = 0;
    npy_intp src_itemsize = src_dtype->elsize;

    /* Check alignment */
    aligned = raw_array_is_aligned(ndim, dst_data, dst_strides,
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
NPY_NO_EXPORT int
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

    PyArray_MaskedStridedUnaryOp *stransfer = NULL;
    NpyAuxData *transferdata = NULL;
    int aligned, needs_api = 0;
    npy_intp src_itemsize = src_dtype->elsize;

    /* Check alignment */
    aligned = raw_array_is_aligned(ndim, dst_data, dst_strides,
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

/*
 * Assigns the scalar value to every element of the destination raw array
 * except for those which are masked as NA according to 'maskna'.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
raw_array_assign_scalar_preservena(int ndim, npy_intp *shape,
        PyArray_Descr *dst_dtype, char *dst_data, npy_intp *dst_strides,
        PyArray_Descr *src_dtype, char *src_data,
        PyArray_Descr *maskna_dtype, char *maskna_data,
        npy_intp *maskna_strides)
{
    int idim;
    npy_intp shape_it[NPY_MAXDIMS], dst_strides_it[NPY_MAXDIMS];
    npy_intp maskna_strides_it[NPY_MAXDIMS];
    npy_intp coord[NPY_MAXDIMS];
    NPY_BEGIN_THREADS_DEF;

    PyArray_MaskedStridedUnaryOp *stransfer = NULL;
    NpyAuxData *transferdata = NULL;
    int aligned, needs_api = 0;
    npy_intp src_itemsize = src_dtype->elsize;

    /* Check alignment */
    aligned = raw_array_is_aligned(ndim, dst_data, dst_strides,
                                    dst_dtype->alignment);
    if (((npy_intp)src_data & (src_dtype->alignment - 1)) != 0) {
        aligned = 0;
    }

    /* Use raw iteration with no heap allocation */
    if (PyArray_PrepareTwoRawArrayIter(
                    ndim, shape,
                    dst_data, dst_strides,
                    maskna_data, maskna_strides,
                    &ndim, shape_it,
                    &dst_data, dst_strides_it,
                    &maskna_data, maskna_strides_it) < 0) {
        return -1;
    }

    /* Get the function to do the casting */
    if (PyArray_GetMaskedDTypeTransferFunction(aligned,
                        0, dst_strides_it[0], maskna_strides_it[0],
                        src_dtype, dst_dtype, maskna_dtype,
                        0,
                        &stransfer, &transferdata,
                        &needs_api) != NPY_SUCCEED) {
        return -1;
    }

    if (!needs_api) {
        NPY_BEGIN_THREADS;
    }

    NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {
        /* Transfer the data based on the NA mask */
        stransfer(dst_data, dst_strides_it[0], src_data, 0,
                    (npy_mask *)maskna_data, maskna_strides_it[0],
                    shape_it[0], src_itemsize, transferdata);
    } NPY_RAW_ITER_TWO_NEXT(idim, ndim, coord, shape_it,
                            dst_data, dst_strides_it,
                            maskna_data, maskna_strides_it);

    if (!needs_api) {
        NPY_END_THREADS;
    }

    NPY_AUXDATA_FREE(transferdata);

    return (needs_api && PyErr_Occurred()) ? -1 : 0;
}

/*
 * Assigns the scalar value to every element of the destination raw array
 * where the 'wheremask' is True, except for those which are masked as NA
 * according to 'maskna'.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
raw_array_wheremasked_assign_scalar_preservena(int ndim, npy_intp *shape,
        PyArray_Descr *dst_dtype, char *dst_data, npy_intp *dst_strides,
        PyArray_Descr *src_dtype, char *src_data,
        PyArray_Descr *maskna_dtype, char *maskna_data,
        npy_intp *maskna_strides,
        PyArray_Descr *wheremask_dtype, char *wheremask_data,
        npy_intp *wheremask_strides)
{
    int idim;
    npy_intp shape_it[NPY_MAXDIMS], dst_strides_it[NPY_MAXDIMS];
    npy_intp maskna_strides_it[NPY_MAXDIMS];
    npy_intp wheremask_strides_it[NPY_MAXDIMS];
    npy_intp coord[NPY_MAXDIMS];
    NPY_BEGIN_THREADS_DEF;

    PyArray_MaskedStridedUnaryOp *stransfer = NULL;
    NpyAuxData *transferdata = NULL;
    int aligned, needs_api = 0;
    npy_intp src_itemsize = src_dtype->elsize;

    PyArray_StridedBinaryOp *maskand_stransfer = NULL;
    NpyAuxData *maskand_transferdata = NULL;

    char *maskna_buffer;
    npy_intp maskna_itemsize;

    /* Check alignment */
    aligned = raw_array_is_aligned(ndim, dst_data, dst_strides,
                                    dst_dtype->alignment);
    if (((npy_intp)src_data & (src_dtype->alignment - 1)) != 0) {
        aligned = 0;
    }

    /* Use raw iteration with no heap allocation */
    if (PyArray_PrepareThreeRawArrayIter(
                    ndim, shape,
                    dst_data, dst_strides,
                    maskna_data, maskna_strides,
                    wheremask_data, wheremask_strides,
                    &ndim, shape_it,
                    &dst_data, dst_strides_it,
                    &maskna_data, maskna_strides_it,
                    &wheremask_data, wheremask_strides_it) < 0) {
        return -1;
    }

    /* Allocate a buffer for inverting/anding the mask */
    maskna_itemsize = maskna_dtype->elsize;
    maskna_buffer = PyArray_malloc(NPY_ARRAY_ASSIGN_BUFFERSIZE *
                                    maskna_itemsize);
    if (maskna_buffer == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    /* Get the function to do the casting */
    if (PyArray_GetMaskedDTypeTransferFunction(aligned,
                        0, dst_strides_it[0], maskna_itemsize,
                        src_dtype, dst_dtype, maskna_dtype,
                        0,
                        &stransfer, &transferdata,
                        &needs_api) != NPY_SUCCEED) {
        PyArray_free(maskna_buffer);
        return -1;
    }

    /*
     * Get the function to invert the mask. The output
     * of the binary operation is the dtype 'maskna_dtype'
     */
    if (PyArray_GetMaskAndFunction(
                        maskna_strides_it[0], maskna_dtype, 0,
                        wheremask_strides_it[0], wheremask_dtype, 0,
                        &maskand_stransfer, &maskand_transferdata) < 0) {
        PyArray_free(maskna_buffer);
        NPY_AUXDATA_FREE(transferdata);
        return -1;
    }

    if (!needs_api) {
        NPY_BEGIN_THREADS;
    }

    NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {
        npy_intp buffered_count, count;
        char *dst_d, *maskna_d, *wheremask_d;
        /* Process the innermost dimension a buffer size at a time */
        count = shape_it[0];
        dst_d = dst_data;
        maskna_d = maskna_data;
        wheremask_d = wheremask_data;
        do {
            buffered_count = count < NPY_ARRAY_ASSIGN_BUFFERSIZE
                                        ? count
                                        : NPY_ARRAY_ASSIGN_BUFFERSIZE;

            /* Prepare the mask into the buffer */
            maskand_stransfer(maskna_buffer, maskna_itemsize,
                        maskna_d, maskna_strides_it[0],
                        wheremask_d, wheremask_strides_it[0],
                        buffered_count, maskand_transferdata);

            /* Transfer the data based on the buffered mask */
            stransfer(dst_d, dst_strides_it[0], src_data, 0,
                        (npy_mask *)maskna_buffer, maskna_itemsize,
                        buffered_count, src_itemsize, transferdata);

            dst_d += buffered_count * dst_strides_it[0];
            maskna_d += buffered_count * maskna_strides_it[0];
            wheremask_d += buffered_count * wheremask_strides_it[0];
            count -= buffered_count;
        } while (count > 0);
    } NPY_RAW_ITER_THREE_NEXT(idim, ndim, coord, shape_it,
                            dst_data, dst_strides_it,
                            maskna_data, maskna_strides_it,
                            wheremask_data, wheremask_strides_it);

    if (!needs_api) {
        NPY_END_THREADS;
    }

    PyArray_free(maskna_buffer);
    NPY_AUXDATA_FREE(transferdata);
    NPY_AUXDATA_FREE(maskand_transferdata);

    return (needs_api && PyErr_Occurred()) ? -1 : 0;
}


/* See array_assign.h for documentation */
NPY_NO_EXPORT int
array_assign_scalar(PyArrayObject *dst,
                    PyArray_Descr *src_dtype, char *src_data,
                    PyArrayObject *wheremask,
                    NPY_CASTING casting,
                    npy_bool preservena, npy_bool *preservewhichna)
{
    int allocated_src_data = 0, dst_has_maskna = PyArray_HASMASKNA(dst);
    npy_longlong scalarbuffer[4];

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

    if (preservewhichna != NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                "multi-NA support is not yet implemented");
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

        /*
         * Use a static buffer to store the aligned/cast version,
         * or allocate some memory if more space is needed.
         */
        if (sizeof(scalarbuffer) >= PyArray_DESCR(dst)->elsize) {
            tmp_src_data = (char *)&scalarbuffer[0];
        }
        else {
            tmp_src_data = PyArray_malloc(PyArray_DESCR(dst)->elsize);
            allocated_src_data = 1;
        }
        if (PyArray_CastRawArrays(1, src_data, tmp_src_data, 0, 0,
                            src_dtype, PyArray_DESCR(dst), 0) != NPY_SUCCEED) {
            goto fail;
        }

        /* Replace src_data/src_dtype */
        src_data = tmp_src_data;
        src_dtype = PyArray_DESCR(dst);
    }

    if (wheremask == NULL) {
        /* A straightforward value assignment */
        if (!preservena || !dst_has_maskna) {
            /* If assigning to an array with an NA mask, set to all exposed */
            if (dst_has_maskna) {
                if (PyArray_AssignMaskNA(dst, NULL, 1) < 0) {
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
        /* A value assignment without overwriting NA values */
        else {
            if (raw_array_assign_scalar_preservena(
                    PyArray_NDIM(dst), PyArray_DIMS(dst),
                    PyArray_DESCR(dst), PyArray_DATA(dst), PyArray_STRIDES(dst),
                    src_dtype, src_data,
                    PyArray_MASKNA_DTYPE(dst), PyArray_MASKNA_DATA(dst),
                    PyArray_MASKNA_STRIDES(dst)) < 0) {
                goto fail;
            }
        }
    }
    else {
        npy_intp wheremask_strides[NPY_MAXDIMS];

        if (PyArray_ContainsNA(wheremask)) {
            if (!dst_has_maskna) {
                PyErr_SetString(PyExc_ValueError,
                        "Cannot assign NA value to an array which "
                        "does not support NAs");
                goto fail;
            }
            else {
                /* TODO: add support for this */
                PyErr_SetString(PyExc_ValueError,
                        "A where mask with NA values is not supported "
                        "yet");
                goto fail;
            }
        }

        /* Broadcast the wheremask to 'dst' for raw iteration */
        if (broadcast_strides(PyArray_NDIM(dst), PyArray_DIMS(dst),
                    PyArray_NDIM(wheremask), PyArray_DIMS(wheremask),
                    PyArray_STRIDES(wheremask), "where mask",
                    wheremask_strides) < 0) {
            goto fail;
        }

        /* A straightforward where-masked assignment */
        if (!preservena || !dst_has_maskna) {
            /* If assigning to an array with an NA mask, set to all exposed */
            if (dst_has_maskna) {
                /*
                 * TODO: If the where mask has NA values, this part
                 *       changes too.
                 */
                if (PyArray_AssignMaskNA(dst, wheremask, 1) < 0) {
                    goto fail;
                }
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
        /* A masked value assignment without overwriting NA values */
        else {
            if (raw_array_wheremasked_assign_scalar_preservena(
                    PyArray_NDIM(dst), PyArray_DIMS(dst),
                    PyArray_DESCR(dst), PyArray_DATA(dst), PyArray_STRIDES(dst),
                    src_dtype, src_data,
                    PyArray_MASKNA_DTYPE(dst), PyArray_MASKNA_DATA(dst),
                    PyArray_MASKNA_STRIDES(dst),
                    PyArray_DESCR(wheremask), PyArray_DATA(wheremask),
                    wheremask_strides) < 0) {
                goto fail;
            }
        }
    }

    if (allocated_src_data) {
        PyArray_free(src_data);
    }

    return 0;

fail:
    if (allocated_src_data) {
        PyArray_free(src_data);
    }

    return -1;
}

