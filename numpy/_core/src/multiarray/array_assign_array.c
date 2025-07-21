/*
 * This file implements assignment from an ndarray to another ndarray.
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/ndarraytypes.h"
#include "numpy/npy_math.h"

#include "npy_config.h"


#include "convert_datatype.h"
#include "methods.h"
#include "shape.h"
#include "lowlevel_strided_loops.h"

#include "array_assign.h"
#include "dtype_transfer.h"

#include "umathmodule.h"

/*
 * Check that array data is both uint-aligned and true-aligned for all array
 * elements, as required by the copy/casting code in lowlevel_strided_loops.c
 */
NPY_NO_EXPORT int
copycast_isaligned(int ndim, npy_intp const *shape,
        PyArray_Descr *dtype, char *data, npy_intp const *strides)
{
    int aligned;
    int big_aln, small_aln;

    int uint_aln = npy_uint_alignment(dtype->elsize);
    int true_aln = dtype->alignment;

    /* uint alignment can be 0, meaning not uint alignable */
    if (uint_aln == 0) {
        return 0;
    }

    /*
     * As an optimization, it is unnecessary to check the alignment to the
     * smaller of (uint_aln, true_aln) if the data is aligned to the bigger of
     * the two and the big is a multiple of the small aln. We check the bigger
     * one first and only check the smaller if necessary.
     */
    if (true_aln >= uint_aln) {
        big_aln = true_aln;
        small_aln = uint_aln;
    }
    else {
        big_aln = uint_aln;
        small_aln = true_aln;
    }

    aligned = raw_array_is_aligned(ndim, shape, data, strides, big_aln);
    if (aligned && big_aln % small_aln != 0) {
        aligned = raw_array_is_aligned(ndim, shape, data, strides, small_aln);
    }
    return aligned;
}

/*
 * Assigns the array from 'src' to 'dst'. The strides must already have
 * been broadcast.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
raw_array_assign_array(int ndim, npy_intp const *shape,
        PyArray_Descr *dst_dtype, char *dst_data, npy_intp const *dst_strides,
        PyArray_Descr *src_dtype, char *src_data, npy_intp const *src_strides,
        int flags)
{
    int idim;
    npy_intp shape_it[NPY_MAXDIMS];
    npy_intp dst_strides_it[NPY_MAXDIMS];
    npy_intp src_strides_it[NPY_MAXDIMS];
    npy_intp coord[NPY_MAXDIMS];

    int aligned = flags & 0x01;
    int same_value_cast = (flags & 0x02) == 0x02;

    NPY_BEGIN_THREADS_DEF;

    /* Use raw iteration with no heap allocation */
    if (PyArray_PrepareTwoRawArrayIter(
                    ndim, shape,
                    dst_data, dst_strides,
                    src_data, src_strides,
                    &ndim, shape_it,
                    &dst_data, dst_strides_it,
                    &src_data, src_strides_it) < 0) {
        return -1;
    }

    /*
     * Overlap check for the 1D case. Higher dimensional arrays and
     * opposite strides cause a temporary copy before getting here.
     */
    if (ndim == 1 && src_data < dst_data &&
                src_data + shape_it[0] * src_strides_it[0] > dst_data) {
        src_data += (shape_it[0] - 1) * src_strides_it[0];
        dst_data += (shape_it[0] - 1) * dst_strides_it[0];
        src_strides_it[0] = -src_strides_it[0];
        dst_strides_it[0] = -dst_strides_it[0];
    }

    /* Get the function to do the casting */
    NPY_cast_info cast_info;
    NPY_ARRAYMETHOD_FLAGS method_flags;
    if (PyArray_GetDTypeTransferFunction(aligned,
                        src_strides_it[0], dst_strides_it[0],
                        src_dtype, dst_dtype,
                        0,
                        &cast_info, &method_flags) != NPY_SUCCEED) {
        return -1;
    }

    if (!(method_flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        npy_clear_floatstatus_barrier((char*)&src_data);
    }

    if (same_value_cast) {
 #if NPY_FEATURE_VERSION > NPY_2_3_API_VERSION
        cast_info.context.flags |= NPY_SAME_VALUE_CASTING;
 #else
        PyErr_SetString(PyExc_NotImplementedError, 
            "raw_array_assign_array with 'same_value' casting not implemented yet");
 #endif
    }

    /* Ensure number of elements exceeds threshold for threading */
    if (!(method_flags & NPY_METH_REQUIRES_PYAPI)) {
        npy_intp nitems = 1, i;
        for (i = 0; i < ndim; i++) {
            nitems *= shape_it[i];
        }
        NPY_BEGIN_THREADS_THRESHOLDED(nitems);
    }

    npy_intp strides[2] = {src_strides_it[0], dst_strides_it[0]};

    int result = 0;
    NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {
        /* Process the innermost dimension */
        char *args[2] = {src_data, dst_data};
        result = cast_info.func(&cast_info.context,
                                args, &shape_it[0], strides,
                                cast_info.auxdata);
        if (result < 0) {
            goto fail;
        }
    } NPY_RAW_ITER_TWO_NEXT(idim, ndim, coord, shape_it,
                            dst_data, dst_strides_it,
                            src_data, src_strides_it);

    NPY_END_THREADS;
    NPY_cast_info_xfree(&cast_info);

    if (!(method_flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        int fpes = npy_get_floatstatus_barrier((char*)&src_data);
        if (fpes && PyUFunc_GiveFloatingpointErrors("cast", fpes) < 0) {
            return -1;
        }
    }

    return 0;
fail:
    NPY_END_THREADS;
    NPY_cast_info_xfree(&cast_info);
    HandleArrayMethodError(result, "astype", method_flags); 
    return -1;
}

/*
 * Assigns the array from 'src' to 'dst, wherever the 'wheremask'
 * value is True. The strides must already have been broadcast.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
raw_array_wheremasked_assign_array(int ndim, npy_intp const *shape,
        PyArray_Descr *dst_dtype, char *dst_data, npy_intp const *dst_strides,
        PyArray_Descr *src_dtype, char *src_data, npy_intp const *src_strides,
        PyArray_Descr *wheremask_dtype, char *wheremask_data,
        npy_intp const *wheremask_strides, int flags)
{
    int idim;
    npy_intp shape_it[NPY_MAXDIMS];
    npy_intp dst_strides_it[NPY_MAXDIMS];
    npy_intp src_strides_it[NPY_MAXDIMS];
    npy_intp wheremask_strides_it[NPY_MAXDIMS];
    npy_intp coord[NPY_MAXDIMS];

    int aligned = flags & 0x01;
    int same_value_cast = (flags & 0x02) == 0x02;

    NPY_BEGIN_THREADS_DEF;

    /* Use raw iteration with no heap allocation */
    if (PyArray_PrepareThreeRawArrayIter(
                    ndim, shape,
                    dst_data, dst_strides,
                    src_data, src_strides,
                    wheremask_data, wheremask_strides,
                    &ndim, shape_it,
                    &dst_data, dst_strides_it,
                    &src_data, src_strides_it,
                    &wheremask_data, wheremask_strides_it) < 0) {
        return -1;
    }

    /*
     * Overlap check for the 1D case. Higher dimensional arrays cause
     * a temporary copy before getting here.
     */
    if (ndim == 1 && src_data < dst_data &&
                src_data + shape_it[0] * src_strides_it[0] > dst_data) {
        src_data += (shape_it[0] - 1) * src_strides_it[0];
        dst_data += (shape_it[0] - 1) * dst_strides_it[0];
        wheremask_data += (shape_it[0] - 1) * wheremask_strides_it[0];
        src_strides_it[0] = -src_strides_it[0];
        dst_strides_it[0] = -dst_strides_it[0];
        wheremask_strides_it[0] = -wheremask_strides_it[0];
    }

    /* Get the function to do the casting */
    NPY_cast_info cast_info;
    NPY_ARRAYMETHOD_FLAGS method_flags;
    if (PyArray_GetMaskedDTypeTransferFunction(aligned,
                        src_strides_it[0],
                        dst_strides_it[0],
                        wheremask_strides_it[0],
                        src_dtype, dst_dtype, wheremask_dtype,
                        0,
                        &cast_info, &method_flags) != NPY_SUCCEED) {
        return -1;
    }
    if (same_value_cast) {
        /* cast_info.context.flags |= NPY_SAME_VALUE_CASTING; */
        PyErr_SetString(PyExc_NotImplementedError,
            "raw_array_wheremasked_assign_array with 'same_value' casting not implemented yet");
        return -1;
    }

    if (!(method_flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        npy_clear_floatstatus_barrier(src_data);
    }
    if (!(method_flags & NPY_METH_REQUIRES_PYAPI)) {
        npy_intp nitems = 1, i;
        for (i = 0; i < ndim; i++) {
            nitems *= shape_it[i];
        }
        NPY_BEGIN_THREADS_THRESHOLDED(nitems);
    }

    npy_intp strides[2] = {src_strides_it[0], dst_strides_it[0]};

    int result = 0;
    NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {
        PyArray_MaskedStridedUnaryOp *stransfer;
        stransfer = (PyArray_MaskedStridedUnaryOp *)cast_info.func;

        /* Process the innermost dimension */
        char *args[2] = {src_data, dst_data};
        result = stransfer(&cast_info.context,
                            args, &shape_it[0], strides,
                            (npy_bool *)wheremask_data, wheremask_strides_it[0],
                            cast_info.auxdata);
        if (result < 0) {
            goto fail;
        }
    } NPY_RAW_ITER_THREE_NEXT(idim, ndim, coord, shape_it,
                            dst_data, dst_strides_it,
                            src_data, src_strides_it,
                            wheremask_data, wheremask_strides_it);

    NPY_END_THREADS;
    NPY_cast_info_xfree(&cast_info);

    if (!(method_flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        int fpes = npy_get_floatstatus_barrier(src_data);
        if (fpes && PyUFunc_GiveFloatingpointErrors("cast", fpes) < 0) {
            return -1;
        }
    }
    return 0;
fail:
    NPY_END_THREADS;
    NPY_cast_info_xfree(&cast_info);
    HandleArrayMethodError(result, "astype", method_flags); 
    return -1;
}

/*
 * An array assignment function for copying arrays, broadcasting 'src' into
 * 'dst'. This function makes a temporary copy of 'src' if 'src' and
 * 'dst' overlap, to be able to handle views of the same data with
 * different strides.
 *
 * dst: The destination array.
 * src: The source array.
 * wheremask: If non-NULL, a boolean mask specifying where to copy.
 * casting: An exception is raised if the copy violates this
 *          casting rule.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_AssignArray(PyArrayObject *dst, PyArrayObject *src,
                    PyArrayObject *wheremask,
                    NPY_CASTING casting)
{
    int copied_src = 0;
    npy_intp src_strides[NPY_MAXDIMS];

    /* Use array_assign_scalar if 'src' NDIM is 0 */
    if (PyArray_NDIM(src) == 0) {
        return PyArray_AssignRawScalar(
                            dst, PyArray_DESCR(src), PyArray_DATA(src),
                            wheremask, casting);
    }

    /*
     * Performance fix for expressions like "a[1000:6000] += x".  In this
     * case, first an in-place add is done, followed by an assignment,
     * equivalently expressed like this:
     *
     *   tmp = a[1000:6000]   # Calls array_subscript in mapping.c
     *   np.add(tmp, x, tmp)
     *   a[1000:6000] = tmp   # Calls array_assign_subscript in mapping.c
     *
     * In the assignment the underlying data type, shape, strides, and
     * data pointers are identical, but src != dst because they are separately
     * generated slices.  By detecting this and skipping the redundant
     * copy of values to themselves, we potentially give a big speed boost.
     *
     * Note that we don't call EquivTypes, because usually the exact same
     * dtype object will appear, and we don't want to slow things down
     * with a complicated comparison.  The comparisons are ordered to
     * try and reject this with as little work as possible.
     */
    if (PyArray_DATA(src) == PyArray_DATA(dst) &&
                        PyArray_DESCR(src) == PyArray_DESCR(dst) &&
                        PyArray_NDIM(src) == PyArray_NDIM(dst) &&
                        PyArray_CompareLists(PyArray_DIMS(src),
                                             PyArray_DIMS(dst),
                                             PyArray_NDIM(src)) &&
                        PyArray_CompareLists(PyArray_STRIDES(src),
                                             PyArray_STRIDES(dst),
                                             PyArray_NDIM(src))) {
        /*printf("Redundant copy operation detected\n");*/
        return 0;
    }

    if (PyArray_FailUnlessWriteable(dst, "assignment destination") < 0) {
        goto fail;
    }

    /* Check the casting rule */
    if (!PyArray_CanCastTypeTo(PyArray_DESCR(src),
                                PyArray_DESCR(dst), casting)) {
        npy_set_invalid_cast_error(
                PyArray_DESCR(src), PyArray_DESCR(dst), casting, NPY_FALSE);
        goto fail;
    }

    /*
     * When ndim is 1 and the strides point in the same direction,
     * the lower-level inner loop handles copying
     * of overlapping data. For bigger ndim and opposite-strided 1D
     * data, we make a temporary copy of 'src' if 'src' and 'dst' overlap.'
     */
    if (((PyArray_NDIM(dst) == 1 && PyArray_NDIM(src) >= 1 &&
                    PyArray_STRIDES(dst)[0] *
                            PyArray_STRIDES(src)[PyArray_NDIM(src) - 1] < 0) ||
                    PyArray_NDIM(dst) > 1 || PyArray_HASFIELDS(dst)) &&
                    arrays_overlap(src, dst)) {
        PyArrayObject *tmp;

        /*
         * Allocate a temporary copy array.
         */
        tmp = (PyArrayObject *)PyArray_NewLikeArray(dst,
                                        NPY_KEEPORDER, NULL, 0);
        if (tmp == NULL) {
            goto fail;
        }

        if (PyArray_AssignArray(tmp, src, NULL, NPY_UNSAFE_CASTING) < 0) {
            Py_DECREF(tmp);
            goto fail;
        }

        src = tmp;
        copied_src = 1;
    }

    /* Broadcast 'src' to 'dst' for raw iteration */
    if (PyArray_NDIM(src) > PyArray_NDIM(dst)) {
        int ndim_tmp = PyArray_NDIM(src);
        npy_intp *src_shape_tmp = PyArray_DIMS(src);
        npy_intp *src_strides_tmp = PyArray_STRIDES(src);
        /*
         * As a special case for backwards compatibility, strip
         * away unit dimensions from the left of 'src'
         */
        while (ndim_tmp > PyArray_NDIM(dst) && src_shape_tmp[0] == 1) {
            --ndim_tmp;
            ++src_shape_tmp;
            ++src_strides_tmp;
        }

        if (broadcast_strides(PyArray_NDIM(dst), PyArray_DIMS(dst),
                    ndim_tmp, src_shape_tmp,
                    src_strides_tmp, "input array",
                    src_strides) < 0) {
            goto fail;
        }
    }
    else {
        if (broadcast_strides(PyArray_NDIM(dst), PyArray_DIMS(dst),
                    PyArray_NDIM(src), PyArray_DIMS(src),
                    PyArray_STRIDES(src), "input array",
                    src_strides) < 0) {
            goto fail;
        }
    }

    /* optimization: scalar boolean mask */
    if (wheremask != NULL &&
            PyArray_NDIM(wheremask) == 0 &&
            PyArray_DESCR(wheremask)->type_num == NPY_BOOL) {
        npy_bool value = *(npy_bool *)PyArray_DATA(wheremask);
        if (value) {
            /* where=True is the same as no where at all */
            wheremask = NULL;
        }
        else {
            /* where=False copies nothing */
            return 0;
        }
    }

    int aligned =
        copycast_isaligned(PyArray_NDIM(dst), PyArray_DIMS(dst), PyArray_DESCR(dst), PyArray_DATA(dst), PyArray_STRIDES(dst)) &&
        copycast_isaligned(PyArray_NDIM(dst), PyArray_DIMS(dst), PyArray_DESCR(src), PyArray_DATA(src), src_strides);
    int flags = ((NPY_SAME_VALUE_CASTING == casting) << 1) | aligned;

    if (wheremask == NULL) {
        /* A straightforward value assignment */
        /* Do the assignment with raw array iteration */
        if (raw_array_assign_array(PyArray_NDIM(dst), PyArray_DIMS(dst),
                PyArray_DESCR(dst), PyArray_DATA(dst), PyArray_STRIDES(dst),
                PyArray_DESCR(src), PyArray_DATA(src), src_strides, flags) < 0){
            goto fail;
        }
    }
    else {
        npy_intp wheremask_strides[NPY_MAXDIMS];

        /* Broadcast the wheremask to 'dst' for raw iteration */
        if (broadcast_strides(PyArray_NDIM(dst), PyArray_DIMS(dst),
                    PyArray_NDIM(wheremask), PyArray_DIMS(wheremask),
                    PyArray_STRIDES(wheremask), "where mask",
                    wheremask_strides) < 0) {
            goto fail;
        }

        /* A straightforward where-masked assignment */
         /* Do the masked assignment with raw array iteration */
        if (raw_array_wheremasked_assign_array(
                PyArray_NDIM(dst), PyArray_DIMS(dst),
                PyArray_DESCR(dst), PyArray_DATA(dst), PyArray_STRIDES(dst),
                PyArray_DESCR(src), PyArray_DATA(src), src_strides,
                PyArray_DESCR(wheremask), PyArray_DATA(wheremask),
                        wheremask_strides, flags) < 0) {
            goto fail;
        }
    }

    if (copied_src) {
        Py_DECREF(src);
    }
    return 0;

fail:
    if (copied_src) {
        Py_DECREF(src);
    }
    return -1;
}
