/*
 * This file implements missing value NA mask support for the NumPy array.
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
#include <numpy/arrayobject.h>

#include "npy_config.h"
#include "numpy/npy_3kcompat.h"

#include "shape.h"
#include "lowlevel_strided_loops.h"
#include "array_assign.h"
#include "na_singleton.h"

/*NUMPY_API
 *
 * Returns true if the array has an NA mask. When
 * NA dtypes are implemented, will also return true
 * if the array's dtype has NA support.
 */
NPY_NO_EXPORT npy_bool
PyArray_HasNASupport(PyArrayObject *arr)
{
    return PyArray_HASMASKNA(arr);
}

/*NUMPY_API
 *
 * Returns false if the array has no NA support. Returns
 * true if the array has NA support AND there is an
 * NA anywhere in the array.
 */
NPY_NO_EXPORT npy_bool
PyArray_ContainsNA(PyArrayObject *arr)
{
    /* Need NA support to contain NA */
    if (PyArray_HASMASKNA(arr)) {
        int idim, ndim;
        char *data;
        npy_intp shape[NPY_MAXDIMS], strides[NPY_MAXDIMS];
        npy_intp i, coord[NPY_MAXDIMS];

        if (PyArray_HASFIELDS(arr)) {
            /* TODO: need to add field-NA support */
            return 1;
        }

        /* Use raw iteration with no heap memory allocation */
        if (PyArray_PrepareOneRawArrayIter(
                        PyArray_NDIM(arr), PyArray_DIMS(arr),
                        PyArray_MASKNA_DATA(arr), PyArray_MASKNA_STRIDES(arr),
                        &ndim, shape,
                        &data, strides) < 0) {
            PyErr_Clear();
            return 1;
        }

        /* Do the iteration */
        NPY_RAW_ITER_START(idim, ndim, coord, shape) {
            char *d = data;
            /* Process the innermost dimension */
            for (i = 0; i < shape[0]; ++i, d += strides[0]) {
                if (!NpyMaskValue_IsExposed((npy_mask)(*d))) {
                    return 1;
                }
            }
        } NPY_RAW_ITER_ONE_NEXT(idim, ndim, coord, shape, data, strides);
    }

    return 0;
}

/*
 * Fills a raw array whose dtype has size one with the specified byte
 *
 * Returns 0 on success, -1 on failure.
 */
static int
fill_raw_byte_array(int ndim, npy_intp *shape,
                char *data, npy_intp *strides, char fillvalue)
{
    int idim;
    npy_intp shape_it[NPY_MAXDIMS], strides_it[NPY_MAXDIMS];
    npy_intp i, coord[NPY_MAXDIMS];

    /* Use raw iteration with no heap memory allocation */
    if (PyArray_PrepareOneRawArrayIter(
                    ndim, shape,
                    data, strides,
                    &ndim, shape_it,
                    &data, strides_it) < 0) {
        PyErr_Clear();
        return 1;
    }

    /* Special case contiguous inner stride */
    if (strides_it[0] == 1) {
        NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {
            /* Process the innermost dimension */
            memset(data, fillvalue, shape_it[0]);
        } NPY_RAW_ITER_ONE_NEXT(idim, ndim, coord, shape_it, data, strides_it);
    }
    /* General inner stride */
    else {
        NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {
            char *d = data;
            /* Process the innermost dimension */
            for (i = 0; i < shape_it[0]; ++i, d += strides_it[0]) {
                *d = fillvalue;
            }
        } NPY_RAW_ITER_ONE_NEXT(idim, ndim, coord, shape_it, data, strides_it);
    }

    return 0;
}

/*NUMPY_API
 *
 * Assigns the mask value to all the NA mask elements of
 * the array. This routine is intended to be used to mask
 * all the elments of an array, or if you will also be assigning
 * values to everything at the same time, to unmask all the elements.
 *
 * If 'wheremask' isn't NULL, it should be a boolean mask which
 * specifies where to do the assignment.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_AssignMaskNA(PyArrayObject *arr, PyArrayObject *wheremask,
                        npy_mask maskvalue)
{
    PyArray_Descr *maskvalue_dtype;
    int retcode = 0;

    /* Need NA support to fill the NA mask */
    if (!PyArray_HASMASKNA(arr)) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot assign to the NA mask of an "
                "array which has no NA mask");
        return -1;
    }

    /*
     * If the mask given has no payload, assign from boolean type, otherwise
     * assign from the mask type.
     */
    if ((maskvalue & (~0x01)) == 0) {
        maskvalue_dtype = PyArray_DescrFromType(NPY_BOOL);
    }
    else {
        maskvalue_dtype = PyArray_DescrFromType(NPY_MASK);
    }
    if (maskvalue_dtype == NULL) {
        return -1;
    }

    if (wheremask == NULL) {
        retcode = raw_array_assign_scalar(
                        PyArray_NDIM(arr), PyArray_DIMS(arr),
                        PyArray_MASKNA_DTYPE(arr),
                        PyArray_MASKNA_DATA(arr),
                        PyArray_MASKNA_STRIDES(arr),
                        maskvalue_dtype, (char *)&maskvalue);
    }
    else {
        npy_intp wheremask_strides[NPY_MAXDIMS];

        /* Broadcast the wheremask to 'arr' */
        if (broadcast_strides(PyArray_NDIM(arr), PyArray_DIMS(arr),
                    PyArray_NDIM(wheremask), PyArray_DIMS(wheremask),
                    PyArray_STRIDES(wheremask), "where mask",
                    wheremask_strides) < 0) {
            Py_DECREF(maskvalue_dtype);
            return -1;
        }

        retcode = raw_array_wheremasked_assign_scalar(
                        PyArray_NDIM(arr), PyArray_DIMS(arr),
                        PyArray_MASKNA_DTYPE(arr),
                        PyArray_MASKNA_DATA(arr),
                        PyArray_MASKNA_STRIDES(arr),
                        maskvalue_dtype, (char *)&maskvalue,
                        PyArray_DESCR(wheremask), PyArray_DATA(wheremask),
                        wheremask_strides);
    }

    Py_DECREF(maskvalue_dtype);
    return retcode;
}

/*NUMPY_API
 *
 * If the array does not have an NA mask already, allocates one for it.
 *
 * If 'ownmaskna' is True, it also allocates one for it if the array does
 * not already own its own mask, then copies the data from the old mask
 * to the new mask.
 *
 * If 'multina' is True, the mask is allocated with an NPY_MASK dtype
 * instead of NPY_BOOL.
 *
 * If a new mask is allocated, and no mask was there to copy from,
 * the mask is filled with the 'defaultmask' value. Normally you
 * set this to 1, so all the values are exposed.
 *
 * Returns -1 on failure, 0 on success.
 */
NPY_NO_EXPORT int
PyArray_AllocateMaskNA(PyArrayObject *arr,
                npy_bool ownmaskna,
                npy_bool multina,
                npy_mask defaultmask)
{
    PyArrayObject_fieldaccess *fa = (PyArrayObject_fieldaccess *)arr;
    PyArray_Descr *maskna_dtype = NULL;
    char *maskna_data = NULL;
    npy_intp size;

    /* If the array already owns a mask, done */
    if (fa->flags & NPY_ARRAY_OWNMASKNA) {
        return 0;
    }

    /* If ownership wasn't requested, and there's already a mask, done */
    if (!ownmaskna && (fa->flags & NPY_ARRAY_MASKNA)) {
        return 0;
    }

    size = PyArray_SIZE(arr);

    /* Create the mask dtype */
    if (PyArray_HASFIELDS(arr)) {
        PyErr_SetString(PyExc_RuntimeError,
                "NumPy field-NA isn't supported yet");
        return -1;
    }
    else {
        maskna_dtype = PyArray_DescrFromType(multina ? NPY_MASK
                                                         : NPY_BOOL);
        if (maskna_dtype == NULL) {
            return -1;
        }
    }

    /* Allocate the mask memory */
    maskna_data = PyArray_malloc(size * maskna_dtype->elsize);
    if (maskna_data == NULL) {
        Py_DECREF(maskna_dtype);
        PyErr_NoMemory();
        return -1;
    }

    /* Copy the data and fill in the strides */
    if (fa->nd == 1) {
        /* If there already was a mask copy it, otherwise set it to all ones */
        if (fa->flags & NPY_ARRAY_MASKNA) {
            if (fa->maskna_strides[0] == 1) {
                memcpy(maskna_data, fa->maskna_data,
                            size * maskna_dtype->elsize);
            }
            else {
                if (PyArray_CastRawArrays(fa->dimensions[0],
                                (char *)fa->maskna_data, maskna_data,
                                fa->maskna_strides[0], maskna_dtype->elsize,
                                fa->maskna_dtype, maskna_dtype, 0) < 0) {
                    Py_DECREF(maskna_dtype);
                    PyArray_free(maskna_data);
                    return -1;
                }
            }
        }
        else {
            memset(maskna_data, defaultmask, size * maskna_dtype->elsize);
        }

        fa->maskna_strides[0] = maskna_dtype->elsize;
    }
    else if (fa->nd > 1) {
        npy_stride_sort_item strideperm[NPY_MAXDIMS];
        npy_intp stride, maskna_strides[NPY_MAXDIMS], *shape;
        int i;

        shape = fa->dimensions;

        /* This causes the NA mask and data memory orderings to match */
        PyArray_CreateSortedStridePerm(fa->nd, fa->strides, strideperm);
        stride = maskna_dtype->elsize;
        for (i = fa->nd-1; i >= 0; --i) {
            npy_intp i_perm = strideperm[i].perm;
            maskna_strides[i_perm] = stride;
            stride *= shape[i_perm];
        }

        /* If there already was a mask copy it, otherwise set it to all ones */
        if (fa->flags & NPY_ARRAY_MASKNA) {
            if (PyArray_CastRawNDimArrays(fa->nd, fa->dimensions,
                            (char *)fa->maskna_data, maskna_data,
                            fa->maskna_strides, maskna_strides,
                            fa->maskna_dtype, maskna_dtype, 0) < 0) {
                Py_DECREF(maskna_dtype);
                PyArray_free(maskna_data);
                return -1;
            }
        }
        else {
            memset(maskna_data, defaultmask, size * maskna_dtype->elsize);
        }

        memcpy(fa->maskna_strides, maskna_strides, fa->nd * sizeof(npy_intp));
    }
    else {
        /* If there already was a mask copy it, otherwise set it to all ones */
        if (fa->flags & NPY_ARRAY_MASKNA) {
            maskna_data[0] = fa->maskna_data[0];
        }
        else {
            maskna_data[0] = defaultmask;
        }
    }

    /* Set the NA mask data in the array */
    fa->maskna_dtype = maskna_dtype;
    fa->maskna_data = maskna_data;
    fa->flags |= (NPY_ARRAY_MASKNA | NPY_ARRAY_OWNMASKNA);

    return 0;
}

/*NUMPY_API
 *
 * Assigns the given NA value to all the elements in the array. If
 * 'arr' has a mask, masks all the elements of the array.
 *
 * In the future, when 'arr' has an NA dtype, will assign the
 * appropriate NA bitpatterns to the elements.
 *
 * Returns -1 on failure, 0 on success.
 */
NPY_NO_EXPORT int
PyArray_AssignNA(PyArrayObject *arr, NpyNA *na)
{
    NpyNA_fields *fna = (NpyNA_fields *)na;
    char maskvalue;

    if (!PyArray_HASMASKNA(arr)) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot assign an NA to an "
                "array with no NA support");
        return -1;
    }

    /* Turn the payload into a mask value */
    if (fna->payload == NPY_NA_NOPAYLOAD) {
        maskvalue = 0;
    }
    else if (PyArray_MASKNA_DTYPE(arr)->type_num !=
                                        NPY_MASK) {
        /* TODO: also handle struct-NA mask dtypes */
        PyErr_SetString(PyExc_ValueError,
                "Cannot assign an NA with a payload to an "
                "NA-array with a boolean mask, requires a multi-NA mask");
        return -1;
    }
    else {
        maskvalue = (char)NpyMaskValue_Create(0, fna->payload);
    }

    return PyArray_AssignMaskNA(arr, NULL, maskvalue);
}

/*
 * A ufunc-like function, which returns a boolean or an array
 * of booleans indicating which values are NA.
 */
NPY_NO_EXPORT PyObject *
PyArray_IsNA(PyObject *obj)
{
    /* NA objects are NA */
    if (NpyNA_Check(obj)) {
        Py_INCREF(Py_True);
        return Py_True;
    }
    /* Otherwise non-array objects are not NA */
    else if (!PyArray_Check(obj)) {
        Py_INCREF(Py_False);
        return Py_False;
    }
    /* Create a boolean array based on the mask */
    else {
        PyArrayObject *ret;
        PyArray_Descr *dtype;

        if (PyArray_HASFIELDS((PyArrayObject *)obj)) {
            PyErr_SetString(PyExc_RuntimeError,
                    "field-NA is not supported yet");
            return NULL;
        }

        dtype = PyArray_DescrFromType(NPY_BOOL);
        if (dtype == NULL) {
            return NULL;
        }

        if (PyArray_HASMASKNA((PyArrayObject *)obj)) {
            NpyIter *iter;
            PyArrayObject *op[2] = {(PyArrayObject *)obj, NULL};
            npy_uint32 flags, op_flags[2];
            PyArray_Descr *op_dtypes[2] = {NULL, dtype};

            flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_ZEROSIZE_OK;
            /*
             * This USE_MASKNA causes there to be 3 operands, where operand
             * 2 is the mask for operand 0
             */
            op_flags[0] = NPY_ITER_READONLY | NPY_ITER_USE_MASKNA;
            op_flags[1] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;

            iter = NpyIter_MultiNew(2, op, flags, NPY_KEEPORDER, NPY_NO_CASTING,
                                    op_flags, op_dtypes);
            if (iter == NULL) {
                Py_DECREF(dtype);
                return NULL;
            }

            if (NpyIter_GetIterSize(iter) > 0) {
                NpyIter_IterNextFunc *iternext;
                npy_intp innersize, *innerstrides;
                npy_intp innerstridemask, innerstride1;
                char **dataptrs, *dataptrmask, *dataptr1;

                iternext = NpyIter_GetIterNext(iter, NULL);
                if (iternext == NULL) {
                    Py_DECREF(dtype);
                    return NULL;
                }
                innerstrides = NpyIter_GetInnerStrideArray(iter);
                innerstridemask = innerstrides[2];
                innerstride1 = innerstrides[1];
                /* Because buffering is disabled, the innersize is fixed */
                innersize = *NpyIter_GetInnerLoopSizePtr(iter);
                dataptrs = NpyIter_GetDataPtrArray(iter);

                do {
                    npy_intp i;
                    dataptrmask = dataptrs[2];
                    dataptr1 = dataptrs[1];

                    for (i = 0; i < innersize; ++i) {
                        /*
                         * Bit 0 of the mask is 0 -> NA, 1 -> available,
                         * so invert it and clear the rest of the bits.
                         */
                        *dataptr1 = ~(*dataptrmask) & 0x01;
                        dataptrmask += innerstridemask;
                        dataptr1 += innerstride1;
                    }
                } while (iternext(iter));
            }

            ret = NpyIter_GetOperandArray(iter)[1];
            Py_INCREF(ret);
            Py_DECREF(dtype);
            NpyIter_Deallocate(iter);
        }
        /* Create an array of all zeros */
        else {
            npy_intp size;
            ret = (PyArrayObject *)PyArray_NewLikeArray(
                            (PyArrayObject *)obj, NPY_KEEPORDER, dtype, 0);
            if (ret == NULL) {
                return NULL;
            }
            /*
             * Can use memset because the newly allocated array is
             * packed tightly in memory
             */
            size = PyArray_SIZE(ret);
            if (size > 0) {
                memset(PyArray_DATA(ret), 0, dtype->elsize * size);
            }
        }

        return (PyObject *)ret;
    }
}

/*NUMPY_API
 *
 * This function performs a reduction on the masks for an array.
 * The masks are provided in raw form, with their strides conformed
 * for the reduction.
 *
 * This is for use with a reduction where 'skipna=False'.
 *
 * ndim, shape: The geometry of the arrays
 * src_dtype, dst_dtype: The NA mask dtypes.
 * src_data, dst_data: The NA mask data pointers.
 * src_strides, dst_strides: The NA mask strides, matching the geometry.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_ReduceMaskNAArray(int ndim, npy_intp *shape,
            PyArray_Descr *src_dtype, char *src_data, npy_intp *src_strides,
            PyArray_Descr *dst_dtype, char *dst_data, npy_intp *dst_strides)
{
    int idim;
    npy_intp i, coord[NPY_MAXDIMS];
    npy_intp shape_it[NPY_MAXDIMS];
    npy_intp src_strides_it[NPY_MAXDIMS];
    npy_intp dst_strides_it[NPY_MAXDIMS];

    /* Confirm that dst is not larger than src */
    for (idim = 0; idim < ndim; ++idim) {
        if (src_strides[idim] == 0 && dst_strides[idim] != 0) {
            PyErr_SetString(PyExc_RuntimeError,
                    "ReduceMaskArray cannot reduce into a larger array");
            return -1;
        }
    }

    if (src_dtype->type_num != NPY_BOOL || dst_dtype->type_num != NPY_BOOL) {
        PyErr_SetString(PyExc_ValueError,
                "multi-NA and field-NA are not yet supported");
        return -1;
    }

    /* Initialize the destination mask to all ones, exposed data */
    if (fill_raw_byte_array(ndim, shape, dst_data, dst_strides, 1) < 0) {
        return -1;
    }

    /*
     * Sort axes based on 'src', which has more non-zero strides,
     * by making it the first operand here
     */
    if (PyArray_PrepareTwoRawArrayIter(ndim, shape,
                                    src_data, src_strides,
                                    dst_data, dst_strides,
                                    &ndim, shape_it,
                                    &src_data, src_strides_it,
                                    &dst_data, dst_strides_it) < 0) {
        return NPY_FAIL;
    }

    /* Special case a reduction in the inner loop */
    if (dst_strides_it[0] == 0) {
        /* Special case a contiguous reduction in the inner loop */
        if (src_strides_it[0] == 1) {
            NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {
                /* If there's a zero in src, set dst to zero */
                if (memchr(src_data, 0, shape_it[0]) != NULL) {
                    *dst_data = 0;
                }
            } NPY_RAW_ITER_TWO_NEXT(idim, ndim, coord, shape_it,
                                        src_data, src_strides_it,
                                        dst_data, dst_strides_it);
        }
        else {
            NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {
                char *src_d = src_data;
                /* If there's a zero in src, set dst to zero */
                for (i = 0; i < shape_it[0]; ++i) {
                    if (*src_d == 0) {
                        *dst_data = 0;
                        break;
                    }
                    src_d += src_strides_it[0];
                }
            } NPY_RAW_ITER_TWO_NEXT(idim, ndim, coord, shape_it,
                                        src_data, src_strides_it,
                                        dst_data, dst_strides_it);
        }
    }
    else {
        NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {
            char *src_d = src_data, *dst_d = dst_data;
            for (i = 0; i < shape_it[0]; ++i) {
                *dst_d &= *src_d;
                src_d += src_strides_it[0];
                dst_d += dst_strides_it[0];
            }
        } NPY_RAW_ITER_TWO_NEXT(idim, ndim, coord, shape_it,
                                    src_data, src_strides_it,
                                    dst_data, dst_strides_it);
    }

    return 0;
}

static void
_strided_bool_mask_inversion(char *dst, npy_intp dst_stride,
                            char *src, npy_intp src_stride,
                            npy_intp N, npy_intp NPY_UNUSED(src_itemsize),
                            NpyAuxData *NPY_UNUSED(opdata))
{
    while (N > 0) {
        *dst = ((*src) ^ 0x01) & 0x01;
        dst += dst_stride;
        src += src_stride;
        --N;
    }
}

NPY_NO_EXPORT int
PyArray_GetMaskInversionFunction(
        npy_intp mask_stride, PyArray_Descr *mask_dtype,
        PyArray_StridedUnaryOp **out_unop, NpyAuxData **out_opdata)
{
    /* Will use the opdata with the field version */
    if (PyDataType_HASFIELDS(mask_dtype)) {
        PyErr_SetString(PyExc_RuntimeError,
                "field-based masks are not supported yet");
        return -1;
    }

    if (mask_dtype->type_num != NPY_BOOL && mask_dtype->type_num != NPY_MASK) {
        PyErr_SetString(PyExc_RuntimeError,
                "unsupported data type for mask");
        return -1;
    }

    /* TODO: Specialize for contiguous data */

    *out_unop = &_strided_bool_mask_inversion;
    *out_opdata = NULL;
    return 0;
}

static void
_strided_bool_mask_noinv0_noinv1_and(char *dst, npy_intp dst_stride,
                            char *src0, npy_intp src0_stride,
                            char *src1, npy_intp src1_stride,
                            npy_intp N, NpyAuxData *NPY_UNUSED(opdata))
{
    while (N > 0) {
        *dst = (*src0) & (*src1);
        dst += dst_stride;
        src0 += src0_stride;
        src1 += src1_stride;
        --N;
    }
}

static void
_strided_bool_mask_inv0_noinv1_and(char *dst, npy_intp dst_stride,
                            char *src0, npy_intp src0_stride,
                            char *src1, npy_intp src1_stride,
                            npy_intp N, NpyAuxData *NPY_UNUSED(opdata))
{
    while (N > 0) {
        *dst = ((*src0) ^ 0x01) & (*src1);
        dst += dst_stride;
        src0 += src0_stride;
        src1 += src1_stride;
        --N;
    }
}

static void
_strided_bool_mask_noinv0_inv1_and(char *dst, npy_intp dst_stride,
                            char *src0, npy_intp src0_stride,
                            char *src1, npy_intp src1_stride,
                            npy_intp N, NpyAuxData *NPY_UNUSED(opdata))
{
    while (N > 0) {
        *dst = (*src0) & ((*src1) ^ 0x01);
        dst += dst_stride;
        src0 += src0_stride;
        src1 += src1_stride;
        --N;
    }
}

static void
_strided_bool_mask_inv0_inv1_and(char *dst, npy_intp dst_stride,
                            char *src0, npy_intp src0_stride,
                            char *src1, npy_intp src1_stride,
                            npy_intp N, NpyAuxData *NPY_UNUSED(opdata))
{
    while (N > 0) {
        *dst = ((*src0) | (*src1)) ^ 0x01;
        dst += dst_stride;
        src0 += src0_stride;
        src1 += src1_stride;
        --N;
    }
}

/*
 * Gets a function which ANDs together two masks, possibly inverting
 * one or both of the masks as well.
 *
 * The dtype of the output must match 'mask0_dtype'.
 */
NPY_NO_EXPORT int
PyArray_GetMaskAndFunction(
        npy_intp mask0_stride, PyArray_Descr *mask0_dtype, int invert_mask0,
        npy_intp mask1_stride, PyArray_Descr *mask1_dtype, int invert_mask1,
        PyArray_StridedBinaryOp **out_binop, NpyAuxData **out_opdata)
{
    /* Will use the opdata with the field version */
    if (PyDataType_HASFIELDS(mask0_dtype) ||
                        PyDataType_HASFIELDS(mask1_dtype)) {
        PyErr_SetString(PyExc_RuntimeError,
                "field-based masks are not supported yet");
        return -1;
    }

    if (mask0_dtype->type_num == NPY_MASK ||
                            mask1_dtype->type_num == NPY_MASK) {
        PyErr_SetString(PyExc_RuntimeError,
                "multi-NA masks are not supported yet");
        return -1;
    }

    if (mask0_dtype->type_num != NPY_BOOL ||
                            mask1_dtype->type_num != NPY_BOOL) {
        PyErr_SetString(PyExc_RuntimeError,
                "unsupported data type for mask");
        return -1;
    }

    /* TODO: Specialize for contiguous data */

    if (invert_mask0) {
        if (invert_mask1) {
            *out_binop = &_strided_bool_mask_inv0_inv1_and;
        }
        else {
            *out_binop = &_strided_bool_mask_inv0_noinv1_and;
        }
    }
    else {
        if (invert_mask1) {
            *out_binop = &_strided_bool_mask_noinv0_inv1_and;
        }
        else {
            *out_binop = &_strided_bool_mask_noinv0_noinv1_and;
        }
    }
    *out_opdata = NULL;
    return 0;
}
