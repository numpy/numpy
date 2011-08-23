#ifndef _NPY_PRIVATE__NA_MASK_H_
#define _NPY_PRIVATE__NA_MASK_H_

#include "lowlevel_strided_loops.h"

/*
 * A ufunc-like function, which returns a boolean or an array
 * of booleans indicating which values are NA.
 */
NPY_NO_EXPORT PyObject *
PyArray_IsNA(PyObject *obj);

/*
 * Gets a strided unary operation which inverts mask values.
 */
NPY_NO_EXPORT int
PyArray_GetMaskInversionFunction(npy_intp dst_mask_stride,
                            npy_intp src_mask_stride,
                            PyArray_Descr *mask_dtype,
                            PyArray_StridedUnaryOp **out_unop,
                            NpyAuxData **out_opdata);

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
        PyArray_StridedBinaryOp **out_binop, NpyAuxData **out_opdata);

/*
 * This function performs a reduction on the masks for an array.
 *
 * This is for use with a reduction where 'skipna=False'.
 *
 * operand: The operand for which the reduction is being done. This array
 *          must have an NA mask.
 * result: The result array, which should have the same 'ndim' as
 *         'operand' but with dimensions of size one for every reduction
 *         axis. This array must have an NA mask.
 * wheremask: NOT SUPPORTED YET, but is for a boolean mask which can
 *            broadcast to 'result', indicating where to do the reduction.
 *            Should pass in NULL.
 * skipwhichna: NOT SUPPORTED YET, but for future expansion to multi-NA,
 *              where reductions can be done on NAs with a subset of
 *              the possible payloads. Should pass in NULL.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_ReduceMaskNAArray(PyArrayObject *operand, PyArrayObject *result,
                            PyArrayObject *wheremask, npy_bool *skipwhichna);

#endif
