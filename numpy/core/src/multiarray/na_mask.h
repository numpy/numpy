#ifndef _NPY_PRIVATE__NA_MASK_H_
#define _NPY_PRIVATE__NA_MASK_H_

#include "lowlevel_strided_loops.h"

/*
 * Assigns the given NA value to all the elements in the array.
 *
 * Returns -1 on failure, 0 on success.
 */
NPY_NO_EXPORT int
PyArray_AssignNA(PyArrayObject *arr, NpyNA *na);

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
PyArray_GetMaskInversionFunction(npy_intp mask_stride,
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

#endif
