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
 * Assigns the mask value to all the NA mask elements of
 * the array.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_AssignMaskNA(PyArrayObject *arr, npy_mask maskvalue);


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
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata);

#endif
