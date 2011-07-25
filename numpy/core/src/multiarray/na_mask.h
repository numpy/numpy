#ifndef _NPY_PRIVATE__NA_MASK_H_
#define _NPY_PRIVATE__NA_MASK_H_

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

#endif
