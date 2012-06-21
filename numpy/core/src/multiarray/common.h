#ifndef _NPY_PRIVATE_COMMON_H_
#define _NPY_PRIVATE_COMMON_H_

#define error_converting(x)  (((x) == -1) && PyErr_Occurred())

/*
 * Recursively examines the object to determine an appropriate dtype
 * to use for converting to an ndarray.
 *
 * 'obj' is the object to be converted to an ndarray.
 *
 * 'maxdims' is the maximum recursion depth.
 *
 * 'out_dtype' should be either NULL or a minimal starting dtype when
 * the function is called. It is updated with the results of type
 * promotion. This dtype does not get updated when processing NA objects.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_DTypeFromObject(PyObject *obj, int maxdims,
                        PyArray_Descr **out_dtype);

NPY_NO_EXPORT int
PyArray_DTypeFromObjectHelper(PyObject *obj, int maxdims,
                              PyArray_Descr **out_dtype, int string_status);

/*
 * Returns NULL without setting an exception if no scalar is matched, a
 * new dtype reference otherwise.
 */
NPY_NO_EXPORT PyArray_Descr *
_array_find_python_scalar_type(PyObject *op);

NPY_NO_EXPORT PyArray_Descr *
_array_typedescr_fromstr(char *str);

/* 
 * Returns -1 and sets an exception if *index is an invalid index for
 * an array of size max_item, otherwise adjusts it in place to be
 * 0 <= *index < max_item, and returns 0.
 * 'axis' should be the array axis that is being indexed over, if known. If
 * unknown, use -1.
 */
NPY_NO_EXPORT int
check_and_adjust_index(npy_intp *index, npy_intp max_item, int axis);

NPY_NO_EXPORT char *
index2ptr(PyArrayObject *mp, npy_intp i);

NPY_NO_EXPORT int
_zerofill(PyArrayObject *ret);

NPY_NO_EXPORT int
_IsAligned(PyArrayObject *ap);

NPY_NO_EXPORT npy_bool
_IsWriteable(PyArrayObject *ap);

#include "ucsnarrow.h"

#endif
