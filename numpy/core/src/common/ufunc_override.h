#ifndef NUMPY_CORE_SRC_COMMON_UFUNC_OVERRIDE_H_
#define NUMPY_CORE_SRC_COMMON_UFUNC_OVERRIDE_H_

#include "npy_config.h"

/*
 * Check whether an object has __array_ufunc__ defined on its class and it
 * is not the default, i.e., the object is not an ndarray, and its
 * __array_ufunc__ is not the same as that of ndarray.
 *
 * Returns a new reference, the value of type(obj).__array_ufunc__ if it
 * exists and is different from that of ndarray, and NULL otherwise.
 */
NPY_NO_EXPORT PyObject *
PyUFuncOverride_GetNonDefaultArrayUfunc(PyObject *obj);

/*
 * Check whether an object has __array_ufunc__ defined on its class and it
 * is not the default, i.e., the object is not an ndarray, and its
 * __array_ufunc__ is not the same as that of ndarray.
 *
 * Returns 1 if this is the case, 0 if not.
 */
NPY_NO_EXPORT int
PyUFunc_HasOverride(PyObject *obj);

/*
 * Get possible out argument from kwds, and returns the number of outputs
 * contained within it: if a tuple, the number of elements in it, 1 otherwise.
 * The out argument itself is returned in out_kwd_obj, and the outputs
 * in the out_obj array (as borrowed references).
 *
 * Returns 0 if no outputs found, -1 if kwds is not a dict (with an error set).
 */
NPY_NO_EXPORT int
PyUFuncOverride_GetOutObjects(PyObject *kwds, PyObject **out_kwd_obj, PyObject ***out_objs);

#endif  /* NUMPY_CORE_SRC_COMMON_UFUNC_OVERRIDE_H_ */
