#ifndef __UFUNC_OVERRIDE_H
#define __UFUNC_OVERRIDE_H

#include "npy_config.h"

/*
 * Check whether an object has __array_ufunc__ defined on its class and it
 * is not the default, i.e., the object is not an ndarray, and its
 * __array_ufunc__ is not the same as that of ndarray.
 *
 * Returns a new reference, the value of type(obj).__array_ufunc__
 *
 * If the __array_ufunc__ matches that of ndarray, or does not exist, return
 * NULL.
 *
 * Note that since this module is used with both multiarray and umath, we do
 * not have access to PyArray_Type and therewith neither to PyArray_CheckExact
 * nor to the default __array_ufunc__ method, so instead we import locally.
 * TODO: Can this really not be done more smartly?
 */
NPY_NO_EXPORT PyObject *
get_non_default_array_ufunc(PyObject *obj);

#endif
