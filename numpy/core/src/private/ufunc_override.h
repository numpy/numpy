#ifndef __UFUNC_OVERRIDE_H
#define __UFUNC_OVERRIDE_H

#include "npy_config.h"

/*
 * Check whether a set of input and output args have a non-default
 *  `__array_ufunc__` method. Returns the number of overrides, setting
 * corresponding objects in PyObject array with_override (if not NULL).
 * returns -1 on failure.
 */
NPY_NO_EXPORT int
PyUFunc_WithOverride(PyObject *args, PyObject *kwds,
                     PyObject **with_override, PyObject **methods);
#endif
