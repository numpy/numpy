#ifndef _NPY_UMATH_OVERRIDE_H
#define _NPY_UMATH_OVERRIDE_H

#include "numpy/ufuncobject.h"

#include "npy_config.h"

NPY_NO_EXPORT int
PyUFunc_CheckOverride(PyUFuncObject *ufunc, char *method, PyObject *in_args,
                      PyObject *out_args, PyObject *const *args,
                      Py_ssize_t len_args, PyObject *kwnames,
                      PyObject **result);

#endif
