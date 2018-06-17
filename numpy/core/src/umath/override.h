#ifndef _NPY_UMATH_OVERRIDE_H
#define _NPY_UMATH_OVERRIDE_H

#include "npy_config.h"
#include "numpy/ufuncobject.h"
#include "ufunc_object.h"

NPY_NO_EXPORT int
PyUFunc_CheckOverride(PyUFuncObject *ufunc, PyObject *method,
                      PyObject *args, PyObject *kwds,
                      PyObject **result);
#endif
