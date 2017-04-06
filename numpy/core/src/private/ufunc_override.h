#ifndef __UFUNC_OVERRIDE_H
#define __UFUNC_OVERRIDE_H

#include "npy_config.h"
#include "numpy/ufuncobject.h"

NPY_NO_EXPORT int
PyUFunc_HasOverride(PyObject *args, PyObject *kwds,
                    PyObject **with_override);

NPY_NO_EXPORT int
PyUFunc_CheckOverride(PyUFuncObject *ufunc, char *method,
                      PyObject *args, PyObject *kwds,
                      PyObject **result);
#endif
