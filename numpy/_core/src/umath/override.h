#ifndef _NPY_UMATH_OVERRIDE_H
#define _NPY_UMATH_OVERRIDE_H

#include "npy_config.h"
#include "numpy/ufuncobject.h"

NPY_NO_EXPORT int
PyUFunc_CheckOverride(PyUFuncObject *ufunc, char *method,
        PyObject *const *in_args, int nin,
        PyObject *const *out_args, int nout,
        PyObject *out_args_tuple,
        PyObject *wheremask_obj,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames,
        PyObject **result);


#endif
