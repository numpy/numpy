#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/arrayobject.h"

static PyObject *
array_dlpack(PyArrayObject *self, PyObject *const *args, Py_ssize_t len_args,
             PyObject *kwnames);


static PyObject *
array_dlpack_device(PyArrayObject *self, PyObject *NPY_UNUSED(args));


NPY_NO_EXPORT PyObject *
from_dlpack(PyObject *NPY_UNUSED(self), PyObject *obj);
