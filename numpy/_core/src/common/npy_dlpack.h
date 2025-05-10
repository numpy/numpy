#include "Python.h"
#include "dlpack/dlpack.h"

#ifndef NPY_DLPACK_H
#define NPY_DLPACK_H

// Part of the Array API specification.
#define NPY_DLPACK_CAPSULE_NAME "dltensor"
#define NPY_DLPACK_VERSIONED_CAPSULE_NAME "dltensor_versioned"
#define NPY_DLPACK_USED_CAPSULE_NAME "used_dltensor"
#define NPY_DLPACK_VERSIONED_USED_CAPSULE_NAME "used_dltensor_versioned"

// Used internally by NumPy to store a base object
// as it has to release a reference to the original
// capsule.
#define NPY_DLPACK_INTERNAL_CAPSULE_NAME "numpy_dltensor"
#define NPY_DLPACK_VERSIONED_INTERNAL_CAPSULE_NAME "numpy_dltensor_versioned"

NPY_NO_EXPORT PyObject *
array_dlpack(PyArrayObject *self, PyObject *const *args, Py_ssize_t len_args,
             PyObject *kwnames);


NPY_NO_EXPORT PyObject *
array_dlpack_device(PyArrayObject *self, PyObject *NPY_UNUSED(args));


NPY_NO_EXPORT PyObject *
from_dlpack(PyObject *NPY_UNUSED(self),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames);

#endif
