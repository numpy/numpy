#include "Python.h"
#include "dlpack/dlpack.h"

#ifndef NPY_DLPACK_H
#define NPY_DLPACK_H

// Part of the Array API specification.
#define NPY_DLPACK_CAPSULE_NAME "dltensor"
#define NPY_DLPACK_USED_CAPSULE_NAME "used_dltensor"

// Used internally by NumPy to store a base object
// as it has to release a reference to the original
// capsule.
#define NPY_DLPACK_INTERNAL_CAPSULE_NAME "numpy_dltensor"

static void array_dlpack_capsule_deleter(PyObject *self)
{
    if (!PyCapsule_IsValid(self, NPY_DLPACK_CAPSULE_NAME) &&
            !PyCapsule_IsValid(self, NPY_DLPACK_INTERNAL_CAPSULE_NAME)) {
        if (!PyCapsule_IsValid(self, NPY_DLPACK_USED_CAPSULE_NAME)) {
            PyErr_SetString(PyExc_RuntimeError, "Invalid capsule name.");
        }
        return;
    }
    DLManagedTensor *managed = 
        (DLManagedTensor *)PyCapsule_GetPointer(self, PyCapsule_GetName(self));
    managed->deleter(managed);
}

#endif