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

/* This is exactly as mandated by dlpack */
static void array_dlpack_capsule_deleter(PyObject *self)
{
    if (PyCapsule_IsValid(self, NPY_DLPACK_USED_CAPSULE_NAME)) {
        return;
    }
    DLManagedTensor *managed = 
        (DLManagedTensor *)PyCapsule_GetPointer(self, NPY_DLPACK_CAPSULE_NAME);
    if (managed == NULL) {
        return;
    }
    managed->deleter(managed);
}

/* used internally */
static void array_dlpack_internal_capsule_deleter(PyObject *self)
{
    DLManagedTensor *managed = 
        (DLManagedTensor *)PyCapsule_GetPointer(self, NPY_DLPACK_INTERNAL_CAPSULE_NAME);
    if (managed == NULL) {
        return;
    }
    managed->deleter(managed);
}

#endif