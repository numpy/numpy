#ifndef NUMPY_CORE_SRC_MULTIARRAY_MODULETYPES_H_
#define NUMPY_CORE_SRC_MULTIARRAY_MODULETYPES_H_

#define PY_SSIZE_T_CLEAN
#include "Python.h"

NPY_NO_EXPORT typedef struct {
    // module state, to hold heap types
    PyTypeObject *arrayflags_type;
} multiarray_umath_state;

NPY_NO_EXPORT multiarray_umath_state *
multiarray_umath_get_state(PyObject *module);

NPY_NO_EXPORT PyType_Spec arrayflags_type_spec;

#endif