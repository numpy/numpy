#ifndef NUMPY_CORE_INCLUDE_NUMPY_MULTIARRAY_UNIQUE_H_
#define NUMPY_CORE_INCLUDE_NUMPY_MULTIARRAY_UNIQUE_H_

#include <Python.h>

#ifdef __cplusplus
extern "C" {
#endif

PyObject* PyArray_Unique(PyObject *NPY_UNUSED(dummy), PyObject *args);

#ifdef __cplusplus
}
#endif

#endif  // NUMPY_CORE_INCLUDE_NUMPY_MULTIARRAY_UNIQUE_H_
