#ifndef NUMPY_CORE_SRC_MULTIARRAY_TYPEINFO_H_
#define NUMPY_CORE_SRC_MULTIARRAY_TYPEINFO_H_

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "npy_config.h"

NPY_VISIBILITY_HIDDEN int
typeinfo_init_structsequences(PyObject *multiarray_dict);

NPY_VISIBILITY_HIDDEN PyObject *
PyArray_typeinfo(
    char typechar, int typenum, int nbits, int align,
    PyTypeObject *type_obj);

NPY_VISIBILITY_HIDDEN PyObject *
PyArray_typeinforanged(
    char typechar, int typenum, int nbits, int align,
    PyObject *max, PyObject *min, PyTypeObject *type_obj);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_TYPEINFO_H_ */
