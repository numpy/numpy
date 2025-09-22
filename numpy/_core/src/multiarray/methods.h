#ifndef NUMPY_CORE_SRC_MULTIARRAY_METHODS_H_
#define NUMPY_CORE_SRC_MULTIARRAY_METHODS_H_

#include "npy_static_data.h"
#include "npy_import.h"

extern NPY_NO_EXPORT PyMethodDef array_methods[];

typedef struct {
    PyArray_DTypeMeta *dtype;
    PyArrayMethodObject *method;
} DType_ArrayMethod;

typedef struct {
    DType_ArrayMethod **sort_impls;
    DType_ArrayMethod **argsort_impls;
} NpySortImpls;

extern NPY_NO_EXPORT NpySortImpls npy_sort_impls;


/*
 * Pathlib support, takes a borrowed reference and returns a new one.
 * The new object may be the same as the old.
 */
static inline PyObject *
NpyPath_PathlikeToFspath(PyObject *file)
{
    if (!PyObject_IsInstance(file, npy_static_pydata.os_PathLike)) {
        Py_INCREF(file);
        return file;
    }
    return PyObject_CallFunctionObjArgs(npy_static_pydata.os_fspath,
                                        file, NULL);
}

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_METHODS_H_ */
