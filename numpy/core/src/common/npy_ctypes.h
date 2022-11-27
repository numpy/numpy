#ifndef NUMPY_CORE_SRC_COMMON_NPY_CTYPES_H_
#define NUMPY_CORE_SRC_COMMON_NPY_CTYPES_H_

#include <Python.h>

#include "npy_import.h"

/*
 * Check if a python type is a ctypes class.
 *
 * Works like the Py<type>_Check functions, returning true if the argument
 * looks like a ctypes object.
 *
 * This entire function is just a wrapper around the Python function of the
 * same name.
 */
static inline int
npy_ctypes_check(PyTypeObject *obj)
{
    static PyObject *py_func = NULL;
    PyObject *ret_obj;
    int ret;

    npy_cache_import("numpy.core._internal", "npy_ctypes_check", &py_func);
    if (py_func == NULL) {
        goto fail;
    }

    ret_obj = PyObject_CallFunctionObjArgs(py_func, (PyObject *)obj, NULL);
    if (ret_obj == NULL) {
        goto fail;
    }

    ret = PyObject_IsTrue(ret_obj);
    Py_DECREF(ret_obj);
    if (ret == -1) {
        goto fail;
    }

    return ret;

fail:
    /* If the above fails, then we should just assume that the type is not from
     * ctypes
     */
    PyErr_Clear();
    return 0;
}

#endif  /* NUMPY_CORE_SRC_COMMON_NPY_CTYPES_H_ */
