#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/arrayobject.h"
#include "npy_pycompat.h"
#include "npy_import.h"
#include "multiarraymodule.h"
#include "strfuncs.h"

static void
npy_PyErr_SetStringChained(PyObject *type, const char *message)
{
    PyObject *exc, *val, *tb;

    PyErr_Fetch(&exc, &val, &tb);
    PyErr_SetString(type, message);
    npy_PyErr_ChainExceptionsCause(exc, val, tb);
}


/*NUMPY_API
 * Set the array print function to be a Python function.
 */
NPY_NO_EXPORT void
PyArray_SetStringFunction(PyObject *op, int repr)
{
    PyErr_SetString(PyExc_ValueError, "PyArray_SetStringFunction was removed");
}


NPY_NO_EXPORT PyObject *
array_repr(PyArrayObject *self)
{
    /*
     * We need to do a delayed import here as initialization on module load
     * leads to circular import problems.
     */
    if (npy_cache_import_runtime("numpy._core.arrayprint", "_default_array_repr",
                                 &npy_runtime_imports._default_array_repr) == -1) {
        npy_PyErr_SetStringChained(PyExc_RuntimeError,
                "Unable to configure default ndarray.__repr__");
        return NULL;
    }
    return PyObject_CallFunctionObjArgs(
            npy_runtime_imports._default_array_repr, self, NULL);
}


NPY_NO_EXPORT PyObject *
array_str(PyArrayObject *self)
{
    /*
     * We need to do a delayed import here as initialization on module load leads
     * to circular import problems.
     */
    if (npy_cache_import_runtime(
                "numpy._core.arrayprint", "_default_array_str",
                &npy_runtime_imports._default_array_str) == -1) {
        npy_PyErr_SetStringChained(PyExc_RuntimeError,
                "Unable to configure default ndarray.__str__");
        return NULL;
    }
    return PyObject_CallFunctionObjArgs(
            npy_runtime_imports._default_array_str, self, NULL);
}


NPY_NO_EXPORT PyObject *
array_format(PyArrayObject *self, PyObject *args)
{
    PyObject *format;
    if (!PyArg_ParseTuple(args, "O:__format__", &format))
        return NULL;

    /* 0d arrays - forward to the scalar type */
    if (PyArray_NDIM(self) == 0) {
        PyObject *item = PyArray_ToScalar(PyArray_DATA(self), self);
        PyObject *res;

        if (item == NULL) {
            return NULL;
        }
        res = PyObject_Format(item, format);
        Py_DECREF(item);
        return res;
    }
    /* Everything else - use the builtin */
    else {
        return PyObject_CallMethod(
            (PyObject *)&PyBaseObject_Type, "__format__", "OO",
            (PyObject *)self, format
        );
    }
}
