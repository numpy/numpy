#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/arrayobject.h"
#include "npy_pycompat.h"
#include "npy_import.h"
#include "strfuncs.h"

static PyObject *PyArray_StrFunction = NULL;
static PyObject *PyArray_ReprFunction = NULL;


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
    if (repr) {
        /* Dispose of previous callback */
        Py_XDECREF(PyArray_ReprFunction);
        /* Add a reference to new callback */
        Py_XINCREF(op);
        /* Remember new callback */
        PyArray_ReprFunction = op;
    }
    else {
        /* Dispose of previous callback */
        Py_XDECREF(PyArray_StrFunction);
        /* Add a reference to new callback */
        Py_XINCREF(op);
        /* Remember new callback */
        PyArray_StrFunction = op;
    }
}


NPY_NO_EXPORT PyObject *
array_repr(PyArrayObject *self)
{
    static PyObject *repr = NULL;

    if (PyArray_ReprFunction != NULL) {
        return PyObject_CallFunctionObjArgs(PyArray_ReprFunction, self, NULL);
    }

    /*
     * We need to do a delayed import here as initialization on module load
     * leads to circular import problems.
     */
    npy_cache_import("numpy.core.arrayprint", "_default_array_repr", &repr);
    if (repr == NULL) {
        npy_PyErr_SetStringChained(PyExc_RuntimeError,
                "Unable to configure default ndarray.__repr__");
        return NULL;
    }
    return PyObject_CallFunctionObjArgs(repr, self, NULL);
}


NPY_NO_EXPORT PyObject *
array_str(PyArrayObject *self)
{
    static PyObject *str = NULL;

    if (PyArray_StrFunction != NULL) {
        return PyObject_CallFunctionObjArgs(PyArray_StrFunction, self, NULL);
    }

    /*
     * We need to do a delayed import here as initialization on module load leads
     * to circular import problems.
     */
    npy_cache_import("numpy.core.arrayprint", "_default_array_str", &str);
    if (str == NULL) {
        npy_PyErr_SetStringChained(PyExc_RuntimeError,
                "Unable to configure default ndarray.__str__");
        return NULL;
    }
    return PyObject_CallFunctionObjArgs(str, self, NULL);
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
