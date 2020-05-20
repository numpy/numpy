#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include <Python.h>
#include <numpy/arrayobject.h>

#include "npy_pycompat.h"

#include "strfuncs.h"

static PyObject *PyArray_StrFunction = NULL;
static PyObject *PyArray_ReprFunction = NULL;

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


/*
 * Extend string. On failure, returns NULL and leaves *strp alone.
 * XXX we do this in multiple places; time for a string library?
 */
static char *
extend_str(char **strp, Py_ssize_t n, Py_ssize_t *maxp)
{
    char *str = *strp;
    Py_ssize_t new_cap;

    if (n >= *maxp - 16) {
        new_cap = *maxp * 2;

        if (new_cap <= *maxp) {     /* overflow */
            return NULL;
        }
        str = PyArray_realloc(*strp, new_cap);
        if (str != NULL) {
            *strp = str;
            *maxp = new_cap;
        }
    }
    return str;
}


static int
dump_data(char **string, Py_ssize_t *n, Py_ssize_t *max_n, char *data, int nd,
          npy_intp const *dimensions, npy_intp const *strides, PyArrayObject* self)
{
    PyObject *op = NULL, *sp = NULL;
    char *ostring;
    npy_intp i, N, ret = 0;

#define CHECK_MEMORY do {                           \
        if (extend_str(string, *n, max_n) == NULL) {    \
            ret = -1;                               \
            goto end;                               \
        }                                           \
    } while (0)

    if (nd == 0) {
        if ((op = PyArray_GETITEM(self, data)) == NULL) {
            return -1;
        }
        sp = PyObject_Repr(op);
        if (sp == NULL) {
            ret = -1;
            goto end;
        }
        ostring = PyString_AsString(sp);
        N = PyString_Size(sp)*sizeof(char);
        *n += N;
        CHECK_MEMORY;
        memmove(*string + (*n - N), ostring, N);
    }
    else {
        CHECK_MEMORY;
        (*string)[*n] = '[';
        *n += 1;
        for (i = 0; i < dimensions[0]; i++) {
            if (dump_data(string, n, max_n,
                          data + (*strides)*i,
                          nd - 1, dimensions + 1,
                          strides + 1, self) < 0) {
                return -1;
            }
            CHECK_MEMORY;
            if (i < dimensions[0] - 1) {
                (*string)[*n] = ',';
                (*string)[*n+1] = ' ';
                *n += 2;
            }
        }
        CHECK_MEMORY;
        (*string)[*n] = ']';
        *n += 1;
    }

#undef CHECK_MEMORY

end:
    Py_XDECREF(op);
    Py_XDECREF(sp);
    return ret;
}


static PyObject *
array_repr_builtin(PyArrayObject *self, int repr)
{
    PyObject *ret;
    char *string;
    /* max_n initial value is arbitrary, dump_data will extend it */
    Py_ssize_t n = 0, max_n = PyArray_NBYTES(self) * 4 + 7;

    if ((string = PyArray_malloc(max_n)) == NULL) {
        return PyErr_NoMemory();
    }

    if (dump_data(&string, &n, &max_n, PyArray_DATA(self),
                  PyArray_NDIM(self), PyArray_DIMS(self),
                  PyArray_STRIDES(self), self) < 0) {
        PyArray_free(string);
        return NULL;
    }

    if (repr) {
        if (PyArray_ISEXTENDED(self)) {
            ret = PyUString_FromFormat("array(%s, '%c%d')",
                                       string,
                                       PyArray_DESCR(self)->type,
                                       PyArray_DESCR(self)->elsize);
        }
        else {
            ret = PyUString_FromFormat("array(%s, '%c')",
                                       string,
                                       PyArray_DESCR(self)->type);
        }
    }
    else {
        ret = PyUString_FromStringAndSize(string, n);
    }

    PyArray_free(string);
    return ret;
}


NPY_NO_EXPORT PyObject *
array_repr(PyArrayObject *self)
{
    PyObject *s;

    if (PyArray_ReprFunction == NULL) {
        s = array_repr_builtin(self, 1);
    }
    else {
        s = PyObject_CallFunctionObjArgs(PyArray_ReprFunction, self, NULL);
    }
    return s;
}


NPY_NO_EXPORT PyObject *
array_str(PyArrayObject *self)
{
    PyObject *s;

    if (PyArray_StrFunction == NULL) {
        s = array_repr_builtin(self, 0);
    }
    else {
        s = PyObject_CallFunctionObjArgs(PyArray_StrFunction, self, NULL);
    }
    return s;
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

