#ifndef _MKLRAND_PY_HELPER_H_
#define _MKLRAND_PY_HELPER_H_

#include <Python.h>

static PyObject *empty_py_bytes(npy_intp length, void **bytesVec)
{
    PyObject *b;
#if PY_MAJOR_VERSION >= 3
    b = PyBytes_FromStringAndSize(NULL, length);
    if (b) {
        *bytesVec = PyBytes_AS_STRING(b);
    }
#else
    b = PyString_FromStringAndSize(NULL, length);
    if (b) {
        *bytesVec = PyString_AS_STRING(b);
    }
#endif
    return b;
}

static char *py_bytes_DataPtr(PyObject *b)
{
#if PY_MAJOR_VERSION >= 3
    return PyBytes_AS_STRING(b);
#else
    return PyString_AS_STRING(b);
#endif
}

static int is_bytes_object(PyObject *b)
{
#if PY_MAJOR_VERSION >= 3
    return PyBytes_Check(b);
#else
    return PyString_Check(b);
#endif
}

#endif /* _MKLRAND_PY_HELPER_H_ */
