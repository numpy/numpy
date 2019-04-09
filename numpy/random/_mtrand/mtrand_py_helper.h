#ifndef _MTRAND_PY_HELPER_H_
#define _MTRAND_PY_HELPER_H_

#include <Python.h>

static PyObject *empty_py_bytes(npy_intp length, void **bytes)
{
    PyObject *b;
#if PY_MAJOR_VERSION >= 3
    b = PyBytes_FromStringAndSize(NULL, length);
    if (b) {
        *bytes = PyBytes_AS_STRING(b);
    }
#else
    b = PyString_FromStringAndSize(NULL, length);
    if (b) {
        *bytes = PyString_AS_STRING(b);
    }
#endif
    return b;
}

#endif /* _MTRAND_PY_HELPER_H_ */
